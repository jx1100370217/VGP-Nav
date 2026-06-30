"""建图质量评估: 对每个数据集算一套全面指标 + good/warn/bad 评分, 输出 outputs/<ds>/quality.json。

指标分 5 类:
  A 覆盖度    : DB节点数 / 轨迹长度 / 地图范围 / 平均DB间距 / 点云数
  B 轨迹一致性: 相邻帧位移中位/p95 (建图错位会有突跳) / scale符号
  C 回环闭合  : 重访对数(空间近+时序远) / 重访空间距离中位 (pgo后应小=闭合好)
  D 定位精度  : 平移误差中位 / <1m / <2m 比例 / 旋转误差中位 (results.npz)
  E 占据连通  : free/occ/unknown 占比 / 最大连通区占比 / DB节点可达率

用法: VGPNAV_DATASET=<ds> python scripts/assess_quality.py   (或不设, 遍历全部)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config
from vgpnav.datasets import DATASETS


def grade(val, good, warn, higher_better=True):
    if val is None:
        return "na"
    if higher_better:
        return "good" if val >= good else ("warn" if val >= warn else "bad")
    return "good" if val <= good else ("warn" if val <= warn else "bad")


def parse_datajs(path):
    txt = open(path).read()
    js = txt[txt.index("{"):txt.rindex("}") + 1]
    return json.loads(js)


def assess(ds):
    os.environ["VGPNAV_DATASET"] = ds
    cfg = Config(dataset=ds)
    db_dir, out_dir = cfg.db_dir, cfg.out_dir
    M = {}
    meta = json.load(open(os.path.join(db_dir, "meta.json")))
    M["数据集"] = ds
    M["相机高度m"] = meta["camera_height_m"]
    M["scale"] = round(meta["scale"], 4)
    M["xy尺度校正"] = meta.get("xy_scale_correction", 1.0)

    # ---- A 覆盖度 ----
    traj = np.load(os.path.join(db_dir, "trajectory.npz"))
    C = traj["centers"]
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    M["A_DB节点数"] = int(meta["n_db"])
    M["A_query数"] = int(meta["n_query"])
    M["A_轨迹长度m"] = round(float(seg.sum()), 1)
    M["A_地图范围m"] = [round(float(C[:, 0].ptp()), 1), round(float(C[:, 1].ptp()), 1)]
    db = np.load(os.path.join(db_dir, "db.npz"))
    DBC = db["centers"]
    dbseg = np.linalg.norm(np.diff(DBC, axis=0), axis=1)
    M["A_平均DB间距m"] = round(float(np.median(dbseg)), 2)
    p4 = os.path.join(db_dir, "global_points_4cam.npz")
    M["A_点云数"] = int(np.load(p4)["points"].shape[0]) if os.path.exists(p4) else None

    # ---- B 轨迹一致性 (相邻帧位移; 建图错位/跳变会拉大 p95) ----
    M["B_帧间位移中位m"] = round(float(np.median(seg)), 2)
    M["B_帧间位移p95m"] = round(float(np.percentile(seg, 95)), 2)
    M["B_位移突跳比"] = round(float(np.percentile(seg, 95) / (np.median(seg) + 1e-6)), 1)
    M["B_scale正"] = bool(meta["scale"] > 0)

    # ---- C 回环闭合 (视觉相似[4路交叉描述子]的远时序帧对 = 同地点不同时经过, 建图空间距离=闭合质量) ----
    desc_p = os.path.join(db_dir, "allframe_desc_4cam.npy")
    tfi = np.array([int(x) for x in traj["frame_idx"]])
    if os.path.exists(desc_p):
        D4 = np.load(desc_p).astype(np.float32)               # (Nf,4,dim)
        D4 /= (np.linalg.norm(D4, axis=2, keepdims=True) + 1e-9)
        Nf = len(D4)
        flat = D4.reshape(Nf * 4, -1)
        S = (flat @ flat.T).reshape(Nf, 4, Nf, 4).max(axis=(1, 3))   # (Nf,Nf) 4路交叉最大相似
        ai, bi = np.triu_indices(Nf, k=1)
        mask = (np.abs(tfi[ai] - tfi[bi]) > 40) & (S[ai, bi] > 0.85)  # 视觉同地点 + 时序远
        ld = np.linalg.norm(C[ai[mask]] - C[bi[mask]], axis=1)
        M["C_视觉回环对数"] = int(mask.sum())
        M["C_回环空间距离中位m"] = round(float(np.median(ld)), 2) if len(ld) else None
        M["C_回环空间距离p90m"] = round(float(np.percentile(ld, 90)), 2) if len(ld) else None
    else:
        M["C_视觉回环对数"] = M["C_回环空间距离中位m"] = M["C_回环空间距离p90m"] = None

    # ---- D 定位精度 (results.npz: errs[:,0]=平移m, errs[:,1]=旋转deg) ----
    rp = os.path.join(out_dir, "query", "results.npz")
    if os.path.exists(rp):
        errs = np.load(rp)["errs"]
        t, r = errs[:, 0], errs[:, 1]
        diag = float(np.hypot(C[:, 0].ptp(), C[:, 1].ptp()))   # 场景对角线
        M["D_平移中位m"] = round(float(np.median(t)), 2)
        M["D_相对误差%"] = round(float(np.median(t) / diag * 100), 2)   # 误差/场景尺寸, 跨尺度可比
        M["D_平移<1m比例"] = round(float((t < 1).mean()), 2)
        M["D_平移<2m比例"] = round(float((t < 2).mean()), 2)
        M["D_旋转中位deg"] = round(float(np.median(r)), 1)
    else:
        M["D_平移中位m"] = M["D_相对误差%"] = M["D_平移<1m比例"] = M["D_平移<2m比例"] = M["D_旋转中位deg"] = None

    # ---- E 占据连通 (从 data.js 的 occ 栅格) ----
    djs = os.path.join(out_dir, "web", "data.js")
    if os.path.exists(djs):
        data = parse_datajs(djs)
        occ = data["occ"]
        nx, ny, res, ox, oy = occ["nx"], occ["ny"], occ["res"], occ["ox"], occ["oy"]
        g = np.array([[int(ch) for ch in row] for row in occ["rows"]], dtype=np.int8)  # ny×nx, 0未知/1free/2占
        tot = g.size
        M["E_free占比"] = round(float((g == 1).mean()), 3)
        M["E_占据占比"] = round(float((g == 2).mean()), 3)
        M["E_未知占比"] = round(float((g == 0).mean()), 3)
        # 最大连通可通行区 (非障碍格 4连通 BFS)
        from collections import deque
        passable = g != 2
        seen = np.zeros_like(g, dtype=bool)
        best = 0
        for si in range(ny):
            for sj in range(nx):
                if passable[si, sj] and not seen[si, sj]:
                    q = deque([(si, sj)])
                    seen[si, sj] = True
                    sz = 0
                    comp = []
                    while q:
                        i, j = q.popleft()
                        sz += 1
                        comp.append((i, j))
                        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < ny and 0 <= nj < nx and passable[ni, nj] and not seen[ni, nj]:
                                seen[ni, nj] = True
                                q.append((ni, nj))
                    if sz > best:
                        best = sz
                        best_comp = comp
        main = np.zeros_like(g, dtype=bool)
        for i, j in best_comp:
            main[i, j] = True
        M["E_最大连通占可通行比"] = round(float(best / max(1, int(passable.sum()))), 3)
        # DB 可达率 (DB中心落在最大连通区)
        reach = 0
        for x, y in DBC[:, :2]:
            j = int((x - ox) / res)
            i = int((y - oy) / res)
            if 0 <= i < ny and 0 <= j < nx and main[i, j]:
                reach += 1
        M["E_DB可达率"] = round(reach / len(DBC), 3)
    else:
        for k in ["E_free占比", "E_占据占比", "E_未知占比", "E_最大连通占可通行比", "E_DB可达率"]:
            M[k] = None

    # ---- 评分 (关键指标) ----
    M["评分"] = {
        "轨迹一致性(位移突跳比)": grade(M["B_位移突跳比"], 3, 6, higher_better=False),
        "scale符号": "good" if M["B_scale正"] else "bad",
        "回环闭合(视觉回环空间距离)": grade(M.get("C_回环空间距离中位m"), 2.0, 5.0, higher_better=False) if M.get("C_回环空间距离中位m") is not None else "na",
        "定位相对误差(误差/场景)": grade(M.get("D_相对误差%"), 1.5, 4.0, higher_better=False),
        "DB可达率": grade(M["E_DB可达率"], 0.95, 0.8),
        "最大连通占比": grade(M["E_最大连通占可通行比"], 0.9, 0.6),
    }
    bad = [k for k, v in M["评分"].items() if v == "bad"]
    warn = [k for k, v in M["评分"].items() if v == "warn"]
    M["总评"] = "差(有bad项)" if bad else ("一般(有warn项)" if warn else "良好")
    M["待优化项"] = bad + warn

    json.dump(M, open(os.path.join(out_dir, "quality.json"), "w"), ensure_ascii=False, indent=1)
    return M


if __name__ == "__main__":
    dss = [os.environ["VGPNAV_DATASET"]] if os.environ.get("VGPNAV_DATASET") else list(DATASETS)
    out = {}
    for ds in dss:
        try:
            out[ds] = assess(ds)
            print(f"=== {ds}: {out[ds]['总评']} ===")
            for k, v in out[ds].items():
                if k not in ("评分", "数据集"):
                    print(f"  {k}: {v}")
            print("  评分:", out[ds]["评分"])
        except Exception as e:
            import traceback
            print(f"{ds}: 失败 {e}")
            traceback.print_exc()
