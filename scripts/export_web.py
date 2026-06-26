"""导出交互网页所需数据 (outputs/web/data.js) 并拷贝 index.html。

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/export_web.py
"""
import base64
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config
from vgpnav.database import load_database
from vgpnav.occupancy import OccupancyGrid

cfg = Config()
WEB_SRC = os.path.join(cfg.proj_root, "web")
WEB_OUT = os.path.join(cfg.out_dir, "web")
os.makedirs(WEB_OUT, exist_ok=True)

db = load_database(cfg)
traj = np.load(os.path.join(cfg.db_dir, "trajectory.npz"))
_p4 = os.path.join(cfg.db_dir, "global_points_4cam.npz")
_pp = _p4 if os.path.exists(_p4) else os.path.join(cfg.db_dir, "global_points.npz")
pts = np.load(_pp)["points"]
print("点云来源:", os.path.basename(_pp))
qz = np.load(os.path.join(cfg.db_dir, "query.npz"))
meta = json.load(open(os.path.join(cfg.db_dir, "meta.json")))
res_path = os.path.join(cfg.out_dir, "query", "results.npz")
qr = np.load(res_path) if os.path.exists(res_path) else None


def cm(a):
    """米 -> 整数厘米 (压缩 JSON)。"""
    return [int(round(float(v) * 100)) for v in np.asarray(a).reshape(-1)]


# ---- 全局占据栅格 (供 JS 端 A*) ----
# 栅格范围用【轨迹 bbox + margin】, 而非点云全部范围 ——
# VGGT 在玻璃/反光/远景处会产生离群远点, 把点云范围撑大几倍(floor1 实测 45m 轨迹被撑到 460m),
# 导致栅格巨大、data.js 臃肿、地图视觉杂乱。改用轨迹包围盒定范围, 并裁剪范围外的离群点。
_tc = traj["centers"]
cx, cy = (_tc[:, 0].min() + _tc[:, 0].max()) / 2, (_tc[:, 1].min() + _tc[:, 1].max()) / 2
rng = max(_tc[:, 0].max() - _tc[:, 0].min(), _tc[:, 1].max() - _tc[:, 1].min()) / 2 + 5
_inr = (np.abs(pts[:, 0] - cx) <= rng) & (np.abs(pts[:, 1] - cy) <= rng)
print(f"点云裁剪: {len(pts)} -> {int(_inr.sum())} (去轨迹范围外离群点; 范围±{rng:.1f}m)")
pts = pts[_inr]
gx, gy = pts[:, 0], pts[:, 1]
RES = 0.25
occ = OccupancyGrid(resolution=RES, range_m=rng, center_xy=(cx, cy),
                    ground_band=cfg.ground_band_m, ceil=cfg.camera_height_m)
occ.integrate(pts, ground_z=0.0)
# 稳健判障: 障碍需 点密度≥6 且 垂直延展≥0.3m(真结构), 滤除 VGGT 悬浮噪声"虚构"障碍。
# 密度阈值 4->6: 4路回环PGO对齐后, 重访区真墙多趟点重合、密度高, 提阈值可滤掉对齐时
# 重合的VGGT单帧噪声点(虚假障碍), 而真墙(高密度)保留; DB可达仍209/209(轨迹free兜底)。
grid = occ.grid_robust(min_hits=6, min_vext=0.3)
# 形态学: 去掉孤立小障碍簇(残余噪声), 保留真墙大块 -> "没有障碍处不虚构"
from scipy import ndimage as _ndi
_lbl, _n = _ndi.label(grid == 2, structure=np.ones((3, 3)))
if _n > 0:
    _sz = _ndi.sum(np.ones_like(_lbl), _lbl, np.arange(1, _n + 1))
    _noise = np.isin(_lbl, np.where(_sz < 5)[0] + 1)
    grid[_noise] = 1
    print(f"占据: 稳健判障 + 清理噪声小簇 {int(_noise.sum())} 格")
# 机器人物理走过的轨迹必然可通行 -> 轨迹带强制标 free, 覆盖"漂移割裂"造成的虚构障碍。
# 左下角等重访区机器人绕行两趟, 单目 VGGT 两趟重建尺度/位姿错位, 同一面墙裂成两个副本
# 弥漫填满房间内部, 把走廊误判成障碍, 致 DB 节点被淹没、无法设起终点 (实测仅 21/209 可达)。
# 轨迹半径 0.75m (机器人尺度) 无条件覆盖障碍 -> DB 节点全在轨迹上, 必连通可设起终点。
_TR = 3
_disk = [(di, dj) for di in range(-_TR, _TR + 1) for dj in range(-_TR, _TR + 1)
         if di * di + dj * dj <= _TR ** 2]
_n_overwrite = 0
for c in traj["centers"]:
    ij = occ.world_to_cell(c[:2])
    j, i = int(ij[0]), int(ij[1])
    for di, dj in _disk:
        ii, jj = i + di, j + dj
        if 0 <= ii < grid.shape[0] and 0 <= jj < grid.shape[1]:
            if grid[ii, jj] == 2:
                _n_overwrite += 1
            grid[ii, jj] = 1
print(f"占据: 轨迹走廊强制可通行, 覆盖虚构障碍 {_n_overwrite} 格 -> DB 节点可设起终点")
rows = ["".join(map(str, row)) for row in grid.tolist()]

# ---- 点云下采样 (按高度着色) ----
h = pts[:, 2]
m = (h > -0.2) & (h < 2.2)
P = pts[m]
if len(P) > 28000:
    P = P[np.random.default_rng(0).choice(len(P), 28000, replace=False)]

# DB 缩略图改用 web/thumbs/ 下的 jpg 文件(export_cams 生成), 不再 base64 内联进 data.js。
# 内联会让 data.js 膨胀(358节点≈9MB, 加载慢); 引用文件后 data.js 只剩几何数据(~1-2MB)。

data = {
    "meta": {
        "n_traj": int(meta["n_traj"]), "n_db": int(meta["n_db"]),
        "n_query": int(meta["n_query"]),
        "scale": round(float(meta["scale"]), 3),
        "camera_height": float(meta["camera_height_m"]),
        "loc_t": (round(float(np.median(qr["errs"][:, 0])), 2) if qr is not None else None),
        "loc_r": (round(float(np.median(qr["errs"][:, 1])), 2) if qr is not None else None),
        "bounds": [float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max())],
    },
    "traj": cm(traj["centers"][:, :2]),
    "dbx": cm(db.centers[:, 0]), "dby": cm(db.centers[:, 1]),
    "dbdx": [int(round(float(v) * 1000)) for v in db.view_dirs[:, 0]],
    "dbdy": [int(round(float(v) * 1000)) for v in db.view_dirs[:, 1]],
    "dbfi": [int(v) for v in db.frame_idx],
    "px": cm(P[:, 0]), "py": cm(P[:, 1]), "ph": cm(P[:, 2]),
    "occ": {"res": RES, "ox": float(occ.origin[0]), "oy": float(occ.origin[1]),
            "nx": int(grid.shape[1]), "ny": int(grid.shape[0]), "rows": rows},
    "cam_layout": cfg.cam_layout,   # 4环视2x2物理布局 [(方位,相机号)] 左上,右上,左下,右下
    "dataset": cfg.dataset,
}
nn_path = os.path.join(cfg.db_dir, "node_names.json")
if os.path.exists(nn_path):
    data["dbname"] = json.load(open(nn_path))
    print(f"已加载节点名称 {len([x for x in data['dbname'] if x])} 个")
if qr is not None:
    data["qgx"] = cm(qr["q_gt"][:, 0, 3])
    data["qgy"] = cm(qr["q_gt"][:, 1, 3])
    data["qex"] = cm(qr["est_poses"][:, 0, 3])
    data["qey"] = cm(qr["est_poses"][:, 1, 3])
    data["qerr"] = [round(float(v), 2) for v in qr["errs"][:, 0]]

with open(os.path.join(WEB_OUT, "data.js"), "w") as f:
    f.write("window.VGPDATA = ")
    json.dump(data, f, separators=(",", ":"))
    f.write(";\n")
# 各数据集页 = 纯导航 (nav.html; 供总应用导航tab嵌入, 也可单独访问 /{ds}/web/)
_nav = os.path.join(WEB_SRC, "nav.html")
shutil.copy(_nav if os.path.exists(_nav) else os.path.join(WEB_SRC, "index.html"),
            os.path.join(WEB_OUT, "index.html"))
# 总入口 = 多地图三tab总应用 (portal.html → outputs/index.html)
_portal = os.path.join(WEB_SRC, "portal.html")
if os.path.exists(_portal):
    shutil.copy(_portal, os.path.join(os.path.dirname(cfg.out_dir.rstrip("/")), "index.html"))
sz = os.path.getsize(os.path.join(WEB_OUT, "data.js")) / 1e6
print(f"导出完成 -> {WEB_OUT} (data.js {sz:.1f} MB, 点{len(P)}, 栅格{grid.shape}); 总应用 -> outputs/index.html")
