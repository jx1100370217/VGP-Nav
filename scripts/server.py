"""VGP-Nav 可视化后端服务 (Flask)。

在原静态 http.server 之上增加:
  · 建图流程: 各步状态 / 触发某步(subprocess 跑建图脚本) / 实时进度日志
  · DB 节点管理: 增 / 删 / 改名 / 查 (读写 db.npz + node_names.json)
仍然托管 outputs/ 下的总入口与各数据集 web。

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/server.py   # 0.0.0.0:8199
"""
import json
import os
import subprocess
import sys
import threading

import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import _PROJ
from vgpnav.datasets import DATASETS

OUTPUTS = os.path.join(_PROJ, "outputs")
PY = sys.executable
PORT = int(os.environ.get("VGPNAV_PORT", "8199"))

app = Flask(__name__)

# 建图流程 (顺序); produces=该步主产物(判断是否完成), viz=可视化产物(png)
PIPELINE = [
    {"key": "build_database", "label": "① 建库 · VGGT滑窗链式+地面锚定", "produces": "db/trajectory.npz", "viz": "database_overview.png", "slow": True},
    {"key": "compute_4cam_desc", "label": "② 4路环视描述子 · 回环检测用", "produces": "db/allframe_desc_4cam.npy", "viz": None, "slow": True},
    {"key": "pgo_fix", "label": "③ 分层回环 PGO · 消漂移/闭环", "produces": "db/trajectory_pgo.npz", "viz": "query/pgo_before_after.png", "args": ["--sim", "0.70", "--dmax", "4"]},
    {"key": "apply_pgo", "label": "④ 写回校正位姿", "produces": "db/trajectory_orig.npz", "viz": None},
    {"key": "build_4cam_points", "label": "⑤ 360°融合点云重建", "produces": "db/global_points_4cam.npz", "viz": "fourcam_test.png", "slow": True},
    {"key": "export_cams", "label": "⑥ 4路环视缩略图", "produces": "web/thumbs", "viz": None},
    {"key": "run_query", "label": "⑦ 定位评测 (端到端)", "produces": "query/results.npz", "viz": "query/localization_overview.png", "slow": True},
    {"key": "export_web", "label": "⑧ 导出最终地图 (占据+导航)", "produces": "web/data.js", "viz": "query/global_occupancy.png"},
]

LABELS = {"Mapping_C8": "深港国际C8", "ChuangfuTower": "创富大厦28楼",
          "Mappingdata_C7": "深港国际C7", "Mappingdata_Firstfloor": "深港国际1楼"}

_jobs = {}        # ds -> {step, log[], done, rc, proc}
_lock = threading.Lock()


def out_dir(ds):
    return os.path.join(OUTPUTS, ds)


def db_dir(ds):
    return os.path.join(OUTPUTS, ds, "db")


def _names_path(ds):
    return os.path.join(db_dir(ds), "node_names.json")


def _load_names(ds, n):
    p = _names_path(ds)
    if os.path.exists(p):
        try:
            names = json.load(open(p))
            if isinstance(names, list) and len(names) == n:
                return names
        except Exception:
            pass
    return [""] * n


def _save_names(ds, names):
    json.dump(names, open(_names_path(ds), "w"), ensure_ascii=False)


# ───────────────────────── 静态 ─────────────────────────
@app.route("/")
def root():
    idx = os.path.join(OUTPUTS, "index.html")
    if not os.path.exists(idx):
        return "总入口未生成, 请先跑 scripts/gen_portal.py", 404
    return send_from_directory(OUTPUTS, "index.html")


@app.route("/<path:p>")
def static_file(p):
    full = os.path.join(OUTPUTS, p)
    if os.path.isdir(full):
        p = p.rstrip("/") + "/index.html"
    return send_from_directory(OUTPUTS, p)


# ───────────────────────── 数据集 ─────────────────────────
@app.route("/api/datasets")
def api_datasets():
    return jsonify([{"name": n, "label": LABELS.get(n, n),
                     "built": os.path.exists(os.path.join(out_dir(n), "web", "data.js"))}
                    for n in DATASETS])


# ──────────────────── 建图流程: 状态/运行/日志 ────────────────────
@app.route("/api/pipeline/<ds>/status")
def api_pipeline_status(ds):
    if ds not in DATASETS:
        abort(404)
    steps = []
    for s in PIPELINE:
        p = os.path.join(out_dir(ds), s["produces"])
        done = os.path.exists(p)
        steps.append({"key": s["key"], "label": s["label"], "viz": s["viz"],
                      "slow": s.get("slow", False), "done": done,
                      "mtime": os.path.getmtime(p) if done else None})
    job = _jobs.get(ds)
    return jsonify({"steps": steps,
                    "running": job["step"] if job and not job["done"] else None})


def _run_job(ds, cmd, env):
    job = _jobs[ds]
    try:
        proc = subprocess.Popen(cmd, cwd=_PROJ, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        job["proc"] = proc
        for line in proc.stdout:
            line = line.rstrip()
            if not line or "Warning" in line or "warn" in line:
                continue
            with _lock:
                job["log"].append(line)
                if len(job["log"]) > 400:
                    del job["log"][:-400]
        proc.wait()
        job["rc"] = proc.returncode
    except Exception as e:
        job["log"].append(f"[错误] {e}")
        job["rc"] = -1
    finally:
        job["done"] = True


@app.route("/api/pipeline/<ds>/run/<step>", methods=["POST"])
def api_run_step(ds, step):
    if ds not in DATASETS:
        abort(404)
    spec = next((s for s in PIPELINE if s["key"] == step), None)
    if not spec:
        abort(404)
    job = _jobs.get(ds)
    if job and not job["done"]:
        return jsonify({"error": f"已有任务运行中: {job['step']}"}), 409
    cmd = [PY, os.path.join("scripts", step + ".py")] + spec.get("args", [])
    env = dict(os.environ, VGPNAV_DATASET=ds)
    _jobs[ds] = {"step": step, "log": [], "done": False, "rc": None, "proc": None}
    threading.Thread(target=_run_job, args=(ds, cmd, env), daemon=True).start()
    return jsonify({"started": True, "step": step})


@app.route("/api/pipeline/<ds>/log")
def api_log(ds):
    job = _jobs.get(ds)
    if not job:
        return jsonify({"log": [], "done": True, "step": None, "rc": None})
    with _lock:
        log = list(job["log"])
    return jsonify({"log": log, "done": job["done"], "step": job["step"], "rc": job.get("rc")})


# ───────────────────────── DB 节点 CRUD ─────────────────────────
@app.route("/api/db/<ds>")
def api_db_list(ds):
    f = os.path.join(db_dir(ds), "db.npz")
    if not os.path.exists(f):
        return jsonify({"nodes": [], "traj_frames": []})
    d = np.load(f)
    fi = [int(x) for x in d["frame_idx"]]
    centers = d["centers"]
    names = _load_names(ds, len(fi))
    nodes = [{"idx": i, "frame_idx": fi[i], "name": names[i],
              "x": round(float(centers[i][0]), 2), "y": round(float(centers[i][1]), 2)}
             for i in range(len(fi))]
    trf = os.path.join(db_dir(ds), "trajectory.npz")
    traj_frames = [int(x) for x in np.load(trf)["frame_idx"]] if os.path.exists(trf) else []
    return jsonify({"nodes": nodes, "traj_frames": traj_frames,
                    "cam_layout": DATASETS[ds].cam_layout, "label": LABELS.get(ds, ds)})


@app.route("/api/db/<ds>/<int:idx>", methods=["PUT"])
def api_db_update(ds, idx):
    f = os.path.join(db_dir(ds), "db.npz")
    if not os.path.exists(f):
        abort(404)
    n = len(np.load(f)["frame_idx"])
    if not (0 <= idx < n):
        abort(404)
    names = _load_names(ds, n)
    names[idx] = (request.json or {}).get("name", "")
    _save_names(ds, names)
    return jsonify({"ok": True})


@app.route("/api/db/<ds>/<int:idx>", methods=["DELETE"])
def api_db_delete(ds, idx):
    f = os.path.join(db_dir(ds), "db.npz")
    if not os.path.exists(f):
        abort(404)
    d = dict(np.load(f))
    n = len(d["frame_idx"])
    if not (0 <= idx < n):
        abort(404)
    for k in list(d):
        a = d[k]
        if isinstance(a, np.ndarray) and a.ndim >= 1 and a.shape[0] == n:
            d[k] = np.delete(a, idx, axis=0)
    np.savez(f, **d)
    names = _load_names(ds, n)
    del names[idx]
    _save_names(ds, names)
    return jsonify({"ok": True, "n": n - 1})


@app.route("/api/db/<ds>", methods=["POST"])
def api_db_add(ds):
    body = request.json or {}
    frame_idx = int(body.get("frame_idx", -1))
    name = body.get("name", "")
    dbf = os.path.join(db_dir(ds), "db.npz")
    trf = os.path.join(db_dir(ds), "trajectory.npz")
    if not (os.path.exists(dbf) and os.path.exists(trf)):
        abort(404)
    d = dict(np.load(dbf))
    tr = np.load(trf)
    tfi = [int(x) for x in tr["frame_idx"]]
    if frame_idx not in tfi:
        return jsonify({"error": "该帧不在轨迹中"}), 400
    if frame_idx in [int(x) for x in d["frame_idx"]]:
        return jsonify({"error": "该帧已是 DB 节点"}), 400
    ti = tfi.index(frame_idx)
    # 描述子: 取该帧 cam1 的 4路描述子(若有), 否则用零向量(检索时该点权重低)
    desc_dim = d["desc"].shape[1]
    descf = os.path.join(db_dir(ds), "allframe_desc_4cam.npy")
    if os.path.exists(descf):
        dd = np.load(descf)[ti, 0].astype(d["desc"].dtype)
        new_desc = dd / (np.linalg.norm(dd) + 1e-9)
    else:
        new_desc = np.zeros(desc_dim, dtype=d["desc"].dtype)
    d["frame_idx"] = np.append(d["frame_idx"], frame_idx)
    for k, src in (("poses", "poses"), ("centers", "centers"), ("view_dirs", "view_dirs")):
        if k in d and src in tr.files:
            d[k] = np.concatenate([d[k], tr[src][ti:ti + 1]], axis=0)
    d["desc"] = np.concatenate([d["desc"], new_desc[None]], axis=0)
    np.savez(dbf, **d)
    names = _load_names(ds, len(d["frame_idx"]) - 1)
    names.append(name)
    _save_names(ds, names)
    return jsonify({"ok": True, "n": len(d["frame_idx"])})


if __name__ == "__main__":
    print(f"VGP-Nav 服务: http://0.0.0.0:{PORT}/  (outputs={OUTPUTS})")
    app.run(host="0.0.0.0", port=PORT, threaded=True)
