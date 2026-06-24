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
gx, gy = pts[:, 0], pts[:, 1]
cx, cy = (gx.min() + gx.max()) / 2, (gy.min() + gy.max()) / 2
rng = max(gx.max() - gx.min(), gy.max() - gy.min()) / 2 + 2
RES = 0.25
occ = OccupancyGrid(resolution=RES, range_m=rng, center_xy=(cx, cy),
                    ground_band=cfg.ground_band_m, ceil=cfg.camera_height_m,
                    occ_min_hits=3)
occ.integrate(pts, ground_z=0.0)
grid = occ.grid()
# 机器人走过的轨迹一定可通行 -> 把轨迹周围一段带标为 free (反映真实可通行走廊,
# 保证自由空间连通、加宽骨架, 使 A* 贴着证实可通行的路线走)
_TR = 3  # 轨迹自由带半径 (cell), ~0.75m -> 走廊宽 ~1.75m (贴合实际走廊, 让A*紧贴已证实可通行区)
_yy, _xx = np.ogrid[-_TR:_TR + 1, -_TR:_TR + 1]
_disk = np.argwhere(_xx ** 2 + _yy ** 2 <= _TR ** 2) - _TR
for c in traj["centers"]:
    ij = occ.world_to_cell(c[:2])
    j, i = int(ij[0]), int(ij[1])
    for di, dj in _disk:
        ii, jj = i + int(di), j + int(dj)
        if 0 <= ii < grid.shape[0] and 0 <= jj < grid.shape[1]:
            grid[ii, jj] = 1
rows = ["".join(map(str, row)) for row in grid.tolist()]

# ---- 点云下采样 (按高度着色) ----
h = pts[:, 2]
m = (h > -0.2) & (h < 2.2)
P = pts[m]
if len(P) > 28000:
    P = P[np.random.default_rng(0).choice(len(P), 28000, replace=False)]

# ---- DB 缩略图 (base64) ----
print(f"生成 {len(db.frame_idx)} 张 DB 缩略图...")
thumbs = []
for fi in db.frame_idx:
    img = db.image(int(fi))
    th = cv2.resize(img, (384, 288), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", th, [cv2.IMWRITE_JPEG_QUALITY, 72])
    thumbs.append("data:image/jpeg;base64," + base64.b64encode(buf).decode())

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
    "thumbs": thumbs,
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
shutil.copy(os.path.join(WEB_SRC, "index.html"),
           os.path.join(WEB_OUT, "index.html"))
sz = os.path.getsize(os.path.join(WEB_OUT, "data.js")) / 1e6
print(f"导出完成 -> {WEB_OUT} (data.js {sz:.1f} MB, 点{len(P)}, 栅格{grid.shape})")
