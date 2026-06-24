"""端到端运行 VGP-Nav: 对 held-out query 定位+感知, 评测+可视化, A* demo。

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/run_query.py
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vgpnav.viz import setup_cjk_font

setup_cjk_font()

from vgpnav import geom
from vgpnav.config import Config
from vgpnav.database import load_database, list_frame_files
from vgpnav.occupancy import OccupancyGrid
from vgpnav.pipeline import VGPNav
from vgpnav.planner import astar
from vgpnav.vggt_runner import VGGTRunner
from vgpnav.vpr import make_vpr

cfg = Config()
OUT = os.path.join(cfg.out_dir, "query")
os.makedirs(OUT, exist_ok=True)

db = load_database(cfg)
q = np.load(os.path.join(cfg.db_dir, "query.npz"))
q_idx, q_gt = q["frame_idx"], q["poses_gt"]
traj = np.load(os.path.join(cfg.db_dir, "trajectory.npz"))
gpts = np.load(os.path.join(cfg.db_dir, "global_points.npz"))["points"]
files = list_frame_files(cfg)

print(f"DB {len(db.frame_idx)} 帧, query {len(q_idx)} 帧")
vpr = make_vpr(cfg)
vggt = VGGTRunner(cfg)
nav = VGPNav(cfg, db, vpr, vggt)

# 4 路环视 query 加载器 (定位统一 4 路: cam1 位姿锚 + cam2/3/4 补 360°几何/检索)
from vgpnav.undistort import PinholeUndistorter, load_camera_params
_QCAMS = ["camera_1", "camera_2", "camera_3", "camera_4"]
_QUND = {c: PinholeUndistorter(*load_camera_params(cfg.cam_params, c),
                               cfg.undist_w, cfg.undist_h, cfg.undist_hfov)
         for c in _QCAMS}


def load_query_cams(fi):
    ts = os.path.basename(db.files[fi]).split("_camera_")[0]
    imgs = []
    for c in _QCAMS:
        im = cv2.imread(os.path.join(cfg.data_dir, f"{ts}_{c}.jpg"))
        if im is not None:
            imgs.append(_QUND[c].undistort(im))
    return imgs if imgs else [db.image(fi)]


# ---------- 逐 query 定位 + 感知 ----------
results, errs = [], []
for n, fi in enumerate(q_idx):
    r = nav.run(load_query_cams(int(fi)))
    gt = q_gt[n]
    te = float(np.linalg.norm(r.T_wc[:3, 3] - gt[:3, 3]))
    re = float(np.degrees(geom.angle_between(r.T_wc[:3, :3], gt[:3, :3])))
    errs.append((te, re))
    results.append(r)
    print(f"  q{n:02d} 帧{fi}: 平移误差={te:.2f} m, 旋转误差={re:.1f} deg, "
          f"参考{len(r.ref_ids)}, 尺度={r.scale:.2f}")

errs = np.array(errs)
print(f"\n定位误差: 平移 中位={np.median(errs[:,0]):.2f} 均值={np.mean(errs[:,0]):.2f} m; "
      f"旋转 中位={np.median(errs[:,1]):.1f} 均值={np.mean(errs[:,1]):.1f} deg")
print(f"平移误差 <1m: {int((errs[:,0]<1).sum())}/{len(errs)}, "
      f"<2m: {int((errs[:,0]<2).sum())}/{len(errs)}")

# ---------- 图1: 定位总览 ----------
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
ax = axs[0]
ob = gpts[(gpts[:, 2] > 0.2) & (gpts[:, 2] < 2.0)]
ax.scatter(ob[:, 0], ob[:, 1], s=0.2, c="gray", alpha=0.2)
ax.plot(traj["centers"][:, 0], traj["centers"][:, 1], "-", c="lightblue", lw=1)
ax.scatter(db.centers[:, 0], db.centers[:, 1], c="g", s=6, label="DB")
for n in range(len(q_idx)):
    g = q_gt[n][:2, 3]
    e = results[n].T_wc[:2, 3]
    ax.plot([g[0], e[0]], [g[1], e[1]], "-r", lw=0.8)
ax.scatter(q_gt[:, 0, 3], q_gt[:, 1, 3], c="k", marker="*", s=80, label="伪GT")
ax.scatter([r.T_wc[0, 3] for r in results], [r.T_wc[1, 3] for r in results],
           c="b", marker="x", s=40, label="估计")
ax.set_aspect("equal")
ax.legend()
ax.set_title("定位: 估计 vs 伪GT (红线=误差)")
ax = axs[1]
ax.hist(errs[:, 0], bins=15, color="steelblue", edgecolor="k")
ax.axvline(np.median(errs[:, 0]), color="r", label=f"中位={np.median(errs[:,0]):.2f}m")
ax.set_xlabel("平移误差 (m)")
ax.set_ylabel("query 数")
ax.legend()
ax.set_title("定位平移误差分布")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "localization_overview.png"), dpi=110)
plt.close()

# ---------- 图2: 几例 query 的检索 + 局部占据 ----------
examples = list(range(0, len(q_idx), max(1, len(q_idx) // 4)))[:4]
fig, axs = plt.subplots(len(examples), 5, figsize=(20, 4 * len(examples)))
if len(examples) == 1:
    axs = axs[None, :]
for row, n in enumerate(examples):
    r = results[n]
    qi = int(q_idx[n])
    axs[row, 0].imshow(cv2.cvtColor(db.image(qi), cv2.COLOR_BGR2RGB))
    axs[row, 0].set_title(f"query 帧{qi}")
    for c in range(2):           # 参考 1,2
        if c < len(r.ref_ids):
            rid = int(db.frame_idx[r.ref_ids[c]])
            axs[row, c + 1].imshow(cv2.cvtColor(db.image(rid), cv2.COLOR_BGR2RGB))
            axs[row, c + 1].set_title(f"参考{c+1} 帧{rid}")
        axs[row, c + 1].axis("off")
    axs[row, 0].axis("off")
    # 鸟瞰度量点云 (按相对地面高度着色)
    P = r.P_world
    h = P[:, 2] - r.ground_z
    ax = axs[row, 3]
    ax.scatter(P[:, 0], P[:, 1], c=np.clip(h, 0, 1.5), s=1, cmap="viridis")
    ct = r.T_wc[:2, 3]
    fwd = r.T_wc[:3, :3] @ np.array([0, 0, 1.0])
    ax.scatter([ct[0]], [ct[1]], c="r", s=60, marker="^")
    ax.arrow(ct[0], ct[1], fwd[0], fwd[1], color="r", width=0.06)
    ax.set_aspect("equal")
    ax.set_title("鸟瞰度量点云(高度色)")
    # 占据图
    ax = axs[row, 4]
    ax.imshow(r.occ.grid(), origin="lower", cmap="gray_r", vmin=0, vmax=2)
    ax.set_title(f"局部占据 尺度{r.scale:.2f}")
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "query_montages.png"), dpi=100)
plt.close()

# ---------- 图3: 局部占据 + A* (论文式 FoV 局部规划) ----------
order = np.argsort(errs[:, 0])
demo_qs = list(order[:2])           # 取定位最准的两帧做 demo
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
for col, n in enumerate(demo_qs):
    r = results[n]
    grid = r.occ.grid().copy()
    ct = r.T_wc[:2, 3]
    fwd = r.T_wc[:3, :3] @ np.array([0, 0, 1.0])
    fwd2 = fwd[:2] / (np.linalg.norm(fwd[:2]) + 1e-9)
    s_ij = r.occ.world_to_cell(ct)[::-1]
    # 机器人脚下标为可通行
    ii, jj = int(s_ij[0]), int(s_ij[1])
    sub = grid[max(0, ii - 2):ii + 3, max(0, jj - 2):jj + 3]
    sub[sub == 0] = 1
    # 目标: 前方最远的非障碍点 (≤3.5m)
    g_ij = None
    for d in np.arange(3.5, 0.6, -0.3):
        cand = r.occ.world_to_cell(ct + fwd2 * d)[::-1]
        if (0 <= cand[0] < grid.shape[0] and 0 <= cand[1] < grid.shape[1]
                and grid[cand[0], cand[1]] != 2):
            g_ij = cand
            break
    ax = axs[col]
    ax.imshow(grid, origin="lower", cmap="gray_r", vmin=0, vmax=2)
    ax.scatter([s_ij[1]], [s_ij[0]], c="g", s=120, marker="o", label="start(机器人)")
    if g_ij is not None:
        path = astar(grid, s_ij, g_ij, robot_radius_cells=2)
        ax.scatter([g_ij[1]], [g_ij[0]], c="r", s=160, marker="*", label="goal(前方)")
        if path:
            p = np.array(path)
            ax.plot(p[:, 1], p[:, 0], "-c", lw=2.5, label="A* 路径")
            print(f"A* q{n} 帧{int(q_idx[n])}: 路径 {len(path)} 步")
        else:
            print(f"A* q{n}: 未找到路径")
    ax.legend()
    ax.set_title(f"q{n} 帧{int(q_idx[n])} 局部占据+A*")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "astar_demo.png"), dpi=110)
plt.close()

# ---------- 图4: 全局度量占据图 (含单目漂移, 仅总览) ----------
gx, gy = gpts[:, 0], gpts[:, 1]
cx, cy = (gx.min() + gx.max()) / 2, (gy.min() + gy.max()) / 2
grng = max(gx.max() - gx.min(), gy.max() - gy.min()) / 2 + 2
gocc = OccupancyGrid(resolution=0.15, range_m=grng, center_xy=(cx, cy),
                     ground_band=cfg.ground_band_m, ceil=cfg.camera_height_m,
                     occ_min_hits=2)
gocc.integrate(gpts, ground_z=0.0)
fig, ax = plt.subplots(figsize=(11, 11))
ax.imshow(gocc.grid(), origin="lower", cmap="gray_r", vmin=0, vmax=2)
ax.set_title("全局度量占据图 (DB点云累积; 含单目漂移)")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "global_occupancy.png"), dpi=110)
plt.close()

# 保存数值
np.savez(os.path.join(OUT, "results.npz"), errs=errs,
         est_poses=np.stack([r.T_wc for r in results]), q_gt=q_gt)
print(f"\n可视化 + 结果 -> {OUT}")
