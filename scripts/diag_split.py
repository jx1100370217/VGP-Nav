"""诊断建图割裂: 原始轨迹(帧序着色)+回环候选+墙体点云, 定位真实割裂与假回环。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vgpnav.config import Config
from vgpnav.viz import setup_cjk_font

setup_cjk_font()
cfg = Config()
traj = np.load(os.path.join(cfg.db_dir, "trajectory_orig.npz"))
C = traj["poses"][:, :3, 3]
N = len(C)
D = np.load(os.path.join(cfg.db_dir, "allframe_desc.npy"))
S = D @ D.T
loops = []
for i in range(N):
    for j in range(i + 40, N):
        if S[i, j] > 0.55 and np.linalg.norm(C[i, :2] - C[j, :2]) < 12:
            loops.append((i, j, float(S[i, j])))
print(f"回环候选 {len(loops)}")

pts = np.load(os.path.join(cfg.db_dir, "global_points.npz"))["points"]
m = (pts[:, 2] > 0.1) & (pts[:, 2] < 2.2)
P = pts[m]
if len(P) > 120000:
    P = P[np.random.default_rng(0).choice(len(P), 120000, replace=False)]

fig, axs = plt.subplots(1, 2, figsize=(26, 13))
ax = axs[0]
sc = ax.scatter(C[:, 0], C[:, 1], c=np.arange(N), cmap="jet", s=10)
ax.plot(C[:, 0], C[:, 1], "-", c="gray", lw=0.4, alpha=0.4)
for i, j, s in loops:
    ax.plot([C[i, 0], C[j, 0]], [C[i, 1], C[j, 1]], "-r", lw=0.3, alpha=0.35)
for k in range(0, N, 25):
    ax.annotate(str(k), (C[k, 0], C[k, 1]), fontsize=6, color="k")
ax.set_aspect("equal")
fig.colorbar(sc, ax=ax, label="帧序")
ax.set_title(f"原始轨迹(色=帧序) + 回环候选{len(loops)}(红) + 帧号标注")
ax = axs[1]
ax.scatter(P[:, 0], P[:, 1], s=0.3, c="gray", alpha=0.4)
ax.plot(C[:, 0], C[:, 1], "-c", lw=1)
for k in range(0, N, 25):
    ax.annotate(str(k), (C[k, 0], C[k, 1]), fontsize=6, color="yellow")
ax.set_aspect("equal")
ax.set_title("墙体点云(灰) + 轨迹(青) + 帧号")
plt.tight_layout()
out = os.path.join(cfg.out_dir, "query", "diag_split.png")
plt.savefig(out, dpi=110)
print("诊断图 ->", out)

# 回环空间分布统计
if loops:
    lc = np.array([[(C[i, 0] + C[j, 0]) / 2, (C[i, 1] + C[j, 1]) / 2] for i, j, _ in loops])
    print(f"回环中点 x:[{lc[:,0].min():.0f},{lc[:,0].max():.0f}] y:[{lc[:,1].min():.0f},{lc[:,1].max():.0f}]")
    di = np.array([abs(i - j) for i, j, _ in loops])
    print(f"回环序号间隔: 中位={np.median(di):.0f} 最大={di.max()}")
