"""全图割裂检测: 4路点云按"生成它的关键帧时间"着色。
同一面墙: 单趟=单色; 两趟对齐=同色重合; 两趟漂移=双色错开(=割裂)。
输出 outputs/query/detect_split.png (全图 + 4象限放大)。
"""
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
d = np.load(os.path.join(cfg.db_dir, "global_points_4cam.npz"))
pts = d["points"]
kt = d["kt"] if "kt" in d.files else np.zeros(len(pts))
traj = np.load(os.path.join(cfg.db_dir, "trajectory.npz"))
C = traj["poses"][:, :3, 3]
m = (pts[:, 2] > 0.2) & (pts[:, 2] < 2.0)
P, K = pts[m], kt[m]
print(f"墙体点 {len(P)}, 关键帧时间范围 [{K.min():.0f},{K.max():.0f}]")

bx = [P[:, 0].min(), P[:, 0].max()]
by = [P[:, 1].min(), P[:, 1].max()]
cx, cy = np.mean(bx), np.mean(by)

fig = plt.figure(figsize=(26, 20))
ax = fig.add_subplot(1, 2, 1)
sc = ax.scatter(P[:, 0], P[:, 1], c=K, cmap="turbo", s=0.6, alpha=0.55)
ax.plot(C[:, 0], C[:, 1], "-k", lw=0.6, alpha=0.6)
for k in range(0, len(C), 20):
    ax.annotate(str(k), (C[k, 0], C[k, 1]), fontsize=6, color="k")
ax.set_aspect("equal")
fig.colorbar(sc, ax=ax, label="关键帧时间(轨迹位置)")
ax.set_title("全图墙体点云 (色=关键帧时间; 双色错开的墙=漂移割裂)")

# 右列: 4 个象限放大
quads = [("左上", bx[0], cx, cy, by[1]), ("右上", cx, bx[1], cy, by[1]),
         ("左下", bx[0], cx, by[0], cy), ("右下", cx, bx[1], by[0], cy)]
for qi, (name, x0, x1, y0, y1) in enumerate(quads):
    ax = fig.add_subplot(4, 2, 2 * qi + 2)
    sel = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)
    ax.scatter(P[sel, 0], P[sel, 1], c=K[sel], cmap="turbo", s=1.0, alpha=0.6,
               vmin=K.min(), vmax=K.max())
    ts = (C[:, 0] >= x0) & (C[:, 0] <= x1) & (C[:, 1] >= y0) & (C[:, 1] <= y1)
    ax.plot(C[ts, 0], C[ts, 1], ".k", ms=1, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"{name}象限")
plt.tight_layout()
out = os.path.join(cfg.out_dir, "query", "detect_split.png")
plt.savefig(out, dpi=120)
print("检测图 ->", out)
