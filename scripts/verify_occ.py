"""占据栅格质量核对: 全量4路点云(墙)叠加占据障碍, 检验"该有的有/没有的不虚构"。
对比 旧法(occ_min_hits) vs 新法(稳健: 密度+垂直结构+形态学)。
输出 outputs/query/verify_occ.png
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

from vgpnav.config import Config
from vgpnav.occupancy import OccupancyGrid
from vgpnav.viz import setup_cjk_font

setup_cjk_font()
cfg = Config()
pts = np.load(os.path.join(cfg.db_dir, "global_points_4cam.npz"))["points"]
gx, gy = pts[:, 0], pts[:, 1]
cx, cy = (gx.min() + gx.max()) / 2, (gy.min() + gy.max()) / 2
rng = max(gx.ptp(), gy.ptp()) / 2 + 2
RES = 0.25


def build(robust):
    occ = OccupancyGrid(RES, rng, (cx, cy), cfg.ground_band_m, cfg.camera_height_m,
                        occ_min_hits=3)
    occ.integrate(pts, ground_z=0.0)
    g = occ.grid_robust(4, 0.3) if robust else occ.grid()
    if robust:
        lbl, n = ndimage.label(g == 2, structure=np.ones((3, 3)))
        if n:
            sz = ndimage.sum(np.ones_like(lbl), lbl, np.arange(1, n + 1))
            g[np.isin(lbl, np.where(sz < 5)[0] + 1)] = 1
    return occ, g


occ, g_new = build(True)
_, g_old = build(False)

# 墙体点 (障碍高度带) 用于核对
m = (pts[:, 2] > cfg.ground_band_m) & (pts[:, 2] < cfg.camera_height_m)
W = pts[m]
if len(W) > 150000:
    W = W[np.random.default_rng(0).choice(len(W), 150000, replace=False)]


def cells_xy(g, val):
    ij = np.argwhere(g == val)
    x = occ.origin[0] + (ij[:, 1] + 0.5) * RES
    y = occ.origin[1] + (ij[:, 0] + 0.5) * RES
    return x, y


fig, axs = plt.subplots(1, 2, figsize=(28, 14))
for ax, g, ttl in [(axs[0], g_old, "旧法(occ_min_hits=3): 易把噪声悬浮点虚构成障碍"),
                   (axs[1], g_new, "新法(稳健: 密度+垂直结构+形态学清理)")]:
    ox, oy = cells_xy(g, 1)
    bx, by = cells_xy(g, 2)
    ax.scatter(ox, oy, s=2, c="#cfe8ff", alpha=0.25, label="free")
    ax.scatter(W[:, 0], W[:, 1], s=0.4, c="0.5", alpha=0.5, label="墙体点云")
    ax.scatter(bx, by, s=6, c="#ff2a6d", alpha=0.8, label="障碍格")
    ax.set_aspect("equal")
    ax.legend(markerscale=3, loc="upper left")
    ax.set_title(f"{ttl}\n障碍格={int((g==2).sum())}")
plt.tight_layout()
out = os.path.join(cfg.out_dir, "query", "verify_occ.png")
plt.savefig(out, dpi=110)
print("核对图 ->", out)
print(f"障碍格: 旧={int((g_old==2).sum())}  新={int((g_new==2).sum())}")
# 量化: 障碍格中"附近无墙点"(虚构)的比例
from scipy.spatial import cKDTree
tree = cKDTree(W[:, :2])
for name, g in [("旧", g_old), ("新", g_new)]:
    bx, by = cells_xy(g, 2)
    if len(bx):
        d, _ = tree.query(np.c_[bx, by], k=1)
        print(f"{name}法 障碍格离最近墙点>{RES}m(虚构嫌疑)占比: "
              f"{100*np.mean(d > RES):.1f}%  >0.5m: {100*np.mean(d > 0.5):.1f}%")
