"""建库入口 + 轨迹/尺度可视化验证。

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/build_database.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vgpnav.config import Config
from vgpnav.database import build_database
from vgpnav.viz import setup_cjk_font

setup_cjk_font()

cfg = Config()
meta = build_database(cfg)
print("meta:", meta)

traj = np.load(os.path.join(cfg.db_dir, "trajectory.npz"))
pts = np.load(os.path.join(cfg.db_dir, "global_points.npz"))["points"]
db = np.load(os.path.join(cfg.db_dir, "db.npz"))
q = np.load(os.path.join(cfg.db_dir, "query.npz"))

fig, axs = plt.subplots(1, 2, figsize=(18, 8))
ax = axs[0]
obst = pts[(pts[:, 2] > 0.15) & (pts[:, 2] < 2.0)]
ax.scatter(obst[:, 0], obst[:, 1], s=0.2, c="gray", alpha=0.25)
ax.plot(traj["centers"][:, 0], traj["centers"][:, 1], "-b", lw=1, label="轨迹")
ax.scatter(db["centers"][:, 0], db["centers"][:, 1], c="g", s=10, label="DB")
ax.scatter(q["poses_gt"][:, 0, 3], q["poses_gt"][:, 1, 3], c="r",
           marker="*", s=90, label="query(伪GT)")
ax.set_aspect("equal")
ax.legend()
ax.set_title("俯视 X-Y (轨迹+障碍点)")

ax = axs[1]
sub = pts[np.random.default_rng(0).choice(len(pts), min(60000, len(pts)), replace=False)]
ax.scatter(sub[:, 0], sub[:, 2], s=0.3, c="gray", alpha=0.3)
ax.axhline(0, color="brown", lw=1.5, label="地面 z=0")
ax.scatter(traj["centers"][:, 0], traj["centers"][:, 2], c="b", s=5, label="相机高度")
ax.axhline(cfg.camera_height_m, color="g", ls="--", label=f"{cfg.camera_height_m}m")
ax.set_title("侧视 X-Z (相机应≈1.3m, 地面≈0)")
ax.legend()

out = os.path.join(cfg.out_dir, "database_overview.png")
plt.tight_layout()
plt.savefig(out, dpi=110)
print("可视化 ->", out)
print(f"相机高度: 中位={np.median(traj['centers'][:,2]):.3f} "
      f"均值={np.mean(traj['centers'][:,2]):.3f} "
      f"std={np.std(traj['centers'][:,2]):.3f}")
print(f"轨迹范围: X[{traj['centers'][:,0].min():.1f},{traj['centers'][:,0].max():.1f}] "
      f"Y[{traj['centers'][:,1].min():.1f},{traj['centers'][:,1].max():.1f}]")
