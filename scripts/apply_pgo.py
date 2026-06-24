"""把 PGO 校正后的位姿(trajectory_pgo.npz)写回 trajectory/db/query, 供重建点云+重导出。

首次运行会把原始 trajectory.npz 备份为 trajectory_orig.npz。
  python scripts/apply_pgo.py
"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config

cfg = Config()
db_dir = cfg.db_dir
pgo = np.load(os.path.join(db_dir, "trajectory_pgo.npz"))
fidx = [int(v) for v in pgo["frame_idx"]]
poses, centers, vdir = pgo["poses"], pgo["centers"], pgo["view_dirs"]
pos = {f: k for k, f in enumerate(fidx)}

# 备份原始轨迹 (仅首次)
orig = os.path.join(db_dir, "trajectory_orig.npz")
if not os.path.exists(orig):
    shutil.copy(os.path.join(db_dir, "trajectory.npz"), orig)
    print("已备份原始轨迹 -> trajectory_orig.npz")

# 写回 trajectory
np.savez(os.path.join(db_dir, "trajectory.npz"),
         frame_idx=pgo["frame_idx"], poses=poses, centers=centers, view_dirs=vdir)

# 更新 db (DB 帧 ⊂ 轨迹帧)
db = np.load(os.path.join(db_dir, "db.npz"))
sel = [pos[int(f)] for f in db["frame_idx"]]
np.savez(os.path.join(db_dir, "db.npz"),
         frame_idx=db["frame_idx"], poses=poses[sel], centers=centers[sel],
         view_dirs=vdir[sel], desc=db["desc"], camera_height=db["camera_height"])

# 更新 query 伪GT
q = np.load(os.path.join(db_dir, "query.npz"))
selq = [pos[int(f)] for f in q["frame_idx"]]
np.savez(os.path.join(db_dir, "query.npz"),
         frame_idx=q["frame_idx"], poses_gt=poses[selq])

print(f"已写回 trajectory({len(fidx)})/db({len(sel)})/query({len(selq)}) 的 PGO 校正位姿")
