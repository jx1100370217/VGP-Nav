"""撤销 PGO: 从 trajectory_orig.npz 还原 trajectory/db/query 的原始位姿。
之后需重建 4 路点云(原始位姿)+重导出。
  python scripts/revert_pgo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config

cfg = Config()
db_dir = cfg.db_dir
orig = np.load(os.path.join(db_dir, "trajectory_orig.npz"))
fidx = [int(v) for v in orig["frame_idx"]]
poses, centers, vdir = orig["poses"], orig["centers"], orig["view_dirs"]
pos = {f: k for k, f in enumerate(fidx)}

np.savez(os.path.join(db_dir, "trajectory.npz"),
         frame_idx=orig["frame_idx"], poses=poses, centers=centers, view_dirs=vdir)
db = np.load(os.path.join(db_dir, "db.npz"))
sel = [pos[int(f)] for f in db["frame_idx"]]
np.savez(os.path.join(db_dir, "db.npz"),
         frame_idx=db["frame_idx"], poses=poses[sel], centers=centers[sel],
         view_dirs=vdir[sel], desc=db["desc"], camera_height=db["camera_height"])
q = np.load(os.path.join(db_dir, "query.npz"))
selq = [pos[int(f)] for f in q["frame_idx"]]
np.savez(os.path.join(db_dir, "query.npz"),
         frame_idx=q["frame_idx"], poses_gt=poses[selq])
print(f"已还原 trajectory/db/query 到原始位姿 (N={len(fidx)})")
