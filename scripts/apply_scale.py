"""水平尺度校正: 把建好的图 xy 全局乘一个因子 k (z 不变, 因为垂直方向地面锚定已对)。

用于修 VGGT 单目室外的"水平运动尺度系统偏小"。k = 真实距离 / 建图当前距离。
幂等(记录 meta.xy_scale_correction, 重复调用按净因子缩放, 不累积)。改完跑 export_web 生效。

  python scripts/apply_scale.py <k>          # 例: apply_scale.py 6.0
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config

k = float(sys.argv[1])
cfg = Config()
d = cfg.db_dir

mp = os.path.join(d, "meta.json")
meta = json.load(open(mp))
old_k = float(meta.get("xy_scale_correction", 1.0))
eff = k / old_k                                  # 净缩放(幂等)
print(f"目标 xy 因子 {k} (已应用 {old_k}) -> 本次净缩放 ×{eff:.4f}")

for f in ["trajectory.npz", "trajectory_orig.npz", "db.npz", "query.npz"]:
    p = os.path.join(d, f)
    if not os.path.exists(p):
        continue
    z = dict(np.load(p))
    if "centers" in z:
        z["centers"][:, :2] *= eff
    if "poses" in z:
        z["poses"][:, :2, 3] *= eff              # 位姿平移 xy
    if "poses_gt" in z:
        z["poses_gt"][:, :2, 3] *= eff
    np.savez(p, **z)
    print(f"  {f}: xy ×{eff:.3f}")

for f in ["global_points.npz", "global_points_4cam.npz"]:
    p = os.path.join(d, f)
    if not os.path.exists(p):
        continue
    z = dict(np.load(p))
    z["points"][:, :2] *= eff                    # 点云 xy (z 不变)
    np.savez(p, **z)
    print(f"  {f}: xy ×{eff:.3f}")

meta["xy_scale_correction"] = k
json.dump(meta, open(mp, "w"))
print(f"完成: xy 全局尺度因子 = {k} (z 保持)。请跑 export_web 重新导出。")
