"""验证 DINOv2 VPR 离线加载 + 描述子合理性 (连续帧应比远帧更相似)。"""
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config
from vgpnav.undistort import PinholeUndistorter, load_camera_params
from vgpnav.vpr import DINOv2VPR

cfg = Config()
intr, dist = load_camera_params(cfg.cam_params, cfg.camera)
und = PinholeUndistorter(intr, dist, cfg.undist_w, cfg.undist_h, cfg.undist_hfov)

files = sorted(glob.glob(os.path.join(cfg.data_dir, f"*_{cfg.camera}.jpg")))[::5][:30]
imgs = [und.undistort(cv2.imread(f)) for f in files]
print(f"取 {len(imgs)} 帧 (每5帧抽1)")

vpr = DINOv2VPR(device=cfg.device)
desc = vpr.describe(imgs)
print("描述子:", desc.shape, "范数(应≈1):", np.linalg.norm(desc[0]))

S = desc @ desc.T
# 连续帧相似度 vs 远帧相似度
consec = np.mean([S[i, i + 1] for i in range(len(imgs) - 1)])
far = np.mean([S[i, (i + 15) % len(imgs)] for i in range(len(imgs))])
print(f"相邻帧平均相似度={consec:.3f}, 远帧(隔15)平均相似度={far:.3f}")
assert consec > far, "连续帧未比远帧更相似, VPR 异常"
# 自相似=1
assert np.allclose(np.diag(S), 1.0, atol=1e-4)
print("DINOv2 VPR 离线加载与描述子合理 ✓")
