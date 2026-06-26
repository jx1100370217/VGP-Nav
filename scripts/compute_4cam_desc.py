"""计算全部轨迹帧的4路环视 SelaVPR 描述子 -> outputs/db/allframe_desc_4cam.npy (N,4,D)。

供 pgo_fix.py 的"4路交叉回环检测"识别反向回环 (机器人重访常反向经过同一地点,
单目 camera_1 看到的景象相反、相似度低会漏检, 如帧621/501 单目0.32 但4路交叉0.77)。

  python scripts/compute_4cam_desc.py
"""
import os
import sys
import glob
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from vgpnav.config import Config
from vgpnav.vpr import make_vpr
from vgpnav.undistort import PinholeUndistorter, load_camera_params

cfg = Config()
vpr = make_vpr(cfg)
cams = ["camera_1", "camera_2", "camera_3", "camera_4"]
und = {c: PinholeUndistorter(load_camera_params(cfg.cam_params, c),
                             cfg.undist_w, cfg.undist_h, cfg.undist_hfov) for c in cams}
files1 = sorted(glob.glob(os.path.join(cfg.data_dir, "*_camera_1.jpg")))
N = len(files1)
print(f"轨迹帧 {N}, 算4路 SelaVPR 描述子...", flush=True)
descs = None
t0 = time.time()
for f in range(N):
    ts = os.path.basename(files1[f]).split("_camera_1")[0]
    ims = [und[c].undistort(cv2.imread(os.path.join(cfg.data_dir, ts + "_" + c + ".jpg")))
           for c in cams]
    d = vpr.describe(ims).astype(np.float32)        # (4, D)
    if descs is None:
        descs = np.zeros((N, 4, d.shape[1]), np.float32)
    descs[f] = d
    if f % 50 == 0:
        print("  %d/%d (%.0fs)" % (f, N, time.time() - t0), flush=True)
out = os.path.join(cfg.db_dir, "allframe_desc_4cam.npy")
np.save(out, descs)
print(f"已保存 {out} {descs.shape} 用时%.0fs" % (time.time() - t0), flush=True)
