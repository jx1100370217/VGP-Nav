"""De-risk 冒烟测试: C8 鱼眼去畸变 -> VGGT 前向, 检查端到端核心依赖。

在 L40 上用 internvla python 运行:
  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/smoke_vggt.py
"""
import os
import sys
import glob
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

MN = "/home/ubuntu/Disk/codes/jianxiong/memory-nav"
sys.path.insert(0, os.path.join(MN, "third_party", "vggt_space"))
sys.path.insert(0, MN)

from vgpnav.undistort import PinholeUndistorter, load_camera_params

CAM = "camera_1"
DATA = os.path.join(MN, "Mapping_C8")
PARAMS = os.path.join(MN, "cam", "params.yaml")
OUT = os.path.join(HERE, "outputs", "smoke")
DEVICE = "cuda:3"
os.makedirs(OUT, exist_ok=True)

intr, dist = load_camera_params(PARAMS, CAM)
print("内参:", intr, "畸变:", dist)
und = PinholeUndistorter(intr, dist, out_w=640, out_h=480, hfov_deg=110.0)
print("虚拟针孔 K=\n", und.K)

files = sorted(glob.glob(os.path.join(DATA, f"*_{CAM}.jpg")))
print("总帧:", len(files))
sel = files[20:28:2]  # 取 4 帧 (间隔 2, 保证有视差又有重叠)
imgs = []
for i, f in enumerate(sel):
    raw = cv2.imread(f)
    u = und.undistort(raw)
    cv2.imwrite(os.path.join(OUT, f"raw_{i}.jpg"), raw)
    cv2.imwrite(os.path.join(OUT, f"undist_{i}.jpg"), u)
    imgs.append(u)
    print(f"  帧{i} {os.path.basename(f)}: raw{raw.shape} -> undist{u.shape}")

from online_mapper.geometry.vggt_backend import VGGTBackend

be = VGGTBackend.get(os.path.join(MN, "pretrained", "vggt-1b", "model.pt"),
                     device=DEVICE)
print("VGGT available:", be.available)
t = time.time()
out = be.infer_bgr_list(imgs)
print(f"VGGT 推理 {len(imgs)} 帧耗时 {time.time() - t:.2f}s")
for k in ["depth", "extri", "intri", "world_points", "depth_conf"]:
    v = out.get(k)
    if v is not None:
        print(f"  {k}: list[{len(v)}], [0].shape={np.asarray(v[0]).shape}, "
              f"dtype={np.asarray(v[0]).dtype}")

print("\nextri[0] (world->cam, 3x4)=\n", np.asarray(out["extri"][0]))
print("intri[0] (3x3)=\n", np.asarray(out["intri"][0]))
print("\n各帧相机中心 (VGGT 公共系, 尺度模糊):")
centers = []
for i in range(len(imgs)):
    E = np.asarray(out["extri"][i])
    R, tt = E[:, :3], E[:, 3]
    C = -R.T @ tt
    centers.append(C)
    print(f"  cam{i} C = {np.round(C, 4)}")
centers = np.array(centers)
print("相邻帧间距 (VGGT 单位):",
      np.round(np.linalg.norm(np.diff(centers, axis=0), axis=1), 4))

d0 = np.asarray(out["depth"][0])
print(f"\ndepth[0]: min={np.nanmin(d0):.3f} max={np.nanmax(d0):.3f} "
      f"median={np.nanmedian(d0):.3f}")
print("冒烟测试完成 ✓")
