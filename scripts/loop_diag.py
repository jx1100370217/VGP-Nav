"""找 floor28 的远距离回环帧对(sim>0.85, 空间距>15m), 拼它们的 4 环视原图判断真假。"""
import glob
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config

cfg = Config()
d = cfg.db_dir
desc = np.load(os.path.join(d, "allframe_desc_4cam.npy"))   # (N,4,4096)
traj = np.load(os.path.join(d, "trajectory_orig.npz"))
centers, fidx = traj["centers"], [int(x) for x in traj["frame_idx"]]
N = len(centers)
dn = desc / (np.linalg.norm(desc, axis=2, keepdims=True) + 1e-9)

pairs = []
for i in range(N):
    for j in range(i + 20, N):
        sim = float((dn[i] @ dn[j].T).max())      # 4路交叉最大相似度
        dist = float(np.linalg.norm(centers[i] - centers[j]))
        if sim > 0.85 and dist > 15:
            pairs.append((i, j, sim, dist))
pairs.sort(key=lambda x: -x[2])
print("远距离回环对(sim>0.85, dist>15m):")
for i, j, s, dd in pairs[:6]:
    print("  帧%d <-> 帧%d  sim=%.3f  dist=%.1fm" % (i, j, s, dd))

if pairs:
    i, j = pairs[0][:2]
    files1 = sorted(glob.glob(os.path.join(cfg.data_dir, "*_camera_1.jpg")))
    ims = []
    for fr in [fidx[i], fidx[j]]:
        for c in [1, 2, 3, 4]:
            p = files1[fr].replace("_camera_1", "_camera_%d" % c)
            ims.append(Image.open(p).resize((240, 180)))
    M = Image.new("RGB", (4 * 240, 2 * 180), "white")
    for k, im in enumerate(ims):
        M.paste(im, ((k % 4) * 240, (k // 4) * 180))
    M.save("/tmp/loop28.jpg")
    print("拼图: 上行=帧%d 下行=帧%d (各 camera_1/2/3/4) -> /tmp/loop28.jpg" % (i, j))
