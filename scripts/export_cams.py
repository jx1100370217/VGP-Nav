"""为每个 DB 节点生成 4 路环视相机(camera_1~4)的去畸变缩略图文件。

供网页导航 FPV 按机器人行进方向选朝向匹配的相机 -> 永远显示"前进视角",消除倒走感。
输出: outputs/web/thumbs/n{节点序号}_c{1..4}.jpg (按需 HTTP 加载, 不进 data.js)

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/export_cams.py
"""
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config
from vgpnav.database import load_database, list_frame_files
from vgpnav.undistort import PinholeUndistorter, load_camera_params

cfg = Config()
db = load_database(cfg)
THUMBS = os.path.join(cfg.out_dir, "web", "thumbs")
os.makedirs(THUMBS, exist_ok=True)

cams = ["camera_1", "camera_2", "camera_3", "camera_4"]
und = {}
for c in cams:
    params = load_camera_params(cfg.cam_params, c)
    und[c] = PinholeUndistorter(params, cfg.undist_w, cfg.undist_h, cfg.undist_hfov)

files1 = list_frame_files(cfg)   # camera_1 文件 (与 db.frame_idx 对应)
nmiss = 0
for k, fi in enumerate(db.frame_idx):
    f1 = files1[int(fi)]
    ts = os.path.basename(f1).split("_camera_")[0]
    for ci, c in enumerate(cams):
        path = os.path.join(cfg.data_dir, f"{ts}_{c}.jpg")
        img = cv2.imread(path)
        if img is None:
            nmiss += 1
            continue
        th = cv2.resize(und[c].undistort(img), (320, 240))
        cv2.imwrite(os.path.join(THUMBS, f"n{k}_c{ci+1}.jpg"), th,
                    [cv2.IMWRITE_JPEG_QUALITY, 72])
    if k % 40 == 0:
        print(f"  {k}/{len(db.frame_idx)}")
print(f"完成: {len(db.frame_idx)} 节点 x4 相机 -> {THUMBS} (缺失 {nmiss})")
