"""4 路环视相机占据融合: 为关键帧把 cam1~4 一起送 VGGT 重建, 用 cam1 已知世界位姿
锚定到世界系, 融合成 360° 全局度量点云 (定位仍单目, 仅感知/占据融合 4 路)。

每关键帧窗口 = cam1 的时序邻帧(给基线) + cam2/3/4 当帧; 以 cam1 当帧为锚,
地面 1.3m 定尺度, 把全部相机的点投到世界系。

  python scripts/build_4cam_points.py [--limit N] [--stride K]
输出: outputs/db/global_points_4cam.npz (+ 测试时 outputs/fourcam_test.png)
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config
from vgpnav.database import list_frame_files
from vgpnav import geom
from vgpnav import scale_recovery as SR
from vgpnav.undistort import PinholeUndistorter, load_camera_params
from vgpnav.vggt_runner import VGGTRunner

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--stride", type=int, default=6)
args = ap.parse_args()

cfg = Config()
traj = np.load(os.path.join(cfg.db_dir, "trajectory.npz"))
traj_idx = [int(v) for v in traj["frame_idx"]]
traj_poses = traj["poses"]
files1 = list_frame_files(cfg)
cams = ["camera_1", "camera_2", "camera_3", "camera_4"]
und = {c: PinholeUndistorter(*load_camera_params(cfg.cam_params, c),
                             cfg.undist_w, cfg.undist_h, cfg.undist_hfov) for c in cams}
vggt = VGGTRunner(cfg)
rng = np.random.default_rng(0)

kf = list(range(0, len(traj_idx), args.stride))
if args.limit > 0:
    kf = kf[:args.limit]
print(f"关键帧 {len(kf)} (stride={args.stride})")

fused, fused_cam, fused_kt = [], [], []
for n, p in enumerate(kf):
    fi = traj_idx[p]
    T_wc1 = traj_poses[p]
    ts = os.path.basename(files1[fi]).split("_camera_")[0]
    imgs, srccam, anchor = [], [], 0
    for dq in (-2, 0, 2):                  # cam1 时序邻帧
        j = fi + dq
        if 0 <= j < len(files1):
            imgs.append(und["camera_1"].undistort(cv2.imread(files1[j])))
            srccam.append(1)
            if dq == 0:
                anchor = len(imgs) - 1
    for ci, c in enumerate(["camera_2", "camera_3", "camera_4"], start=2):
        img = cv2.imread(os.path.join(cfg.data_dir, f"{ts}_{c}.jpg"))
        if img is not None:
            imgs.append(und[c].undistort(img))
            srccam.append(ci)
    out = vggt.infer(imgs)
    Ta = geom.extri_to_Tcw(out["extri"][anchor])
    Ra, ta = Ta[:3, :3], Ta[:3, 3]
    Cav = -Ra.T @ ta
    R_S = T_wc1[:3, :3] @ Ra
    Pcam_a = (out["world_points"][anchor].reshape(-1, 3) @ Ra.T) + ta
    _, scale, _ = SR.anchor_query_points(Pcam_a, T_wc1[:3, :3], T_wc1[:3, 3],
                                         cfg.camera_height_m)
    for f in range(len(imgs)):
        P = out["world_points"][f].reshape(-1, 3)
        cf = out["depth_conf"][f].reshape(-1)
        keep = cf >= np.percentile(cf, 55)
        P = P[keep]
        if len(P) > 2500:
            P = P[rng.choice(len(P), 2500, replace=False)]
        Pw = scale * ((P - Cav) @ R_S.T) + T_wc1[:3, 3]
        fused.append(Pw)
        fused_cam.append(np.full(len(Pw), srccam[f]))
        fused_kt.append(np.full(len(Pw), p))    # 关键帧轨迹位置(时间), 供割裂检测着色
    if n % 10 == 0:
        print(f"  {n}/{len(kf)}")

pts = np.vstack(fused)
camid = np.concatenate(fused_cam)
ktid = np.concatenate(fused_kt)
print(f"融合点数: {len(pts)}")

if args.limit > 0:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vgpnav.viz import setup_cjk_font
    setup_cjk_font()
    m = (pts[:, 2] > -0.2) & (pts[:, 2] < 2.2)
    P, C = pts[m], camid[m]
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = {1: "#2de2e6", 2: "#5cff9d", 3: "#ff2a6d", 4: "#ffcf48"}
    for c in (1, 2, 3, 4):
        q = C == c
        ax.scatter(P[q, 0], P[q, 1], s=1, c=colors[c], label=f"cam{c}", alpha=.5)
    ax.plot(traj["centers"][:, 0], traj["centers"][:, 1], "-w", lw=.5, alpha=.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.set_title("4 路融合点云 (按相机着色)")
    out_png = os.path.join(cfg.out_dir, "fourcam_test.png")
    plt.savefig(out_png, dpi=110)
    print("测试图 ->", out_png)
else:
    if len(pts) > 600000:
        idx = rng.choice(len(pts), 600000, replace=False)
        pts, ktid = pts[idx], ktid[idx]
    np.savez(os.path.join(cfg.db_dir, "global_points_4cam.npz"), points=pts, kt=ktid)
    print("已写出 -> global_points_4cam.npz")
