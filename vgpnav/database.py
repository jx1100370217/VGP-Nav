"""离线建库: 把 C8 单目序列建成度量地图 (伪 GT) + 检索数据库。

流程:
 1. 选轨迹帧 (map_stride), 鱼眼去畸变成针孔单目。
 2. VGGT 滑窗重建, 各窗口用 align_sim3_from_poses 链接成全局 (up-to-scale) 轨迹,
    同时累积下采样的稠密点。
 3. 重力对齐 + 地面峰 + 1.3m 相机高度 -> 全图度量尺度 (地面 z=0)。
 4. 划分 DB / query (query 留出做定位评测, 伪 GT = 轨迹位姿)。
 5. 为 DB 帧算 VPR 描述子; 存库 (位姿/中心/朝向/描述子/全局点/划分)。

这一步对应论文里 "已知环境的带位姿数据库" 的构建 (用 VGGT 自建伪 GT)。
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from . import geom
from . import scale_recovery as SR
from .undistort import PinholeUndistorter, load_camera_params
from .vggt_runner import VGGTRunner
from .vpr import make_vpr


def list_frame_files(cfg):
    return sorted(glob.glob(os.path.join(cfg.data_dir, f"*_{cfg.camera}.jpg")))


def _subsample_points(world_pts, conf, n=1500):
    """从一帧 VGGT 稠密点中按置信度过滤并随机下采样。"""
    P = world_pts.reshape(-1, 3)
    c = conf.reshape(-1)
    thr = np.percentile(c, 50)
    keep = c >= thr
    P = P[keep]
    if len(P) > n:
        idx = np.random.default_rng(0).choice(len(P), n, replace=False)
        P = P[idx]
    return P


def _chain_trajectory(frame_files, traj_idx, und, vggt, cfg):
    """VGGT 滑窗 -> 逐窗重力对齐+地面1.3m度量尺度锚定(两遍稳健平滑) -> 刚性拼接。

    单目 VGGT 每窗尺度独立模糊, 仅末尾一次全局锚定无法消除逐窗尺度差累积成的漂移
    (重建地面随轨迹抬升、整体被撑大、重访区错位割裂)。故:
      Pass1: 每窗 VGGT + 重力对齐 + 地面1.3m 估"原始度量尺度"(缓存, VGGT 只跑一遍)。
      平滑: 对所有窗口的原始尺度做鲁棒去极值(限于 [0.4,2.5]×中位) + 滑动中值,
            消除个别窗口地面估计失败/噪声, 同时保留缓慢漂移修正。
      Pass2: 应用平滑尺度, 只做刚性 (R,t) 拼接 -> 全局均匀消漂移、不引入窗口边界扭曲。
    """
    chunk, overlap = cfg.chunk_size, cfg.chunk_overlap
    step = max(1, chunk - overlap)
    starts = list(range(0, len(traj_idx), step))
    # ---- Pass 1: VGGT + 逐窗重力对齐 + 原始度量尺度 (缓存) ----
    wins = []
    for w, st in enumerate(starts):
        local = traj_idx[st:st + chunk]
        if len(local) < 2:
            continue
        imgs = [und.undistort(cv2.imread(frame_files[i])) for i in local]
        out = vggt.infer(imgs)
        T = {i: geom.T_wc_from_extri(out["extri"][k]) for k, i in enumerate(local)}
        pts = {i: _subsample_points(out["world_points"][k], out["depth_conf"][k])
               for k, i in enumerate(local)}
        Rg = SR.rotation_align(
            SR.estimate_gravity_down([T[i][:3, :3] for i in local]),
            np.array([0, 0, -1.0]))
        for i in local:
            T[i][:3, :3] = Rg @ T[i][:3, :3]
            T[i][:3, 3] = Rg @ T[i][:3, 3]
            pts[i] = pts[i] @ Rg.T
        centers = np.array([T[i][:3, 3] for i in local])
        raw = SR.anchor_map(centers, np.vstack([pts[i] for i in local]),
                            cfg.camera_height_m)["scale"]
        wins.append(dict(local=local, T=T, pts=pts, raw=float(raw)))
        if (w % 10) == 0:
            print(f"  窗口 {w+1}/{len(starts)} (帧 {st}..), 原始窗尺度={raw:.2f}")
    # ---- 统一稳健尺度 ----
    # 逐窗地面尺度估计噪声大(杂乱区误判地面->虚高), 各自生效会把地图撑爆/扭曲。
    # VGGT 各窗原生尺度近似一致, 故用所有窗的中位作"统一尺度", 消除噪声、保证窗间一致;
    # 绝对尺度由末尾全局地面锚定按相机高 1.3m 统一确定 (此处统一尺度仅需内部一致)。
    raws = np.array([wd["raw"] for wd in wins])
    s_uni = float(np.median(raws))
    print(f"逐窗原始尺度: 中位={s_uni:.3f} std={raws.std():.3f} -> 统一采用中位 {s_uni:.3f}")
    # ---- Pass 2: 应用统一尺度 + 刚性拼接 ----
    global_T = {}
    pts_global = []
    for w, wd in enumerate(wins):
        local, T, pts, s_w = wd["local"], wd["T"], wd["pts"], s_uni
        for i in local:
            T[i] = T[i].copy()
            T[i][:3, 3] = T[i][:3, 3] * s_w
            pts[i] = pts[i] * s_w
        if not global_T:
            R, t = np.eye(3), np.zeros(3)
        else:
            shared = [i for i in local if i in global_T]
            if len(shared) >= 2:
                _, R, t = geom.align_sim3_from_poses(
                    [T[i] for i in shared],
                    [global_T[i] for i in shared], fix_scale=True)
            else:
                R, t = np.eye(3), np.zeros(3)
        for i in local:
            if i in global_T:
                continue
            Ti = T[i].copy()
            Ti[:3, :3] = R @ Ti[:3, :3]
            Ti[:3, 3] = R @ Ti[:3, 3] + t
            global_T[i] = Ti
            pts_global.append((pts[i] @ R.T) + t)
    return (global_T,
            (np.vstack(pts_global) if pts_global else np.zeros((0, 3))))


def build_database(cfg):
    os.makedirs(cfg.db_dir, exist_ok=True)
    files = list_frame_files(cfg)
    n = len(files)
    print(f"C8 共 {n} 帧 (相机 {cfg.camera})")
    traj_idx = list(range(0, n, cfg.map_stride))
    print(f"轨迹帧 {len(traj_idx)} (map_stride={cfg.map_stride})")

    params = load_camera_params(cfg.cam_params, cfg.camera)
    und = PinholeUndistorter(params, cfg.undist_w, cfg.undist_h,
                             cfg.undist_hfov, cfg.undist_pitch_down)
    vggt = VGGTRunner(cfg)

    # ---- 链式轨迹 (up-to-scale, 刚性拼接) ----
    print("VGGT 滑窗链式重建中...")
    global_T, pts = _chain_trajectory(files, traj_idx, und, vggt, cfg)
    placed = [i for i in traj_idx if i in global_T]
    poses = np.stack([global_T[i] for i in placed])     # (M,4,4)

    # ---- 全局重力对齐 (IMU 自标定优先; 否则视觉) ----
    # 头盔头部运动让"相机平均朝向"估的重力抖动(floor1 相机高度 std 偏大主因)。
    # imu.csv 存在时: 用 IMU 加速度计真重力 + VGGT 位姿自标定 R_cam<-imu(外参不全),
    # 输出 world 重力方向 -> 更稳的地面/z 对齐。
    imu_csv = os.path.join(cfg.data_dir, "imu.csv")
    poses_R = [T[:3, :3] for T in poses]
    if os.path.exists(imu_csv):
        from . import imu as IMU
        ts_i, lin_i = IMU.load_imu(imu_csv)
        frame_ts = [float(os.path.basename(files[i]).split("_camera")[0]) for i in placed]
        g_imu = IMU.gravity_imu_for(frame_ts, ts_i, lin_i)
        g_down, _, _resid = IMU.solve_world_gravity(poses_R, g_imu)
        print(f"重力对齐: IMU 自标定 (帧间一致性残差 {_resid:.1f}°)")
    else:
        g_down = SR.estimate_gravity_down(poses_R)
        print("重力对齐: 视觉(相机平均朝向)")
    R_align = SR.rotation_align(g_down, np.array([0, 0, -1.0]))
    poses_a = poses.copy()
    for j in range(len(poses_a)):
        poses_a[j, :3, :3] = R_align @ poses[j, :3, :3]
        poses_a[j, :3, 3] = R_align @ poses[j, :3, 3]
    pts_a = pts @ R_align.T

    # ---- 单次全局地面峰 + 1.3m 尺度锚定 (鲁棒) ----
    anc = SR.anchor_map(poses_a[:, :3, 3], pts_a, cfg.camera_height_m)
    if anc["scale"] < 0:
        # 重力 down 被标反(IMU 自标定符号歧义 + VGGT 位姿噪声, floor28 室内位姿差时触发):
        # 地面峰落到相机上方 -> scale<0 -> 整图镜像。物理上相机必在地面上方, 故翻转重力重对齐。
        g_down = -g_down
        R_align = SR.rotation_align(g_down, np.array([0, 0, -1.0]))
        for j in range(len(poses_a)):
            poses_a[j, :3, :3] = R_align @ poses[j, :3, :3]
            poses_a[j, :3, 3] = R_align @ poses[j, :3, 3]
        pts_a = pts @ R_align.T
        anc = SR.anchor_map(poses_a[:, :3, 3], pts_a, cfg.camera_height_m)
        print("  (检测到重力方向标反, 已翻转重对齐 -> scale 转正)")
    s, z_shift = anc["scale"], anc["z_shift"]
    poses_m = poses_a.copy()
    poses_m[:, :3, 3] *= s
    poses_m[:, 2, 3] += z_shift
    pts_m = pts_a * s
    pts_m[:, 2] += z_shift
    # x,y 原点平移到首帧
    xy0 = poses_m[0, :2, 3].copy()
    poses_m[:, :2, 3] -= xy0
    pts_m[:, :2] -= xy0
    print(f"全局尺度锚定: scale={s:.4f}, 地面峰(对齐系)={anc['ground_z']:.3f}")
    print(f"锚定后相机高度: 中位={np.median(poses_m[:,2,3]):.3f} "
          f"std={np.std(poses_m[:,2,3]):.3f} m (应≈{cfg.camera_height_m}±小)")

    centers_m = poses_m[:, :3, 3]
    view_dirs = np.stack([poses_m[j, :3, :3] @ np.array([0, 0, 1.0])
                          for j in range(len(poses_m))])  # 相机前向(世界)

    # ---- DB / query 划分 ----
    pos = np.arange(len(placed))
    if cfg.db_sub <= 1:
        # 密 DB: query 均匀 held-out n_query 个, 其余全做 DB。
        # DB 越密, 检索到的最近参考越贴近 query, 运动平均定位越准(短序列强混叠场景尤其有效)。
        q_sel = np.linspace(0, len(pos) - 1, cfg.n_query + 2)[1:-1].astype(int)
        query_mask = np.zeros(len(pos), dtype=bool)
        query_mask[q_sel] = True
        db_local = pos[~query_mask]
        query_local = pos[query_mask]
    else:
        db_mask = (pos % cfg.db_sub == 0)
        db_local = pos[db_mask]
        nondb_local = pos[~db_mask]
        if len(nondb_local) > cfg.n_query:
            q_sel = np.linspace(0, len(nondb_local) - 1, cfg.n_query).astype(int)
            query_local = nondb_local[q_sel]
        else:
            query_local = nondb_local
    print(f"DB 帧 {len(db_local)}, query 帧 {len(query_local)}")

    # ---- DB 描述子 ----
    print("计算 DB VPR 描述子...")
    vpr = make_vpr(cfg)
    db_imgs = [und.undistort(cv2.imread(files[placed[i]])) for i in db_local]
    db_desc = vpr.describe(db_imgs)

    # ---- 存库 ----
    np.savez(os.path.join(cfg.db_dir, "db.npz"),
             frame_idx=np.array([placed[i] for i in db_local]),
             poses=poses_m[db_local],
             centers=centers_m[db_local],
             view_dirs=view_dirs[db_local],
             desc=db_desc,
             camera_height=cfg.camera_height_m)
    np.savez(os.path.join(cfg.db_dir, "query.npz"),
             frame_idx=np.array([placed[i] for i in query_local]),
             poses_gt=poses_m[query_local])
    np.savez(os.path.join(cfg.db_dir, "trajectory.npz"),
             frame_idx=np.array(placed), poses=poses_m,
             centers=centers_m, view_dirs=view_dirs)
    # 全局点下采样存 (供占据/可视化)
    if len(pts_m) > 400000:
        idx = np.random.default_rng(0).choice(len(pts_m), 400000, replace=False)
        pts_save = pts_m[idx]
    else:
        pts_save = pts_m
    np.savez(os.path.join(cfg.db_dir, "global_points.npz"), points=pts_save)
    meta = dict(camera=cfg.camera, undist_w=cfg.undist_w, undist_h=cfg.undist_h,
                undist_hfov=cfg.undist_hfov, undist_pitch_down=cfg.undist_pitch_down,
                camera_height_m=cfg.camera_height_m, scale=s,
                n_traj=len(placed), n_db=len(db_local), n_query=len(query_local))
    with open(os.path.join(cfg.db_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"建库完成 -> {cfg.db_dir}")
    return meta


@dataclass
class Database:
    frame_idx: np.ndarray
    poses: np.ndarray
    centers: np.ndarray
    view_dirs: np.ndarray
    desc: np.ndarray
    camera_height: float
    files: list
    cfg: object
    _und: object = None

    @property
    def und(self):
        if self._und is None:
            params = load_camera_params(self.cfg.cam_params, self.cfg.camera)
            self._und = PinholeUndistorter(
                params, self.cfg.undist_w, self.cfg.undist_h,
                self.cfg.undist_hfov, self.cfg.undist_pitch_down)
        return self._und

    def image(self, frame_index: int):
        return self.und.undistort(cv2.imread(self.files[frame_index]))


def load_database(cfg) -> Database:
    d = np.load(os.path.join(cfg.db_dir, "db.npz"))
    files = list_frame_files(cfg)
    return Database(frame_idx=d["frame_idx"], poses=d["poses"],
                    centers=d["centers"], view_dirs=d["view_dirs"],
                    desc=d["desc"], camera_height=float(d["camera_height"]),
                    files=files, cfg=cfg)
