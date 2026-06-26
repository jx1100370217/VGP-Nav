"""IMU 重力方向提取 + 自标定,用于替代/校正视觉重力对齐。

动机:头盔采集时头部不停转动,VGP-Nav 原本"假设相机 y 轴=down"估重力会被严重干扰
(头一低这帧的 down 就偏了),这是 floor1 相机高度 std 偏大的主因之一。IMU 加速度计直接测
真重力(机体系),不受头朝向影响。

难点:IMU-相机旋转外参不全(device_info 无, PDF 仅部分角度)。解法:用【IMU重力 + VGGT位姿
自标定】—— 找固定旋转 R_ci(cam←imu) 使各帧的 world 重力 R_wc·R_ci·g_imu 尽量一致(Wahba 迭代),
输出 world 系重力 down 方向(供 rotation_align 用)。

imu.csv 列: timestamp_sec, ..., lin_x/y/z (specific force; 静止时 ≈ -g, 指向上)。
重力 down(IMU机体系) = -normalize(lin)。
"""
import csv

import numpy as np


def load_imu(csv_path):
    """返回 (ts[N] 秒, lin[N,3] 线加速度)。"""
    ts, lin = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp_sec"]))
            lin.append([float(row["lin_x"]), float(row["lin_y"]), float(row["lin_z"])])
    return np.array(ts), np.array(lin, dtype=np.float64)


def gravity_imu_for(frame_ts, ts, lin, win=0.5):
    """对每个图像帧时间戳, 取附近 ±win 秒的 IMU lin 平均(低通去头部运动加速度),
    返回重力 down 单位向量(IMU机体系) (M,3)。"""
    out = np.zeros((len(frame_ts), 3))
    for i, t in enumerate(frame_ts):
        m = np.abs(ts - t) < win
        if m.sum() < 3:
            m = np.abs(ts - t) < 2.0
        g = lin[m].mean(axis=0) if m.any() else lin[np.argmin(np.abs(ts - t))]
        n = np.linalg.norm(g)
        out[i] = (-g / n) if n > 1e-6 else np.array([0, 0, -1.0])   # lin 指上, down=-lin
    return out


def _wahba(a, b):
    """求旋转 R 使 R·b_i ≈ a_i (单位向量集, SVD 解)。"""
    H = a.T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def solve_world_gravity(R_wc_list, g_imu, iters=12):
    """自标定 R_ci(cam←imu) 与 world 重力方向, 交替迭代:
        g_world = normalize(mean_t R_wc(t)·R_ci·g_imu(t));
        R_ci    = wahba(R_wc(t)ᵀ·g_world, g_imu(t))。
    返回 world 系重力 down 单位向量(可直接喂 rotation_align(g_down,[0,0,-1]))。"""
    R = np.stack([np.asarray(r, dtype=np.float64) for r in R_wc_list])   # (N,3,3) cam->world
    g_imu = np.asarray(g_imu, dtype=np.float64)                          # (N,3)
    R_ci = np.eye(3)
    g_world = np.array([0.0, 0.0, -1.0])
    for _ in range(iters):
        gw = np.einsum("nij,jk,nk->ni", R, R_ci, g_imu)     # R_wc·R_ci·g_imu
        g_world = gw.mean(axis=0)
        g_world /= (np.linalg.norm(g_world) + 1e-9)
        a = np.einsum("nji,j->ni", R, g_world)              # R_wcᵀ·g_world (重力在 cam 系)
        R_ci = _wahba(a, g_imu)
    # 残差(各帧 world 重力方向与均值的夹角中位, 越小说明 IMU 一致性越好)
    gw = np.einsum("nij,jk,nk->ni", R, R_ci, g_imu)
    gw /= (np.linalg.norm(gw, axis=1, keepdims=True) + 1e-9)
    cos = np.clip(gw @ g_world, -1, 1)
    resid_deg = float(np.degrees(np.arccos(cos)).mean())
    return g_world, R_ci, resid_deg
