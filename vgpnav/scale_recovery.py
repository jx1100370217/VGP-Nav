"""地面锚定尺度恢复 (论文 III-E)。

地平面是导航环境中静态、全局一致的几何参考, 不随相机抖动改变, 因此是解单目
尺度模糊的天然度量锚。

- 建库一次性初始化: 用相机姿态估重力 -> 重力对齐 -> 沿重力轴找地面峰 ->
  用已知相机离地高度 (1.3m) 把地面高度对齐到真实物理地面 -> 得到全图度量尺度。
- 在线 (每个 query): 用定位旋转 R_wc 把 VGGT 点云转到世界朝向, 沿重力轴找地面峰,
  用 1.3m 锚定本帧尺度 s, 放到世界系。这维持了同一支撑面上的度量尺度。
"""
from __future__ import annotations

import numpy as np


def rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """返回旋转 R 使 R @ a = b (a,b 单位向量, Rodrigues)。"""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else -np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def estimate_gravity_down(R_wc_list) -> np.ndarray:
    """由相机姿态估世界系下的重力(向下)方向。

    相机系 y 轴向下; 各相机 down 轴在世界系 = R_wc @ [0,1,0]。平均即重力向下。
    """
    downs = [np.asarray(R)[:3, :3] @ np.array([0, 1.0, 0]) for R in R_wc_list]
    g = np.mean(downs, axis=0)
    return g / (np.linalg.norm(g) + 1e-12)


def find_ground_peak(z: np.ndarray, n_bins: int = 160,
                     low_frac: float = 0.6, min_pts: int = 50) -> float:
    """沿重力轴 (z) 的点高度分布中, 取下半部分的主峰作为地面高度。

    论文: "ground plane manifests as the dominant peak in the height statistics
    along the gravity axis (lower than camera height)"。
    """
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if len(z) < min_pts:
        return float(np.median(z)) if len(z) else 0.0
    lo, hi = np.percentile(z, [1, 99])
    zc = z[(z >= lo) & (z <= hi)]
    hist, edges = np.histogram(zc, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # 只在 z 的下半部分 (更接近地面侧) 找主峰
    mask = centers <= lo + low_frac * (hi - lo)
    if mask.sum() < 3:
        mask[:] = True
    idx = int(np.argmax(hist * mask))
    return float(centers[idx])


def anchor_map(centers: np.ndarray, points: np.ndarray, camera_height: float):
    """建库一次性锚定。

    Args:
        centers: (Nc,3) 相机中心 (up-to-scale 全局系)
        points: (Np,3) 稠密点 (同系, 用于估地面)
        camera_height: 相机离地高度 (m)
    Returns:
        dict(R_align, scale, z_shift, ground_z), 以及应用变换的闭包 apply()。
        世界点: X_w = scale * (R_align @ X) + [0,0,z_shift]
    """
    # 这里 centers/points 已是 (R_align 之后)? 不: 需要先有 R_wc 才能估重力。
    # 故 anchor_map 接收的 centers/points 已假定在重力对齐系 (调用方先对齐)。
    ground_z = find_ground_peak(points[:, 2])
    cam_z = float(np.median(centers[:, 2]))
    height_arb = cam_z - ground_z
    if abs(height_arb) < 1e-6:
        height_arb = 1.0
    scale = camera_height / height_arb
    z_shift = -ground_z * scale     # 地面 -> z=0
    return dict(scale=scale, z_shift=z_shift, ground_z=ground_z,
                height_arb=height_arb)


def anchor_query_points(P_cam: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray,
                        camera_height: float):
    """在线: 把 query 的 VGGT 相机系点云锚定到世界度量系。

    Args:
        P_cam: (N,3) query 相机系点 (VGGT 尺度)
        R_wc, t_wc: 定位得到的 query 世界位姿 (cam->world)
        camera_height: 1.3 m
    Returns:
        P_world (N,3) 度量世界点, scale, ground_z_rel
    """
    P_or = P_cam @ R_wc.T                       # 转到世界朝向 (相机仍在原点附近)
    ground_z_rel = find_ground_peak(P_or[:, 2])  # 相对相机的地面高度 (应为负)
    denom = -ground_z_rel
    if abs(denom) < 1e-6:
        denom = camera_height
    scale = camera_height / denom
    P_world = scale * P_or + np.asarray(t_wc).reshape(1, 3)
    return P_world, float(scale), float(ground_z_rel)
