"""SE(3)/旋转 几何工具。

约定:
- VGGT 的 extri 是 3x4 [R|t], 表示 world->cam (X_cam = R X_world + t)。
- T_cw: 4x4, world->cam; T_wc = inv(T_cw), cam->world。
- 相机中心 C = -R^T t (在对应 world 系下)。
- 相机系: x 右, y 下, z 前 (OpenCV)。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def extri_to_Tcw(extri: np.ndarray) -> np.ndarray:
    """3x4 [R|t] (world->cam) -> 4x4 T_cw。"""
    T = np.eye(4)
    T[:3, :4] = np.asarray(extri, dtype=np.float64)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def Rt_to_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


def cam_center(T_cw: np.ndarray) -> np.ndarray:
    """world->cam 的相机中心 (world 系)。"""
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    return -R.T @ t


def T_wc_from_extri(extri: np.ndarray) -> np.ndarray:
    """extri (world->cam) -> T_wc (cam->world)。"""
    return inv_T(extri_to_Tcw(extri))


def rotation_angle(R: np.ndarray) -> float:
    """旋转矩阵对应的旋转角 (弧度)。"""
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def angle_between(R1: np.ndarray, R2: np.ndarray) -> float:
    """两旋转间的测地角 (弧度)。"""
    return rotation_angle(R1.T @ R2)


def quat_average(Rs, weights=None) -> np.ndarray:
    """加权四元数平均 (Markley 法: Σwᵢqᵢqᵢᵀ 的最大特征向量)。

    Args:
        Rs: list/array of (3,3) 旋转矩阵
        weights: 每个旋转的权重 (None=均权)
    Returns:
        平均旋转矩阵 (3,3)
    """
    Rs = [np.asarray(R, dtype=np.float64) for R in Rs]
    n = len(Rs)
    if weights is None:
        weights = np.ones(n)
    weights = np.asarray(weights, dtype=np.float64)
    quats = Rot.from_matrix(np.stack(Rs)).as_quat()  # (n,4) [x,y,z,w]
    # 半球对齐, 避免 q 与 -q 抵消
    ref = quats[0]
    for i in range(n):
        if np.dot(quats[i], ref) < 0:
            quats[i] = -quats[i]
    M = np.zeros((4, 4))
    for i in range(n):
        q = quats[i]
        M += weights[i] * np.outer(q, q)
    w_eig, v_eig = np.linalg.eigh(M)
    q_mean = v_eig[:, np.argmax(w_eig)]
    return Rot.from_quat(q_mean).as_matrix()


def make_sim3(s: float, R: np.ndarray, t: np.ndarray):
    """构造一个 Sim(3) 变换 (s,R,t), 作用: y = s R x + t。返回函数。"""
    R = np.asarray(R)
    t = np.asarray(t).reshape(3)

    def f(X):
        X = np.asarray(X)
        return (s * (X @ R.T)) + t

    return f


def align_sim3_from_poses(T_src_list, T_dst_list, fix_scale=False):
    """由成对相机位姿估 Sim(3): dst ≈ s R src + t。

    旋转用相机姿态求 (R = mean(R_dst R_src^T)), 尺度用中心间距比, 平移用残差均值。
    相比仅用中心点的 Umeyama, 对相机中心近共线 (机器人直行) 鲁棒。
    fix_scale=True 时强制 s=1 (两端已度量, 只求刚性 R,t)。
    """
    from scipy.spatial.distance import pdist
    Rs = [np.asarray(Td)[:3, :3] @ np.asarray(Ts)[:3, :3].T
          for Ts, Td in zip(T_src_list, T_dst_list)]
    R = quat_average(Rs)
    Cs = np.array([np.asarray(T)[:3, 3] for T in T_src_list])
    Cd = np.array([np.asarray(T)[:3, 3] for T in T_dst_list])
    if fix_scale:
        s = 1.0
    elif len(Cs) >= 2:
        ds, dd = pdist(Cs), pdist(Cd)
        m = ds > 1e-6
        s = float(np.median(dd[m] / ds[m])) if m.any() else 1.0
    else:
        s = 1.0
    t = np.mean(Cd - s * (Cs @ R.T), axis=0)
    return s, R, t


def umeyama_sim3(src: np.ndarray, dst: np.ndarray):
    """估计把 src 对齐到 dst 的 Sim(3): dst ≈ s R src + t (Umeyama 1991)。

    Args:
        src, dst: (N,3) 对应点
    Returns:
        (s, R, t)
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    Xs = src - mu_s
    Xd = dst - mu_d
    Sigma = (Xd.T @ Xs) / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / n
    s = np.trace(np.diag(D) @ S) / var_s
    t = mu_d - s * R @ mu_s
    return float(s), R, t
