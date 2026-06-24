"""加权运动平均 (论文 III-D)。

输入: k 个参考帧的已知世界位姿 T_wc_i (度量) + VGGT 对 {query + k refs} 的相对
几何 (extri)。输出: query 的全局位姿 T_wc。

两阶段 (解耦旋转/平移):
 (a) 旋转平均: 每个参考给一个 query 世界旋转假设
        R_wc_q^(i) = R_wc_i · R_i · R_q^T
     四元数空间求共识 + 高斯核加权 (w_rot)。
 (b) 平移 = 多视角射线三角化 (式1):
        每个参考提供一条世界系射线 (起点 c_i=参考度量中心, 方向 v_i=ref->query),
        query 中心 = argmin_t Σ Ω_i ||(I - v_i v_iᵀ)(t - c_i)||²
     Ω_i = w_rot · w_geo · w_res, IRLS 求解。
     注意: 射线起点是度量的参考中心, 故交点天然是度量的, 与 VGGT 尺度无关。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geom


@dataclass
class PoseEstimate:
    T_wc: np.ndarray            # 4x4 query 世界位姿
    w_rot: np.ndarray           # (k,) 旋转一致性权重
    w_geo: np.ndarray           # (k,) 几何条件权重
    omega: np.ndarray           # (k,) 最终复合权重
    ray_origins: np.ndarray     # (k,3)
    ray_dirs: np.ndarray        # (k,3)
    rot_dev_deg: np.ndarray     # (k,) 各旋转假设相对共识的角偏差


def _ray_geometry(ref_world_poses, extris_ref, extri_query):
    """由 VGGT extri + 参考世界位姿, 算每个参考给出的 query 旋转假设与射线。

    Returns: R_hyps (k,3,3), origins c_i (k,3), dirs v_i (k,3)
    """
    Tq = geom.extri_to_Tcw(extri_query)        # world_v -> cam_q
    Rq = Tq[:3, :3]
    Cq_v = geom.cam_center(Tq)                  # query 中心 (VGGT 系)

    R_hyps, origins, dirs = [], [], []
    for T_wc_i, extri_i in zip(ref_world_poses, extris_ref):
        Ti = geom.extri_to_Tcw(extri_i)        # world_v -> cam_i
        Ri = Ti[:3, :3]
        Ci_v = geom.cam_center(Ti)             # 参考中心 (VGGT 系)
        R_wc_i = np.asarray(T_wc_i)[:3, :3]    # cam_i -> world (已知度量)
        c_i = np.asarray(T_wc_i)[:3, 3]        # 参考世界中心 (度量)

        # world_v -> world 的旋转 (经参考 i)
        rot_v2w = R_wc_i @ Ri
        # query 世界旋转假设: cam_q -> world = (world_v->world) @ (cam_q->world_v)
        R_hyps.append(rot_v2w @ Rq.T)
        # ref_i -> query 的方向 (VGGT 系下 Cq-Ci), 转到世界系
        d_world = rot_v2w @ (Cq_v - Ci_v)
        nrm = np.linalg.norm(d_world)
        v_i = d_world / nrm if nrm > 1e-9 else np.array([1.0, 0, 0])
        origins.append(c_i)
        dirs.append(v_i)
    return (np.stack(R_hyps), np.stack(origins), np.stack(dirs))


def _rotation_consensus(R_hyps):
    """四元数共识 + 高斯核加权。返回 (R_wc, w_rot, dev_rad)。"""
    R0 = geom.quat_average(R_hyps)                       # 初始均权平均
    dev = np.array([geom.angle_between(R0, R) for R in R_hyps])
    sigma = max(np.median(dev), np.radians(1.0))        # 鲁棒尺度
    w_rot = np.exp(-(dev / sigma) ** 2)
    R_wc = geom.quat_average(R_hyps, weights=w_rot)      # 加权再平均
    dev2 = np.array([geom.angle_between(R_wc, R) for R in R_hyps])
    return R_wc, w_rot, dev2


def _geometric_conditioning(dirs):
    """w_geo_i = 平均 |sin(夹角(v_i, v_j))|, 偏好大基线/非平行的射线。"""
    k = len(dirs)
    if k < 2:
        return np.ones(k)
    w = np.zeros(k)
    for i in range(k):
        sins = []
        for j in range(k):
            if i == j:
                continue
            cross = np.linalg.norm(np.cross(dirs[i], dirs[j]))
            sins.append(min(cross, 1.0))   # |sin| (单位向量)
        w[i] = np.mean(sins) if sins else 1.0
    return w


def _irls_triangulate(origins, dirs, w_prior, iters=10, tukey_c=4.685):
    """加权多视角射线三角化 (式1) + Tukey IRLS。

    min_t Σ Ω_i ||P_i (t - c_i)||², P_i = I - v_i v_iᵀ。
    Ω_i = w_prior_i · w_res_i, w_res 由残差经 Tukey 自适应下权。
    """
    k = len(origins)
    P = np.stack([np.eye(3) - np.outer(v, v) for v in dirs])   # (k,3,3)
    w_res = np.ones(k)

    def solve(weights):
        A = np.zeros((3, 3))
        b = np.zeros(3)
        for i in range(k):
            Wi = weights[i] * P[i]
            A += Wi
            b += Wi @ origins[i]
        return np.linalg.lstsq(A, b, rcond=None)[0]

    t = solve(w_prior)
    for _ in range(iters):
        # 残差 = 点到射线 i 的垂距
        res = np.array([np.linalg.norm(P[i] @ (t - origins[i])) for i in range(k)])
        mad = np.median(res) + 1e-6
        u = res / (tukey_c * 1.4826 * mad)
        w_res = np.where(u < 1.0, (1.0 - u ** 2) ** 2, 0.0)    # Tukey biweight
        omega = w_prior * w_res
        if omega.sum() < 1e-9:
            omega = w_prior
        t = solve(omega)
    return t, w_res


def estimate_world_pose(ref_world_poses, extris_ref, extri_query,
                        irls_iters=10, tukey_c=4.685) -> PoseEstimate:
    """加权运动平均主入口。

    Args:
        ref_world_poses: list of (4,4) T_wc_i, 参考帧世界位姿 (度量, cam->world)
        extris_ref: list of (3,4) VGGT extri (world_v->cam_i), 与 refs 一一对应
        extri_query: (3,4) VGGT extri (world_v->cam_q)
    """
    R_hyps, origins, dirs = _ray_geometry(ref_world_poses, extris_ref, extri_query)
    R_wc, w_rot, dev = _rotation_consensus(R_hyps)
    w_geo = _geometric_conditioning(dirs)
    w_geo = w_geo / (w_geo.max() + 1e-9)
    w_prior = w_rot * w_geo
    t_wc, w_res = _irls_triangulate(origins, dirs, w_prior,
                                    iters=irls_iters, tukey_c=tukey_c)
    omega = w_prior * w_res
    return PoseEstimate(
        T_wc=geom.Rt_to_T(R_wc, t_wc),
        w_rot=w_rot, w_geo=w_geo, omega=omega,
        ray_origins=origins, ray_dirs=dirs,
        rot_dev_deg=np.degrees(dev),
    )
