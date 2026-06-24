"""合成数据验证加权运动平均的数学正确性。

构造: 已知世界位姿的 k refs + 1 query; 用随机 Sim(3) (s,Rg,tg) 把世界系映射到
"VGGT 公共系" 生成 extri。理论上无噪声时应精确恢复 query 世界位姿。
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav import geom
from vgpnav import motion_averaging as MA


def make_vggt_extri(T_wc, s, Rg, tg, rot_noise_deg=0.0, c_noise=0.0, rng=None):
    """由相机世界位姿 T_wc (cam->world) + 世界->VGGT 的 Sim3, 生成 VGGT extri。

    世界->VGGT 作用于点: X_v = s Rg X_w + tg。
    cam->world_v 旋转 = Rg @ R_wc; 中心 C_v = s Rg C_w + tg。
    """
    R_wc = T_wc[:3, :3]
    C_w = T_wc[:3, 3]
    R_c2v = Rg @ R_wc
    C_v = s * Rg @ C_w + tg
    if rot_noise_deg > 0 and rng is not None:
        n = rng.normal(0, np.radians(rot_noise_deg), 3)
        R_c2v = R_c2v @ Rot.from_rotvec(n).as_matrix()
    if c_noise > 0 and rng is not None:
        C_v = C_v + rng.normal(0, c_noise, 3)
    T_c2v = geom.Rt_to_T(R_c2v, C_v)
    return geom.inv_T(T_c2v)[:3, :4]   # world_v -> cam


def random_world_poses(k, rng):
    poses = []
    for _ in range(k + 1):
        R = Rot.from_rotvec(rng.normal(0, 1.0, 3)).as_matrix()
        t = rng.uniform(-4, 4, 3)
        poses.append(geom.Rt_to_T(R, t))
    return poses[:-1], poses[-1]   # refs, query


def pose_err(T_est, T_gt):
    te = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
    re = np.degrees(geom.angle_between(T_est[:3, :3], T_gt[:3, :3]))
    return te, re


def run_case(name, rot_noise_deg=0.0, c_noise=0.0, k=8, seed=0):
    rng = np.random.default_rng(seed)
    refs, query = random_world_poses(k, rng)
    # 随机 Sim3 (世界->VGGT)
    s = rng.uniform(0.2, 5.0)
    Rg = Rot.from_rotvec(rng.normal(0, 1.0, 3)).as_matrix()
    tg = rng.uniform(-2, 2, 3)
    extris_ref = [make_vggt_extri(T, s, Rg, tg, rot_noise_deg, c_noise * s, rng)
                  for T in refs]
    extri_q = make_vggt_extri(query, s, Rg, tg, rot_noise_deg, c_noise * s, rng)
    est = MA.estimate_world_pose(refs, extris_ref, extri_q)
    te, re = pose_err(est.T_wc, query)
    print(f"[{name}] 平移误差={te:.4g} m, 旋转误差={re:.4g} deg, "
          f"sim3 尺度 s={s:.3f}")
    return te, re, est


def test_noiseless():
    te, re, _ = run_case("无噪声", 0.0, 0.0)
    assert te < 1e-6, f"无噪声平移误差过大: {te}"
    assert re < 1e-4, f"无噪声旋转误差过大: {re}"


def test_noisy():
    tes, res = [], []
    for seed in range(8):
        te, re, _ = run_case(f"含噪 seed{seed}", rot_noise_deg=1.0,
                             c_noise=0.01, seed=seed)
        tes.append(te)
        res.append(re)
    print(f"含噪均值: 平移={np.mean(tes):.4g} m, 旋转={np.mean(res):.4g} deg")
    assert np.mean(tes) < 0.2, f"含噪平移误差过大: {np.mean(tes)}"
    assert np.mean(res) < 2.0, f"含噪旋转误差过大: {np.mean(res)}"


def test_outlier_downweight():
    """注入一个离群参考 (大旋转噪声), 验证它在 omega 中被显著下权。"""
    rng = np.random.default_rng(3)
    refs, query = random_world_poses(8, rng)
    s = 2.0
    Rg = Rot.from_rotvec(rng.normal(0, 1.0, 3)).as_matrix()
    tg = np.array([1.0, -1.0, 0.5])
    extris_ref = [make_vggt_extri(T, s, Rg, tg) for T in refs]
    # 把第 3 个参考改成离群 (45° 旋转 + 大平移噪声)
    bad = make_vggt_extri(refs[3], s, Rg, tg, rot_noise_deg=45.0,
                          c_noise=1.0, rng=rng)
    extris_ref[3] = bad
    extri_q = make_vggt_extri(query, s, Rg, tg)
    est = MA.estimate_world_pose(refs, extris_ref, extri_q)
    te, re = pose_err(est.T_wc, query)
    print(f"[离群] 平移误差={te:.4g} m, 旋转误差={re:.4g} deg")
    print(f"[离群] omega={np.round(est.omega, 3)}, "
          f"离群帧(idx3) omega={est.omega[3]:.3f}, 其余中位={np.median(np.delete(est.omega,3)):.3f}")
    assert est.omega[3] < np.median(np.delete(est.omega, 3)), \
        "离群参考未被下权"
    assert te < 0.5, f"含离群时定位崩溃: {te}"


if __name__ == "__main__":
    test_noiseless()
    test_noisy()
    test_outlier_downweight()
    print("\n所有运动平均测试通过 ✓")
