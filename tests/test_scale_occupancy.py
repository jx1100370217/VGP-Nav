"""合成数据验证地面锚定尺度恢复 + 占据图。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav import scale_recovery as SR
from vgpnav.occupancy import OccupancyGrid


def make_scene(rng):
    # 地面 z=0 (密集) + 一个障碍盒 (2,0) z∈[0.2,0.8]
    g = np.zeros((40000, 3))
    g[:, 0] = rng.uniform(-3, 5, 40000)
    g[:, 1] = rng.uniform(-3, 3, 40000)
    g[:, 2] = rng.normal(0, 0.01, 40000)
    box = np.zeros((600, 3))
    box[:, 0] = rng.uniform(1.8, 2.2, 600)
    box[:, 1] = rng.uniform(-0.3, 0.3, 600)
    box[:, 2] = rng.uniform(0.2, 0.8, 600)
    return np.vstack([g, box])


def cam_pose_forward():
    """相机在 (0,0,1.3), 朝世界 +x 水平前看, 相机 y 轴向下(-z)。"""
    C = np.array([0.0, 0.0, 1.3])
    cam_z = np.array([1.0, 0, 0])      # forward
    cam_y = np.array([0, 0, -1.0])     # down
    cam_x = np.cross(cam_y, cam_z)     # right
    R_wc = np.column_stack([cam_x, cam_y, cam_z])
    return R_wc, C


def test_query_scale_recovery():
    rng = np.random.default_rng(0)
    P_w = make_scene(rng)
    R_wc, C = cam_pose_forward()
    alpha = 0.37                       # VGGT 任意尺度
    P_cam_true = (P_w - C) @ R_wc      # = R_wc^T (P_w - C)
    P_cam_vggt = alpha * P_cam_true
    P_world, s, gz_rel = SR.anchor_query_points(P_cam_vggt, R_wc, C,
                                                camera_height=1.3)
    print(f"[query尺度] 估计 scale={s:.4f}, 理论 1/alpha={1/alpha:.4f}, "
          f"地面相对高={gz_rel:.3f}")
    err = np.abs(P_world - P_w).max()
    print(f"[query尺度] 点云恢复最大误差={err:.4f} m")
    assert abs(s - 1 / alpha) < 0.05 * (1 / alpha), "尺度恢复偏差大"
    assert err < 0.1, f"点云恢复误差大: {err}"
    # 地面峰应 ≈ 0
    assert abs(SR.find_ground_peak(P_world[:, 2])) < 0.05


def test_occupancy():
    rng = np.random.default_rng(1)
    P_w = make_scene(rng)
    occ = OccupancyGrid(resolution=0.1, range_m=6.0, center_xy=(0, 0),
                        ground_band=0.15, ceil=1.3)
    occ.integrate(P_w, ground_z=0.0)
    g = occ.grid()
    # 障碍盒 (2,0) 处应为占据
    box_cell = occ.world_to_cell([2.0, 0.0])
    assert g[box_cell[1], box_cell[0]] == 2, "障碍盒未被标占据"
    # 地面 (0,0) 处应为可通行 (无障碍点投影)
    free_cell = occ.world_to_cell([0.0, 1.5])
    assert g[free_cell[1], free_cell[0]] == 1, "空地未被标可通行"
    n_occ = int((g == 2).sum())
    n_free = int((g == 1).sum())
    print(f"[占据图] 障碍格={n_occ}, 可通行格={n_free}, 障碍盒处={g[box_cell[1],box_cell[0]]}")
    print("[占据图] 障碍/可通行分类正确 ✓")


if __name__ == "__main__":
    test_query_scale_recovery()
    test_occupancy()
    print("\n所有尺度/占据测试通过 ✓")
