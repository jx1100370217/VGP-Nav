"""回环闭合 + 位姿图优化(PGO): 修复单目 VGGT 链式建图的累计漂移/割裂。

思路:
 1. 对全部轨迹关键帧算 SelaVPR++ 外观描述子(缓存), 检测回环(外观相似+序号远+当前空间近以抗混叠)。
 2. 把轨迹投到 SE(2) (x,y,航向ψ), 建位姿图: 里程计边(相邻+窗内跳边, 来自链式位姿)
    + 回环边(外观匹配的两帧约束为同一位姿)。scipy.least_squares + 稀疏Jacobian + 鲁棒核求解。
 3. 楼层为平面 -> z 拉平到相机高度 1.3m, 消除垂直弯曲。
 4. 用校正后的 (Δψ, x, y, z) 重构每帧 T_wc (Rz(Δψ)@R_old 保留俯仰/横滚), 写出校正轨迹。

输出: outputs/db/trajectory_pgo.npz (poses/centers/view_dirs) + outputs/db/pgo_before_after.png
后续: 用校正位姿重建 4 路点云 + 重导出。

  python scripts/pgo_fix.py [--sim 0.55] [--gap 40] [--dmax 12] [--wloop 3.0]
"""
import argparse
import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vgpnav.config import Config
from vgpnav.database import list_frame_files
from vgpnav.undistort import PinholeUndistorter, load_camera_params
from vgpnav.viz import setup_cjk_font
from vgpnav.vpr import make_vpr

setup_cjk_font()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(R):
    """相机前向(世界)的航向角: view_dir = R[:,2]。"""
    return np.arctan2(R[1, 2], R[0, 2])


def rel_se2(xi, yi, pi, xj, yj, pj):
    """i 帧坐标系下 j 的相对 SE(2): (dx_local, dy_local, dψ)。"""
    dx, dy = xj - xi, yj - yi
    c, s = np.cos(pi), np.sin(pi)
    return c * dx + s * dy, -s * dx + c * dy, wrap(pj - pi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", type=float, default=0.55, help="回环外观相似度阈值")
    ap.add_argument("--gap", type=int, default=40, help="回环最小序号间隔")
    ap.add_argument("--dmax", type=float, default=12.0, help="回环当前空间距离上限(抗混叠)")
    ap.add_argument("--wloop", type=float, default=3.0, help="回环边权重(相对里程计)")
    ap.add_argument("--topk", type=int, default=4, help="每帧保留前k个回环")
    args = ap.parse_args()

    cfg = Config()
    # 始终从原始轨迹出发优化 (避免在已校正结果上叠加)
    traj_path = os.path.join(cfg.db_dir, "trajectory_orig.npz")
    if not os.path.exists(traj_path):
        traj_path = os.path.join(cfg.db_dir, "trajectory.npz")
    print(f"输入轨迹: {os.path.basename(traj_path)}")
    traj = np.load(traj_path)
    poses = traj["poses"].astype(np.float64)      # (N,4,4) T_wc
    frame_idx = [int(v) for v in traj["frame_idx"]]
    N = len(poses)
    files = list_frame_files(cfg)
    print(f"轨迹 {N} 帧")

    # ---- 1. 描述子 + 回环相似度 (4路环视交叉) ----
    # 机器人重访常"反向"经过同一地点 (如帧621/501): 单目 camera_1 此时看到的景象相反,
    # 相似度低 (0.32) 会漏检回环 -> 该处回环不被 PGO 对齐 -> 建图割裂出虚假障碍。
    # 改用4路环视交叉最大相似度 (任一路对任一路), 帧621/501 -> 0.77, 能识别反向回环。
    desc4_path = os.path.join(cfg.db_dir, "allframe_desc_4cam.npy")
    if os.path.exists(desc4_path):
        D4 = np.load(desc4_path).astype(np.float32)            # (N,4,Dd)
        D4 /= (np.linalg.norm(D4, axis=2, keepdims=True) + 1e-9)
        print(f"载入4路描述子 {D4.shape}, 回环相似度=4路交叉最大")
        Dflat = D4.reshape(len(D4) * 4, -1)
        S = (Dflat @ Dflat.T).reshape(N, 4, N, 4).max(axis=(1, 3))  # (N,N) 4路交叉
    else:
        # 回退: 单目 camera_1 (旧行为, 漏检反向回环)
        desc_path = os.path.join(cfg.db_dir, "allframe_desc.npy")
        if not os.path.exists(desc_path):
            raise SystemExit("缺 allframe_desc_4cam.npy, 请先跑 scripts/compute_4cam_desc.py")
        D = np.load(desc_path).astype(np.float32)
        D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
        print(f"载入单目描述子 {D.shape} (回退, 会漏检反向回环)")
        S = D @ D.T

    # ---- 2. 回环检测 ----
    C = poses[:, :3, 3]
    loops = []
    for i in range(N):
        cand = [(j, S[i, j]) for j in range(i + args.gap, N)
                if S[i, j] > args.sim and np.linalg.norm(C[i, :2] - C[j, :2]) < args.dmax]
        cand.sort(key=lambda t: -t[1])
        for j, sim in cand[:args.topk]:
            loops.append((i, j, float(sim)))
    # 去重 (i,j)
    loops = list({(min(i, j), max(i, j)): (i, j, s) for i, j, s in loops}.values())
    print(f"回环边 {len(loops)} (sim>{args.sim}, gap>{args.gap}, dmax<{args.dmax}m)")
    if loops:
        dd = [np.linalg.norm(C[i, :2] - C[j, :2]) for i, j, _ in loops]
        print(f"  回环对当前空间距离: 中位={np.median(dd):.2f} 最大={max(dd):.2f} m (=待消除的漂移)")

    # ---- 3. SE(2) 位姿图 ----
    x = C[:, 0].copy()
    y = C[:, 1].copy()
    psi = np.array([yaw_of(poses[i, :3, :3]) for i in range(N)])

    # 里程计边: 窗内跳边 (保形刚度)
    odom = []
    for gap in (1, 2, 3, 5, 8):
        for i in range(N - gap):
            j = i + gap
            zx, zy, zp = rel_se2(x[i], y[i], psi[i], x[j], y[j], psi[j])
            odom.append((i, j, zx, zy, zp, 1.0, 1.0))    # aw=1: 约束相对航向
    edges = list(odom)
    # 回环: 仅约束"同地点"(位置), 不约束航向 (走廊多为往返反向, 航向差~180°由里程计决定)
    for i, j, _ in loops:
        edges.append((i, j, 0.0, 0.0, 0.0, args.wloop, 0.0))   # aw=0: 不约束航向
    n_anchor = 1  # 固定 0 号位姿 (规约自由度)

    def unpack(p):
        return p[0::3], p[1::3], p[2::3]

    def resid(p):
        X, Y, P = unpack(p)
        r = np.empty(len(edges) * 3 + 3)
        for k, (i, j, zx, zy, zp, w, aw) in enumerate(edges):
            dx, dy, dp = rel_se2(X[i], Y[i], P[i], X[j], Y[j], P[j])
            sw = np.sqrt(w)
            r[3 * k] = sw * (dx - zx)
            r[3 * k + 1] = sw * (dy - zy)
            r[3 * k + 2] = sw * aw * wrap(dp - zp)
        # 锚 0 号
        wa = 100.0
        r[-3] = wa * (X[0] - C[0, 0])
        r[-2] = wa * (Y[0] - C[0, 1])
        r[-1] = wa * wrap(P[0] - psi[0])
        return r

    # 稀疏 Jacobian
    M = len(edges) * 3 + 3
    spar = lil_matrix((M, 3 * N), dtype=np.uint8)
    for k, (i, j, *_), in enumerate(edges):
        for rr in range(3):
            for c in (3 * i, 3 * i + 1, 3 * i + 2, 3 * j, 3 * j + 1, 3 * j + 2):
                spar[3 * k + rr, c] = 1
    for rr, c in enumerate((0, 1, 2)):
        spar[M - 3 + rr, c] = 1

    p0 = np.empty(3 * N)
    p0[0::3], p0[1::3], p0[2::3] = x, y, psi
    print(f"PGO 求解: {N} 位姿, {len(edges)} 边, {M} 残差...")
    sol = least_squares(resid, p0, jac_sparsity=spar, loss="soft_l1",
                        f_scale=0.5, max_nfev=400, verbose=1)
    X, Y, P = unpack(sol.x)

    # ---- 4. 重构校正后 T_wc (Rz(Δψ)@R_old, z=1.3) ----
    def Rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    poses_new = poses.copy()
    for i in range(N):
        dpsi = wrap(P[i] - psi[i])
        poses_new[i, :3, :3] = Rz(dpsi) @ poses[i, :3, :3]
        poses_new[i, :3, 3] = [X[i], Y[i], cfg.camera_height_m]
    centers_new = poses_new[:, :3, 3]
    view_new = np.stack([poses_new[i, :3, :3] @ np.array([0, 0, 1.0]) for i in range(N)])

    np.savez(os.path.join(cfg.db_dir, "trajectory_pgo.npz"),
             frame_idx=np.array(frame_idx), poses=poses_new,
             centers=centers_new, view_dirs=view_new)

    # 报告
    ext0 = (C[:, 0].ptp(), C[:, 1].ptp())
    ext1 = (centers_new[:, 0].ptp(), centers_new[:, 1].ptp())
    print(f"范围 前: {ext0[0]:.1f}x{ext0[1]:.1f}m  后: {ext1[0]:.1f}x{ext1[1]:.1f}m")
    if loops:
        dd1 = [np.linalg.norm(centers_new[i, :2] - centers_new[j, :2]) for i, j, _ in loops]
        print(f"回环残余距离: 前中位={np.median(dd):.2f}m 后中位={np.median(dd1):.2f}m")

    # ---- 5. 前后对比图 ----
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for ax, (cc, ttl) in zip(axs, [(C, "PGO 前 (含漂移割裂)"), (centers_new, "PGO 后 (回环闭合)")]):
        ax.plot(cc[:, 0], cc[:, 1], "-", c="#2de2e6", lw=1)
        ax.scatter(cc[:, 0], cc[:, 1], c="#5cff9d", s=4)
        for i, j, _ in loops:
            ax.plot([cc[i, 0], cc[j, 0]], [cc[i, 1], cc[j, 1]], "-r", lw=0.4, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(ttl)
    plt.tight_layout()
    out_png = os.path.join(cfg.out_dir, "query", "pgo_before_after.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=110)
    print(f"对比图 -> {out_png}")


if __name__ == "__main__":
    main()
