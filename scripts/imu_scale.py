"""用 IMU 步频 PDR 标定 VGGT 轨迹全局尺度 k (绝对尺度, 室内外通用, 无需外部地图)。

连续行走、几乎无静止段 -> 不能 ZUPT 双积分。改用步频法(手机计步器原理):
|lin|(比力模长, 不依赖姿态)带通滤波 -> 数行走步数 -> ×平均步长 = 真实路程 ->
/ VGGT 轨迹长度 = 尺度因子 k。步数检测准, 步长用经验值(主要不确定来源)。
"""
import csv
import glob
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config


def load_imu(path):
    ts, lin = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ts.append(float(row["timestamp_sec"]))
            lin.append([float(row["lin_x"]), float(row["lin_y"]), float(row["lin_z"])])
    return np.array(ts), np.array(lin)


cfg = Config()
ts_i, lin = load_imu(os.path.join(cfg.data_dir, "imu.csv"))
fs = (len(ts_i) - 1) / (ts_i[-1] - ts_i[0])
print("IMU %d 样本, %.1fHz, %.1fs" % (len(ts_i), fs, ts_i[-1] - ts_i[0]))

gmag = np.linalg.norm(lin, axis=1)

# 运动时段: 短窗 lin 方差大 (行走加速度波动明显)
W = 40
std_roll = np.array([lin[max(0, i - W):i + W].std(0).max() for i in range(len(ts_i))])
mov = std_roll > 0.5
print("运动时段占比 %.1f%%  (累计 %.0fs)" % (100 * mov.mean(), mov.sum() / fs))

# 带通 0.7-2.8Hz (步频范围), 数步
b, a = butter(2, [0.7, 2.8], btype="band", fs=fs)
sig = filtfilt(b, a, gmag - gmag.mean())
peaks, _ = find_peaks(sig, height=0.4, distance=int(fs * 0.3))   # 步间≥0.3s
peaks = peaks[mov[peaks]]                                        # 仅运动时段
nstep = len(peaks)
walk_t = mov.sum() / fs
print("检出步数 %d, 行走时长 %.0fs, 步频 %.2f Hz" % (nstep, walk_t, nstep / max(walk_t, 1)))

# VGGT 轨迹长度 (当前尺度)
traj = np.load(os.path.join(cfg.db_dir, "trajectory.npz"))
centers = traj["centers"]
L_vggt = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1)))
print("VGGT 轨迹长 %.1fm (%d 关键帧)" % (L_vggt, len(centers)))

print("\n步长假设 -> 真实路程 -> 尺度 k:")
for SL in [0.55, 0.60, 0.65, 0.70, 0.75]:
    L_real = nstep * SL
    print("  步长 %.2fm: 路程 %5.0fm, k = %.2f" % (SL, L_real, L_real / L_vggt))
