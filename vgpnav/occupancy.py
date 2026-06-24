"""从度量点云生成 2D 占据栅格 (论文 III-E 末)。

按点相对地面的高度分类:
  - 贴地 (h < ground_band)            -> 可通行
  - (ground_band, camera_height) 之间 -> 障碍, 投影到 2D 地面
把世界系的点光栅化进栅格; 可累积多帧 (全局地图), 也可单帧 (局部地图)。
栅格值: 0=未知, 1=可通行, 2=障碍。
"""
from __future__ import annotations

import numpy as np


class OccupancyGrid:
    def __init__(self, resolution=0.05, range_m=8.0, center_xy=(0.0, 0.0),
                 ground_band=0.15, ceil=1.3, occ_min_hits=1):
        self.res = float(resolution)
        self.range_m = float(range_m)
        self.ground_band = float(ground_band)
        self.ceil = float(ceil)
        self.occ_min_hits = int(occ_min_hits)
        self.n = int(2 * range_m / resolution)
        self.origin = np.array([center_xy[0] - range_m,
                                center_xy[1] - range_m], dtype=np.float64)
        self.occ_count = np.zeros((self.n, self.n), dtype=np.int32)
        self.free_count = np.zeros((self.n, self.n), dtype=np.int32)
        # 每格障碍点高度范围 (用于"垂直结构"判据: 真障碍跨高度, 噪声悬浮点不跨)
        self.occ_hmin = np.full((self.n, self.n), np.inf, dtype=np.float32)
        self.occ_hmax = np.full((self.n, self.n), -np.inf, dtype=np.float32)

    def _cells_h(self, xy, h):
        """世界 xy -> 栅格 (i,j), 同步返回对齐的高度 (越界已剔除)。"""
        c = np.floor((xy - self.origin) / self.res).astype(int)
        valid = (c[:, 0] >= 0) & (c[:, 0] < self.n) & \
                (c[:, 1] >= 0) & (c[:, 1] < self.n)
        return c[valid], h[valid]

    def _to_cells(self, xy):
        c = np.floor((xy - self.origin) / self.res).astype(int)
        valid = (c[:, 0] >= 0) & (c[:, 0] < self.n) & \
                (c[:, 1] >= 0) & (c[:, 1] < self.n)
        return c[valid]

    def integrate(self, P_world: np.ndarray, ground_z: float = 0.0):
        """把一组世界系度量点累积进栅格 (含每格障碍高度范围)。"""
        P = np.asarray(P_world, dtype=np.float64)
        P = P[np.isfinite(P).all(axis=1)]
        h = P[:, 2] - ground_z
        free = P[(h >= -self.ground_band) & (h < self.ground_band)]
        om = (h >= self.ground_band) & (h <= self.ceil)
        fc = self._to_cells(free[:, :2])
        oc, oh = self._cells_h(P[om][:, :2], h[om])
        if len(fc):
            np.add.at(self.free_count, (fc[:, 1], fc[:, 0]), 1)
        if len(oc):
            np.add.at(self.occ_count, (oc[:, 1], oc[:, 0]), 1)
            np.minimum.at(self.occ_hmin, (oc[:, 1], oc[:, 0]), oh.astype(np.float32))
            np.maximum.at(self.occ_hmax, (oc[:, 1], oc[:, 0]), oh.astype(np.float32))

    def grid(self) -> np.ndarray:
        """0=未知, 1=可通行, 2=障碍 (障碍优先)。"""
        g = np.zeros((self.n, self.n), dtype=np.int8)
        g[self.free_count > 0] = 1
        g[self.occ_count >= self.occ_min_hits] = 2
        return g

    def grid_robust(self, min_hits=4, min_vext=0.3) -> np.ndarray:
        """稳健占据: 障碍需 足够点密度 且 跨高度(垂直结构), 滤除悬浮噪声"虚构"障碍。
        0=未知, 1=可通行, 2=障碍。"""
        g = np.zeros((self.n, self.n), dtype=np.int8)
        g[self.free_count > 0] = 1
        vext = np.where(self.occ_count > 0, self.occ_hmax - self.occ_hmin, 0.0)
        obstacle = (self.occ_count >= min_hits) & (vext >= min_vext)
        g[obstacle] = 2
        return g

    def world_to_cell(self, xy):
        return np.floor((np.asarray(xy) - self.origin) / self.res).astype(int)

    def cell_to_world(self, ij):
        ij = np.asarray(ij)
        return self.origin + (ij[..., ::-1] + 0.5) * self.res  # (i,j)->(x,y)
