"""A* 路径规划 (论文用 A* 在度量占据栅格上规划无碰撞轨迹)。"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.ndimage import binary_dilation


def inflate_obstacles(occ_grid, radius_cells):
    """把障碍 (==2) 按机器人半径膨胀, 返回布尔阻挡图。"""
    obst = (occ_grid == 2)
    if radius_cells <= 0:
        return obst
    r = int(radius_cells)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disk = (xx ** 2 + yy ** 2) <= r ** 2
    return binary_dilation(obst, structure=disk)


def astar(occ_grid, start_ij, goal_ij, robot_radius_cells=3,
          allow_unknown=True):
    """8 连通 A*。栅格值 0=未知,1=可通行,2=障碍。

    返回 path [(i,j),...] (含起终点) 或 None。i=行(y), j=列(x)。
    """
    blocked = inflate_obstacles(occ_grid, robot_radius_cells)
    n_rows, n_cols = occ_grid.shape

    def passable(ij):
        i, j = ij
        if not (0 <= i < n_rows and 0 <= j < n_cols):
            return False
        if blocked[i, j]:
            return False
        if not allow_unknown and occ_grid[i, j] == 0:
            return False
        return True

    start, goal = tuple(start_ij), tuple(goal_ij)
    if not passable(goal):
        return None
    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def h(a):
        return np.hypot(a[0] - goal[0], a[1] - goal[1])

    open_heap = [(h(start), 0.0, start)]
    came = {}
    gscore = {start: 0.0}
    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return path[::-1]
        for di, dj in nbrs:
            nb = (cur[0] + di, cur[1] + dj)
            if not passable(nb):
                continue
            step = np.hypot(di, dj)
            ng = g + step
            if ng < gscore.get(nb, np.inf):
                gscore[nb] = ng
                came[nb] = cur
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))
    return None
