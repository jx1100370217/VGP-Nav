"""几何感知检索策略 (论文 Algorithm 1)。

在 VPR 相似度基础上, 先做空间离群剔除, 再做多样性驱动选择, 得到一组
"既相似又视角分散" 的参考帧, 为下游 VGGT 重建提供大基线、含地面的良态约束。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def geometric_median(X: np.ndarray, iters: int = 200, eps: float = 1e-6) -> np.ndarray:
    """Weiszfeld 算法求几何中位数 (对离群鲁棒)。X: (N,3)。"""
    y = X.mean(axis=0)
    for _ in range(iters):
        d = np.linalg.norm(X - y, axis=1)
        nz = d > eps
        if not nz.any():
            break
        w = 1.0 / d[nz]
        y_new = (X[nz] * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(y_new - y) < eps:
            break
        y = y_new
    return y


@dataclass
class RetrievalResult:
    selected: list          # 选中的 DB 索引 (含 best, 长度 ≤ k)
    c_init: list            # top-N 候选索引
    c_valid: list           # 离群剔除后的有效候选索引
    best: int               # VPR 最相似帧索引


def geometry_aware_retrieval(sims, centers, view_dirs, k=10, search_pool_N=40,
                             lambda_pos=0.2, lambda_ang=0.6,
                             outlier_percentile=95.0, radius_m=None,
                             anchor_topm=5) -> RetrievalResult:
    """论文 Algorithm 1 (含针对强感知混叠的鲁棒化离群剔除)。

    Args:
        sims: (M,) query 对每个 DB 帧的 VPR 相似度
        centers: (M,3) DB 帧相机中心 (世界系)
        view_dirs: (M,3) DB 帧朝向 (世界系单位向量)
        k: 目标参考数
        search_pool_N: 搜索池 N (=4k)
        radius_m: 若给定, 以最可信 top-M 匹配的几何中位为锚, 只保留半径内候选
                  (强混叠场景下比"全池几何中位+百分位"鲁棒; 论文亦显式保留 I_best)。
        anchor_topm: 取 VPR 前 m 个作锚 (最相似者最可能为真实位置)。
    """
    sims = np.asarray(sims, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    view_dirs = np.asarray(view_dirs, dtype=np.float64)
    M = len(sims)

    # ---- Phase 1: 空间离群剔除 ----
    N = min(search_pool_N, M)
    c_init = list(np.argsort(-sims)[:N])           # top-N by VPR
    best = int(c_init[0])                           # I_best = argmax sim
    if radius_m is not None:
        # 以最可信 top-M 的几何中位为锚 (抗单帧 top1 误匹配), 半径门控
        m = min(anchor_topm, N)
        anchor = geometric_median(centers[c_init[:m]])
        d = np.linalg.norm(centers[c_init] - anchor, axis=1)
        c_valid = [c_init[i] for i in range(N) if d[i] <= radius_m]
        if len(c_valid) < k:                        # 半径内不足, 取最近的若干
            order = np.argsort(d)
            c_valid = [c_init[i] for i in order[:max(k, len(c_valid))]]
        if best not in c_valid:
            c_valid.append(best)
        med = anchor
    else:
        t_init = centers[c_init]
        med = geometric_median(t_init)
        d = np.linalg.norm(t_init - med, axis=1)    # d_i
        tau = np.percentile(d, outlier_percentile)  # τ = 95 分位
        c_valid = [c_init[i] for i in range(N) if d[i] <= tau]
        if best not in c_valid:
            c_valid.append(best)                    # 显式保留 best

    # ---- Phase 2: 多样性驱动选择 (类 FPS 贪心) ----
    sv = sims[c_valid]
    s_norm = (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)   # 归一到 [0,1]
    snorm = {c: float(s_norm[i]) for i, c in enumerate(c_valid)}
    # L_scale: 候选相对中位数的最大距离 (位置项归一化尺度)
    dv = np.linalg.norm(centers[c_valid] - med, axis=1)
    L_scale = float(dv.max()) + 1e-9

    selected = [best]
    remaining = [c for c in c_valid if c != best]
    while len(selected) < k and remaining:
        best_c, best_J = None, -np.inf
        for c in remaining:
            d_geo = np.inf
            for s in selected:
                pos = np.linalg.norm(centers[c] - centers[s]) / L_scale
                cosang = np.clip(view_dirs[c] @ view_dirs[s], -1, 1)
                ang = np.arccos(cosang) / np.pi
                delta = lambda_pos * pos + lambda_ang * ang
                d_geo = min(d_geo, delta)
            J = d_geo * snorm[c]                    # J = D_geo · S_norm
            if J > best_J:
                best_J, best_c = J, c
        selected.append(best_c)
        remaining.remove(best_c)

    return RetrievalResult(selected=selected, c_init=c_init,
                           c_valid=c_valid, best=best)
