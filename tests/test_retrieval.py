"""合成数据验证几何感知检索 (Algorithm 1)。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.retrieval import geometry_aware_retrieval, geometric_median


def _rand_dirs(n, rng):
    v = rng.normal(0, 1, (n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_geometric_median():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 0.3, (40, 3))
    X[0] = [100, 0, 0]      # 离群点
    med = geometric_median(X)
    assert np.linalg.norm(med) < 1.0, f"几何中位数被离群点带偏: {med}"
    print(f"[几何中位数] {np.round(med,3)} (离群点不影响) ✓")


def test_outlier_rejection():
    rng = np.random.default_rng(1)
    M = 40
    centers = rng.normal(0, 0.5, (M, 3))    # 主簇
    centers[5] = [50, 0, 0]                  # 远离群 1
    centers[7] = [0, -60, 0]                 # 远离群 2
    dirs = _rand_dirs(M, rng)
    sims = rng.uniform(0.5, 0.85, M)         # best 落在主簇 (非离群)
    sims[20] = 0.95
    res = geometry_aware_retrieval(sims, centers, dirs, k=10, search_pool_N=40)
    assert 5 not in res.c_valid and 7 not in res.c_valid, \
        f"远离群未被剔除: c_valid={res.c_valid}"
    print(f"[离群剔除] 远点 5,7 被剔除; c_valid 数={len(res.c_valid)} ✓")


def test_diversity_selection():
    """高相似度点聚成一簇 + 低相似度点空间分散; 验证选择会跳出簇取分散参考。"""
    rng = np.random.default_rng(2)
    cluster = rng.normal(0, 0.3, (12, 3))            # x≈0 紧簇, 高 sim
    spread = np.zeros((18, 3))
    spread[:, 0] = np.linspace(2, 15, 18)            # 沿 x 分散, 低 sim
    spread[:, 1] = rng.normal(0, 0.3, 18)
    centers = np.vstack([cluster, spread])
    dirs = _rand_dirs(30, rng)
    sims = np.concatenate([rng.uniform(0.85, 0.95, 12),
                           rng.uniform(0.55, 0.75, 18)])
    k = 8
    res = geometry_aware_retrieval(sims, centers, dirs, k=k, search_pool_N=30,
                                   lambda_pos=0.2, lambda_ang=0.6)

    def extent(idx):
        pts = centers[idx]
        from scipy.spatial.distance import pdist
        return pdist(pts).max()

    topk_sim = list(np.argsort(-sims)[:k])
    e_sel = extent(res.selected)
    e_topk = extent(topk_sim)
    print(f"[多样性] 选中帧空间跨度={e_sel:.2f} vs 纯top-k-sim跨度={e_topk:.2f}")
    print(f"         选中索引={sorted(res.selected)}")
    assert len(res.selected) == k
    assert res.best in res.selected
    assert e_sel > 2.0 * e_topk, "多样性选择未显著扩大空间覆盖"
    print("[多样性] 选择显著扩大空间覆盖 ✓")


if __name__ == "__main__":
    test_geometric_median()
    test_outlier_rejection()
    test_diversity_selection()
    print("\n所有检索测试通过 ✓")
