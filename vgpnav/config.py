"""VGP-Nav 复现的集中配置。

路径默认指向 L40 上的 memory-nav 资产 (VGGT 源码/权重、cam 内参、C8 数据),
不复制重资产。超参数与论文一致 (k=10, N=4k, λ_pos=0.2, λ_ang=0.6, λ_sim=0.2)。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ---- 路径 ----
    memory_nav_root: str = "/home/ubuntu/Disk/codes/jianxiong/memory-nav"
    proj_root: str = _PROJ
    data_dir: str = ""          # 默认 {mn}/Mapping_C8
    cam_params: str = ""        # 默认 {mn}/cam/params.yaml
    vggt_weights: str = ""      # 默认 {mn}/pretrained/vggt-1b/model.pt
    out_dir: str = ""           # 默认 {proj}/outputs
    db_dir: str = ""            # 默认 {out}/db
    device: str = "cuda:3"
    camera: str = "camera_1"
    vpr_backend: str = "selavpr"   # selavpr (SelaVPR++) | dinov2

    # ---- 鱼眼->针孔 去畸变 ----
    undist_w: int = 640
    undist_h: int = 480
    undist_hfov: float = 110.0
    undist_pitch_down: float = 0.0

    # ---- 物理先验 (尺度锚) ----
    camera_height_m: float = 1.3   # 相机离地高度 (用户提供)

    # ---- 离线建库 (VGGT 分块链式重建) ----
    map_stride: int = 1            # 建轨迹时对 626 帧的抽样步长 (1=全帧, ~1m间距, VGGT重叠更好)
    db_sub: int = 3                # 轨迹帧中每 db_sub 取 1 作为 DB
    n_query: int = 25              # 从非 DB 轨迹帧中均匀抽多少作 query 评测
    chunk_size: int = 10           # VGGT 滑窗大小 (~10m, 重叠充分)
    chunk_overlap: int = 6         # 相邻窗口重叠帧数 (用于 sim3 对齐链接)

    # ---- 几何感知检索 (Algorithm 1) ----
    k_refs: int = 10               # 目标参考数 k
    search_pool_N: int = 40        # 搜索池 N = 4k
    lambda_pos: float = 0.2        # 平移多样性权重
    lambda_ang: float = 0.6        # 朝向多样性权重
    lambda_sim: float = 0.2        # VPR 相似度权重 (J 中 S_norm 的作用)
    outlier_percentile: float = 95.0
    retrieval_radius_m: float = 7.0  # 以最可信匹配为锚的半径门控 (抗强感知混叠)
    anchor_topm: int = 5             # 取 VPR 前 m 个的几何中位作锚

    # ---- 加权运动平均 ----
    irls_iters: int = 10
    tukey_c: float = 4.685         # Tukey robust M-estimator 常数 (单位: 残差/MAD)

    # ---- 地面锚定尺度恢复 + 占据图 ----
    ground_band_m: float = 0.15    # 贴地(可通行)高度阈值
    occ_resolution: float = 0.05   # 占据栅格分辨率 m/cell
    occ_range_m: float = 8.0       # 以相机为中心的占据图半径 (m)
    point_conf_thresh: float = 0.0 # VGGT depth_conf 过滤阈值 (0=不过滤, 后续按分位)

    def __post_init__(self):
        mn = self.memory_nav_root
        self.data_dir = self.data_dir or os.path.join(mn, "Mapping_C8")
        self.cam_params = self.cam_params or os.path.join(mn, "cam", "params.yaml")
        self.vggt_weights = self.vggt_weights or os.path.join(
            mn, "pretrained", "vggt-1b", "model.pt")
        self.out_dir = self.out_dir or os.path.join(self.proj_root, "outputs")
        self.db_dir = self.db_dir or os.path.join(self.out_dir, "db")
        os.makedirs(self.out_dir, exist_ok=True)
