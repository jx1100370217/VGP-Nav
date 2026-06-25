"""VGP-Nav 复现的集中配置 (多数据集)。

数据集相关参数 (数据路径 / 相机高度 / 环视布局 / 内参 / 抽样) 集中在 `datasets.py` 的 DATASETS;
此处 `Config(dataset="名字")` 选择, 产物隔离到 `outputs/{名字}/`。**加新数据集只改 datasets.py。**

非数据集相关的超参 (检索 k=10/N=4k、运动平均、占据阈值等) 与论文一致, 所有数据集共用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .datasets import get_dataset

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    # ---- 数据集选择 (见 datasets.DATASETS) ----
    # 默认读环境变量 VGPNAV_DATASET (没有则 Mapping_C8); 也可显式 Config(dataset="名字")。
    # 这样所有脚本里的 cfg=Config() 无需改动, 跑别的数据集只需 export VGPNAV_DATASET=名字。
    dataset: str = field(default_factory=lambda: os.environ.get("VGPNAV_DATASET", "Mapping_C8"))

    # ---- 路径 (留空 → 按 dataset 自动填) ----
    memory_nav_root: str = "/home/ubuntu/Disk/codes/jianxiong/memory-nav"
    proj_root: str = _PROJ
    data_dir: str = ""          # 默认 = dataset.data_dir
    cam_params: str = ""        # 默认 = dataset.cam_params 或 {mn}/cam/params.yaml
    vggt_weights: str = ""      # 默认 {mn}/pretrained/vggt-1b/model.pt
    out_dir: str = ""           # 默认 {proj}/outputs/{dataset}
    db_dir: str = ""            # 默认 {out}/db
    device: str = "cuda:3"
    camera: str = "camera_1"    # 定位主锚相机
    vpr_backend: str = "selavpr"   # selavpr (SelaVPR++) | dinov2

    # ---- 鱼眼->针孔 去畸变 ----
    undist_w: int = 640
    undist_h: int = 480
    undist_hfov: float = 110.0
    undist_pitch_down: float = 0.0

    # ---- 物理先验 (尺度锚; 0 → 按 dataset 填) ----
    camera_height_m: float = 0.0

    # ---- 4 环视物理布局 (web 用; None → 按 dataset 填) ----
    cam_layout: list = None

    # ---- 离线建库 (0 → 按 dataset 填) ----
    map_stride: int = 0            # 轨迹帧抽样步长
    db_sub: int = 3                # 每 db_sub 帧取 1 作 DB
    n_query: int = 0               # 评测 query 帧数
    chunk_size: int = 10           # VGGT 滑窗大小
    chunk_overlap: int = 6         # 相邻窗口重叠帧数

    # ---- 几何感知检索 (Algorithm 1) ----
    k_refs: int = 10
    search_pool_N: int = 40
    lambda_pos: float = 0.2
    lambda_ang: float = 0.6
    lambda_sim: float = 0.2
    outlier_percentile: float = 95.0
    retrieval_radius_m: float = 7.0
    anchor_topm: int = 5

    # ---- 加权运动平均 ----
    irls_iters: int = 10
    tukey_c: float = 4.685

    # ---- 地面锚定尺度恢复 + 占据图 ----
    ground_band_m: float = 0.15
    occ_resolution: float = 0.05
    occ_range_m: float = 8.0
    point_conf_thresh: float = 0.0

    def __post_init__(self):
        mn = self.memory_nav_root
        spec = get_dataset(self.dataset)
        # 数据集相关项: 显式给值则尊重, 否则用 spec
        self.data_dir = self.data_dir or spec.data_dir
        self.cam_params = self.cam_params or spec.cam_params or os.path.join(mn, "cam", "params.yaml")
        self.camera_height_m = self.camera_height_m or spec.camera_height_m
        self.cam_layout = self.cam_layout if self.cam_layout is not None else spec.cam_layout
        self.map_stride = self.map_stride or spec.map_stride
        self.n_query = self.n_query or spec.n_query
        # 共用资产
        self.vggt_weights = self.vggt_weights or os.path.join(mn, "pretrained", "vggt-1b", "model.pt")
        # 产物按数据集隔离: outputs/{dataset}/{db,query,web}
        self.out_dir = self.out_dir or os.path.join(self.proj_root, "outputs", self.dataset)
        self.db_dir = self.db_dir or os.path.join(self.out_dir, "db")
        os.makedirs(self.out_dir, exist_ok=True)
