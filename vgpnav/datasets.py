"""数据集注册表 —— 多数据集兼容的核心。

**加入新数据集 = 在 DATASETS 增加一条 DatasetSpec, 无需改动任何核心代码。**
各脚本通过 `Config(dataset="名字")` 选择; 产物隔离到 `outputs/{名字}/`。

约定: 每个数据集的图片目录里, 4 路环视鱼眼命名为 `{timestamp}_camera_{1..4}.jpg`
(与 C8 一致); 不同数据集可有不同的物理布局/相机高度/内参。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetSpec:
    name: str                       # 数据集标识 (= outputs 下的子目录名)
    data_dir: str                   # 4 路环视图片目录 (含 {ts}_camera_{1..4}.jpg)
    camera_height_m: float          # 相机离地高度 (米), 地面锚定尺度用
    cam_layout: list                # web 4 环视 2x2 布局: [(方位标签, 相机号)],
                                    #   顺序 = 左上, 右上, 左下, 右下 (前进方向恒在上排)
    cam_params: str = ""            # 相机内参 yaml ("" → 用 memory-nav/cam/params.yaml)
    map_stride: int = 1             # 轨迹帧抽样步长 (短序列用 1)
    n_query: int = 25               # 留出的评测 query 帧数


# ────────────────────────────────────────────────────────────────────────
# 已注册数据集。加新数据集: 复制一条, 改 name/data_dir/camera_height_m/cam_layout 即可。
# ────────────────────────────────────────────────────────────────────────
DATASETS = {
    # DEEPROUTE 办公室一层, 环视采集, 相机离地 1.3m。
    # 物理布局: camera_1=前左, camera_2=前右, camera_3=后右, camera_4=后左。
    "Mapping_C8": DatasetSpec(
        name="Mapping_C8",
        data_dir="/home/ubuntu/Disk/codes/jianxiong/memory-nav/Mapping_C8",
        camera_height_m=1.3,
        cam_layout=[("前左", 1), ("前右", 2), ("后左", 4), ("后右", 3)],
        map_stride=1, n_query=25,
    ),
    # 创富大厦, 头盔环视采集 (采集者身高 173cm, 头戴头盔)。
    # 相机为同型号 AVM 鱼眼 (1920x1536), 复用 C8 内参 (camera_1/4 一组、2/3 一组, 与本数据 AVM 分组一致)。
    # extracted camera_1..4 = 外参表 AVM CAM 1..4:
    #   camera_1=左后(Yaw 142.5°), camera_2=左前(37.5°), camera_3=右前(-37.5°), camera_4=右后(-142.5°)。
    # 相机离地 ≈ 1.70m: 主锚 camera_1=AVM CAM1, "头盔相机高度"图示其在头盔底部边缘上方 95mm;
    #   头盔底部边缘(帽壳下沿)约在头顶下方 12cm, 故离地 ≈ 身高173cm - 12cm + 9.5cm ≈ 170cm。
    #   (若身高按 175cm 则 ≈ 1.72m; 最准应实测相机离地高度后改此值。)
    "ChuangfuTower": DatasetSpec(
        name="ChuangfuTower",
        data_dir="/home/ubuntu/Disk/codes/jianxiong/VGP-Nav/data/ChuangfuTower/extracted_data",
        camera_height_m=1.70,
        cam_layout=[("左前", 2), ("右前", 3), ("左后", 1), ("右后", 4)],
        map_stride=1, n_query=8,
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"未知数据集 '{name}', 可选: {list(DATASETS)}")
    return DATASETS[name]
