"""数据集注册表 —— 多数据集兼容的核心。

**加入新数据集 = 在 DATASETS 增加一条 DatasetSpec, 无需改动核心代码。**
各脚本通过 `Config(dataset="名字")` 选择 (默认读环境变量 `VGPNAV_DATASET`); 产物隔离到 `outputs/{名字}/`。

约定: 图片目录里 4 路环视命名 `{timestamp}_camera_{1..4}.jpg`; 不同数据集可有不同布局/高度/内参/相机模型。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DatasetSpec:
    name: str                       # 数据集标识 (= outputs 下子目录名)
    data_dir: str                   # 4 路环视图片目录 (含 {ts}_camera_{1..4}.jpg)
    camera_height_m: float          # 相机离地高度 (米), 地面锚定尺度用
    cam_layout: list                # web 4环视 2x2 布局 [(方位, 相机号)], 顺序=左上,右上,左下,右下
    cam_params: str = ""            # 相机内参 ("" → memory-nav/cam/params.yaml(MEI); 或 camera_param.json(KB4鱼眼))
    map_stride: int = 1             # 轨迹帧抽样步长 (长序列调大)
    n_query: int = 25               # 评测 query 帧数


DATASETS = {
    # 创富大厦 1 楼, 头盔环视采集 (采集者身高 170cm)。
    # 相机 = KB4 鱼眼 (ISX031, 1920x1536), 内参在 camera_param.json (efl/cod/k1..4)。
    # extracted camera_1..4 = 外参表 AVM CAM 1..4: camera_1左后/camera_2左前/camera_3右前/camera_4右后。
    # 相机离地 ≈ 1.67m: 身高170cm - 头顶到头盔帽沿~12cm + 相机在帽沿上9.5cm (可后续校准)。
    # 3215 帧密集采集, map_stride=5 抽样 ~643 帧建图。
    "ChuangfuTower_floor1": DatasetSpec(
        name="ChuangfuTower_floor1",
        data_dir="/home/ubuntu/Disk/codes/jianxiong/VGP-Nav/data/ChuangfuTower_floor1",
        camera_height_m=1.67,
        cam_layout=[("左前", 2), ("右前", 3), ("左后", 1), ("右后", 4)],
        cam_params="/home/ubuntu/Disk/codes/jianxiong/VGP-Nav/data/ChuangfuTower_floor1/camera_param.json",
        map_stride=3, n_query=25,   # 3215→~1072帧(间距0.24m), 兼顾 DB 密度与 VGGT 基线
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"未知数据集 '{name}', 可选: {list(DATASETS)}")
    return DATASETS[name]
