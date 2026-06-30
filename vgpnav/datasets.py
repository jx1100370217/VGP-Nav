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
    # 检索覆盖 (0 → 用 config 全局默认): 强感知混叠场景(开放工位高度重复)调严,
    # 以 VPR best 为锚 + 小半径 + 少 refs, 剔除视觉像但物理远的误匹配工位。
    retrieval_radius_m: float = 0.0
    anchor_topm: int = 0
    k_refs: int = 0


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
        map_stride=5, n_query=25,   # 643帧(间距0.40m); 实测 stride=3(0.24m)基线太小、点云更杂乱
        # DB 稀疏(间距12.8m)+室外街区局部重复: 默认半径7m<间距, refs 不足会补到远街角污染定位。
        # best锚+半径15m+8refs(适配稀疏DB)把定位中位 12.61→6.14m(相对1.97%→0.96%)。
        retrieval_radius_m=15.0, anchor_topm=1, k_refs=8,
    ),
    # 创富大厦 28 楼, 同头盔设备 (复用同款 KB4 鱼眼内参 camera_param.json)。
    # 短序列: 180 帧 / 3 分钟, map_stride=1 全用; 相机离地 1.67m, 布局同 floor1。
    "ChuangfuTower_floor28": DatasetSpec(
        name="ChuangfuTower_floor28",
        data_dir="/home/ubuntu/Disk/codes/jianxiong/VGP-Nav/data/ChuangfuTower_floor28",
        camera_height_m=1.67,
        cam_layout=[("左前", 2), ("右前", 3), ("左后", 1), ("右后", 4)],
        cam_params="/home/ubuntu/Disk/codes/jianxiong/VGP-Nav/data/ChuangfuTower_floor28/camera_param.json",
        map_stride=1, n_query=10,
        # 开放办公区工位高度重复 → 强感知混叠: VPR best 准但多样性选择会拖入远工位污染定位。
        # 实测 best锚+r3+k5 把定位中位 5.40→2.25m, 选中的远工位ref 6→1.6个。
        retrieval_radius_m=3.0, anchor_topm=1, k_refs=5,
    ),
    # 深港国际 C8 (Mapping_C8): DEEPROUTE 一层, 626帧×4 鱼眼, MEI 相机模型(params.yaml)。
    # 相机离地 1.3m; cam_layout 前左1/前右2/后左4/后右3。VGP-Nav 最早的测试数据集。
    "Mapping_C8": DatasetSpec(
        name="Mapping_C8",
        data_dir="/home/ubuntu/Disk/codes/jianxiong/VGP-Nav/data/Mapping_C8",
        camera_height_m=1.3,
        cam_layout=[("前左", 1), ("前右", 2), ("后左", 4), ("后右", 3)],
        cam_params="/home/ubuntu/Disk/codes/jianxiong/memory-nav/cam/params.yaml",
        map_stride=1, n_query=25,
    ),
}


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"未知数据集 '{name}', 可选: {list(DATASETS)}")
    return DATASETS[name]
