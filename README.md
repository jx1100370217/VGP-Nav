# VGP-Nav 复现

复现论文 **VGP-Nav: Metric-Aware Visual Geometric Perception for Robot Navigation**
(Pan et al., arXiv 2606.09268)。纯单目 RGB,同时做**度量定位**与**度量障碍感知**,
核心是用**地平面几何**把单目重建锚定到真实物理尺度。

在 L40 服务器、真实机器人数据上端到端复现 + 定性验证,扩展为**多数据集**系统(三个场景),
含 **IMU 重力对齐 / 步频尺度校正、分层回环 PGO、灰度占据栅格可视化、轨迹引导 A* 导航**,
以及一套 Flask 后端的可视化总应用(建图流程监控 + 导航演示 + DB 管理)。

## 多数据集

`vgpnav/datasets.py` 注册表统一管理;切数据集只需 `VGPNAV_DATASET=<名字>`,产物隔离到 `outputs/<名字>/`。

| 数据集 | 显示名 | 相机模型 | 帧数 / 抽样 | 相机离地 | IMU | 场景 |
|---|---|---|---|---|---|---|
| `Mapping_C8` | 深港国际C8 | **MEI** (params.yaml) | 626 / stride1 | 1.3m | 无(视觉重力) | DEEPROUTE 办公一层 |
| `ChuangfuTower_floor1` | 创富大厦1楼 | **KB4** 鱼眼 (camera_param.json) | 3215 / stride5≈643 | 1.67m(头盔) | 有 | 室外梅林街区 |
| `ChuangfuTower_floor28` | 创富大厦28楼 | **KB4** 鱼眼 | 180 / stride1 | 1.67m(头盔) | 有 | 室内写字楼办公区 |

`undistort.py` 的 `load_camera_params` 按文件后缀自动识别相机模型(`.json`→KB4 Kannala-Brandt,
`.yaml`→MEI 全向),统一去畸变成针孔供 VGGT。

## 论文 pipeline 与本仓库模块对应

| 论文组件 (Fig.2 / §III) | 本仓库 | 说明 |
|---|---|---|
| 单目输入 (去畸变针孔) | `vgpnav/undistort.py` | KB4 鱼眼 + MEI 全向两种模型 → 针孔 |
| 带位姿数据库 D={(Iᵢ,T_wi)} | `vgpnav/database.py` | VGGT 滑窗链式重建自建**伪位姿库** + 重力对齐 + 地面锚定尺度 |
| VPR 检索 (NetVLAD) | `vgpnav/vpr.py` | 功能复现改用全局描述子 (离线缓存) |
| 几何感知检索 (Alg.1) | `vgpnav/retrieval.py` | 空间离群剔除 + 多样性选择 (含抗强混叠的半径门控) |
| 前馈重建 (VGGT) | `vgpnav/vggt_runner.py` | 复用 memory-nav 的 VGGT-1B 后端 |
| 加权运动平均 (§III-D) | `vgpnav/motion_averaging.py` | 旋转共识 + 多因子 IRLS 射线三角化 (式1) |
| 地面锚定尺度恢复 (§III-E) | `vgpnav/scale_recovery.py` | 重力对齐 + 地面峰 + 相机高度锚定 |
| 占据图 + A* | `vgpnav/occupancy.py`, `web/nav.html` | 高度分类 → 2D 占据栅格 → 轨迹引导 A* |
| 端到端 | `vgpnav/pipeline.py` | 串联以上, 单帧 query → 位姿 + 度量点云 + 占据图 |

**本复现相对论文的增强模块**:

- **IMU 重力对齐** (`vgpnav/imu.py`):有 IMU 时用加速度计真重力 + VGGT 位姿自标定 R_cam←imu(Wahba 迭代)
  求 world 重力方向,比"相机平均朝向估重力"稳(头盔头部转动会扰动后者);无 IMU(C8)回退视觉。
  地面锚定要求相机在地面上方(物理约束,保证尺度符号正确)。
- **分层回环 PGO** (`scripts/pgo_fix.py`):近距离(<`--dmax`)用 `--sim` 阈值、远距离用高阈值 `--simfar`
  抓大漂移真闭环;`--max_nfev` 控制 PGO 迭代上限(大漂移闭环需足够迭代才收敛)。
- **IMU 步频 PDR 尺度校正** (`scripts/imu_scale.py` + `apply_scale.py`):用 IMU |加速度| 带通数步 × 步长
  估真实路程,标定全局尺度因子 k(校正 VGGT 室外偏小的水平尺度);`apply_scale` 应用 xy×k(z 保持)。

## 数据流 (单帧 query)

```
query 图 ─VPR─> top-N ─几何感知检索(Alg.1)─> k 个参考(带度量位姿)
        └─> VGGT([query]+refs) ─> 相对位姿 + 稠密点(尺度模糊)
              ├─加权运动平均─> 全局位姿 T_wc   (定位; 度量来自参考射线三角化)
              └─地面锚定尺度─> 度量点云 ─> 2D 占据图 ─> 轨迹引导 A* 无碰撞路径
```

## 导航:轨迹引导的栅格 A*

导航在 **2D 占据栅格 + 机器人轨迹**上做 A*(`web/nav.html`,JS 端实时):

- **栅格 A***:在占据 free 空间上找物理可走的最短路(沿真实走廊)。
- **轨迹连通格 `trajCell`**:机器人走过的轨迹(相邻关键帧插值连线、±1 格)标为强可通行、低代价。
  它补回 VGGT 占据碎片处断裂的连通性,并让 A* 优先沿走过的路。
- **起终点吸附 DB 节点**:点击地图吸附到最近 DB 节点,再 snap 到其 free 栅格作为 A* 端点。

## 运行 (L40, internvla 环境)

```bash
PY=/home/ubuntu/miniconda3/envs/internvla/bin/python
cd /home/ubuntu/Disk/codes/jianxiong/VGP-Nav
export VGPNAV_DATASET=Mapping_C8      # 选数据集(默认 Mapping_C8);产物进 outputs/$VGPNAV_DATASET/

# 建图 8 步(详见 docs/02_代码执行指南.md)
$PY scripts/build_database.py         # ① VGGT 滑窗+重力对齐+地面锚定 -> db/
$PY scripts/compute_4cam_desc.py      # ② 4 路环视描述子(回环用)
$PY scripts/pgo_fix.py --sim 0.70 --dmax 4 --simfar 0.85 --max_nfev 4000  # ③ 分层回环 PGO
$PY scripts/apply_pgo.py              # ④ 写回校正位姿
$PY scripts/build_4cam_points.py      # ⑤ 360° 融合点云
$PY scripts/export_cams.py            # ⑥ 4 路环视缩略图
$PY scripts/run_query.py              # ⑦ 定位评测(端到端)
$PY scripts/export_web.py             # ⑧ 导出最终占据地图 + web

# (室外尺度偏小的数据集) IMU 步频标定 + 校正
$PY scripts/imu_scale.py              # 看尺度因子 k
$PY scripts/apply_scale.py <k> && $PY scripts/export_web.py

$PY scripts/gen_portal.py             # 多数据集总入口
$PY scripts/server.py                 # Flask 服务(静态+建图控制+DB管理), 0.0.0.0:8199
```

本地开发:在 `/home/jianxiong/vgp-nav/` 编辑,`./deploy.sh` rsync 到 L40 执行(L40 为主仓库)。

## 结果

- **C8(深港国际C8,25 query)**:定位平移中位 **0.32m**、旋转 0.97°;<1m 23/25、<2m 24/25。
  209 DB 节点,分层回环 190 条、残余 0.31m。
- **floor1(创富1楼)**:户外两街区,IMU 步频 PDR 把 VGGT 偏小的水平尺度校正到 ~470m 真实尺度。
- **floor28(创富28楼)**:室内办公单程,真回环闭合残余 0.76m;定位受开放办公区**强感知混叠**限制
  (相似工位区让 VPR 误匹配)——单目 VGP-Nav 在有视觉差异室内表现好、强混叠场景吃力。

## 交互网页 (灰度占据栅格 + 动态导航 + Flask 总应用)

总入口 `http://192.168.50.72:8199/`(Flask `scripts/server.py`),**顶层三 tab + 每 tab 内切换地图**:

1. **建图流程**:8 步 pipeline 卡片(状态灯 / 看产物图 / ▶运行某步 / 实时日志 / ▶一键全流程)。
2. **导航演示**(`web/nav.html`):**灰度占据栅格图**(暗灰蓝=未知 / 浅灰=可通行 / 深=墙,柔和护眼)
   + 轨迹 + DB 节点;点选起终点 → 轨迹引导 A* → 发光路径 + 机器人沿路径动画 + 4 路环视实时切到最近 DB 帧。
3. **DB 节点管理**:表格增删改查(改名 / 删 / 从轨迹帧增 / 查 4 环视)。

```bash
$PY scripts/server.py    # 启动后浏览器开 http://192.168.50.72:8199/
```
> Chrome 对 data.js 会强缓存,改后需 Ctrl+Shift+R 硬刷新。

## 与论文的差异 (功能复现取舍)

1. **VPR**:论文 NetVLAD,此处用全局描述子(权重本机缓存,离线)。
2. **数据库位姿**:论文用手持扫描仪建精确地图;此处无真值,用 VGGT 滑窗重建 + 地面锚定自建伪位姿库
   (本身即论文思想的延伸)。定位评测是对"伪 GT"的一致性。
3. **检索半径门控**:办公室走廊高度自相似 → 强感知混叠,论文"几何中位+95分位"离群剔除在此失效;
   改为"以最可信 top-M 匹配为锚 + 半径门控"。
4. **导航 A***:论文未细化全局规划;本复现用轨迹引导栅格 A*。

## 已知限制

- **VGGT 单目尺度**:跨窗尺度不完全一致;室外水平运动尺度系统偏小(远景视差小),靠 IMU 步频 PDR 校正。
- **玻璃 / 透明结构**:被动视觉重建不出玻璃,VGGT 透过玻璃重建出室内地面、占据误判为 free,
  栅格 A* 可能穿过玻璃隔挡的会议室(物理不通)。占据无法区分真走廊与玻璃室,如需规避可手动把玻璃墙补成障碍。
- **室内强混叠**:开放办公区工位高度重复 → VPR 定位易误匹配(floor28),单目方法固有,需更稠密重建/主动传感。

## 依赖

internvla 环境 (torch 2.6+cu124)。重资产按路径引用 memory-nav:VGGT 源码、权重、
C8 相机内参 (`cam/params.yaml`,MEI)、`Mapping_C8` 数据。创富数据集相机内参用各自
`camera_param.json`(KB4)。VPR 权重取自本机缓存。
