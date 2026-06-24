# VGP-Nav 复现

复现论文 **VGP-Nav: Metric-Aware Visual Geometric Perception for Robot Navigation**
(Pan et al., arXiv 2606.09268)。纯单目 RGB, 同时做**度量定位**与**度量障碍感知**,
核心是用**地平面几何**把单目重建锚定到真实物理尺度。

在 L40 服务器、真实机器人数据 **Mapping_C8**(DEEPROUTE 办公室一层, 4 路环视鱼眼之
camera_1) 上端到端功能复现 + 定性验证。

## 论文 pipeline 与本仓库模块对应

| 论文组件 (Fig.2 / §III) | 本仓库 | 说明 |
|---|---|---|
| 单目输入 (去畸变针孔) | `vgpnav/undistort.py` | MEI 鱼眼模型 → 针孔 (移植 cam/ 的投影公式) |
| 带位姿数据库 D={(Iᵢ,T_wi)} | `vgpnav/database.py` | VGGT 滑窗链式重建自建**伪位姿库** + 1.3m 地面锚定尺度 |
| VPR 检索 (NetVLAD) | `vgpnav/vpr.py` | 功能复现改用 **DINOv2** 全局描述子 (离线缓存) |
| 几何感知检索 (Alg.1) | `vgpnav/retrieval.py` | 空间离群剔除 + 多样性选择 (含抗强混叠的半径门控) |
| 前馈重建 (VGGT) | `vgpnav/vggt_runner.py` | 复用 memory-nav 的 VGGT-1B 后端 |
| 加权运动平均 (§III-D) | `vgpnav/motion_averaging.py` | 旋转共识 + 多因子 IRLS 射线三角化 (式1) |
| 地面锚定尺度恢复 (§III-E) | `vgpnav/scale_recovery.py` | 重力对齐 + 地面峰 + 1.3m 锚定 |
| 占据图 + A* | `vgpnav/occupancy.py`, `planner.py` | 高度分类 → 2D 占据栅格 → A* 规划 |
| 端到端 | `vgpnav/pipeline.py` | 串联以上, 单帧 query → 位姿 + 度量点云 + 占据图 |

## 数据流 (单帧 query)

```
query 图 ─VPR─> top-N ─几何感知检索(Alg.1)─> k=10 参考(带度量位姿)
        └─> VGGT([query]+refs) ─> 相对位姿 + 稠密点(尺度模糊)
              ├─加权运动平均─> 全局位姿 T_wc   (定位; 度量来自参考射线三角化)
              └─地面锚定尺度─> 度量点云 ─> 2D 占据图 ─> A* 无碰撞路径
```

## 运行 (L40, internvla 环境)

```bash
PY=/home/ubuntu/miniconda3/envs/internvla/bin/python
cd /home/ubuntu/Disk/codes/jianxiong/VGP-Nav
$PY scripts/build_database.py   # 建伪位姿库 (VGGT 滑窗+1.3m锚定+VPR) -> outputs/db/
$PY scripts/run_query.py        # 端到端: 定位+感知+A*, 评测+可视化 -> outputs/query/
$PY scripts/bench_timing.py     # 单帧耗时分解 (对应 Table IV)
# 单元测试 (合成数据验证各模块数学):
$PY tests/test_motion_averaging.py
$PY tests/test_retrieval.py
$PY tests/test_scale_occupancy.py
$PY tests/test_vpr.py
```

本地开发: 在 `/home/jianxiong/vgp-nav/` 编辑, `./deploy.sh` rsync 到 L40 执行。

## 结果 (C8, 25 个 held-out query)

- **定位**: 平移误差中位 **1.02 m**, 旋转中位 **0.9°**; <1m 12/25, <2m 16/25。
- **感知**: 多视角稠密度量点云 + 2D 占据图 (地面≈1.3m 锚定, 见 `outputs/query/query_montages.png`)。
- **规划**: A* 在局部度量占据栅格上规划出无碰撞路径 (`astar_demo.png`)。
- **耗时** (L40, 单帧): VGGT ~0.62s, 运动平均 ~0.003s, 地面锚定+占据 ~0.043s
  (运动平均/感知与论文 0.005/0.040s 基本吻合)。

## 交互网页 (科技感可视化 + 动态导航演示)

`web/index.html` 是一个自包含的 Canvas 单页应用 (暗色霓虹科技风, 离线可开):

- **建图详情**: 度量点云(高度着色)/轨迹/DB 节点/占据障碍/定位评测 分层可视, 平移缩放,
  悬停 DB 节点看对应相机帧 (FPV) 与位姿, 侧栏统计 (帧数/尺度/栅格/定位误差)。
- **动态导航**: 点选起点终点 → JS 端 A* 在度量占据栅格上规划 → 发光路径 + 机器人沿路径
  动画 (朝向/FoV 锥/扫描环), 机器人视图实时切换到最近 DB 帧。

生成数据并启动:
```bash
$PY scripts/export_web.py     # 生成 outputs/web/data.js (+ 拷贝 index.html, 含DB缩略图)
bash web/serve.sh             # http://localhost:8765/index.html
```
data.js 与 index.html 自包含, 拷到任意机器用 `python3 -m http.server` 即可打开。

## 与论文的差异 (功能复现取舍)

1. **VPR**: 论文 NetVLAD, 此处 DINOv2 全局描述子 (权重本机缓存, 离线; AnyLoc 等亦基于 DINOv2)。
2. **数据库位姿**: 论文用手持扫描仪建精确地图; 此处无 C8 真值, 用 **VGGT 滑窗重建 + 1.3m
   地面锚定**自建伪位姿库 (本身即论文思想的延伸)。故定位评测是对"伪 GT"的一致性, 非绝对真值。
3. **检索半径门控**: C8 办公室走廊高度自相似 → 强感知混叠, 论文"几何中位+95分位"离群剔除
   (假设内点占多数) 在此失效; 改为"以最可信 top-M 匹配为锚 + 半径门控"(论文亦显式保留 I_best)。

## 已知限制

- **全局建图漂移**: 单目长序列无回环优化, VGGT 跨窗尺度不完全一致 → 全局地图有尺度/旋转漂移
  (局部度量一致, 相邻帧高度差中位 0.08m)。**对每帧定位/局部占据/局部 A* 无碍**(均只用局部参考/FoV,
  与论文一致); 仅全局地图总览受影响。可加回环+位姿图优化改善 (future work)。
- **定位离群**: 少数 query (强混叠/漂移处) 误差大, 与论文报告的失效模式 (狭窄通道、无纹理墙面
  导致定位丢失) 一致。

## 依赖

internvla 环境 (torch 2.6+cu124)。重资产按路径引用 memory-nav: VGGT 源码
(`third_party/vggt_space`)、权重 (`pretrained/vggt-1b/model.pt`)、相机内参 (`cam/params.yaml`)、
数据 (`Mapping_C8`)。DINOv2 权重取自本机 torch.hub 缓存。
