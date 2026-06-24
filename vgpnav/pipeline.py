"""VGP-Nav 端到端单帧推理 (论文 Fig.2 主流程)。

query 图 -> VPR 检索 -> 几何感知检索(Alg.1) -> VGGT 重建 -> 加权运动平均(定位 T_wc)
-> 地面锚定尺度恢复(度量点云) -> 2D 占据图。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import geom
from . import motion_averaging as MA
from . import scale_recovery as SR
from .occupancy import OccupancyGrid
from .retrieval import geometry_aware_retrieval


@dataclass
class QueryResult:
    T_wc: np.ndarray            # 估计的 query 世界位姿
    ref_ids: list               # 选中的 DB 参考 (DB 内索引)
    P_world: np.ndarray         # 度量世界点云 (多视角, 单次前向所有帧)
    scale: float                # 本帧地面锚定尺度
    ground_z: float             # 占据图所用地面高度
    occ: OccupancyGrid          # 局部占据图
    est: MA.PoseEstimate        # 运动平均细节 (权重等)


class VGPNav:
    def __init__(self, cfg, db, vpr, vggt):
        self.cfg = cfg
        self.db = db
        self.vpr = vpr
        self.vggt = vggt
        self.timing = {}        # 各阶段耗时 (秒), 对应论文 Table IV

    def _query_points_cam(self, world_points_vggt, extri_query, conf, conf_pct=55):
        """VGGT world_points(world_v 系) -> query 相机系, 并按置信度过滤。"""
        Tq = geom.extri_to_Tcw(extri_query)      # world_v -> cam_q
        R, t = Tq[:3, :3], Tq[:3, 3]
        P = world_points_vggt.reshape(-1, 3)
        c = conf.reshape(-1)
        thr = np.percentile(c, conf_pct)
        keep = c >= thr
        P = P[keep]
        return (P @ R.T) + t                     # cam 系点

    def run(self, query_cams) -> QueryResult:
        """query_cams: 单张 BGR 或 4 路环视 [cam1,cam2,cam3,cam4] (cam1 为位姿锚)。"""
        cfg = self.cfg
        db = self.db
        tm = {}
        if isinstance(query_cams, np.ndarray):
            query_cams = [query_cams]
        nq = len(query_cams)
        # 1) VPR: 4 路各算描述子, 对 DB 取"最大相似"(任一路认出该地点即可, 抗朝向差)
        _t = time.perf_counter()
        q_descs = self.vpr.describe(query_cams)          # (nq, D)
        sims = (q_descs @ db.desc.T).max(axis=0)         # (M,)
        tm["VPR"] = time.perf_counter() - _t
        # 2) 几何感知检索 (Alg.1)
        _t = time.perf_counter()
        res = geometry_aware_retrieval(
            sims, db.centers, db.view_dirs, k=cfg.k_refs,
            search_pool_N=cfg.search_pool_N, lambda_pos=cfg.lambda_pos,
            lambda_ang=cfg.lambda_ang, outlier_percentile=cfg.outlier_percentile,
            radius_m=cfg.retrieval_radius_m, anchor_topm=cfg.anchor_topm)
        ref_ids = res.selected
        tm["几何感知检索"] = time.perf_counter() - _t
        # 3) 取参考图 + VGGT: [query 4路] + refs (360°覆盖, 约束更强)
        _t = time.perf_counter()
        ref_imgs = [db.image(int(db.frame_idx[i])) for i in ref_ids]
        ref_poses = [db.poses[i] for i in ref_ids]
        out = self.vggt.infer(list(query_cams) + ref_imgs)
        extri_q = out["extri"][0]                         # cam1 = rig 位姿锚
        extris_ref = out["extri"][nq:]
        tm["VGGT"] = time.perf_counter() - _t
        # 4) 加权运动平均 -> 定位
        _t = time.perf_counter()
        est = MA.estimate_world_pose(ref_poses, extris_ref, extri_q,
                                     irls_iters=cfg.irls_iters, tukey_c=cfg.tukey_c)
        T_wc = est.T_wc
        tm["加权运动平均"] = time.perf_counter() - _t
        _t = time.perf_counter()
        # 5) 地面锚定尺度恢复 (用 query 帧点估本帧尺度 s)
        P_cam_q = self._query_points_cam(out["world_points"][0], extri_q,
                                         out["depth_conf"][0])
        _, scale, _ = SR.anchor_query_points(
            P_cam_q, T_wc[:3, :3], T_wc[:3, 3], cfg.camera_height_m)
        # 6) 多视角稠密点云: 单次前向的所有帧 (同一 VGGT 尺度) 一并投到世界系
        #    VGGT->世界 sim3: X_w = s·R_S·(X_v - C_q^v) + t_wc, R_S = R_wc·R_q
        Tq = geom.extri_to_Tcw(extri_q)
        Rq, tq = Tq[:3, :3], Tq[:3, 3]
        Cqv = -Rq.T @ tq
        R_S = T_wc[:3, :3] @ Rq
        rng = np.random.default_rng(0)
        allP = []
        for f in range(len(out["world_points"])):
            P = out["world_points"][f].reshape(-1, 3)
            c = out["depth_conf"][f].reshape(-1)
            keep = c >= np.percentile(c, 55)
            P = P[keep]
            if len(P) > 5000:
                P = P[rng.choice(len(P), 5000, replace=False)]
            allP.append(scale * ((P - Cqv) @ R_S.T) + T_wc[:3, 3])
        P_world = np.vstack(allP)
        # 7) 局部占据图 (地面高度由点云峰值定)
        ground_z = SR.find_ground_peak(P_world[:, 2])
        occ = OccupancyGrid(resolution=cfg.occ_resolution, range_m=cfg.occ_range_m,
                            center_xy=T_wc[:2, 3], ground_band=cfg.ground_band_m,
                            ceil=cfg.camera_height_m, occ_min_hits=2)
        occ.integrate(P_world, ground_z=ground_z)
        tm["地面锚定尺度+占据图"] = time.perf_counter() - _t
        tm["总计"] = sum(tm.values())
        self.timing = tm
        return QueryResult(T_wc=T_wc, ref_ids=ref_ids, P_world=P_world,
                           scale=scale, ground_z=ground_z, occ=occ, est=est)
