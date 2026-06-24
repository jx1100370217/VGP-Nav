"""VGGT 前馈重建适配器。

复用 memory-nav 已封装好的 VGGT-1B 后端 (源码 third_party/vggt_space, 权重
pretrained/vggt-1b/model.pt), 只暴露一个干净的 infer 接口给本项目使用。
"""
from __future__ import annotations

import os
import sys


class VGGTRunner:
    def __init__(self, cfg):
        sys.path.insert(0, os.path.join(cfg.memory_nav_root, "third_party",
                                        "vggt_space"))
        sys.path.insert(0, cfg.memory_nav_root)
        from online_mapper.geometry.vggt_backend import VGGTBackend
        self.be = VGGTBackend.get(cfg.vggt_weights, device=cfg.device)
        assert self.be.available, "VGGT 后端不可用"

    def infer(self, bgr_list):
        """返回 dict: depth/extri/intri/world_points/depth_conf, 每项长度=N 的 list。

        extri: (3,4) world_v->cam; intri: (3,3); world_points: (H,W,3) VGGT 公共系。
        """
        return self.be.infer_bgr_list(bgr_list)
