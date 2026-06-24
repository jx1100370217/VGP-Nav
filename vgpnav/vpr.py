"""VPR 全局描述子 (用于检索 top-N)。

论文用 NetVLAD; 这里功能复现改用 DINOv2 全局描述子 (CLS token), 权重已缓存在
本机 torch.hub, 完全离线; AnyLoc 等 SOTA VPR 亦基于 DINOv2, 强度足够。
描述子归一化后用余弦相似度。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOv2VPR:
    def __init__(self, device="cuda:0", model_name="dinov2_vitb14",
                 hub_dir=None, img_size=224):
        hub_dir = hub_dir or os.path.expanduser(
            "~/.cache/torch/hub/facebookresearch_dinov2_main")
        self.device = device
        self.img_size = img_size
        self.model = torch.hub.load(hub_dir, model_name, source="local",
                                    pretrained=True).to(device).eval()
        self.mean = _IMAGENET_MEAN.to(device)
        self.std = _IMAGENET_STD.to(device)

    def _preprocess(self, bgr_list):
        xs = []
        for bgr in bgr_list:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.img_size, self.img_size))
            xs.append(rgb)
        x = np.stack(xs).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def describe(self, bgr_list, batch=32) -> np.ndarray:
        """返回 (N, D) 归一化全局描述子。"""
        if isinstance(bgr_list, np.ndarray) and bgr_list.ndim == 3:
            bgr_list = [bgr_list]
        out = []
        for i in range(0, len(bgr_list), batch):
            x = self._preprocess(bgr_list[i:i + batch])
            feat = self.model(x)            # (B, D) CLS token
            feat = F.normalize(feat, dim=-1)
            out.append(feat.cpu().numpy())
        return np.concatenate(out, axis=0)

    @staticmethod
    def similarity(q_desc: np.ndarray, db_desc: np.ndarray) -> np.ndarray:
        """余弦相似度 (描述子已归一化)。q:(D,) 或 (Q,D); db:(M,D) -> (M,) 或 (Q,M)。"""
        return q_desc @ db_desc.T


class SelaVPRVPR:
    """SelaVPR++ 全局描述子 (复用 memory-nav 的 SelaVPRExtractor, dinov2-large 4096维)。

    论文检索默认 NetVLAD, 这里按用户要求换成 memory-nav 的 SelaVPR++。
    注: SelaVPRExtractor 内部用 DataParallel 钉在 cuda:0, 这里解开以跑在指定 device。
    """

    def __init__(self, cfg):
        sys.path.insert(0, cfg.memory_nav_root)
        from memory_nav.selavpr_extractor import SelaVPRExtractor
        self.ex = SelaVPRExtractor(backbone="dinov2-large", use_hashing=False,
                                   use_rerank=False, device=cfg.device)
        if isinstance(self.ex.model, torch.nn.DataParallel):
            self.ex.model = self.ex.model.module.to(cfg.device)

    def describe(self, bgr_list, batch=12) -> np.ndarray:
        if isinstance(bgr_list, np.ndarray) and bgr_list.ndim == 3:
            bgr_list = [bgr_list]
        out = []
        for i in range(0, len(bgr_list), batch):
            out.append(self.ex.extract_batch(bgr_list[i:i + batch]))
        D = np.concatenate(out, axis=0).astype(np.float32)
        return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def similarity(q_desc, db_desc):
        return q_desc @ db_desc.T


def make_vpr(cfg):
    """按 cfg.vpr_backend 选择 VPR 实现 (selavpr | dinov2)。"""
    if getattr(cfg, "vpr_backend", "selavpr") == "selavpr":
        return SelaVPRVPR(cfg)
    return DINOv2VPR(device=cfg.device)
