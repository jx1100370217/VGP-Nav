"""单帧耗时分解 (对应论文 Table IV)。

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/bench_timing.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config
from vgpnav.database import load_database
from vgpnav.pipeline import VGPNav
from vgpnav.vggt_runner import VGGTRunner
from vgpnav.vpr import make_vpr

cfg = Config()
db = load_database(cfg)
q = np.load(os.path.join(cfg.db_dir, "query.npz"))["frame_idx"]
vpr = make_vpr(cfg)
vggt = VGGTRunner(cfg)
nav = VGPNav(cfg, db, vpr, vggt)

# 预热 (CUDA / 模型首帧慢)
nav.run(db.image(int(q[0])))

stages = ["VPR", "几何感知检索", "VGGT", "加权运动平均", "地面锚定尺度+占据图", "总计"]
acc = {s: [] for s in stages}
N = min(15, len(q))
for fi in q[:N]:
    nav.run(db.image(int(fi)))
    for s in stages:
        acc[s].append(nav.timing[s])

print(f"\n单帧耗时分解 (均值, {N} 帧, {cfg.device}) —— 对应论文 Table IV:")
print(f"{'模块':<22}{'本复现 (s)':<14}{'论文VGP-Nav (s)'}")
paper = {"VPR": "-", "几何感知检索": "0.027", "VGGT": "0.420",
         "加权运动平均": "0.005", "地面锚定尺度+占据图": "0.040", "总计": "0.492"}
for s in stages:
    print(f"{s:<22}{np.mean(acc[s]):<14.3f}{paper.get(s,'-')}")
print("\n注: 论文用 RTX 5090, 本复现用 L40; VPR 论文用 NetVLAD(此处 DINOv2)。")
