"""部署多数据集总入口 outputs/index.html = web/portal.html(顶层三tab + 质量评估应用)。

portal.html 自身 fetch /api/datasets 动态列已建好的数据集, 故只需复制为 index.html。
(早期的 iframe 多地图入口已被 c6fea79 的三tab portal 取代。)

  /home/ubuntu/miniconda3/envs/internvla/bin/python scripts/gen_portal.py
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import _PROJ
from vgpnav.datasets import DATASETS

OUT = os.path.join(_PROJ, "outputs")
shutil.copy(os.path.join(_PROJ, "web", "portal.html"), os.path.join(OUT, "index.html"))

built = [n for n in DATASETS if os.path.exists(os.path.join(OUT, n, "web", "data.js"))]
print(f"总入口(三tab portal) -> {OUT}/index.html  已建好: {built}")
