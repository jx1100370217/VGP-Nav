"""matplotlib 中文字体配置 (修复中文显示为方框的问题)。"""
from __future__ import annotations

import matplotlib
from matplotlib import font_manager as fm

# 候选中文字体 (按优先级), L40 上有 Noto Sans CJK SC
_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "AR PL UMing CN", "SimHei",
]


def setup_cjk_font():
    """让 matplotlib 能正确渲染中文 + 负号。返回选中的字体名 (或 None)。"""
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in _CJK_CANDIDATES if c in available), None)
    if chosen:
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["font.sans-serif"] = (
            [chosen] + list(matplotlib.rcParams.get("font.sans-serif", [])))
    matplotlib.rcParams["axes.unicode_minus"] = False  # 负号用 ASCII, 避免缺字形
    return chosen
