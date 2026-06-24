"""用 Qwen3-VL-4B (≤4B 本地视觉语言模型) 给每个 DB 节点按其相机画面起中文地点名。

注: Qwen3.5-0.8B 是纯文本模型(意图分类/路径叙述), 无法看图; 故改用项目内 ≤4B 的
Qwen 视觉模型 Qwen3-VL-4B-Thinking 实现"按图命名"。

  /home/ubuntu/miniconda3/envs/qwen3/bin/python scripts/name_nodes.py [--limit N]
输出: outputs/db/node_names.json (与 db.frame_idx 等长的中文名列表)
"""
import argparse
import json
import os
import sys

import cv2
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vgpnav.config import Config
from vgpnav.database import load_database

MODEL = "/home/ubuntu/Disk/models/vlm/Qwen/Qwen3-VL-4B-Thinking"
DEVICE = "cuda:3"
PROMPT = ("这是机器人在室内导航时某个位置的相机画面。请用4到8个汉字给这个地点起一个"
          "简洁的功能性名称(例如:前台接待台、玻璃走廊、茶水间、电梯厅、开放办公区、"
          "会议室门口、洗手间走廊、休息沙发区)。直接输出名称本身,不要任何解释或标点。")

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--fix", action="store_true")
args = ap.parse_args()

cfg = Config()
db = load_database(cfg)
n = len(db.frame_idx)
NN_PATH = os.path.join(cfg.db_dir, "node_names.json")

def isbad(x):
    return (not x) or len(x) < 2 or len(x) > 8 or any(
        k in x for k in ["用户", "现在", "需要", "场景", "名称", "图片",
                         "这个", "应该", "先看", "参考", "我"])

names = [""] * n
MAXTOK = 224
if args.fix:
    names = json.load(open(NN_PATH))
    todo = [i for i in range(n) if isbad(names[i])]
    MAXTOK = 600
    print(f"fix 模式: 重跑 {len(todo)} 个坏名")
else:
    todo = list(range(n if args.limit <= 0 else min(args.limit, n)))

print(f"加载 {MODEL} ...")
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    MODEL, dtype=torch.bfloat16, trust_remote_code=True).to(DEVICE).eval()
print("模型就绪, 开始命名", len(todo), "个节点, max_new_tokens =", MAXTOK)


def clean(t):
    if "</think>" in t:
        t = t.split("</think>")[-1]
    t = t.strip().strip('"\'“”。.，,：: \n\t')
    t = t.split("\n")[0].strip()
    return (t[:12] if t else "未命名")


def name_one(bgr):
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil}, {"type": "text", "text": PROMPT}]}]
    inputs = proc.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAXTOK, do_sample=False)
    out = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                            skip_special_tokens=True)[0]
    return clean(out), out.strip()[:80]


for c, k in enumerate(todo):
    fi = int(db.frame_idx[k])
    nm, raw = name_one(db.image(fi))
    names[k] = nm
    print(f"  [{c+1}/{len(todo)}] DB帧#{fi} -> 「{nm}」   (raw: {raw})")

if args.limit <= 0 or args.fix:
    # 兜底: 仍坏的名用最近的好邻居名 (连续节点多为同一区域)
    nfix = 0
    for i in range(n):
        if isbad(names[i]):
            for d in range(1, n):
                if i - d >= 0 and not isbad(names[i - d]):
                    names[i] = names[i - d]; nfix += 1; break
                if i + d < n and not isbad(names[i + d]):
                    names[i] = names[i + d]; nfix += 1; break
    print(f"邻居兜底修复 {nfix} 个")
    json.dump(names, open(NN_PATH, "w"), ensure_ascii=False)
    print("已写出 ->", NN_PATH)
else:
    print("(smoke 模式, 未写文件)")
