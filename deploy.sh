#!/usr/bin/env bash
# 部署本地 VGP-Nav 代码到 L40 (仅同步代码, 不碰远端 data/outputs/权重)
set -e
LOCAL="$(cd "$(dirname "$0")" && pwd)/"
REMOTE="L40:/home/ubuntu/Disk/codes/jianxiong/VGP-Nav/"
rsync -az \
  --exclude '__pycache__' --exclude '*.pyc' --exclude '.git' \
  --exclude 'outputs' --exclude 'data' \
  "$LOCAL" "$REMOTE"
echo "已部署 -> $REMOTE"
