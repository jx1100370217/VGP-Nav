#!/usr/bin/env bash
# 启动 VGP-Nav 交互网页本地服务。
# 优先服务 outputs/web (含 export_web.py 生成的 data.js); 否则服务脚本所在目录。
DIR="$(cd "$(dirname "$0")/.." && pwd)/outputs/web"
[ -f "$DIR/data.js" ] || DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${1:-8765}"
echo "VGP-Nav 网页: http://localhost:$PORT/index.html   (Ctrl+C 停止)"
cd "$DIR" && python3 -m http.server "$PORT"
