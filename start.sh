#!/usr/bin/env bash
set -e
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 优先使用项目专属虚拟环境
if [ -d "$DIR/.venv" ]; then
    PYTHON="$DIR/.venv/bin/python3"
else
    PYTHON="python3"
fi

# 确保二进制权限
chmod +x "$DIR/download_models.py" 2>/dev/null || true
if [ -d "$DIR/bin" ]; then
    chmod +x "$DIR/bin/cloudflared" 2>/dev/null || true
    chmod +x "$DIR/bin/breeze-tts-2-v0.1.7-linux-x64-vulkan"/* 2>/dev/null || true
fi

# 检查超大模型文件，若首次运行缺失则自动极速补全
if [ ! -f "$DIR/models/breeze-tts-2-q4_k.gguf" ] || [ ! -f "$DIR/data/sensevoice_int8/model.int8.onnx" ]; then
    echo ">> 检测到超大模型权重缺失，正在自动多线程高速下载..."
    "$PYTHON" "$DIR/download_models.py"
fi

echo ">> 启动全息桌面伴生萌宠 (端口 7860)..."
exec "$PYTHON" "$DIR/app.py" "$@"
