#!/usr/bin/env bash
# =================================================================
# 全息桌面伴生萌宠 · 新机一键自动化部署脚本
# =================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "================================================================="
echo "   全息桌面伴生萌宠 · 一键部署与依赖安装向导"
echo "================================================================="

# 1. 检查 Python 版本
echo "[1/6] 检查 Python 运行环境..."
if ! command -v python3 &>/dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.10 或更高版本。"
    exit 1
fi
PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "检测到 Python 版本: $PY_VER"

# 2. 检查系统依赖并友好提示
echo "[2/6] 检查 Linux 系统多媒体与底层依赖..."
SYS_DEPS=("ffmpeg" "parecord" "arecord")
MISSING_SYS=()
for dep in "${SYS_DEPS[@]}"; do
    if ! command -v "$dep" &>/dev/null; then
        MISSING_SYS+=("$dep")
    fi
done

if [ ${#MISSING_SYS[@]} -gt 0 ]; then
    echo "提示: 检测到以下系统音频/编解码工具可能未安装: ${MISSING_SYS[*]}"
    echo "建议在 Ubuntu/Debian 上执行以下命令补全:"
    echo "  sudo apt update && sudo apt install -y ffmpeg libvulkan1 pulseaudio-utils alsa-utils libsndfile1 portaudio19-dev"
else
    echo "核心系统音频工具均已就绪。"
fi

# 3. 创建与激活虚拟环境
echo "[3/6] 初始化独立虚拟环境 (.venv)..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "已创建新虚拟环境: $DIR/.venv"
else
    echo "虚拟环境 .venv 已存在，直接复用。"
fi

VENV_PY="$DIR/.venv/bin/python3"
VENV_PIP="$DIR/.venv/bin/pip"

# 4. 安装 Python 核心依赖
echo "[4/6] 安装 Python 依赖包 (requirements.txt)..."
"$VENV_PIP" install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
"$VENV_PIP" install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 下载超大权重模型
echo "[5/6] 校验与下载超大模型权重 (自动多线程极速下载)..."
"$VENV_PY" download_models.py

# 6. 设置执行权限
echo "[6/6] 配置可执行权限..."
chmod +x "$DIR/start.sh" "$DIR/download_models.py" 2>/dev/null || true
if [ -d "$DIR/bin" ]; then
    chmod +x "$DIR/bin/cloudflared" 2>/dev/null || true
    chmod +x "$DIR/bin/breeze-tts-2-v0.1.7-linux-x64-vulkan"/* 2>/dev/null || true
fi

echo ""
echo "================================================================="
echo "   部署完成！全息桌面伴生萌宠环境已就绪！"
echo "================================================================="
echo "推荐启动方式:"
echo "  1. 快速一键启动: ./start.sh"
echo "  2. 手动直接启动: source .venv/bin/activate && python3 app.py"
echo ""
echo "访问地址: http://127.0.0.1:7860"
echo "手机扫码: 点击主屏右上角【手机扫码遥控】即可无线连接"
echo "================================================================="
