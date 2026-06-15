#!/bin/bash
# =============================================================================
# install_trackeval.sh
# 在服务器 venv 中安装 TrackEval 及追踪评估依赖
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
cd "$PROJECT_ROOT"
source venv/bin/activate

echo "[INFO] 安装 TrackEval..."
pip install trackeval

echo "[INFO] 安装其他依赖..."
pip install pillow scipy

echo "[INFO] 安装完成，验证导入..."
python -c "import trackeval; print('trackeval OK:', trackeval.__version__ if hasattr(trackeval, '__version__') else 'installed')"

echo ""
echo "TrackEval 安装完成。"
