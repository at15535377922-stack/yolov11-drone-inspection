#!/bin/bash
# 服务器安装脚本 - 适配本地上传的 ultralytics 压缩包
# 用法：chmod +x install.sh && ./install.sh

set -e

PROJECT_DIR="/root/autodl-tmp/yolov11-drone-inspection"
export OMP_NUM_THREADS=1
cd $PROJECT_DIR

echo "=========================================="
echo "Step 1: 创建虚拟环境"
echo "=========================================="

echo "当前方案：venv 继承系统包（复用现有 torch / torchvision / CUDA 环境）"
if [ -d "venv" ]; then
    echo "虚拟环境已存在，跳过"
else
    python3 -m venv venv --system-site-packages
    echo "虚拟环境创建成功"
fi

source venv/bin/activate
echo "虚拟环境激活成功"

echo ""
echo "=========================================="
echo "Step 2: 准备 ultralytics 源码"
echo "=========================================="

if [ -d "ultralytics" ]; then
    echo "找到 ultralytics 目录，跳过解压"
elif [ -d "ultralytics-main" ]; then
    rm -rf ultralytics
    mv ultralytics-main ultralytics
    echo "已将 ultralytics-main 重命名为 ultralytics"
elif [ -f "ultralytics-main.zip" ]; then
    rm -rf ultralytics ultralytics-main
    unzip -q ultralytics-main.zip
    mv ultralytics-main ultralytics
    echo "已从 ultralytics-main.zip 解压并整理目录"
elif [ -f "ultralytics.zip" ]; then
    rm -rf ultralytics ultralytics-main
    unzip -q ultralytics.zip
    if [ -d "ultralytics-main" ]; then
        mv ultralytics-main ultralytics
    fi
    echo "已从 ultralytics.zip 解压并整理目录"
elif [ -f "ultralytics.tar.gz" ]; then
    rm -rf ultralytics ultralytics-main
    tar -xzf ultralytics.tar.gz
    if [ -d "ultralytics-main" ]; then
        mv ultralytics-main ultralytics
    fi
    echo "已从 ultralytics.tar.gz 解压并整理目录"
else
    echo "✗ 错误：未找到 ultralytics-main.zip、ultralytics.zip、ultralytics.tar.gz 或源码目录"
    echo "请先上传源码压缩包到 $PROJECT_DIR"
    exit 1
fi

if [ ! -f "ultralytics/pyproject.toml" ]; then
    echo "✗ 错误：ultralytics 目录不完整，未找到 pyproject.toml"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 3: 安装 ultralytics"
echo "=========================================="

cd ultralytics
pip install -e . --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "ultralytics 安装成功"

echo ""
echo "=========================================="
echo "Step 4: 安装额外依赖"
echo "=========================================="

pip install opencv-python tqdm thop pycocotools --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "依赖安装成功"

echo ""
echo "=========================================="
echo "Step 5: 验证环境"
echo "=========================================="

python -c "
import torch
import ultralytics
from ultralytics import YOLO

print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Ultralytics: {ultralytics.__version__}')

model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
print('YOLOv11 配置加载成功')
"

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "说明：安装脚本只验证环境与 YOLOv11 配置，不在此阶段下载预训练权重。"
echo "如需单独下载权重，可在后续手动执行：python -c \"from ultralytics import YOLO; YOLO('yolo11n.pt')\""
echo ""
echo "使用方式："
echo "cd /root/autodl-tmp/yolov11-drone-inspection"
echo "source venv/bin/activate"
echo "cd ultralytics"
echo "python train.py ..."
