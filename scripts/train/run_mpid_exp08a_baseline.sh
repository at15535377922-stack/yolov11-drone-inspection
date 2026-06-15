#!/bin/bash
# =============================================================================
# EXP-08a：MPID 上 YOLOv11n 基线训练（从 yolo11n.pt 预训练权重 fine-tune）
#
# 目标：建立 MPID 数据集上的原始 YOLOv11n 基线，用于与 ASE-YOLOv11 对比。
# 数据集：MPID（glass + porcelain + composite 合并）
# 模型：yolo11n（原始结构，无 P2/LWFusion/CBAM/TADetect/NWD 改动）
#
# 使用前提：
#   1. 已在服务器解压并合并 MPID（scripts/dataset/merge_mpid.sh）
#   2. configs/datasets/mpid.yaml 中 nc 和 names 已核实
#   3. 已激活 venv: source /root/autodl-tmp/yolov11-drone-inspection/venv/bin/activate
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
PRETRAIN="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
DATA="${PROJECT_ROOT}/configs/datasets/mpid.yaml"
RUN_NAME="mpid_exp08a_baseline"

# 数据集说明：
# MPID 三子集合并后随机 8:1:1 切分，路径 data/mpid/merged/
# 合并脚本：bash scripts/dataset/merge_mpid.sh

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "=============================="
echo " EXP-08a: MPID YOLOv11n 基线"
echo "=============================="
echo "预训练权重: ${PRETRAIN}"
echo "数据配置:   ${DATA}"
echo "输出目录:   runs/detect/${RUN_NAME}"
echo ""

yolo detect train \
    model="${PRETRAIN}" \
    data="${DATA}" \
    imgsz=640 \
    batch=32 \
    epochs=100 \
    device=0 \
    workers=8 \
    project="${PROJECT_ROOT}/runs/detect" \
    name="${RUN_NAME}" \
    exist_ok=True \
    patience=30 \
    save_period=10 \
    val=True \
    plots=True

echo ""
echo "EXP-08a 训练完成。"
echo "结果目录: ${PROJECT_ROOT}/runs/detect/${RUN_NAME}"
echo "最佳权重: ${PROJECT_ROOT}/runs/detect/${RUN_NAME}/weights/best.pt"
echo ""
echo "请将 mAP50 / mAP50-95 / Precision / Recall / FPS 记录到 logs/EXP-08_mpid_transfer.md"
