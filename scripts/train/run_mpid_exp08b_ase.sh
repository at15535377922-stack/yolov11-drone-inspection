#!/bin/bash
# =============================================================================
# EXP-08b：MPID 上 ASE-YOLOv11（EXP-07 全量模型）迁移训练
#
# 目标：验证 ASE-YOLOv11 在 MPID 绝缘子数据集上的迁移适应能力，
#       与 EXP-08a YOLOv11n 基线对比。
#
# 模型：yolo11-p2-full.yaml（TADetect ta_layers=2，与 VisDrone EXP-07 一致）
# 预训练权重：yolo11n.pt（重新从预训练开始，保证公平对比；
#              如需从 VisDrone best.pt 迁移，可改为下方的注释行）
#
# 使用前提：
#   1. 已完成 EXP-08a 基线
#   2. MPID 数据集已合并，mpid.yaml nc/names 已核实
#   3. ultralytics/cfg/models/11/yolo11-p2-full.yaml 存在
#   4. 已激活 venv
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"

# 使用 yolo11n.pt 作为起点（与基线公平对比）
PRETRAIN="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
# 如改为从 VisDrone 最优权重迁移（跨域迁移场景），取消下行注释并注释上行：
# PRETRAIN="${PROJECT_ROOT}/runs/detect/visdrone10_exp07_full/weights/best.pt"

# 数据集说明：
# MPID 三子集合并后随机 8:1:1 切分，路径 data/mpid/merged/
# 合并脚本：bash scripts/dataset/merge_mpid.sh

MODEL_CFG="${PROJECT_ROOT}/ultralytics/ultralytics/cfg/models/11/yolo11-p2-full.yaml"
DATA="${PROJECT_ROOT}/configs/datasets/mpid.yaml"
RUN_NAME="mpid_exp08b_ase_yolov11"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-08b: MPID ASE-YOLOv11 迁移训练"
echo "======================================="
echo "模型配置:   ${MODEL_CFG}"
echo "预训练权重: ${PRETRAIN}"
echo "数据配置:   ${DATA}"
echo "输出目录:   runs/detect/${RUN_NAME}"
echo ""

yolo detect train \
    model="${MODEL_CFG}" \
    data="${DATA}" \
    pretrained="${PRETRAIN}" \
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
    plots=True \
    nwd_weight=0.4

echo ""
echo "EXP-08b 训练完成。"
echo "结果目录: ${PROJECT_ROOT}/runs/detect/${RUN_NAME}"
echo "最佳权重: ${PROJECT_ROOT}/runs/detect/${RUN_NAME}/weights/best.pt"
echo ""
echo "请将 mAP50 / mAP50-95 / Precision / Recall / FPS 记录到 logs/EXP-08_mpid_transfer.md"
