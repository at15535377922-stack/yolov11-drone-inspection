#!/usr/bin/env bash
# =============================================================================
# EXP-03: Lightweight Weighted Cross-Scale Fusion
# Model  : yolo11-p2-lwf.yaml (P2 branch + weighted fusion)
# Dataset: VisDrone2019-DET 10-class
# Goal   : Ablation — replace Concat with lightweight weighted fusion on EXP-02
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone10.yaml"
MODEL_CFG="yolo11-p2-lwf.yaml"
PRETRAINED="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone10_exp03_lwf"

cd "${PROJECT_ROOT}"
source venv/bin/activate
cd ultralytics

echo "=============================================="
echo " EXP-03: Lightweight Weighted Fusion Ablation"
echo " Model  : ${MODEL_CFG}"
echo " Data   : ${DATA_CFG}"
echo " Output : ${RUN_DIR}/${RUN_NAME}"
echo "=============================================="

yolo detect train \
    data="${DATA_CFG}" \
    model="${MODEL_CFG}" \
    pretrained="${PRETRAINED}" \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    workers=8 \
    project="${RUN_DIR}" \
    name="${RUN_NAME}" \
    exist_ok=False \
    cache=False

echo ""
echo "Training complete. Validating best.pt ..."

yolo detect val \
    data="${DATA_CFG}" \
    model="${RUN_DIR}/${RUN_NAME}/weights/best.pt" \
    imgsz=640 \
    batch=16 \
    device=0 \
    split=val

echo ""
echo "EXP-03 done. Results saved to ${RUN_DIR}/${RUN_NAME}/"
