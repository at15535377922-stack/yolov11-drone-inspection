#!/usr/bin/env bash
# =============================================================================
# EXP-04: Lightweight Channel-Spatial Attention on P2+LWFusion
# Model  : yolo11-p2-lwf-cbam.yaml (P2 branch + weighted fusion + CBAM)
# Dataset: VisDrone2019-DET 10-class
# Goal   : Ablation — add lightweight attention after key fusion outputs on EXP-03
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone10.yaml"
MODEL_CFG="yolo11-p2-lwf-cbam.yaml"
PRETRAINED="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone10_exp04_cbam"

cd "${PROJECT_ROOT}"
source venv/bin/activate
cd ultralytics

echo "=============================================="
echo " EXP-04: Lightweight Attention Ablation"
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
echo "EXP-04 done. Results saved to ${RUN_DIR}/${RUN_NAME}/"
