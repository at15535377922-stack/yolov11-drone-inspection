#!/usr/bin/env bash
# =============================================================================
# EXP-02: P2 Small-Object Detection Branch
# Model  : yolo11-p2.yaml (4-head: P2/P3/P4/P5)
# Dataset: VisDrone2019-DET 10-class
# Goal   : Ablation — add P2 branch only, all other conditions same as EXP-01
#
# Run on server:
#   cd /root/autodl-tmp/yolov11-drone-inspection
#   bash scripts/train/run_visdrone10_exp02_p2.sh
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone10.yaml"
MODEL_CFG="${PROJECT_ROOT}/ultralytics/ultralytics/cfg/models/11/yolo11-p2.yaml"
PRETRAINED="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone10_exp02_p2branch"

cd "${PROJECT_ROOT}"

echo "=============================================="
echo " EXP-02: P2 Branch Ablation"
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
    batch=32 \
    device=0 \
    workers=8 \
    project="${RUN_DIR}" \
    name="${RUN_NAME}" \
    exist_ok=False \
    verbose=True

echo ""
echo "Training complete. Validating best.pt ..."

yolo detect val \
    data="${DATA_CFG}" \
    model="${RUN_DIR}/${RUN_NAME}/weights/best.pt" \
    imgsz=640 \
    batch=32 \
    device=0 \
    split=val \
    verbose=True

echo ""
echo "EXP-02 done. Results saved to ${RUN_DIR}/${RUN_NAME}/"
