#!/usr/bin/env bash
# =============================================================================
# EXP-05: Small-Object Box Loss Optimisation (CIoU + NWD + Size-Weighting)
# Model  : yolo11-p2.yaml  (P2 branch ONLY — same as EXP-02)
# Dataset: VisDrone2019-DET 10-class
# Goal   : Ablation — verify loss-function contribution independent of
#          structural changes (LWFusion / CBAM not included).
#
# Key changes vs EXP-02:
#   - nwd_weight=0.4  → L_box = 0.6*(1-CIoU) + 0.4*(1-NWD)  per fg anchor
#   - nwd_constant=12.8 → C in NWD = imgsz * 0.02  (640 * 0.02)
#   - nwd_small_rho=0.5 → extra up-weighting for micro-boxes
#   All other hyper-parameters identical to EXP-02.
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone10.yaml"
MODEL_CFG="yolo11-p2.yaml"                         # reuse EXP-02 backbone — loss is the only variable
PRETRAINED="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone10_exp05_nwd"

cd "${PROJECT_ROOT}"
source venv/bin/activate
cd ultralytics

echo "=============================================="
echo " EXP-05: NWD + Small-Object Loss Ablation"
echo " Model  : ${MODEL_CFG}"
echo " Data   : ${DATA_CFG}"
echo " Output : ${RUN_DIR}/${RUN_NAME}"
echo " Loss   : CIoU (alpha=0.6) + NWD (beta=0.4) + size-weight (rho=0.5)"
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
    exist_ok=True \
    cache=False \
    nwd_weight=0.4 \
    nwd_constant=12.8 \
    nwd_small_rho=0.5

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
echo "EXP-05 done. Results saved to ${RUN_DIR}/${RUN_NAME}/"
