#!/usr/bin/env bash
# =============================================================================
# EXP-06: Task-Aligned Detection Head (TADetect) on P2/P3
# Model  : yolo11-p2-tahead.yaml  (P2 branch + TADetect, no LWFusion/CBAM)
# Dataset: VisDrone2019-DET 10-class
# Goal   : Ablation — verify TADetect contribution on top of EXP-05 (NWD loss)
#          Single variable: only the detection head changes vs EXP-05.
#
# TADetect applies bidirectional cls/reg interaction on P2 and P3 scales:
#   F_cls' = F_cls * Spatial_Attn(F_reg)
#   F_reg' = F_reg * Channel_Attn(F_cls)
# P4 and P5 run the standard Detect forward path unchanged.
#
# NWD loss is kept (same hyp as EXP-05) to maintain consistent loss baseline.
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone10.yaml"
MODEL_CFG="yolo11-p2-tahead.yaml"
PRETRAINED="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone10_exp06_tahead"

cd "${PROJECT_ROOT}"
source venv/bin/activate
cd ultralytics

echo "=============================================="
echo " EXP-06: Task-Aligned Detection Head (TADetect)"
echo " Model  : ${MODEL_CFG}"
echo " Data   : ${DATA_CFG}"
echo " Output : ${RUN_DIR}/${RUN_NAME}"
echo " Head   : TADetect (ta_layers=2, P2+P3 bidirectional cls/reg interaction)"
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
echo "EXP-06 done. Results saved to ${RUN_DIR}/${RUN_NAME}/"
