#!/usr/bin/env bash
# =============================================================================
# EXP-07b: ASE-YOLOv11 Full Integration — Tuned Variant
# Model  : yolo11-p2-full-b.yaml
#          = P2 branch + LWFusion(×6) + CBAM(×4) + TADetect(ta_layers=1) + NWD loss
# Dataset: VisDrone2019-DET 10-class
#
# Changes vs EXP-07:
#   1. ta_layers: 2 → 1  (TA interaction on P2 only, skip P3)
#      Motivation: EXP-07 showed tricycle(−0.016) and awning-tricycle(−0.014)
#      regression, likely caused by LWFusion noise amplified by P3 TA interaction
#      on densely packed small objects. Restricting to P2 (finest scale) reduces
#      this interference while preserving the cls/reg alignment benefit on the
#      smallest targets.
#   2. nwd_weight: 0.4 → 0.5  (strengthen NWD contribution)
#      Motivation: Recover tricycle/awning-tricycle AP that EXP-07 lost vs EXP-06.
#      Higher NWD weight improves localization smoothness for objects < 20px.
#
# All other hyperparameters identical to EXP-07.
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone10.yaml"
MODEL_CFG="yolo11-p2-full-b.yaml"
PRETRAINED="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone10_exp07b_full"

cd "${PROJECT_ROOT}"
source venv/bin/activate

# 清除整个项目下所有 Python 字节码缓存
echo "Clearing pycache..."
find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}" -name "*.pyc" -delete 2>/dev/null || true

# 重新安装 ultralytics 源码包
echo "Reinstalling ultralytics from source..."
pip install -e "${PROJECT_ROOT}/ultralytics" -q

cd "${PROJECT_ROOT}/ultralytics"

# 验证关键模块
python -c "
from ultralytics.nn.modules.head import TADetect
from ultralytics.nn.modules import LWFusion, CBAM
import inspect
sig = inspect.signature(TADetect.__init__)
params = list(sig.parameters.keys())
assert params == ['self', 'nc', 'reg_max', 'ta_layers', 'ch'], f'WRONG TADetect sig: {params}'
print('[CHECK] TADetect sig OK:', sig)
print('[CHECK] LWFusion OK:', LWFusion)
print('[CHECK] CBAM OK:', CBAM)
print('[CHECK] All modules verified.')
"

echo "=============================================="
echo " EXP-07b: ASE-YOLOv11 Full Integration (Tuned)"
echo " Model   : ${MODEL_CFG}"
echo " Data    : ${DATA_CFG}"
echo " Output  : ${RUN_DIR}/${RUN_NAME}"
echo " Stack   : P2 + LWFusion(x6) + CBAM(x4) + TADetect(ta_layers=1) + NWD loss"
echo " Changes : ta_layers 2→1 | nwd_weight 0.4→0.5"
echo " Loss    : CIoU(alpha=0.5) + NWD(beta=0.5) + size-weight(rho=0.5)"
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
    nwd_weight=0.5 \
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
echo "EXP-07b done. Results saved to ${RUN_DIR}/${RUN_NAME}/"
