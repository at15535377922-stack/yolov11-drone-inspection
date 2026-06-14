#!/usr/bin/env bash
# =============================================================================
# EXP-07: ASE-YOLOv11 Full Integration
# Model  : yolo11-p2-full.yaml
#          = P2 branch + LWFusion (6 nodes) + CBAM (P2/P3/P4/P5) + TADetect + NWD loss
# Dataset: VisDrone2019-DET 10-class
# Goal   : Verify additive effect of all ablated modules; target mAP50 >= 0.345
#
# Module stack vs baselines:
#   EXP-02 added : P2 branch         (+0.015 mAP50)
#   EXP-03 added : LWFusion          ( 0.000)
#   EXP-04 added : CBAM              ( 0.000)
#   EXP-05 added : NWD loss          (+0.001)
#   EXP-06 added : TADetect          (+0.001)
#   EXP-07       : all of the above combined
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone10.yaml"
MODEL_CFG="yolo11-p2-full.yaml"
PRETRAINED="${PROJECT_ROOT}/weights/pretrained/yolo11n.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone10_exp07_full"

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

# 验证关键模块能正常导入
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
echo " EXP-07: ASE-YOLOv11 Full Integration"
echo " Model  : ${MODEL_CFG}"
echo " Data   : ${DATA_CFG}"
echo " Output : ${RUN_DIR}/${RUN_NAME}"
echo " Stack  : P2 + LWFusion(x6) + CBAM(x4) + TADetect(ta_layers=2) + NWD loss"
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
echo "EXP-07 done. Results saved to ${RUN_DIR}/${RUN_NAME}/"
