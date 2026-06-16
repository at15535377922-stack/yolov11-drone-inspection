#!/usr/bin/env bash
# =============================================================================
# EXP-09d：VisDrone pedestrian 专用检测器训练
#
# 目标：以 ASE-YOLOv11（EXP-07 结构）微调训练 pedestrian 单类检测器，
#       作为 EXP-09e/09f 追踪实验的高召回检测前端，替换 10 类通用检测器。
#
# 设计依据：
#   EXP-09b/09c 实验揭示，追踪性能瓶颈在 pedestrian 检测端（DetA≈12，CLR_Re≈18%）。
#   专用单类检测器可大幅提升 pedestrian 召回率，从而使追踪器关联能力的差异得以显现。
#
# 训练策略：
#   - 使用 EXP-07 best.pt 作为预训练权重（迁移学习，而非从头训练）
#   - 仅保留 pedestrian（class=0）标签，nc=1
#   - 较低 conf 阈值（0.25），较多 epochs（50，因已有好的初始化）
#   - 保持 640 输入，batch=32，与 EXP-07 一致
#
# 前提：
#   已执行 scripts/dataset/filter_visdrone_ped.py 生成 VisDrone2019-DET-ped/
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone_ped.yaml"
MODEL_CFG="yolo11-p2-full.yaml"
# 从 EXP-07 最优权重迁移，而非 yolo11n.pt 从头来
PRETRAINED="${PROJECT_ROOT}/runs/detect/visdrone10_exp07_full/weights/best.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone_ped_exp09d"

cd "${PROJECT_ROOT}"
source venv/bin/activate

echo "Clearing pycache..."
find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}" -name "*.pyc" -delete 2>/dev/null || true

echo "Reinstalling ultralytics from source..."
pip install -e "${PROJECT_ROOT}/ultralytics" -q

echo "========================================"
echo " EXP-09d: pedestrian 专用检测器训练"
echo "========================================"
echo "数据配置:  ${DATA_CFG}"
echo "模型结构:  ${MODEL_CFG}"
echo "预训练权重: ${PRETRAINED}（EXP-07）"
echo "输出目录:  ${RUN_DIR}/${RUN_NAME}"
echo ""

# Step 1: 生成 pedestrian 单类标签（如未生成）
if [ ! -d "${PROJECT_ROOT}/data/VisDrone2019-DET-ped/labels/train" ]; then
    echo "[INFO] 生成 pedestrian 单类标签..."
    python scripts/dataset/filter_visdrone_ped.py \
        --src "${PROJECT_ROOT}/data/VisDrone2019-DET" \
        --dst "${PROJECT_ROOT}/data/VisDrone2019-DET-ped"
else
    echo "[INFO] pedestrian 单类标签已存在，跳过生成"
fi

# Step 2: 训练
python -c "
from ultralytics import YOLO
model = YOLO('${MODEL_CFG}')
model.train(
    data='${DATA_CFG}',
    pretrained='${PRETRAINED}',
    epochs=50,
    imgsz=640,
    batch=32,
    workers=8,
    device=0,
    project='${RUN_DIR}',
    name='${RUN_NAME}',
    exist_ok=True,
    patience=20,
    optimizer='AdamW',
    lr0=1e-4,
    lrf=0.01,
    warmup_epochs=3,
    conf=0.001,
    iou=0.6,
    amp=True,
    cache=False,
    verbose=True,
)
"

echo ""
echo "========================================"
echo " EXP-09d 训练完成"
echo " 权重: ${RUN_DIR}/${RUN_NAME}/weights/best.pt"
echo "========================================"
