#!/usr/bin/env bash
# =============================================================================
# EXP-09n：基线 YOLOv11n（EXP-01）+ VisDrone-MOT 域内微调
#
# 与 09j 的唯一区别：预训练权重来自 09m（基线 YOLOv11n 的 pedestrian 专用版），
# 而不是 09d（ASE-YOLOv11 的 pedestrian 专用版）。数据集与 09j 完全一致
# （VisDrone2019-MOT-ped-det，已由 09j 生成，无需重新构建）。
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone_mot_ped.yaml"
PRETRAINED="${PROJECT_ROOT}/runs/detect/visdrone_ped_exp09m_baseline/weights/best.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone_mot_ped_exp09n_baseline"

cd "${PROJECT_ROOT}"
source venv/bin/activate

echo "Clearing pycache..."
find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}" -name "*.pyc" -delete 2>/dev/null || true

echo "Reinstalling ultralytics from source..."
pip install -e "${PROJECT_ROOT}/ultralytics" -q

echo "========================================"
echo " EXP-09n: 基线 YOLOv11n + VisDrone-MOT 域内微调"
echo "========================================"
echo "预训练权重: ${PRETRAINED}（EXP-09m，基线 pedestrian 专用）"
echo "输出目录:  ${RUN_DIR}/${RUN_NAME}"
echo ""

# 数据集应已由 09j 生成；若未生成则先构建
if [ ! -d "${PROJECT_ROOT}/data/VisDrone2019-MOT-ped-det/labels/train" ]; then
    echo "[INFO] 生成 VisDrone-MOT 域内 pedestrian 检测数据集..."
    python scripts/dataset/build_visdrone_mot_ped_detset.py \
        --mot_train_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-train" \
        --dst "${PROJECT_ROOT}/data/VisDrone2019-MOT-ped-det" \
        --sample_stride 5 \
        --val_seq_ratio 0.15 \
        --seed 42
else
    echo "[INFO] VisDrone-MOT-ped-det 已存在（09j 生成），跳过"
fi

python -c "
from ultralytics import YOLO
model = YOLO('${PRETRAINED}')
model.train(
    data='${DATA_CFG}',
    epochs=30,
    imgsz=640,
    batch=32,
    workers=8,
    device=0,
    project='${RUN_DIR}',
    name='${RUN_NAME}',
    exist_ok=True,
    patience=15,
    optimizer='AdamW',
    lr0=5e-5,
    lrf=0.01,
    warmup_epochs=2,
    conf=0.001,
    iou=0.6,
    amp=True,
    cache=False,
    verbose=True,
)
"

echo ""
echo "========================================"
echo " EXP-09n 训练完成"
echo " 权重: ${RUN_DIR}/${RUN_NAME}/weights/best.pt"
echo "========================================"
