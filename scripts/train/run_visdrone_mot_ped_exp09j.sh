#!/usr/bin/env bash
# =============================================================================
# EXP-09j：VisDrone-MOT 域内 pedestrian 检测器训练（排查训练/评估域差异假设）
#
# 背景：
#   EXP-09d 用 VisDrone2019-DET（静态图像）微调 pedestrian 专用检测器，在
#   VisDrone-MOT val 上 CLR_Re 只从 17.4% 提升到 19.7%，提升有限。09e~09i 的
#   完整追踪器消融（GMC/ReID/双阈值/遮挡记忆）均未见收益，瓶颈定位在检测召回率本身。
#   一个尚未排除的假设：VisDrone-DET（静态图）与 VisDrone-MOT（视频帧）之间存在
#   拍摄场景/压缩/运动模糊等域差异，导致检测器泛化不足。
#
#   本实验直接用 VisDrone2019-MOT-train 的视频帧（而不是 DET 静态图）训练，
#   彻底消除训练/评估域差异，验证召回率能否进一步提升。
#
# 训练策略：
#   - 从 EXP-09d best.pt 继续微调（已经是 pedestrian 专用，此步只解决域差异，
#     不重新学习类别本身）
#   - 数据来自 scripts/dataset/build_visdrone_mot_ped_detset.py 构建的
#     VisDrone2019-MOT-ped-det（按序列划分 train/val，MOT-val 不参与训练/验证）
#   - epochs 减少（30，因为已经是二次微调，数据量也更小）
#
# 前提：
#   已执行 scripts/dataset/build_visdrone_mot_ped_detset.py 生成
#   data/VisDrone2019-MOT-ped-det/
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone_mot_ped.yaml"
MODEL_CFG="yolo11-p2-full.yaml"
PRETRAINED="${PROJECT_ROOT}/runs/detect/visdrone_ped_exp09d/weights/best.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone_mot_ped_exp09j"

cd "${PROJECT_ROOT}"
source venv/bin/activate

echo "Clearing pycache..."
find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}" -name "*.pyc" -delete 2>/dev/null || true

echo "Reinstalling ultralytics from source..."
pip install -e "${PROJECT_ROOT}/ultralytics" -q

echo "========================================"
echo " EXP-09j: VisDrone-MOT 域内 pedestrian 检测器训练"
echo "========================================"
echo "数据配置:  ${DATA_CFG}"
echo "预训练权重: ${PRETRAINED}（EXP-09d，pedestrian 专用，DET 域）"
echo "输出目录:  ${RUN_DIR}/${RUN_NAME}"
echo ""

# Step 1: 构建 MOT 域内检测数据集（如未生成）
if [ ! -d "${PROJECT_ROOT}/data/VisDrone2019-MOT-ped-det/labels/train" ]; then
    echo "[INFO] 生成 VisDrone-MOT 域内 pedestrian 检测数据集..."
    python scripts/dataset/build_visdrone_mot_ped_detset.py \
        --mot_train_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-train" \
        --dst "${PROJECT_ROOT}/data/VisDrone2019-MOT-ped-det" \
        --sample_stride 5 \
        --val_seq_ratio 0.15 \
        --seed 42
else
    echo "[INFO] VisDrone-MOT-ped-det 已存在，跳过生成"
fi

# Step 2: 训练
python -c "
from ultralytics import YOLO
model = YOLO('${MODEL_CFG}')
model.train(
    data='${DATA_CFG}',
    pretrained='${PRETRAINED}',
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
echo " EXP-09j 训练完成"
echo " 权重: ${RUN_DIR}/${RUN_NAME}/weights/best.pt"
echo "========================================"
