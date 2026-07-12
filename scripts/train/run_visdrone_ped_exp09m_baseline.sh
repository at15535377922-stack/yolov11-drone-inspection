#!/usr/bin/env bash
# =============================================================================
# EXP-09m：基线 YOLOv11n（EXP-01，无任何改进）pedestrian 专用检测器训练
#
# 目的：把"检测器创新点"（ASE-YOLOv11 相比基线 YOLOv11n）和"追踪性能"挂钩。
#       09d 是从 EXP-07（全量改进的 ASE-YOLOv11）微调出的 pedestrian 专用检测器；
#       本实验用完全相同的流程和超参数，但预训练权重换成 EXP-01 基线 YOLOv11n，
#       之后走一遍和 09d->09j->09k 完全一致的流程（09m->09n->09o），
#       与 09k 直接对比，检验检测器架构改进能否穿透到下游追踪指标。
#
# 与 09d 的唯一区别：预训练权重 EXP-07（ASE-YOLOv11） -> EXP-01（基线 YOLOv11n）
# 注意：EXP-01 是标准 yolo11n 架构（无 P2 分支等改进），直接从其 best.pt 加载
#       架构+权重继续训练，不需要额外指定模型结构 yaml。
#
# 前提：
#   已执行 scripts/dataset/filter_visdrone_ped.py 生成 VisDrone2019-DET-ped/
#   （与 09d 共用同一份数据，无需重新生成）
# =============================================================================

set -e

export OMP_NUM_THREADS=1

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="${PROJECT_ROOT}/configs/datasets/visdrone_ped.yaml"
# 注意：请先确认 EXP-01 基线权重的实际路径（训练脚本与日志记录的目录不完全一致，
# 需要用 find 命令核实），如与下方默认值不同请自行替换。
PRETRAINED="${PROJECT_ROOT}/runs/detect/visdrone10_yolo11n_baseline/weights/best.pt"
RUN_DIR="${PROJECT_ROOT}/runs/detect"
RUN_NAME="visdrone_ped_exp09m_baseline"

cd "${PROJECT_ROOT}"
source venv/bin/activate

echo "Clearing pycache..."
find "${PROJECT_ROOT}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}" -name "*.pyc" -delete 2>/dev/null || true

echo "Reinstalling ultralytics from source..."
pip install -e "${PROJECT_ROOT}/ultralytics" -q

if [ ! -f "${PRETRAINED}" ]; then
    echo "[ERROR] 预训练权重不存在: ${PRETRAINED}"
    echo "        请先用 find 命令核实 EXP-01 基线权重的实际路径，并修改本脚本的 PRETRAINED 变量"
    exit 1
fi

echo "========================================"
echo " EXP-09m: 基线 YOLOv11n pedestrian 专用检测器训练"
echo "========================================"
echo "数据配置:  ${DATA_CFG}"
echo "预训练权重: ${PRETRAINED}（EXP-01 基线，无 ASE-YOLOv11 改进）"
echo "输出目录:  ${RUN_DIR}/${RUN_NAME}"
echo ""

# Step 1: 生成 pedestrian 单类标签（如未生成，与 09d 共用）
if [ ! -d "${PROJECT_ROOT}/data/VisDrone2019-DET-ped/labels/train" ]; then
    echo "[INFO] 生成 pedestrian 单类标签..."
    python scripts/dataset/filter_visdrone_ped.py \
        --src "${PROJECT_ROOT}/data/VisDrone2019-DET" \
        --dst "${PROJECT_ROOT}/data/VisDrone2019-DET-ped"
else
    echo "[INFO] pedestrian 单类标签已存在，跳过生成"
fi

# Step 2: 训练（直接从 EXP-01 best.pt 加载架构+权重继续训练，与 09d 的超参数保持一致以便公平对比）
python -c "
from ultralytics import YOLO
model = YOLO('${PRETRAINED}')
model.train(
    data='${DATA_CFG}',
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
echo " EXP-09m 训练完成"
echo " 权重: ${RUN_DIR}/${RUN_NAME}/weights/best.pt"
echo "========================================"
