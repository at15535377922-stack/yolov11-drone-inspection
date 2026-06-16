#!/bin/bash
# =============================================================================
# EXP-09f：pedestrian 专用检测器 + MCTrack（本文方法）
#
# 目标：用 EXP-09d pedestrian 专用检测器 + MCTrack 配置，
#       与 EXP-09e（ByteTrack）形成公平对比，验证 MCTrack 追踪器创新点效果。
#
# 这是论文追踪章节的核心对比实验：
#   EXP-09e: ped检测器 + ByteTrack  → 追踪基线
#   EXP-09f: ped检测器 + MCTrack    → 本文方法
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_ped_exp09d/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09f_ped_mctrack"
TRACKER_CFG="${PROJECT_ROOT}/configs/trackers/mctrack.yaml"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09f: ped检测器 + MCTrack 本文方法"
echo "======================================="
echo "检测权重: ${DETECTOR_WEIGHTS}（pedestrian专用）"
echo "追踪配置: ${TRACKER_CFG}"

mkdir -p "${OUTPUT_DIR}/mot_results"

for SEQ_DIR in "${MOT_DATA}"/*/; do
    SEQ_NAME=$(basename "${SEQ_DIR}")
    echo "[INFO] 处理序列: ${SEQ_NAME}"

    yolo track \
        model="${DETECTOR_WEIGHTS}" \
        source="${SEQ_DIR}" \
        tracker="${TRACKER_CFG}" \
        imgsz=640 \
        conf=0.25 \
        iou=0.45 \
        device=0 \
        save=False \
        save_txt=True \
        project="${OUTPUT_DIR}/raw" \
        name="${SEQ_NAME}" \
        exist_ok=True \
        stream_buffer=False

    python scripts/tracking/convert_to_mot.py \
        --input "${OUTPUT_DIR}/raw/${SEQ_NAME}/labels" \
        --output "${OUTPUT_DIR}/mot_results/${SEQ_NAME}.txt" \
        --seq_name "${SEQ_NAME}"
done

python scripts/tracking/eval_trackeval.py \
    --tracker_name "exp09f_ped_mctrack" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
