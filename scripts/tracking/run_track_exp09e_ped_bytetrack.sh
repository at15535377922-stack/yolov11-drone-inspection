#!/bin/bash
# =============================================================================
# EXP-09e：pedestrian 专用检测器 + ByteTrack（基线对比）
#
# 目标：用 EXP-09d 训练的 pedestrian 专用检测器替换通用 10 类检测器，
#       配合 ByteTrack，作为 EXP-09f（MCTrack）的公平对比基线。
#
# 与 EXP-09b 的区别：检测器从 EXP-07（10类）换为 EXP-09d（pedestrian单类）
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_ped_exp09d/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09e_ped_bytetrack"
TRACKER="bytetrack"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09e: ped检测器 + ByteTrack 基线"
echo "======================================="
echo "检测权重: ${DETECTOR_WEIGHTS}（pedestrian专用）"

mkdir -p "${OUTPUT_DIR}/mot_results"

for SEQ_DIR in "${MOT_DATA}"/*/; do
    SEQ_NAME=$(basename "${SEQ_DIR}")
    echo "[INFO] 处理序列: ${SEQ_NAME}"

    yolo track \
        model="${DETECTOR_WEIGHTS}" \
        source="${SEQ_DIR}" \
        tracker="${TRACKER}.yaml" \
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
    --tracker_name "exp09e_ped_bytetrack" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
