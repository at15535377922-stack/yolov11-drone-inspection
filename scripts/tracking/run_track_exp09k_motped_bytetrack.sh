#!/bin/bash
# =============================================================================
# EXP-09k：VisDrone-MOT 域内检测器（EXP-09j）+ ByteTrack（新基线）
#
# 目的：用域内微调的检测器（EXP-09j）替换 EXP-09d（DET 域），检验域内训练能否
#       真正提升 pedestrian 召回率；作为 EXP-09l（MCTrack）的公平对比基线。
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_mot_ped_exp09j/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09k_motped_bytetrack"
TRACKER="bytetrack"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09k: MOT域内检测器 + ByteTrack（conf=0.05）"
echo "======================================="
echo "检测权重: ${DETECTOR_WEIGHTS}（VisDrone-MOT 域内微调）"

mkdir -p "${OUTPUT_DIR}/mot_results"

for SEQ_DIR in "${MOT_DATA}"/*/; do
    SEQ_NAME=$(basename "${SEQ_DIR}")
    echo "[INFO] 处理序列: ${SEQ_NAME}"

    yolo track \
        model="${DETECTOR_WEIGHTS}" \
        source="${SEQ_DIR}" \
        tracker="${TRACKER}.yaml" \
        imgsz=640 \
        conf=0.05 \
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
    --tracker_name "exp09k_motped_bytetrack" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
