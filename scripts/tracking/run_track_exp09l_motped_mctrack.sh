#!/bin/bash
# =============================================================================
# EXP-09l：VisDrone-MOT 域内检测器（EXP-09j）+ YOLOv11-MCTrack（本文方法）
#
# 目的：与 09k 组成公平对比（唯一变量是追踪器）。若域内训练真的提升了检测密度，
#       使关联任务出现足够的歧义场景，GMC/ReID/双阈值/遮挡记忆这次才有可能
#       展现出 09e~09i 中一直未见的收益。
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_mot_ped_exp09j/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09l_motped_mctrack"
TRACKER_CFG="${PROJECT_ROOT}/configs/trackers/mctrack.yaml"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09l: MOT域内检测器 + MCTrack（conf=0.05）"
echo "======================================="
echo "检测权重: ${DETECTOR_WEIGHTS}（VisDrone-MOT 域内微调）"
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
    --tracker_name "exp09l_motped_mctrack" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
