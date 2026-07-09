#!/bin/bash
# =============================================================================
# EXP-09g：pedestrian 专用检测器 + ByteTrack（低 conf，激活双阈值分流）
#
# 目的：09e/09f 用 conf=0.25 时，MCTrack 的 track_low_thresh=0.05 分支永远不会
#       被触发（因为外层 conf=0.25 已经高于 track_high_thresh=0.15，凡是能到达
#       追踪器的框都被算作"高置信度"）。这里把外层 conf 降到 track_low_thresh
#       附近（0.05），让 0.05~0.25 区间的检测框真正进入追踪器的二次匹配逻辑，
#       09g（ByteTrack）与 09h（MCTrack）用完全相同的外层 conf 做公平对比。
#
# 与 09e 的唯一区别：conf 0.25 -> 0.05
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_ped_exp09d/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09g_ped_bytetrack_lowconf"
TRACKER="bytetrack"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09g: ped检测器 + ByteTrack（conf=0.05）"
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
    --tracker_name "exp09g_ped_bytetrack_lowconf" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
