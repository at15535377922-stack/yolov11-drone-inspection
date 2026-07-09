#!/bin/bash
# =============================================================================
# EXP-09h：pedestrian 专用检测器 + YOLOv11-MCTrack（低 conf，激活双阈值分流）
#
# 目的：与 09g 组成公平对比（唯一变量是追踪器：ByteTrack vs MCTrack），
#       外层 conf=0.05 使 0.05~0.25 区间的检测框能进入 MCTrack 的
#       track_low_thresh=0.05 / track_high_thresh=0.15 双阈值分流逻辑，
#       真正激活双阈值创新点，而不是像 09f 那样被外层 conf=0.25 完全屏蔽。
#
# 与 09f 的唯一区别：conf 0.25 -> 0.05
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_ped_exp09d/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09h_ped_mctrack_lowconf"
TRACKER_CFG="${PROJECT_ROOT}/configs/trackers/mctrack.yaml"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09h: ped检测器 + MCTrack（conf=0.05）"
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
    --tracker_name "exp09h_ped_mctrack_lowconf" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
