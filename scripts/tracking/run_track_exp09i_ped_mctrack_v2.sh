#!/bin/bash
# =============================================================================
# EXP-09i：pedestrian 专用检测器 + MCTrack v2（严格新建轨迹阈值，隔离验证遮挡恢复价值）
#
# 目的：09h 证实 MCTrack 放宽 new_track_thresh 后引入更多误检轨迹，净效果为负。
#       本实验用 configs/trackers/mctrack_v2_strictnew.yaml（new_track_thresh 调回
#       与 ByteTrack 一致的 0.25，只保留更低的 track_low_thresh 用于挽救已有轨迹 +
#       GMC + ReID + 遮挡记忆），单独验证"遮挡恢复能力"本身是否有正向价值。
#
# 与 09h 的唯一区别：mctrack.yaml -> mctrack_v2_strictnew.yaml（new_track_thresh 0.20->0.25）
# 与 09g（ByteTrack, conf=0.05）是本实验真正的对比基线
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_ped_exp09d/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09i_ped_mctrack_v2"
TRACKER_CFG="${PROJECT_ROOT}/configs/trackers/mctrack_v2_strictnew.yaml"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09i: ped检测器 + MCTrack v2（new_track_thresh=0.25）"
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
    --tracker_name "exp09i_ped_mctrack_v2" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
