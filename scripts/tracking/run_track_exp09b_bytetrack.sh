#!/bin/bash
# =============================================================================
# EXP-09b：VisDrone-MOT 上 ASE-YOLOv11 + ByteTrack 对比实验
#
# 目标：以相同的 ASE-YOLOv11 (EXP-07 best.pt) 为检测前端，替换追踪器为
#       ByteTrack，与 EXP-09a(BoT-SORT) 形成公平对比。
#
# 关键改进（相比 EXP-09a）：
#   - tracker=bytetrack.yaml
#   - classes=0（只追踪 pedestrian，与 TrackEval 默认评估类对齐，消除 FP 污染）
#
# 使用前提：
#   1. ASE-YOLOv11 best.pt 已训练完成（EXP-07）
#   2. VisDrone-MOT 已解压到 data/visdrone-mot/
#   3. 已激活 venv，ultralytics 已安装
#   4. TrackEval 已安装（见 scripts/tracking/install_trackeval.sh）
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone10_exp07_full/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09b_bytetrack"
TRACKER="bytetrack"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09b: VisDrone-MOT ByteTrack 对比"
echo "======================================="
echo "检测权重: ${DETECTOR_WEIGHTS}"
echo "数据目录: ${MOT_DATA}"
echo "追踪器:   ${TRACKER}"
echo "输出目录: ${OUTPUT_DIR}"
echo "过滤类别: 0 (pedestrian only)"
echo ""

mkdir -p "${OUTPUT_DIR}/mot_results"

# 对每个视频序列逐个运行追踪
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
        classes=0 \
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

    echo "[INFO] 序列 ${SEQ_NAME} 完成"
done

echo ""
echo "======================================="
echo " 所有序列追踪完成，开始 TrackEval 评估"
echo "======================================="

python scripts/tracking/eval_trackeval.py \
    --tracker_name "exp09b_bytetrack" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
