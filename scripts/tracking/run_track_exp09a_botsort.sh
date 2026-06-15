#!/bin/bash
# =============================================================================
# EXP-09a：VisDrone-MOT 上 ASE-YOLOv11 + BoT-SORT 追踪基线
#
# 目标：以 ASE-YOLOv11 (EXP-07 best.pt) 为检测前端，搭配 Ultralytics 内置
#       BoT-SORT 追踪器，在 VisDrone-MOT val 集上运行 tracking-by-detection，
#       输出 MOT Challenge 格式结果，供 TrackEval 评估。
#
# 评估指标：HOTA / MOTA / IDF1 / ID Switch / Fragmentation
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
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09a_botsort"
TRACKER="botsort"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09a: VisDrone-MOT BoT-SORT 基线"
echo "======================================="
echo "检测权重: ${DETECTOR_WEIGHTS}"
echo "数据目录: ${MOT_DATA}"
echo "追踪器:   ${TRACKER}"
echo "输出目录: ${OUTPUT_DIR}"
echo ""

mkdir -p "${OUTPUT_DIR}/mot_results"

# 对每个视频序列逐个运行追踪，输出 MOT Challenge 格式 txt
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

    # Ultralytics save_txt 输出格式：frame,id,x,y,w,h,conf,cls,_
    # 转换为标准 MOT Challenge 格式（见 scripts/tracking/convert_to_mot.py）
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
    --tracker_name "exp09a_botsort" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"

echo ""
echo "EXP-09a 评估完成。"
echo "结果目录: ${OUTPUT_DIR}/eval"
echo "请将 HOTA / MOTA / IDF1 / IDS / Frag 记录到 logs/EXP-09_tracking.md"
