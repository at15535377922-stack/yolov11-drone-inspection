#!/bin/bash
# =============================================================================
# EXP-09o：基线 YOLOv11n（域内微调）+ ByteTrack —— 与 09k 直接对比
#
# 目的：09k 是 ASE-YOLOv11（EXP-07 全量改进）经过 pedestrian 专项 + MOT 域内
#       两次微调后 + ByteTrack 的追踪结果（MOTA=-1.92，本轮实验最佳）。
#       本实验用完全相同的流程微调基线 YOLOv11n（EXP-01，无 P2 分支/加权融合/
#       CBAM/NWD loss/TA head 等改进），与 09k 直接对比，检验"检测器架构创新"
#       本身能否穿透到下游追踪指标——这是把检测章节和追踪章节挂钩的关键实验。
#
# 与 09k 的唯一区别：检测器来自 09n（基线架构）而不是 09j（ASE-YOLOv11 架构）
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone_mot_ped_exp09n_baseline/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09o_baseline_bytetrack"
TRACKER="bytetrack"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09o: 基线检测器（域内微调）+ ByteTrack（conf=0.05）"
echo "======================================="
echo "检测权重: ${DETECTOR_WEIGHTS}（基线 YOLOv11n，无 ASE-YOLOv11 改进）"

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
    --tracker_name "exp09o_baseline_bytetrack" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
