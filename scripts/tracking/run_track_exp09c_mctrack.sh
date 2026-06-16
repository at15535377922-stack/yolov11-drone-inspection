#!/bin/bash
# =============================================================================
# EXP-09c：YOLOv11-MCTrack 自研追踪方法
#
# 目标：以 ASE-YOLOv11 为检测前端，搭配自研 MCTrack 配置，
#       在 VisDrone-MOT val 上评估，与 09a/09b 形成三方对比。
#
# MCTrack 相对 BoT-SORT/ByteTrack 的改进：
#   1. GMC sparseOptFlow 相机运动补偿（补偿无人机飞行偏差）
#   2. 双阈值降低（0.15/0.05）提升小目标 pedestrian 召回
#   3. ReID 开启（model=auto 用检测器特征，零额外参数）
#   4. track_buffer=60 扩大遮挡记忆
#   5. fuse_score=True 融合置信度与 IoU 关联
#   6. classes=0 只追踪 pedestrian（与 09b 统一评估口径）
#
# 使用前提：
#   1. ASE-YOLOv11 best.pt 已训练完成（EXP-07）
#   2. VisDrone-MOT 已解压到 data/visdrone-mot/
#   3. 已激活 venv，ultralytics 已安装
#   4. TrackEval 已安装
# =============================================================================

set -e

PROJECT_ROOT="/root/autodl-tmp/yolov11-drone-inspection"
DETECTOR_WEIGHTS="${PROJECT_ROOT}/runs/detect/visdrone10_exp07_full/weights/best.pt"
MOT_DATA="${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/sequences"
OUTPUT_DIR="${PROJECT_ROOT}/runs/track/exp09c_mctrack"
TRACKER_CFG="${PROJECT_ROOT}/configs/trackers/mctrack.yaml"

cd "$PROJECT_ROOT"
source venv/bin/activate

echo "======================================="
echo " EXP-09c: YOLOv11-MCTrack 自研方法"
echo "======================================="
echo "检测权重:   ${DETECTOR_WEIGHTS}"
echo "数据目录:   ${MOT_DATA}"
echo "追踪器配置: ${TRACKER_CFG}"
echo "输出目录:   ${OUTPUT_DIR}"
echo "过滤类别:   0 (pedestrian only)"
echo ""

mkdir -p "${OUTPUT_DIR}/mot_results"

# 对每个视频序列逐个运行追踪
for SEQ_DIR in "${MOT_DATA}"/*/; do
    SEQ_NAME=$(basename "${SEQ_DIR}")
    echo "[INFO] 处理序列: ${SEQ_NAME}"

    yolo track \
        model="${DETECTOR_WEIGHTS}" \
        source="${SEQ_DIR}" \
        tracker="${TRACKER_CFG}" \
        imgsz=640 \
        conf=0.10 \
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
    --tracker_name "exp09c_mctrack" \
    --results_dir "${OUTPUT_DIR}/mot_results" \
    --gt_dir "${PROJECT_ROOT}/data/visdrone-mot/VisDrone2019-MOT-val/annotations" \
    --output_dir "${OUTPUT_DIR}/eval"
