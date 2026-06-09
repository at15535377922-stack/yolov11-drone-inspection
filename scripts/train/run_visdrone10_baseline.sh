#!/bin/bash
set -e

PROJECT_DIR="/root/autodl-tmp/yolov11-drone-inspection"
DATA_CFG="$PROJECT_DIR/configs/datasets/visdrone10.yaml"
MODEL_WEIGHTS="yolo11n.pt"
RUN_NAME="visdrone10_yolo11n_baseline"
IMGSZ=640
BATCH=32
EPOCHS=100
DEVICE=0
WORKERS=8

cd "$PROJECT_DIR"
source venv/bin/activate
cd ultralytics

python -m ultralytics cfg checks >/dev/null 2>&1 || true

yolo detect train \
  data="$DATA_CFG" \
  model="$MODEL_WEIGHTS" \
  epochs=$EPOCHS \
  imgsz=$IMGSZ \
  batch=$BATCH \
  device=$DEVICE \
  workers=$WORKERS \
  project="$PROJECT_DIR/runs/detect" \
  name="$RUN_NAME" \
  pretrained=True \
  cache=False
