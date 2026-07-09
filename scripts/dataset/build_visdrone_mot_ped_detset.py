#!/usr/bin/env python3
"""
build_visdrone_mot_ped_detset.py

从 VisDrone2019-MOT-train 的视频序列 + MOT 格式标注，构建一个
pedestrian 单类检测训练集，直接来自追踪任务的视频域（而不是 VisDrone-DET 静态图像域），
用于排查 EXP-09d（用 VisDrone-DET 微调）可能存在的训练/评估域差异问题。

背景：
  EXP-09d 用 VisDrone2019-DET（静态图像，拍摄场景/高度/压缩与 MOT 视频不同）微调出的
  pedestrian 专用检测器，在 VisDrone-MOT val 上的 CLR_Re 仅从 17.4% 提升到 19.7%，
  提升有限。一个未排除的假设是训练域（DET 静态图）与评估域（MOT 视频帧）存在差异，
  直接用 MOT-train 的视频帧训练可能进一步提升召回率。

关键设计：
  - 按【序列】切分 train/val（而不是按帧），避免同一段视频的相邻帧分别落入
    train 和 val 造成数据泄漏。
  - VisDrone2019-MOT-val（本实验最终追踪评估用的验证集）全程不参与检测器的
    训练或验证，避免评估泄漏。
  - 按固定间隔抽帧（默认每 5 帧取 1 帧），减少视频连续帧的高度冗余，
    同时保留足够的场景/时序多样性。
  - 只保留 category=1（pedestrian，对应 VisDrone-DET 的 class=0），
    与 EXP-09d/TrackEval 的评估口径保持一致。

标注格式（VisDrone-MOT，MOT Challenge 兼容）：
  frame_id, target_id, bb_left, bb_top, bb_width, bb_height, score, category, truncation, occlusion
  frame_id 从 1 开始，category=1 为 pedestrian。

用法（服务器执行）：
  python scripts/dataset/build_visdrone_mot_ped_detset.py \
      --mot_train_dir /root/autodl-tmp/yolov11-drone-inspection/data/visdrone-mot/VisDrone2019-MOT-train \
      --dst /root/autodl-tmp/yolov11-drone-inspection/data/VisDrone2019-MOT-ped-det \
      --sample_stride 5 \
      --val_seq_ratio 0.15 \
      --seed 42
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path


def parse_mot_gt(gt_path: Path, keep_category: int = 1):
    """按 frame_id 分组，返回 {frame_id: [(bb_left, bb_top, bb_w, bb_h), ...]}（像素坐标）。"""
    frame_boxes = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            frame_id = int(parts[0])
            bb_left, bb_top, bb_w, bb_h = map(float, parts[2:6])
            category = int(parts[7])
            if category != keep_category:
                continue
            if bb_w <= 0 or bb_h <= 0:
                continue
            frame_boxes[frame_id].append((bb_left, bb_top, bb_w, bb_h))
    return frame_boxes


def write_yolo_label(label_path: Path, boxes, img_w: int, img_h: int):
    lines = []
    for bb_left, bb_top, bb_w, bb_h in boxes:
        cx = (bb_left + bb_w / 2) / img_w
        cy = (bb_top + bb_h / 2) / img_h
        w = bb_w / img_w
        h = bb_h / img_h
        # 裁剪到 [0, 1]，防止标注越界
        cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in (cx, cy, w, h))
        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_path, "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def get_image_size(img_path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(img_path) as im:
        return im.size  # (w, h)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mot_train_dir", required=True, help="VisDrone2019-MOT-train 根目录")
    parser.add_argument("--dst", required=True, help="输出检测数据集根目录")
    parser.add_argument("--sample_stride", type=int, default=5, help="抽帧间隔（每 N 帧取 1 帧）")
    parser.add_argument("--val_seq_ratio", type=float, default=0.15, help="按序列划为 val 的比例")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    mot_train_dir = Path(args.mot_train_dir)
    dst = Path(args.dst)
    seq_root = mot_train_dir / "sequences"
    ann_root = mot_train_dir / "annotations"

    seq_names = sorted(p.name for p in seq_root.iterdir() if p.is_dir())
    print(f"[INFO] 共发现 {len(seq_names)} 个训练序列")

    random.seed(args.seed)
    shuffled = seq_names[:]
    random.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * args.val_seq_ratio))
    val_seqs = set(shuffled[:n_val])
    train_seqs = set(shuffled[n_val:])
    print(f"[INFO] 按序列划分: train={len(train_seqs)} 个序列, val={len(val_seqs)} 个序列")
    print(f"[INFO] val 序列: {sorted(val_seqs)}")

    stats = {"train": {"frames": 0, "boxes": 0}, "val": {"frames": 0, "boxes": 0}}

    for seq_name in seq_names:
        split = "val" if seq_name in val_seqs else "train"
        gt_path = ann_root / f"{seq_name}.txt"
        if not gt_path.exists():
            print(f"[WARN] 标注文件不存在，跳过: {gt_path}")
            continue

        frame_boxes = parse_mot_gt(gt_path, keep_category=1)
        seq_img_dir = seq_root / seq_name
        img_files = sorted(seq_img_dir.glob("*.jpg"))
        if not img_files:
            print(f"[WARN] {seq_name} 无图像，跳过")
            continue

        img_w, img_h = get_image_size(img_files[0])

        for idx, img_path in enumerate(img_files):
            if idx % args.sample_stride != 0:
                continue
            frame_id = int(img_path.stem)  # 文件名即帧号，如 0000001.jpg -> 1
            boxes = frame_boxes.get(frame_id, [])

            dst_img_dir = dst / "images" / split
            dst_lbl_dir = dst / "labels" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)

            dst_img_name = f"{seq_name}_{img_path.name}"
            dst_img_path = dst_img_dir / dst_img_name
            if not dst_img_path.exists():
                if dst_img_path.is_symlink():
                    dst_img_path.unlink()
                os.symlink(img_path.resolve(), dst_img_path)

            dst_lbl_path = dst_lbl_dir / (dst_img_name.rsplit(".", 1)[0] + ".txt")
            write_yolo_label(dst_lbl_path, boxes, img_w, img_h)

            stats[split]["frames"] += 1
            stats[split]["boxes"] += len(boxes)

    print()
    print("[DONE] 数据集构建完成")
    for split in ("train", "val"):
        print(f"  {split}: {stats[split]['frames']} 帧, {stats[split]['boxes']} 个 pedestrian 框")


if __name__ == "__main__":
    main()
