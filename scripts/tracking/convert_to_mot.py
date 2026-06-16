#!/usr/bin/env python3
"""
convert_to_mot.py
将 Ultralytics `yolo track --save_txt` 输出的 labels/ 目录
转换为 MOT Challenge 标准格式 txt 文件。

Ultralytics track save_txt 格式（每行）：
  frame_idx  track_id  x_center  y_center  w  h  conf  cls
  （均为归一化坐标，frame_idx 从 0 开始）

MOT Challenge 格式（每行）：
  frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1
  （像素坐标，frame 从 1 开始，conf 保留原始置信度）

用法：
  python scripts/tracking/convert_to_mot.py \
      --input  runs/track/exp09a_botsort/raw/uav0000013_00000_v/labels \
      --output runs/track/exp09a_botsort/mot_results/uav0000013_00000_v.txt \
      --seq_name uav0000013_00000_v \
      --img_w 1920 --img_h 1080   # 可选，若不填则读第一张图推断
"""

import argparse
import os
import glob
from pathlib import Path
from PIL import Image


def get_image_size(seq_root: str, seq_name: str) -> tuple[int, int]:
    """从序列图片目录读取图像尺寸。"""
    # 尝试找序列图像目录
    candidates = [
        os.path.join(seq_root, seq_name),                             # data/visdrone-mot/.../sequences/SEQ/
        os.path.join(seq_root, "VisDrone2019-MOT-val", "sequences", seq_name),
        os.path.join(seq_root, "VisDrone2019-MOT-train", "sequences", seq_name),
    ]
    for d in candidates:
        imgs = sorted(glob.glob(os.path.join(d, "*.jpg")))
        if imgs:
            img = Image.open(imgs[0])
            return img.size  # (width, height)
    raise FileNotFoundError(f"无法找到序列 {seq_name} 的图像目录，请手动指定 --img_w / --img_h")


def convert(input_dir: str, output_path: str, seq_name: str,
            img_w: int | None, img_h: int | None, data_root: str) -> None:
    label_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    if not label_files:
        print(f"[WARN] {input_dir} 下没有 txt 文件，跳过 {seq_name}")
        return

    # 推断图像尺寸
    if img_w is None or img_h is None:
        img_w, img_h = get_image_size(data_root, seq_name)
        print(f"[INFO] 图像尺寸自动推断: {img_w}x{img_h}")

    rows = []
    for lf in label_files:
        # 文件名格式：000001.txt（帧号，从 1 开始）
        frame_id = int(Path(lf).stem)
        with open(lf) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                # Ultralytics track txt: track_id cx cy w h conf [cls]
                # 注意：frame 信息编码在文件名中，不在行内
                # Ultralytics track save_txt 实际格式：cls cx cy w h track_id（共6列）
                cls_id   = int(float(parts[0]))
                cx       = float(parts[1])
                cy       = float(parts[2])
                w        = float(parts[3])
                h        = float(parts[4])
                track_id = int(float(parts[5])) if len(parts) > 5 else 0
                conf     = 1.0  # save_txt 不含置信度，填 1.0

                # 反归一化
                bb_left  = (cx - w / 2) * img_w
                bb_top   = (cy - h / 2) * img_h
                bb_w     = w * img_w
                bb_h     = h * img_h

                rows.append(
                    f"{frame_id},{track_id},{bb_left:.2f},{bb_top:.2f},"
                    f"{bb_w:.2f},{bb_h:.2f},{conf:.4f},-1,-1,-1"
                )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(rows))
    print(f"[INFO] {seq_name}: {len(rows)} 行 -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",    required=True, help="Ultralytics labels/ 目录")
    parser.add_argument("--output",   required=True, help="输出 MOT txt 路径")
    parser.add_argument("--seq_name", required=True, help="序列名称（用于查找图像目录）")
    parser.add_argument("--img_w",    type=int, default=None, help="图像宽度（像素）")
    parser.add_argument("--img_h",    type=int, default=None, help="图像高度（像素）")
    parser.add_argument("--data_root", default="/root/autodl-tmp/yolov11-drone-inspection/data/visdrone-mot",
                        help="VisDrone-MOT 数据根目录（用于自动推断图像尺寸）")
    args = parser.parse_args()
    convert(args.input, args.output, args.seq_name, args.img_w, args.img_h, args.data_root)


if __name__ == "__main__":
    main()
