#!/usr/bin/env python3
"""
filter_visdrone_ped.py
从 VisDrone2019-DET 的 YOLO 格式标签中，只保留 class=0（pedestrian），
输出到新目录，供 pedestrian 单类检测器训练使用。

用法（服务器执行）：
  python scripts/dataset/filter_visdrone_ped.py \
      --src  /root/autodl-tmp/yolov11-drone-inspection/data/VisDrone2019-DET \
      --dst  /root/autodl-tmp/yolov11-drone-inspection/data/VisDrone2019-DET-ped

目录结构：
  src/
    images/{train,val,test}/
    labels/{train,val,test}/   ← 每行: cls cx cy w h（YOLO 格式，归一化）
  dst/
    images/{train,val,test}/   ← 软链接到 src images（节省磁盘）
    labels/{train,val,test}/   ← 过滤后仅含 class=0 行，并重映射为 0

说明：
  - 原始多类标签中 class=0 为 pedestrian，直接过滤保留即可
  - class id 不需要重映射（本来就是 0）
  - 如果某张图片没有 pedestrian，输出空标签文件（Ultralytics 支持）
"""

import argparse
import os
import shutil
from pathlib import Path


def filter_labels(src_label_dir: Path, dst_label_dir: Path, keep_cls: int = 0) -> tuple[int, int, int]:
    """过滤标签目录，只保留指定类别。返回 (总文件数, 有目标文件数, 总目标框数)。"""
    dst_label_dir.mkdir(parents=True, exist_ok=True)
    total_files = 0
    files_with_target = 0
    total_boxes = 0

    for lf in sorted(src_label_dir.glob("*.txt")):
        total_files += 1
        kept_lines = []
        with open(lf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5 and int(parts[0]) == keep_cls:
                    kept_lines.append(line)

        dst_lf = dst_label_dir / lf.name
        with open(dst_lf, "w") as f:
            f.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))

        if kept_lines:
            files_with_target += 1
            total_boxes += len(kept_lines)

    return total_files, files_with_target, total_boxes


def link_images(src_img_dir: Path, dst_img_dir: Path) -> None:
    """创建软链接（节省磁盘空间，图片无需复制）。"""
    dst_img_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_img_dir.exists() or dst_img_dir.is_symlink():
        dst_img_dir.unlink() if dst_img_dir.is_symlink() else shutil.rmtree(dst_img_dir)
    os.symlink(src_img_dir.resolve(), dst_img_dir)
    print(f"  [LINK] {dst_img_dir} -> {src_img_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="原始 VisDrone2019-DET 根目录")
    parser.add_argument("--dst", required=True, help="输出 pedestrian 单类数据集根目录")
    parser.add_argument("--keep_cls", type=int, default=0, help="保留的类别 id（默认 0=pedestrian）")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    print(f"[INFO] 源目录: {src}")
    print(f"[INFO] 目标目录: {dst}")
    print(f"[INFO] 保留类别 id: {args.keep_cls}")
    print()

    for split in ["train", "val", "test"]:
        src_img = src / "images" / split
        src_lbl = src / "labels" / split
        dst_img = dst / "images" / split
        dst_lbl = dst / "labels" / split

        if not src_lbl.exists():
            print(f"  [SKIP] {split}: 标签目录不存在 {src_lbl}")
            continue

        print(f"[{split}]")
        # 图片软链接
        if src_img.exists():
            link_images(src_img, dst_img)
        else:
            print(f"  [WARN] 图片目录不存在: {src_img}")

        # 过滤标签
        total, with_target, boxes = filter_labels(src_lbl, dst_lbl, keep_cls=args.keep_cls)
        ratio = with_target / total * 100 if total > 0 else 0
        print(f"  [LABEL] {total} 文件 -> {with_target} 含 pedestrian（{ratio:.1f}%），共 {boxes} 框")
        print()

    print("[DONE] 过滤完成")


if __name__ == "__main__":
    main()
