"""
select_test_images.py
=====================
从 VisDrone 测试集中挑选适合论文可视化对比的图片。

挑选策略：
  1. 包含"小目标密集"场景（bicycle / awning-tricycle / tricycle 实例多）
  2. 包含"行人密集"场景（pedestrian / people 实例多）
  3. 包含"混合多类"场景（同时出现 4 类以上）
  4. 目标尺度差异大（同图出现 car + bicycle）

用法（在服务器上运行）：
    cd /root/autodl-tmp/yolov11-drone-inspection
    python scripts/vis/select_test_images.py \
        --label_dir data/VisDrone2019-DET/labels/test \
        --image_dir data/VisDrone2019-DET/images/test \
        --output_dir scripts/vis/selected_images.txt \
        --n_per_scene 3
"""

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path

# VisDrone 10 类 id
CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

# 四个场景对应的优先类 id
SCENE_CRITERIA = {
    "small_dense":    [2, 6, 7],    # bicycle / tricycle / awning-tricycle
    "pedestrian":     [0, 1],       # pedestrian / people
    "multi_class":    None,         # 同时出现类别数 >= 5
    "scale_diverse":  [2, 3],       # car + bicycle 同图
}


def parse_label(label_path):
    """返回 {class_id: count} 字典"""
    counter = defaultdict(int)
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                counter[int(parts[0])] += 1
    return counter


def score_small_dense(counter):
    return sum(counter.get(c, 0) for c in [2, 6, 7])


def score_pedestrian(counter):
    return sum(counter.get(c, 0) for c in [0, 1])


def score_multi_class(counter):
    return len([v for v in counter.values() if v > 0])


def score_scale_diverse(counter):
    # 要求同时有 car(3) 和 bicycle(2)
    if counter.get(2, 0) > 0 and counter.get(3, 0) > 0:
        return counter.get(2, 0) + counter.get(3, 0)
    return 0


SCORE_FN = {
    "small_dense":   score_small_dense,
    "pedestrian":    score_pedestrian,
    "multi_class":   score_multi_class,
    "scale_diverse": score_scale_diverse,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir",  required=True)
    parser.add_argument("--image_dir",  required=True)
    parser.add_argument("--output_dir", default="scripts/vis/selected_images.txt")
    parser.add_argument("--n_per_scene", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    label_dir = Path(args.label_dir)
    image_dir = Path(args.image_dir)

    label_files = sorted(label_dir.glob("*.txt"))
    print(f"[INFO] 标注文件总数：{len(label_files)}")

    # 解析所有标注
    records = []
    for lf in label_files:
        img_stem = lf.stem
        # 支持 jpg / png
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = image_dir / (img_stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            continue
        counter = parse_label(lf)
        if not counter:
            continue
        records.append((img_path, counter))

    print(f"[INFO] 有效图片数：{len(records)}")

    selected = {}  # scene -> [img_path, ...]

    for scene, score_fn in SCORE_FN.items():
        # 按得分降序排列
        scored = [(score_fn(r[1]), r[0]) for r in records]
        scored.sort(key=lambda x: -x[0])

        # 取得分前 15 名，随机挑 n_per_scene 张（保证多样性）
        top_pool = [x[1] for x in scored[:15] if x[0] > 0]
        chosen = random.sample(top_pool, min(args.n_per_scene, len(top_pool)))
        selected[scene] = chosen

        print(f"\n[场景: {scene}]")
        for img in chosen:
            score = score_fn(parse_label(label_dir / (img.stem + ".txt")))
            counter = parse_label(label_dir / (img.stem + ".txt"))
            class_info = ", ".join(
                f"{CLASS_NAMES[k]}×{v}" for k, v in sorted(counter.items())
            )
            print(f"  {img.name}  得分={score}  [{class_info}]")

    # 汇总去重
    all_selected = []
    seen = set()
    for scene, imgs in selected.items():
        for img in imgs:
            if img.name not in seen:
                seen.add(img.name)
                all_selected.append(img)

    # 写入文件列表
    out_path = Path(args.output_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for img in all_selected:
            f.write(str(img) + "\n")

    print(f"\n[INFO] 共挑选 {len(all_selected)} 张图，列表已写入 {out_path}")
    print("[INFO] 可直接将此列表传给 visualize_compare.py 使用")


if __name__ == "__main__":
    main()
