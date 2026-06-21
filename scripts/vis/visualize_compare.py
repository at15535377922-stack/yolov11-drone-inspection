"""
visualize_compare.py
====================
使用 EXP-01（基线）与 EXP-06（TADetect）的 best.pt 对同一批测试图片
进行推理，并将检测结果保存为独立图像，供后续拼图论文使用。

用法（在服务器上运行）：
    cd /root/autodl-tmp/yolov11-drone-inspection
    python scripts/vis/visualize_compare.py \
        --image_list scripts/vis/selected_images.txt \
        --weight_baseline runs/detect/visdrone10_baseline/yolo11n_640/weights/best.pt \
        --weight_exp06   runs/detect/visdrone10_exp06_tahead/weights/best.pt \
        --output_dir     results/vis_compare \
        --conf 0.25 \
        --iou  0.45 \
        --imgsz 640

    # 若没有 selected_images.txt，也可直接传图片目录（随机采样）：
    python scripts/vis/visualize_compare.py \
        --image_dir data/VisDrone2019-DET/images/test \
        --n_sample  12 \
        --weight_baseline runs/detect/visdrone10_yolo11n_baseline/weights/best.pt \
        --weight_exp06    runs/detect/visdrone10_exp06_tahead/weights/best.pt \
        --output_dir      results/vis_compare \
        --conf 0.25 --iou 0.45 --imgsz 640

输出结构：
    results/vis_compare/
    ├── baseline/          # EXP-01 检测结果图
    │   ├── img001.jpg
    │   └── ...
    └── exp06/             # EXP-06 检测结果图
        ├── img001.jpg
        └── ...
"""

import argparse
import random
from pathlib import Path

from ultralytics import YOLO


CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# 每个类别固定颜色（BGR，供参考，Ultralytics 内部会自动配色）
# 这里不强制覆盖，使用 Ultralytics 默认渲染


def load_image_list(args) -> list[Path]:
    if args.image_list:
        with open(args.image_list) as f:
            paths = [Path(line.strip()) for line in f if line.strip()]
        print(f"[INFO] 从列表载入 {len(paths)} 张图片")
        return paths

    if args.image_dir:
        all_imgs = sorted(
            list(Path(args.image_dir).glob("*.jpg")) +
            list(Path(args.image_dir).glob("*.png"))
        )
        random.seed(args.seed)
        chosen = random.sample(all_imgs, min(args.n_sample, len(all_imgs)))
        print(f"[INFO] 从目录随机采样 {len(chosen)} 张图片")
        return chosen

    raise ValueError("必须指定 --image_list 或 --image_dir")


def run_inference(model: YOLO, img_paths: list[Path],
                  out_dir: Path, conf: float, iou: float, imgsz: int):
    """对图片列表逐张推理并保存可视化结果"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for img_path in img_paths:
        results = model.predict(
            source=str(img_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            save=False,          # 不用 Ultralytics 自动保存，手动控制路径
            verbose=False,
        )
        for r in results:
            # 绘制检测框并保存
            annotated = r.plot(
                line_width=2,
                font_size=10,
                labels=True,
                conf=True,
            )
            import cv2
            save_path = out_dir / img_path.name
            cv2.imwrite(str(save_path), annotated)
            print(f"  [SAVE] {save_path}")


def main():
    parser = argparse.ArgumentParser()
    # 图片来源（二选一）
    parser.add_argument("--image_list",  default=None, help="由 select_test_images.py 生成的图片路径列表")
    parser.add_argument("--image_dir",   default=None, help="测试集图片目录，配合 --n_sample 使用")
    parser.add_argument("--n_sample",    type=int, default=12)
    parser.add_argument("--seed",        type=int, default=42)
    # 权重
    parser.add_argument("--weight_baseline", required=True, help="EXP-01 基线 best.pt 路径")
    parser.add_argument("--weight_exp06",    required=True, help="EXP-06 TADetect best.pt 路径")
    # 推理参数（baseline 与 exp06 可分别设置置信度阈值，突出对比效果）
    parser.add_argument("--conf_baseline", type=float, default=0.35,
                        help="EXP-01 基线置信度阈值，适当调高可让基线漏检更多小目标（默认 0.35）")
    parser.add_argument("--conf_exp06",    type=float, default=0.20,
                        help="EXP-06 TADetect 置信度阈值，保持较低以充分显示改进效果（默认 0.20）")
    parser.add_argument("--iou",   type=float, default=0.45)
    parser.add_argument("--imgsz", type=int,   default=640)
    # 输出
    parser.add_argument("--output_dir", default="results/vis_compare")
    args = parser.parse_args()

    img_paths = load_image_list(args)
    out_root = Path(args.output_dir)

    print(f"\n{'='*50}")
    print(f"[STEP 1/2] 使用 EXP-01 基线模型推理 ...（conf={args.conf_baseline}）")
    model_baseline = YOLO(args.weight_baseline)
    run_inference(model_baseline, img_paths,
                  out_root / "baseline", args.conf_baseline, args.iou, args.imgsz)

    print(f"\n{'='*50}")
    print(f"[STEP 2/2] 使用 EXP-06 TADetect 模型推理 ...（conf={args.conf_exp06}）")
    model_exp06 = YOLO(args.weight_exp06)
    run_inference(model_exp06, img_paths,
                  out_root / "exp06", args.conf_exp06, args.iou, args.imgsz)

    print(f"\n[DONE] 结果保存在：{out_root.resolve()}")
    print(f"  baseline/ → EXP-01 结果图")
    print(f"  exp06/    → EXP-06 结果图")
    print(f"\n下一步：运行 make_comparison_grid.py 生成论文级拼图。")


if __name__ == "__main__":
    main()
