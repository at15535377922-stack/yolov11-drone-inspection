"""
make_comparison_grid.py
=======================
将 visualize_compare.py 生成的 baseline/ 与 exp06/ 目录中的同名图片
拼合为论文级双栏对比图（左=基线，右=TADetect），并附带标题标注。

用法（服务器或本地均可运行）：
    cd /root/autodl-tmp/yolov11-drone-inspection
    python scripts/vis/make_comparison_grid.py \
        --baseline_dir results/vis_compare/baseline \
        --exp06_dir    results/vis_compare/exp06 \
        --output_dir   results/vis_compare/grid \
        --cols 2 \
        --dpi  300

    # 同时生成单张大拼图（所有图拼在一起）：
    python scripts/vis/make_comparison_grid.py \
        --baseline_dir results/vis_compare/baseline \
        --exp06_dir    results/vis_compare/exp06 \
        --output_dir   results/vis_compare/grid \
        --merge_all    \
        --cols 2 \
        --dpi  300

输出结构：
    results/vis_compare/grid/
    ├── cmp_img001.png      # 每张图的左右对比（横向拼）
    ├── cmp_img002.png
    ├── ...
    └── merged_all.png      # 所有图合并为一张大图（--merge_all 时生成）
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


# ─── 配置区（可按需调整）────────────────────────────────────────────────────
LABEL_FONT       = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE      = 1.2          # 标题字号（相对 px）
LABEL_THICKNESS  = 2
LABEL_COLOR      = (255, 255, 255)   # 白色文字
LABEL_BG_COLOR   = (30, 30, 30)     # 深色背景条
LABEL_BAR_H      = 50               # 标题栏高度（px）
SEP_W            = 6                # 左右图之间分隔线宽度（px）
SEP_COLOR        = (200, 200, 200)  # 分隔线颜色
# ────────────────────────────────────────────────────────────────────────────


def add_title_bar(img: np.ndarray, text: str, bar_h: int = LABEL_BAR_H) -> np.ndarray:
    """在图像顶部添加深色标题栏"""
    h, w = img.shape[:2]
    bar = np.full((bar_h, w, 3), LABEL_BG_COLOR, dtype=np.uint8)
    text_size, _ = cv2.getTextSize(text, LABEL_FONT, LABEL_SCALE, LABEL_THICKNESS)
    tx = max(0, (w - text_size[0]) // 2)
    ty = (bar_h + text_size[1]) // 2
    cv2.putText(bar, text, (tx, ty), LABEL_FONT,
                LABEL_SCALE, LABEL_COLOR, LABEL_THICKNESS, cv2.LINE_AA)
    return np.vstack([bar, img])


def make_side_by_side(img_left: np.ndarray, img_right: np.ndarray,
                      label_left: str = "EXP-01  Baseline (YOLOv11n)",
                      label_right: str = "EXP-06  TADetect (P2/P3 cls/reg 交互)") -> np.ndarray:
    """生成左右对比图（高度对齐，中间加分隔线）"""
    # 统一高度
    h = max(img_left.shape[0], img_right.shape[0])
    w_l, w_r = img_left.shape[1], img_right.shape[1]

    def pad_height(img, target_h):
        pad = target_h - img.shape[0]
        if pad > 0:
            bottom = np.full((pad, img.shape[1], 3), 240, dtype=np.uint8)
            return np.vstack([img, bottom])
        return img

    img_left  = pad_height(img_left, h)
    img_right = pad_height(img_right, h)

    # 添加标题栏
    img_left  = add_title_bar(img_left,  label_left)
    img_right = add_title_bar(img_right, label_right)

    # 分隔线
    sep = np.full((img_left.shape[0], SEP_W, 3), SEP_COLOR, dtype=np.uint8)

    return np.hstack([img_left, sep, img_right])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--exp06_dir",    required=True)
    parser.add_argument("--output_dir",   default="results/vis_compare/grid")
    parser.add_argument("--merge_all",    action="store_true",
                        help="额外生成一张包含所有对比图的大拼图")
    parser.add_argument("--cols",  type=int, default=2,
                        help="大拼图每行的对比对数（--merge_all 生效）")
    parser.add_argument("--dpi",   type=int, default=300,
                        help="仅影响最终图片的保存质量提示，cv2 保存不强制 DPI，"
                             "若需嵌入 DPI 元数据请改用 PIL 模式")
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    exp06_dir    = Path(args.exp06_dir)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 取两目录中同名文件
    baseline_imgs = {p.name: p for p in sorted(baseline_dir.glob("*.*"))
                     if p.suffix.lower() in {".jpg", ".png", ".jpeg"}}
    exp06_imgs    = {p.name: p for p in sorted(exp06_dir.glob("*.*"))
                     if p.suffix.lower() in {".jpg", ".png", ".jpeg"}}
    common = sorted(set(baseline_imgs) & set(exp06_imgs))

    if not common:
        print("[ERROR] baseline 与 exp06 目录中没有同名图片，请先运行 visualize_compare.py")
        return

    print(f"[INFO] 找到 {len(common)} 对同名图片，开始生成对比图 ...")

    all_grids = []
    for name in common:
        img_b = cv2.imread(str(baseline_imgs[name]))
        img_e = cv2.imread(str(exp06_imgs[name]))
        if img_b is None or img_e is None:
            print(f"  [SKIP] 读取失败：{name}")
            continue

        grid = make_side_by_side(img_b, img_e)
        save_path = out_dir / f"cmp_{Path(name).stem}.png"
        cv2.imwrite(str(save_path), grid,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"  [SAVE] {save_path}")
        all_grids.append(grid)

    # ── 大拼图（可选）──────────────────────────────────────────────────────
    if args.merge_all and all_grids:
        # 统一宽度（取最大宽度，不足补白边）
        max_w = max(g.shape[1] for g in all_grids)
        def pad_w(img, target_w):
            pad = target_w - img.shape[1]
            if pad > 0:
                right = np.full((img.shape[0], pad, 3), 240, dtype=np.uint8)
                return np.hstack([img, right])
            return img
        all_grids = [pad_w(g, max_w) for g in all_grids]

        # 按 cols 排列行
        cols = args.cols
        rows_imgs = []
        for i in range(0, len(all_grids), cols):
            row_imgs = all_grids[i:i+cols]
            # 如果最后一行不足 cols，补空白
            while len(row_imgs) < cols:
                blank = np.full_like(row_imgs[0], 240)
                row_imgs.append(blank)
            # 统一行内高度
            max_h = max(g.shape[0] for g in row_imgs)
            def pad_h_fn(img, target_h):
                pad = target_h - img.shape[0]
                if pad > 0:
                    bottom = np.full((pad, img.shape[1], 3), 240, dtype=np.uint8)
                    return np.vstack([img, bottom])
                return img
            row_imgs = [pad_h_fn(g, max_h) for g in row_imgs]
            rows_imgs.append(np.hstack(row_imgs))

        # 统一列宽后纵向拼接
        merged = np.vstack(rows_imgs)
        merge_path = out_dir / "merged_all.png"
        cv2.imwrite(str(merge_path), merged, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        print(f"\n[INFO] 大拼图已保存：{merge_path}")
        print(f"       尺寸：{merged.shape[1]}×{merged.shape[0]} px")

    print(f"\n[DONE] 所有对比图保存在：{out_dir.resolve()}")
    print("论文使用建议：")
    print("  - 单图对比：使用 cmp_*.png，截取关键区域放大展示")
    print("  - 大拼图：  使用 merged_all.png，适合附录或补充材料")
    print("  - 推荐在 Word/LaTeX 中以 ≥ 300 DPI 嵌入 PNG")


if __name__ == "__main__":
    main()
