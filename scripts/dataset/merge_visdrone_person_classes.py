from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

SOURCE_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

TARGET_NAMES = [
    "person",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]

CLASS_ID_MAP = {
    0: 0,
    1: 0,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge VisDrone classes 0(pedestrian) and 1(people) into person and create a derived 9-class dataset."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("data/visdrone-det"),
        help="Source YOLO-format VisDrone dataset root.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("data/visdrone-det-person9"),
        help="Destination dataset root for the derived 9-class dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to process, e.g. train val test.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images into destination instead of creating hard links.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete destination directory before processing.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_source_subdir(root: Path, name: str) -> Path:
    direct = root / name
    if direct.exists():
        return direct
    raw = root / "raw" / name
    if raw.exists():
        return raw
    raise FileNotFoundError(f"Cannot find required directory: {root / name} or {root / 'raw' / name}")


def hardlink_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def remap_label_line(line: str, split: str, label_path: Path, stats: Counter) -> str:
    stripped = line.strip()
    if not stripped:
        return ""

    parts = stripped.split()
    if len(parts) < 5:
        raise ValueError(f"Invalid YOLO label line in {split}: {label_path}: {line!r}")

    old_class_id = int(parts[0])
    if old_class_id not in CLASS_ID_MAP:
        raise ValueError(f"Unknown class id {old_class_id} in {label_path}")

    new_class_id = CLASS_ID_MAP[old_class_id]
    stats[f"src_{old_class_id}"] += 1
    stats[f"dst_{new_class_id}"] += 1
    parts[0] = str(new_class_id)
    return " ".join(parts)


def write_label_file(src_label: Path, dst_label: Path, split: str, stats: Counter) -> None:
    ensure_dir(dst_label.parent)
    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        stats["missing_label_files"] += 1
        return

    remapped_lines: list[str] = []
    for line in src_label.read_text(encoding="utf-8").splitlines():
        new_line = remap_label_line(line, split, src_label, stats)
        if new_line:
            remapped_lines.append(new_line)

    content = "\n".join(remapped_lines)
    if remapped_lines:
        content += "\n"
    dst_label.write_text(content, encoding="utf-8")
    stats["label_files"] += 1
    stats["objects"] += len(remapped_lines)


def process_split(src_root: Path, dst_root: Path, split: str, copy_images: bool) -> Counter:
    src_images_dir = src_root / "images" / split
    src_labels_dir = src_root / "labels" / split
    if not src_images_dir.exists():
        raise FileNotFoundError(f"Missing images split directory: {src_images_dir}")
    if not src_labels_dir.exists():
        raise FileNotFoundError(f"Missing labels split directory: {src_labels_dir}")

    dst_images_dir = dst_root / "images" / split
    dst_labels_dir = dst_root / "labels" / split
    ensure_dir(dst_images_dir)
    ensure_dir(dst_labels_dir)

    stats: Counter = Counter()
    image_files = sorted(path for path in src_images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    for image_path in image_files:
        dst_image_path = dst_images_dir / image_path.name
        hardlink_or_copy(image_path, dst_image_path, copy_images)
        stats["images"] += 1

        src_label_path = src_labels_dir / f"{image_path.stem}.txt"
        dst_label_path = dst_labels_dir / src_label_path.name
        write_label_file(src_label_path, dst_label_path, split, stats)

    return stats


def write_dataset_yaml(dst_root: Path) -> Path:
    yaml_path = dst_root / "visdrone_person9.yaml"
    lines = [
        f"path: {dst_root.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(TARGET_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {idx}: {name}" for idx, name in enumerate(TARGET_NAMES))
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return yaml_path


def print_summary(all_stats: dict[str, Counter], yaml_path: Path) -> None:
    print("[done] generated derived dataset: person9")
    print(f"[done] dataset yaml: {yaml_path}")
    print("[class-map] 0(pedestrian) + 1(people) -> 0(person)")
    print("[class-map] 2..9 -> 1..8")
    print()
    for split, stats in all_stats.items():
        print(
            f"[{split}] images={stats['images']} label_files={stats['label_files']} "
            f"missing_labels={stats['missing_label_files']} objects={stats['objects']}"
        )
    print()
    print("[target-names]", ", ".join(f"{idx}:{name}" for idx, name in enumerate(TARGET_NAMES)))


def main() -> None:
    args = parse_args()
    src_root = args.src.resolve()
    dst_root = args.dst.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset root not found: {src_root}")

    resolve_source_subdir(src_root, "images")
    resolve_source_subdir(src_root, "labels")

    if args.overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    ensure_dir(dst_root)

    actual_src_root = src_root
    if (src_root / "images").exists() and (src_root / "labels").exists():
        actual_src_root = src_root
    elif (src_root / "raw" / "images").exists() and (src_root / "raw" / "labels").exists():
        actual_src_root = src_root / "raw"

    all_stats: dict[str, Counter] = {}
    for split in args.splits:
        all_stats[split] = process_split(actual_src_root, dst_root, split, args.copy_images)

    yaml_path = write_dataset_yaml(dst_root)
    print_summary(all_stats, yaml_path)


if __name__ == "__main__":
    main()
