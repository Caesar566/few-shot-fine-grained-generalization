"""Validate CUB split manifests against the official metadata."""

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read_pairs(path: Path):
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, value = line.split(maxsplit=1)
        if key in pairs:
            raise ValueError(f"{path}: 重复 ID {key}")
        pairs[key] = value
    return pairs


def validate(data_root: Path, splits_dir: Path, check_images: bool = False):
    classes = read_pairs(data_root / "classes.txt")
    images = read_pairs(data_root / "images.txt")
    image_labels = read_pairs(data_root / "image_class_labels.txt")
    if set(images) != set(image_labels):
        raise ValueError("官方 images.txt 与 image_class_labels.txt 的 ID 不一致")
    if not classes or not images:
        raise ValueError("官方元数据为空")

    seen_classes, seen_images = set(), set()
    results = {}
    for split in ("train", "val", "test"):
        split_classes = read_pairs(splits_dir / f"{split}_classes.txt")
        class_names = set(split_classes.values())
        if not split_classes or not class_names:
            raise ValueError(f"{split}: 类别清单为空")
        if set(split_classes) & seen_classes:
            raise ValueError(f"{split}: 类别 ID 与其他划分重叠")
        seen_classes.update(split_classes)
        for cid, name in split_classes.items():
            if classes.get(cid) != name:
                raise ValueError(f"{split}: 类别 {cid} 与官方元数据不符")

        counts = Counter()
        manifest = splits_dir / f"{split}_images.txt"
        lines = [line.split(maxsplit=2) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        for parts in lines:
            if len(parts) != 3:
                raise ValueError(f"{manifest}: 图像清单行格式错误")
            image_id, name, rel_path = parts
            if image_id in seen_images:
                raise ValueError(f"图像 ID {image_id} 重复或跨划分重叠")
            seen_images.add(image_id)
            if images.get(image_id) != rel_path:
                raise ValueError(f"图像 {image_id} 路径与官方元数据不符")
            cid = image_labels[image_id]
            if split_classes.get(cid) != name:
                raise ValueError(f"图像 {image_id} 类别与官方元数据不符")
            if check_images and not (data_root / "images" / rel_path).is_file():
                raise FileNotFoundError(data_root / "images" / rel_path)
            counts[name] += 1
        if set(counts) != class_names:
            raise ValueError(f"{split}: 存在无图像类别或图像类别未列入清单")
        results[split] = {"classes": len(split_classes), "images": len(lines), "min_images_per_class": min(counts.values())}

    if seen_classes != set(classes) or seen_images != set(images):
        raise ValueError("划分清单未覆盖全部官方类别或图像")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw/CUB_200_2011")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--check-images", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.data_root, args.splits_dir, args.check_images), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
