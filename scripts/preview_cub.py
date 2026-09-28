"""检查 CUB-200-2011 索引，并生成带边界框和部位点的图像预览。

运行方式：uv run python scripts/preview_cub.py
默认读取仓库中的 cub2002011/versions/7/CUB_200_2011，结果写入 output/cub_preview。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "cub2002011/versions/7/CUB_200_2011"
DEFAULT_OUTPUT = PROJECT_ROOT / "output/cub_preview"


def read_id_table(path: Path, expected_fields: int) -> dict[int, list[str]]:
    """读取以整数 ID 开头的标注表，并检查列数与重复 ID。"""
    rows: dict[int, list[str]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            # 名称字段可能含空格，例如部位名 "left eye"。
            fields = line.strip().split(maxsplit=1) if expected_fields == 2 else line.split()
            if len(fields) != expected_fields:
                raise ValueError(f"{path}:{line_number} 应有 {expected_fields} 列，实际 {len(fields)} 列")
            image_id = int(fields[0])
            if image_id in rows:
                raise ValueError(f"{path}:{line_number} 出现重复 ID {image_id}")
            rows[image_id] = fields[1:]
    return rows


def read_parts(path: Path) -> dict[int, list[tuple[int, float, float, bool]]]:
    """按图像 ID 汇总 15 个身体部位的位置和可见性。"""
    parts: dict[int, list[tuple[int, float, float, bool]]] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"{path}:{line_number} 应有 5 列")
            image_id, part_id = map(int, fields[:2])
            x, y = map(float, fields[2:4])
            if fields[4] not in {"0", "1"}:
                raise ValueError(f"{path}:{line_number} 可见性不是 0 或 1")
            parts.setdefault(image_id, []).append((part_id, x, y, fields[4] == "1"))
    return parts


def draw_preview(
    source: Path,
    destination: Path,
    box: list[str],
    parts: list[tuple[int, float, float, bool]],
) -> dict[str, object]:
    """在原图坐标系上绘制红色边界框和黄色可见部位点。"""
    x, y, width, height = map(float, box)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image_width, image_height = image.size
        box_in_bounds = (
            0 <= x < image_width
            and 0 <= y < image_height
            and width > 0
            and height > 0
            and x + width <= image_width + 1
            and y + height <= image_height + 1
        )
        painter = ImageDraw.Draw(image)
        stroke = max(3, round(min(image.size) / 100))
        painter.rectangle((x, y, x + width, y + height), outline="red", width=stroke)

        visible_count = 0
        out_of_bounds_parts = 0
        radius = max(3, round(min(image.size) / 120))
        for part_id, part_x, part_y, visible in parts:
            if not visible:
                continue
            visible_count += 1
            if not (0 <= part_x < image_width and 0 <= part_y < image_height):
                out_of_bounds_parts += 1
                continue
            painter.ellipse(
                (part_x - radius, part_y - radius, part_x + radius, part_y + radius),
                fill="yellow",
                outline="black",
                width=1,
            )
            painter.text((part_x + radius + 2, part_y - radius), str(part_id), fill="yellow", font=ImageFont.load_default())

        image.save(destination)
    return {
        "image_size": [image_width, image_height],
        "bounding_box": [x, y, width, height],
        "bounding_box_in_bounds": box_in_bounds,
        "visible_parts": visible_count,
        "visible_parts_out_of_bounds": out_of_bounds_parts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=8, help="随机预览图像数量，默认 8")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，默认 42")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples 必须大于 0")

    dataset = args.dataset_root.resolve()
    output = args.output_dir.resolve()
    images = read_id_table(dataset / "images.txt", 2)
    classes = read_id_table(dataset / "classes.txt", 2)
    labels = read_id_table(dataset / "image_class_labels.txt", 2)
    split = read_id_table(dataset / "train_test_split.txt", 2)
    boxes = read_id_table(dataset / "bounding_boxes.txt", 5)
    part_names = read_id_table(dataset / "parts/parts.txt", 2)
    parts = read_parts(dataset / "parts/part_locs.txt")

    # 用 images.txt 作为样本全集；逐项验证其他标注表能否按 image_id 正确连接。
    image_ids = set(images)
    issues: list[str] = []
    for name, table in [("class labels", labels), ("train/test", split), ("bounding boxes", boxes), ("parts", parts)]:
        missing = image_ids - set(table)
        extra = set(table) - image_ids
        if missing or extra:
            issues.append(f"{name}: 缺少 {len(missing)} 个图像 ID，多出 {len(extra)} 个图像 ID")

    missing_files: list[str] = []
    inconsistent_labels: list[int] = []
    invalid_splits: list[int] = []
    invalid_part_counts: list[int] = []
    counts: Counter[int] = Counter()
    split_counts: Counter[str] = Counter()
    
    for image_id, (relative_path,) in images.items():
        if not (dataset / "images" / relative_path).is_file():
            missing_files.append(relative_path)
        if image_id in labels:
            class_id = int(labels[image_id][0])
            counts[class_id] += 1
            if class_id not in classes or relative_path.split("/", 1)[0] != classes.get(class_id, [None])[0]:
                inconsistent_labels.append(image_id)
        if image_id in split:
            value = split[image_id][0]
            split_counts[value] += 1
            if value not in {"0", "1"}:
                invalid_splits.append(image_id)
        if image_id in parts and {item[0] for item in parts[image_id]} != set(part_names):
            invalid_part_counts.append(image_id)

    indexed_paths = {record[0] for record in images.values()}
    disk_paths = {str(path.relative_to(dataset / "images")) for path in (dataset / "images").rglob("*.jpg")}
    extra_files = sorted(disk_paths - indexed_paths)
    if missing_files:
        issues.append(f"索引中的图像文件缺失：{len(missing_files)} 张")
    if inconsistent_labels:
        issues.append(f"类别标签与图像目录不一致：{len(inconsistent_labels)} 张")
    if invalid_splits:
        issues.append(f"train/test 值无效：{len(invalid_splits)} 张")
    if invalid_part_counts:
        issues.append(f"身体部位 ID 不完整：{len(invalid_part_counts)} 张")

    # 只有各类标注都存在且文件可读的样本，才能被选入可视化预览。
    candidates = sorted(
        (image_ids & set(labels) & set(split) & set(boxes) & set(parts))
        - set(inconsistent_labels)
        - set(invalid_splits)
        - set(invalid_part_counts)
    )
    candidates = [image_id for image_id in candidates if (dataset / "images" / images[image_id][0]).is_file()]
    if not candidates:
        raise ValueError("没有同时具备图像、类别、边界框和部位标注的样本")

    output.mkdir(parents=True, exist_ok=True)
    preview_dir = output / "previews"
    preview_dir.mkdir(exist_ok=True)
    with (output / "class_counts.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["class_id", "class_name", "image_count"])
        for class_id, (class_name,) in sorted(classes.items()):
            writer.writerow([class_id, class_name, counts[class_id]])
    class_sizes = [counts[class_id] for class_id in classes]

    random_ids = random.Random(args.seed).sample(candidates, min(args.samples, len(candidates)))
    previews: list[dict[str, object]] = []
    for number, image_id in enumerate(random_ids, 1):
        relative_path = images[image_id][0]
        class_id = int(labels[image_id][0])
        preview_name = f"sample_{number:02d}_image_{image_id}.png"
        detail = draw_preview(dataset / "images" / relative_path, preview_dir / preview_name, boxes[image_id], parts[image_id])
        previews.append({
            "image_id": image_id,
            "class_id": class_id,
            "class_name": classes[class_id][0],
            "image_path": relative_path,
            "preview_path": f"previews/{preview_name}",
            "train_or_test": "train" if split[image_id][0] == "1" else "test",
            **detail,
        })

    summary = {
        "dataset_root": str(dataset),
        "class_count": len(classes),
        "indexed_image_count": len(images),
        "disk_jpg_count": len(disk_paths),
        "train_count": split_counts["1"],
        "test_count": split_counts["0"],
        "class_count_min": min(class_sizes),
        "class_count_max": max(class_sizes),
        "class_count_mean": round(len(images) / len(classes), 2),
        "extra_jpg_not_in_index": extra_files,
        "issues": issues,
        "samples": previews,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # 报告保留完整的输入路径说明；逐类精确数量放在 CSV，便于后续筛选和绘图。
    lines = [
        "# CUB-200-2011 数据预览",
        "",
        f"- 类别数：**{len(classes)}**；索引样本数：**{len(images)}**。",
        f"- 每类图像数：平均 **{len(images) / len(classes):.2f}** 张，最少 **{min(class_sizes)}** 张，最多 **{max(class_sizes)}** 张。完整逐类统计见 [class_counts.csv](class_counts.csv)。",
        f"- 官方按图像划分：训练 **{split_counts['1']}** 张，测试 **{split_counts['0']}** 张。",
        f"- 磁盘 JPG：**{len(disk_paths)}** 个；其中 **{len(extra_files)}** 个不在 `images.txt` 中。",
        f"- 图像、标签及标注对应关系：**{'通过' if not issues else '发现问题'}**。",
        "",
        "## 数据位置和连接方式",
        "",
        f"数据根目录：`{dataset}`",
        "",
        "| 内容 | 相对数据根目录路径 | 字段与用途 |",
        "| --- | --- | --- |",
        "| 图像 | `images/<类别目录>/<图片名>.jpg` | 按类别分目录存放 |",
        "| 图像索引 | `images.txt` | `image_id 相对图像路径` |",
        "| 类别名称 | `classes.txt` | `class_id 类别名称` |",
        "| 图像标签 | `image_class_labels.txt` | `image_id class_id` |",
        "| 训练/测试 | `train_test_split.txt` | `image_id is_train`；1 为训练，0 为测试 |",
        "| 边界框 | `bounding_boxes.txt` | `image_id x y width height`；单位为像素 |",
        "| 身体部位 | `parts/part_locs.txt`、`parts/parts.txt` | `image_id part_id x y visible`，以及部位名称 |",
        "",
        "以上文件用 `image_id` 对应；`image_class_labels.txt` 中的 `class_id` 再连接 `classes.txt`。本报告检查了索引图像是否存在、标签是否与类别目录一致、各标注表是否覆盖图像 ID，以及每张图是否有完整的 15 个部位 ID。",
        "",
        "## 校验详情",
        "",
    ]
    lines.extend(f"- {issue}" for issue in issues) if issues else lines.append("- 全部索引图像与类别标签、train/test、边界框和部位标注对应成功。")
    if extra_files:
        lines += ["", "未列入 `images.txt` 的 JPG：", ""] + [f"- `{name}`" for name in extra_files]
    lines += [
        "",
        "## 随机样本标注预览",
        "",
        f"随机种子：`{args.seed}`。红框表示鸟的边界框；黄点及数字表示可见的身体部位及其 ID。预览基于原图像素坐标绘制，没有缩放坐标。",
        "",
    ]
    for sample in previews:
        lines += [
            f"### image_id {sample['image_id']}：{sample['class_name']}",
            "",
            f"原图：`{sample['image_path']}`；划分：`{sample['train_or_test']}`；尺寸：`{sample['image_size']}`；框 `(x, y, width, height)`：`{sample['bounding_box']}`；可见部位：{sample['visible_parts']} 个。",
            f"边界框位于图像内：**{'是' if sample['bounding_box_in_bounds'] else '否'}**；可见部位越界数：**{sample['visible_parts_out_of_bounds']}**。",
            "",
            f"![image_id {sample['image_id']} 标注预览]({sample['preview_path']})",
            "",
        ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"数据预览已生成：{output / 'report.md'}")
    print(f"类别 {len(classes)}，索引图像 {len(images)}，对应关系问题 {len(issues)}，预览图 {len(previews)} 张")


if __name__ == "__main__":
    main()
