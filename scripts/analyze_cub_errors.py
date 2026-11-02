"""Audit existing CUB test predictions by official bird bounding-box size.

The script only reads saved predictions. It never runs or selects a model.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from pattern_recognition_coursework.engine.experiment import PROJECT_ROOT, project_output, sha256_file


def official_image_metadata(data_root: Path, test_manifest: Path):
    images = {}
    for line in (data_root / "images.txt").read_text(encoding="utf-8").splitlines():
        image_id, relative = line.split(maxsplit=1)
        images[relative.replace("\\", "/")] = image_id
    boxes = {}
    for line in (data_root / "bounding_boxes.txt").read_text(encoding="utf-8").splitlines():
        image_id, x, y, width, height = line.split()
        boxes[image_id] = (float(x), float(y), float(width), float(height))
    test_paths = {line.split(maxsplit=2)[2] for line in test_manifest.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not test_paths or not test_paths <= images.keys():
        raise ValueError("CUB 测试清单与官方图像表不一致")
    metadata = {}
    for relative in sorted(test_paths):
        path = data_root / "images" / relative
        with Image.open(path) as image:
            image_width, image_height = image.size
        _, _, box_width, box_height = boxes[images[relative]]
        fraction = box_width * box_height / (image_width * image_height)
        if not 0 < fraction <= 1.05:
            raise ValueError(f"边界框与图像尺寸异常: {relative}: {fraction}")
        metadata[relative] = {"fraction": min(fraction, 1.0), "path": path}
    return metadata


def quartile_boundaries(values):
    values = sorted(values)
    return [values[(len(values) * q) // 4] for q in (1, 2, 3)]


def quartile(value, boundaries):
    return 1 + sum(value >= boundary for boundary in boundaries)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/CUB_200_2011")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--evaluations-dir", type=Path, default=PROJECT_ROOT / "outputs/cub_study_v1/evaluations")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "reports/records/cub_error_audit_v1.json")
    parser.add_argument("--review-csv", type=Path, default=PROJECT_ROOT / "reports/records/cub_error_review_v1.csv")
    args = parser.parse_args(argv)
    args.output_json = project_output(args.output_json)
    args.review_csv = project_output(args.review_csv)
    if args.output_json.exists() or args.review_csv.exists():
        parser.error("结果已存在；拒绝覆盖已有审核记录")
    manifest = args.splits_dir / "test_images.txt"
    metadata = official_image_metadata(args.data_root, manifest)
    boundaries = quartile_boundaries(row["fraction"] for row in metadata.values())
    image_root = (args.data_root / "images").resolve()
    results = {}
    review_rows = []
    expected = {(method, shot, seed) for method in ("protonet", "global_local") for shot in (1, 5) for seed in (42, 43, 44)}
    seen = set()
    for path in sorted(args.evaluations_dir.rglob("*.json")):
        if path.name.endswith("_queries.jsonl"):
            continue
        evaluation = json.loads(path.read_text(encoding="utf-8"))
        if evaluation.get("evaluation_protocol", {}).get("target_dataset") != "cub":
            continue
        method = evaluation["model_config"]["method"]
        shot = evaluation["evaluation_protocol"]["k_shot"]
        seed = evaluation["seed"]
        key = (method, shot, seed)
        if key not in expected or key in seen:
            raise ValueError(f"意外或重复的第一轮评估: {path}")
        seen.add(key)
        predictions = Path(evaluation["predictions"])
        if not predictions.is_file() or predictions.parent != path.parent:
            raise ValueError(f"预测记录路径异常: {predictions}")
        counts = defaultdict(lambda: {"correct": 0, "total": 0, "images": set()})
        total = correct = 0
        for line in predictions.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            query = Path(row["query_image"]).resolve()
            if not query.is_relative_to(image_root):
                raise ValueError(f"查询图像不在 CUB 图像目录: {query}")
            relative = query.relative_to(image_root).as_posix()
            if relative not in metadata or row["true_class"] != relative.split("/")[0]:
                raise ValueError(f"查询图像或标签不属于测试清单: {relative}")
            bucket = quartile(metadata[relative]["fraction"], boundaries)
            is_correct = row["true_class"] == row["predicted_class"]
            counts[bucket]["correct"] += is_correct
            counts[bucket]["total"] += 1
            counts[bucket]["images"].add(relative)
            total += 1
            correct += is_correct
            if not is_correct:
                review_rows.append({
                    "method": method, "shot": shot, "seed": seed,
                    "episode_id": row["episode_id"], "query_image": relative,
                    "true_class": row["true_class"], "predicted_class": row["predicted_class"],
                    "confidence": float(row["confidence"]),
                    "bbox_area_fraction": metadata[relative]["fraction"], "size_quartile": bucket,
                    "pose": "", "lighting": "", "background": "", "occlusion": "",
                    "label_issue": "", "review_notes": "",
                })
        if abs(correct / total - evaluation["cub_novel_test"]["accuracy"]) > 1e-6:
            raise ValueError(f"预测记录与保存的准确率不符: {path}")
        results[f"{method}_{shot}shot_seed{seed}"] = {
            "evaluation_sha256": sha256_file(path), "predictions_sha256": sha256_file(predictions),
            "accuracy": correct / total, "query_decisions": total,
            "by_size_quartile": {str(bucket): {
                "accuracy": counts[bucket]["correct"] / counts[bucket]["total"],
                "correct": counts[bucket]["correct"], "query_decisions": counts[bucket]["total"],
                "unique_query_images": len(counts[bucket]["images"]),
            } for bucket in range(1, 5)},
        }
    if seen != expected:
        raise ValueError(f"第一轮评估不完整: {sorted(expected - seen)}")
    output = {
        "schema_version": 1, "source": "saved CUB study v1 test predictions; no new inference",
        "test_manifest_sha256": sha256_file(manifest), "bbox_source_sha256": sha256_file(args.data_root / "bounding_boxes.txt"),
        "size_metric": "official bbox width*height / original image width*height; capped at 1",
        "quartile_boundaries": boundaries,
        "interpretation": "Descriptive association only; repeated queries and episodes are not independent. Pose, lighting, background, occlusion, and label issues require manual review.",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    review_rows.sort(key=lambda row: (row["method"], row["shot"], row["seed"], -row["confidence"], row["episode_id"]))
    with args.review_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    print(json.dumps({"summary": str(args.output_json), "review": str(args.review_csv), "error_decisions": len(review_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
