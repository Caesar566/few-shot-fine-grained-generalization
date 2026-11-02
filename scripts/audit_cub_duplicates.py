"""Audit exact CUB image duplicates and their use in saved test episodes."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def manifest_rows(path: Path, split: str) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            raise ValueError(f"Malformed manifest row in {path}: {line}")
        image_id, class_name, relative_path = parts
        rows.append({"split": split, "id": image_id, "class": class_name, "path": relative_path})
    return rows


def audit(data_root: Path, split_root: Path, plan_root: Path) -> dict:
    image_root = (data_root / "images").resolve()
    split_rows = {
        split: manifest_rows(split_root / f"{split}_images.txt", split)
        for split in ("train", "val", "test")
    }
    hashes = defaultdict(list)
    for rows in split_rows.values():
        for row in rows:
            image_path = (image_root / row["path"]).resolve()
            if not image_path.is_relative_to(image_root):
                raise ValueError(f"Image path escapes data root: {row['path']}")
            with image_path.open("rb") as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
            hashes[digest].append(row)

    duplicate_groups = [
        {"sha256": digest, "items": items}
        for digest, items in sorted(hashes.items())
        if len(items) > 1
    ]
    cross_split = [
        group for group in duplicate_groups
        if len({row["split"] for row in group["items"]}) > 1
    ]

    test_index = {row["id"]: index for index, row in enumerate(split_rows["test"])}
    test_groups = [
        group for group in duplicate_groups
        if all(item["split"] == "test" for item in group["items"])
    ]
    plans = []
    for path in sorted(plan_root.glob("cub_test_*.json")):
        plan = json.loads(path.read_text(encoding="utf-8"))
        support_count = plan["n_way"] * plan["k_shot"]
        events = []
        for episode_index, batch in enumerate(plan["batches"]):
            for group in test_groups:
                indices = [test_index[item["id"]] for item in group["items"]]
                present = [index for index in indices if index in batch]
                if len(present) < 2:
                    continue
                positions = [batch.index(index) for index in present]
                events.append({
                    "episode_index_zero_based": episode_index,
                    "sha256": group["sha256"],
                    "positions": positions,
                    "support_query_overlap": any(p < support_count for p in positions)
                    and any(p >= support_count for p in positions),
                })
        plans.append({
            "plan": path.relative_to(PROJECT_ROOT).as_posix(),
            "shot": plan["k_shot"],
            "seed": plan["seed"],
            "duplicate_cooccurrence": events,
        })
    return {
        "scope": "exact byte-for-byte SHA256 only; perceptual or source-image overlap not audited",
        "split_counts": {split: len(rows) for split, rows in split_rows.items()},
        "unique_sha256_count": len(hashes),
        "duplicate_sha256_groups": duplicate_groups,
        "cross_split_duplicate_sha256_groups": cross_split,
        "saved_test_plan_duplicate_cooccurrence": plans,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/CUB_200_2011")
    parser.add_argument("--split-root", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--plan-root", type=Path, default=PROJECT_ROOT / "outputs/cub_study_v1/episodes")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "reports/records/cub_duplicate_audit_v1.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Audit output must stay inside the project directory")
    result = audit(args.data_root, args.split_root, args.plan_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "split_counts": result["split_counts"],
        "duplicate_groups": len(result["duplicate_sha256_groups"]),
        "cross_split_groups": len(result["cross_split_duplicate_sha256_groups"]),
        "saved_plans": len(result["saved_test_plan_duplicate_cooccurrence"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
