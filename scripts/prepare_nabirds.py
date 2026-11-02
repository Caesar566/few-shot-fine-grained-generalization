"""Create a pending NABirds overlap review, then build an audited manifest."""

import argparse
import json
from pathlib import Path

from pattern_recognition_coursework.data.nabirds import build_manifest, create_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["review", "build"])
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/nabirds")
    parser.add_argument("--cub-train-classes", type=Path, default=PROJECT_ROOT / "data/splits/train_classes.txt")
    parser.add_argument("--review", type=Path, default=PROJECT_ROOT / "data/splits/nabirds_review.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/splits/nabirds_test.json")
    parser.add_argument("--min-images", type=int, default=20, help="At least shot + query for each target species")
    args = parser.parse_args(argv)
    destination = args.review if args.action == "review" else args.output
    if not destination.resolve().is_relative_to(PROJECT_ROOT):
        parser.error("Generated files must stay inside the project directory")
    if destination.exists():
        parser.error(f"Refusing to overwrite an existing review/manifest: {destination}; choose a new output path")
    try:
        result = (create_review(args.data_root, args.cub_train_classes) if args.action == "review" else
                  build_manifest(args.data_root, args.cub_train_classes, args.review, args.min_images))
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.error(str(error))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {destination}")
    if args.action == "review":
        print(f"{len(result['rows'])} categories remain pending explicit human review; no evaluation manifest created.")
    else:
        print(f"{len(result['classes'])} reviewed species, {len(result['samples'])} images")
    return result


if __name__ == "__main__":
    main()
