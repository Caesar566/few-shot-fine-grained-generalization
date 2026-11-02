import json

import pytest
from PIL import Image

from pattern_recognition_coursework.data.episodes import load_or_create_plan
from pattern_recognition_coursework.data.nabirds import (
    NABirdsDataset, build_manifest, create_review,
)


class TinyDataset:
    data_root = None

    def __init__(self, root):
        self.data_root = root
        self.samples = [(root / f"img{i}.jpg", i // 4) for i in range(12)]
        self.class_to_indices = {c: list(range(c * 4, (c + 1) * 4)) for c in range(3)}

    def __len__(self):
        return len(self.samples)


def test_saved_episode_plan_rejects_tampering(tmp_path):
    dataset = TinyDataset(tmp_path)
    path = tmp_path / "episodes.json"
    first = load_or_create_plan(path, dataset, 2, 1, 1, 3, 42)
    assert first == load_or_create_plan(path, dataset, 2, 1, 1, 3, 42)
    first["batches"][0][2] = first["batches"][0][0]
    path.write_text(json.dumps(first), encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        load_or_create_plan(path, dataset, 2, 1, 1, 3, 42)


def _tiny_nabirds(root):
    (root / "images").mkdir(parents=True)
    (root / "classes.txt").write_text(
        "1 Birds\n2 Other species\n3 Black footed Albatross\n4 Adult\n5 Juvenile\n", encoding="utf-8")
    (root / "hierarchy.txt").write_text("2 1\n3 1\n4 2\n5 3\n", encoding="utf-8")
    (root / "images.txt").write_text("one a.jpg\ntwo b.jpg\n", encoding="utf-8")
    (root / "image_class_labels.txt").write_text("one 4\ntwo 5\n", encoding="utf-8")
    for name in ("a.jpg", "b.jpg"):
        Image.new("RGB", (8, 8)).save(root / "images" / name)


def test_nabirds_requires_explicit_species_review(tmp_path):
    root = tmp_path / "nabirds"
    _tiny_nabirds(root)
    cub = tmp_path / "train_classes.txt"
    cub.write_text("1 001.Black_footed_Albatross\n", encoding="utf-8")
    review = create_review(root, cub)
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="No human-approved"):
        build_manifest(root, cub, path)
    rows = {row["class_id"]: row for row in review["rows"]}
    rows[4].update(decision="non_overlap", species_class_id=2,
                   reviewer="taxonomy reviewer", evidence="checked source taxonomy and synonyms")
    rows[5].update(decision="exclude", reviewer="taxonomy reviewer", evidence="overlaps CUB species")
    path.write_text(json.dumps(review), encoding="utf-8")
    manifest = build_manifest(root, cub, path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    dataset = NABirdsDataset(root, manifest_path, review_file=path, cub_train_classes=cub)
    assert len(dataset) == 1
    assert len(dataset.classes) == 1
    rows[5].update(decision="non_overlap", species_class_id=3)
    path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="审核记录哈希"):
        NABirdsDataset(root, manifest_path, review_file=path, cub_train_classes=cub)
    with pytest.raises(ValueError, match="overlap candidate"):
        build_manifest(root, cub, path)


def test_nabirds_rejects_partial_visual_subclasses(tmp_path):
    root = tmp_path / "nabirds"
    _tiny_nabirds(root)
    with (root / "classes.txt").open("a", encoding="utf-8") as stream:
        stream.write("6 Female\n")
    with (root / "hierarchy.txt").open("a", encoding="utf-8") as stream:
        stream.write("6 2\n")
    with (root / "images.txt").open("a", encoding="utf-8") as stream:
        stream.write("three c.jpg\n")
    with (root / "image_class_labels.txt").open("a", encoding="utf-8") as stream:
        stream.write("three 6\n")
    Image.new("RGB", (8, 8)).save(root / "images" / "c.jpg")
    cub = tmp_path / "train_classes.txt"
    cub.write_text("1 001.Black_footed_Albatross\n", encoding="utf-8")
    review = create_review(root, cub)
    rows = {row["class_id"]: row for row in review["rows"]}
    rows[4].update(decision="non_overlap", species_class_id=2,
                   reviewer="taxonomy reviewer", evidence="checked species")
    rows[5].update(decision="exclude", reviewer="taxonomy reviewer", evidence="overlap")
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed or excluded visual subclass"):
        build_manifest(root, cub, review_path)
