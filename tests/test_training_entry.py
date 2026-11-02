import json
from pathlib import Path

from PIL import Image

from pattern_recognition_coursework.engine.evaluate import main as evaluate
from pattern_recognition_coursework.engine.experiment import main as train


def _tiny_cub(root: Path):
    data = root / "data"
    splits = root / "splits"
    (data / "images").mkdir(parents=True)
    splits.mkdir()
    for split, start in (("train", 1), ("val", 3), ("test", 5)):
        class_rows, image_rows = [], []
        for class_id in (start, start + 1):
            name = f"{class_id:03d}.Bird_{class_id}"
            (data / "images" / name).mkdir()
            class_rows.append(f"{class_id} {name}\n")
            for number in range(3):
                relative = f"{name}/{number}.jpg"
                Image.new("RGB", (64, 64), (class_id * 30, number * 50, 80)).save(data / "images" / relative)
                image_rows.append(f"{class_id * 10 + number} {name} {relative}\n")
        (splits / f"{split}_classes.txt").write_text("".join(class_rows), encoding="utf-8")
        (splits / f"{split}_images.txt").write_text("".join(image_rows), encoding="utf-8")
    return data, splits


def test_meta_baseline_checkpoint_can_be_evaluated_independently(tmp_path):
    data, splits = _tiny_cub(tmp_path)
    output = tmp_path / "outputs"
    result = train([
        "--data-root", str(data), "--splits-dir", str(splits), "--output-dir", str(output),
        "--stage", "meta_baseline", "--method", "protonet", "--metric", "cosine",
        "--supervised-epochs", "1", "--epochs", "1", "--batch-size", "3",
        "--train-episodes", "1", "--eval-episodes", "1", "--n-way", "2",
        "--k-shot", "1", "--n-query", "1", "--image-size", "64", "--device", "cpu",
    ])
    repeated = evaluate([
        "--checkpoint", result["checkpoint"], "--dataset", "cub", "--data-root", str(data),
        "--splits-dir", str(splits), "--output-dir", str(output), "--device", "cpu",
    ])
    assert repeated["evaluation_protocol"]["episode_plan_hash"] == result["evaluation_protocol"]["episode_plan_hash"]
    assert repeated["cub_novel_test"]["accuracy"] == result["cub_novel_test"]["accuracy"]
    assert json.loads(Path(result["checkpoint"]).with_name("metrics.json").read_text())["selected_phase"] == "episodic"


def test_validation_only_training_does_not_read_test_images(tmp_path):
    data, splits = _tiny_cub(tmp_path)
    for image in (data / "images" / "005.Bird_5").glob("*.jpg"):
        image.unlink()
    result = train([
        "--data-root", str(data), "--splits-dir", str(splits),
        "--output-dir", str(tmp_path / "outputs"), "--skip-test",
        "--epochs", "1", "--train-episodes", "1", "--eval-episodes", "1",
        "--n-way", "2", "--k-shot", "1", "--n-query", "1",
        "--image-size", "64", "--device", "cpu",
    ])
    assert result["status"] == "validation_only"
    assert "cub_novel_test" not in result
    assert not list((tmp_path / "outputs" / "episodes").glob("cub_test*"))
    validation = evaluate([
        "--checkpoint", result["checkpoint"], "--dataset", "cub", "--split", "val",
        "--data-root", str(data), "--splits-dir", str(splits),
        "--output-dir", str(tmp_path / "outputs"), "--device", "cpu",
    ])
    assert validation["status"] == "validation_only"
    assert "cub_validation" in validation and "cub_novel_test" not in validation
    assert validation["evaluation_protocol"]["split"] == "val"
    assert not list((tmp_path / "outputs" / "episodes").glob("cub_test*"))
