import pytest

from pattern_recognition_coursework.analysis.select import select_fusion_weight


def record(seed, weight, accuracy):
    return {
        "schema_version": 2, "status": "validation_only", "seed": seed,
        "model_config": {"method": "global_local", "metric": "cosine", "local_weight": weight},
        "training_config": {"seed": seed, "epochs": 10, "image_size": 224},
        "validation_plan_hash": f"plan-{seed}", "best_val_accuracy": accuracy,
    }


def test_fusion_selection_uses_only_shared_validation_tasks():
    records = [record(seed, weight, .5 + weight / 10) for seed in (1, 2, 3) for weight in (0.0, 0.5)]
    assert select_fusion_weight(records)["selected_local_weight"] == .5
    records[0]["cub_novel_test"] = {"accuracy": .9}
    with pytest.raises(ValueError, match="validation_only"):
        select_fusion_weight(records)
