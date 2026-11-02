import pytest

from pattern_recognition_coursework.analysis.summarize import generalization_gap, paired_summary


def record(seed, accuracy, *, target="cub", method="protonet", plan=None):
    key = "cub_novel_test" if target == "cub" else "cross_dataset_test"
    return {
        "schema_version": 2,
        "seed": seed,
        "model_config": {"method": method, "metric": "cosine"},
        "training_config": {"seed": seed, "epochs": 10, "split_hashes": {"train": "a"}},
        "evaluation_protocol": {
            "source_dataset": "cub", "target_dataset": target,
            "target_manifest_hash": "cub-test" if target == "cub" else "reviewed-nabirds",
            "episode_plan_hash": plan or f"{target}-plan-{seed}",
            "test_episode_seed": seed + 20_000,
            "n_way": 5, "k_shot": 1, "n_query": 5,
            "episodes": 100, "image_size": 224,
            "preprocessing": "imagenet-resize-center-v1",
        },
        key: {"accuracy": accuracy, "episodes": 100},
    }


def test_cub_improvement_is_not_cross_dataset_goal():
    baseline = [record(seed, acc) for seed, acc in [(1, .40), (2, .41), (3, .42)]]
    candidate = [record(seed, acc, method="global_local") for seed, acc in [(1, .46), (2, .47), (3, .48)]]
    result = paired_summary(baseline, candidate)
    assert result["paired_improvement"]["mean_pp"] == pytest.approx(6)
    assert result["meets_5pp_goal"] is None


def test_nabirds_improvement_and_domain_gap():
    cub = [record(seed, acc) for seed, acc in [(1, .65), (2, .66), (3, .67)]]
    cross = [record(seed, acc, target="nabirds") for seed, acc in [(1, .40), (2, .41), (3, .42)]]
    candidate = [record(seed, acc, target="nabirds", method="global_local")
                 for seed, acc in [(1, .46), (2, .47), (3, .48)]]
    assert paired_summary(cross, candidate)["meets_5pp_goal"]
    assert generalization_gap(cub, cross)["generalization_gap"]["mean_pp"] == pytest.approx(25)


def test_paired_summary_rejects_different_episode_plan():
    baseline = [record(seed, .40) for seed in (1, 2, 3)]
    candidate = [record(seed, .46, method="global_local") for seed in (1, 2, 3)]
    candidate[0]["evaluation_protocol"]["episode_plan_hash"] = "different"
    with pytest.raises(ValueError, match="测试 episode 计划"):
        paired_summary(baseline, candidate)


def test_paired_summary_rejects_invalid_accuracy():
    baseline = [record(seed, .40) for seed in (1, 2, 3)]
    candidate = [record(seed, .46, method="global_local") for seed in (1, 2, 3)]
    candidate[1]["cub_novel_test"]["accuracy"] = float("nan")
    with pytest.raises(ValueError, match="准确率"):
        paired_summary(baseline, candidate)


def test_paired_summary_rejects_different_source_splits():
    baseline = [record(seed, .40) for seed in (1, 2, 3)]
    candidate = [record(seed, .46, method="global_local") for seed in (1, 2, 3)]
    for row in candidate:
        row["training_config"]["split_hashes"] = {"train": "changed"}
    with pytest.raises(ValueError, match="相同的 CUB"):
        paired_summary(baseline, candidate)
