"""Paired, seed-level accuracy intervals for method gains and domain gaps."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_FIELDS = (
    "source_dataset", "target_dataset", "target_manifest_hash", "n_way",
    "k_shot", "n_query", "episodes", "image_size", "preprocessing",
)
TASK_FIELDS = ("n_way", "k_shot", "n_query", "episodes", "image_size", "preprocessing")


def _records_by_seed(records: list[dict], target: str) -> dict[int, dict]:
    if len(records) < 3:
        raise ValueError("每组至少需要 3 个不同的训练随机种子")
    indexed = {}
    for record in records:
        if record.get("schema_version") != 2:
            raise ValueError("需要包含固定 episode 计划的版本 2 结果；旧指标无法证明配对协议")
        seed = record.get("seed")
        if type(seed) is not int or seed in indexed:
            raise ValueError("训练随机种子缺失或重复")
        protocol = record.get("evaluation_protocol", {})
        if protocol.get("source_dataset") != "cub" or protocol.get("target_dataset") != target:
            raise ValueError("结果场景与请求不一致")
        if not all(protocol.get(field) is not None for field in (*PROTOCOL_FIELDS, "episode_plan_hash", "test_episode_seed")):
            raise ValueError("评估协议字段不完整")
        value = record.get("cub_novel_test" if target == "cub" else "cross_dataset_test", {})
        accuracy = value.get("accuracy")
        if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("准确率必须是 [0, 1] 范围的有限数")
        if value.get("episodes") != protocol["episodes"]:
            raise ValueError("实际评估 episode 数与协议不一致")
        if protocol["test_episode_seed"] != seed + 20_000:
            raise ValueError("测试 episode 种子与训练种子不一致")
        if not isinstance(record.get("model_config"), dict) or not isinstance(record.get("training_config"), dict):
            raise ValueError("模型或训练配置缺失")
        indexed[seed] = record
    first = next(iter(indexed.values()))
    for record in indexed.values():
        if record["model_config"] != first["model_config"]:
            raise ValueError("同组实验模型配置不一致")
        training = dict(record["training_config"])
        first_training = dict(first["training_config"])
        training.pop("seed", None)
        first_training.pop("seed", None)
        if training != first_training:
            raise ValueError("同组实验训练配置不一致")
        if any(record["evaluation_protocol"][key] != first["evaluation_protocol"][key]
               for key in PROTOCOL_FIELDS):
            raise ValueError("同组实验评估协议不一致")
    return indexed


def _stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    mean = float(data.mean())
    std = float(data.std(ddof=1))
    half = float(t.ppf(0.975, len(data) - 1) * std / np.sqrt(len(data)))
    return {"mean_pp": mean, "seed_std_pp": std,
            "ci95_pp": [mean - half, mean + half]}


def _paired(left: list[dict], right: list[dict], left_target: str, right_target: str,
            *, same_model: bool) -> tuple[list[int], list[float], list[float], dict, dict]:
    first = _records_by_seed(left, left_target)
    second = _records_by_seed(right, right_target)
    if set(first) != set(second):
        raise ValueError("两组实验必须使用完全相同的训练随机种子")
    seeds = sorted(first)
    for seed in seeds:
        a, b = first[seed], second[seed]
        p, q = a["evaluation_protocol"], b["evaluation_protocol"]
        if any(p[key] != q[key] for key in TASK_FIELDS):
            raise ValueError("配对实验的任务协议不一致")
        if p["test_episode_seed"] != q["test_episode_seed"]:
            raise ValueError("配对实验测试 episode 种子不同")
        if left_target == right_target and p["episode_plan_hash"] != q["episode_plan_hash"]:
            raise ValueError("同场景配对实验必须使用完全相同的测试 episode 计划")
        if a["training_config"].get("split_hashes") != b["training_config"].get("split_hashes"):
            raise ValueError("配对实验必须使用相同的 CUB 训练、验证和测试划分")
        if same_model and (a["model_config"] != b["model_config"] or
                           a["training_config"] != b["training_config"]):
            raise ValueError("泛化差距必须由同一个训练模型在两场景评估")
    left_key = "cub_novel_test" if left_target == "cub" else "cross_dataset_test"
    right_key = "cub_novel_test" if right_target == "cub" else "cross_dataset_test"
    a_values = [100 * first[s][left_key]["accuracy"] for s in seeds]
    b_values = [100 * second[s][right_key]["accuracy"] for s in seeds]
    return seeds, a_values, b_values, first[seeds[0]], second[seeds[0]]


def paired_summary(baseline: list[dict], candidate: list[dict]) -> dict:
    if not baseline or not candidate:
        raise ValueError("结果列表不能为空")
    target = baseline[0].get("evaluation_protocol", {}).get("target_dataset")
    if target not in {"cub", "nabirds"}:
        raise ValueError("只支持 CUB 未见类或人工审核的 NABirds 场景")
    seeds, base, new, a, b = _paired(baseline, candidate, target, target, same_model=False)
    gain = _stats([candidate_value - base_value for base_value, candidate_value in zip(base, new)])
    is_cross = target == "nabirds"
    return {
        "comparison": "candidate minus baseline", "scenario": target,
        "seeds": seeds, "confidence_unit": "paired training seed; two-sided 95% Student t interval",
        "baseline_model_config": a["model_config"],
        "candidate_model_config": b["model_config"],
        "baseline_training_config": a["training_config"],
        "candidate_training_config": b["training_config"],
        "baseline": _stats(base), "candidate": _stats(new),
        "paired_improvement": gain,
        "meets_5pp_goal": (gain["mean_pp"] >= 5 and gain["ci95_pp"][0] > 0) if is_cross else None,
        "ci_lower_at_least_5pp": gain["ci95_pp"][0] >= 5 if is_cross else None,
    }


def generalization_gap(cub_results: list[dict], nabirds_results: list[dict]) -> dict:
    seeds, cub, nabirds, a, _ = _paired(cub_results, nabirds_results, "cub", "nabirds", same_model=True)
    return {
        "comparison": "CUB novel minus NABirds", "seeds": seeds,
        "confidence_unit": "paired training seed; two-sided 95% Student t interval",
        "model_config": a["model_config"], "cub_novel": _stats(cub),
        "cross_dataset": _stats(nabirds),
        "generalization_gap": _stats([source - target for source, target in zip(cub, nabirds)]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["improvement", "gap"], default="improvement")
    parser.add_argument("--baseline", type=Path, nargs="+", required=True,
                        help="Baseline or CUB novel result JSON files")
    parser.add_argument("--candidate", type=Path, nargs="+", required=True,
                        help="Candidate or NABirds result JSON files")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    left = [json.loads(path.read_text(encoding="utf-8")) for path in args.baseline]
    right = [json.loads(path.read_text(encoding="utf-8")) for path in args.candidate]
    result = paired_summary(left, right) if args.mode == "improvement" else generalization_gap(left, right)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        destination = args.output.resolve()
        if not destination.is_relative_to(ROOT):
            parser.error("汇总结果必须保存在工程目录内")
        if destination.exists():
            parser.error(f"汇总结果已存在: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return result


if __name__ == "__main__":
    main()
