"""Choose a global/local fusion weight using validation results only."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from pattern_recognition_coursework.analysis.summarize import ROOT


def select_fusion_weight(records: list[dict]) -> dict:
    groups = defaultdict(dict)
    reference_model = reference_training = None
    plans_by_seed = {}
    for record in records:
        if record.get("schema_version") != 2 or record.get("status") != "validation_only" or "cub_novel_test" in record:
            raise ValueError("融合权重选择只接受未读取测试集的 validation_only 结果")
        model = dict(record["model_config"])
        if model.get("method") != "global_local":
            raise ValueError("只接受 global_local 模型")
        weight = model.pop("local_weight")
        if not isinstance(weight, (int, float)) or not math.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError("局部分支权重无效")
        training = dict(record["training_config"])
        seed = training.pop("seed")
        if type(seed) is not int or record["seed"] != seed:
            raise ValueError("训练随机种子缺失或不一致")
        accuracy = record.get("best_val_accuracy")
        if not isinstance(accuracy, (int, float)) or not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
            raise ValueError("验证准确率无效")
        if reference_model is None:
            reference_model, reference_training = model, training
        if model != reference_model or training != reference_training:
            raise ValueError("候选融合权重以外的模型或训练配置不一致")
        if seed in groups[weight]:
            raise ValueError("相同权重与随机种子的运行重复")
        plan = record.get("validation_plan_hash")
        if not plan:
            raise ValueError("缺少固定验证 episode 计划哈希")
        if seed in plans_by_seed and plans_by_seed[seed] != plan:
            raise ValueError("相同随机种子使用了不同验证 episode 计划")
        plans_by_seed[seed] = plan
        groups[weight][seed] = float(accuracy)
    if len(groups) < 2:
        raise ValueError("至少需要两个融合权重候选")
    seeds = set(next(iter(groups.values())))
    if len(seeds) < 3 or any(set(group) != seeds for group in groups.values()):
        raise ValueError("每个权重需要相同的至少 3 个随机种子")
    means = {weight: sum(group.values()) / len(group) for weight, group in groups.items()}
    chosen = min(means, key=lambda weight: (-means[weight], weight))
    return {
        "selection_source": "fixed CUB validation episodes only",
        "seeds": sorted(seeds), "candidate_mean_validation_accuracy": {
            str(weight): means[weight] for weight in sorted(means)
        },
        "selected_local_weight": chosen,
        "selected_mean_validation_accuracy": means[chosen],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, nargs="+", required=True,
                        help="validation_only metrics.json files for all candidate weights and seeds")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = select_fusion_weight([json.loads(path.read_text(encoding="utf-8")) for path in args.results])
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        destination = args.output.resolve()
        if not destination.is_relative_to(ROOT):
            parser.error("选择结果必须保存在工程目录内")
        if destination.exists():
            parser.error(f"文件已存在: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return result


if __name__ == "__main__":
    main()
