"""Evaluate a saved CUB model on fixed CUB or reviewed NABirds episodes."""

import argparse
import json
from pathlib import Path

import torch

from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset, get_cub_transforms
from pattern_recognition_coursework.data.episodes import load_or_create_plan, plan_hash
from pattern_recognition_coursework.data.nabirds import NABirdsDataset
from pattern_recognition_coursework.engine.experiment import (
    PROJECT_ROOT, episode_loader, model_from_config, project_output,
    run_episodes, sha256_file,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["cub", "nabirds"], default="cub")
    parser.add_argument("--split", choices=["val", "test"], default="test",
                        help="CUB 可复评验证或测试类别；NABirds 仅支持 test")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path, help="Required human-reviewed NABirds JSON manifest")
    parser.add_argument("--review", type=Path, help="NABirds reviewer decisions used to build the manifest")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--n-way", type=int)
    parser.add_argument("--k-shot", type=int)
    parser.add_argument("--n-query", type=int)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--export-errors", action="store_true")
    args = parser.parse_args(argv)
    if args.dataset == "nabirds" and (args.manifest is None or args.review is None):
        parser.error("NABirds 评估必须指定人工审核后的 --manifest 和 --review")
    if args.dataset == "nabirds" and args.split != "test":
        parser.error("NABirds 仅支持 test；CUB 可使用 --split val")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    if args.workers < 0:
        parser.error("workers 不能为负")
    args.output_dir = project_output(args.output_dir)
    device = torch.device("cuda" if args.device == "cuda" or
                          args.device == "auto" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if not {"model", "model_config", "training_config"} <= checkpoint.keys():
        parser.error("检查点缺少模型或训练配置；请使用新训练入口生成的检查点")
    model = model_from_config(checkpoint["model_config"], load_initial_weights=False).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    training = checkpoint["training_config"]
    n_way = training["n_way"] if args.n_way is None else args.n_way
    k_shot = training["k_shot"] if args.k_shot is None else args.k_shot
    n_query = training["n_query"] if args.n_query is None else args.n_query
    episodes = training["eval_episodes"] if args.episodes is None else args.episodes
    seed = args.seed if args.seed is not None else training["seed"] + 20_000
    if min(n_way, k_shot, n_query, episodes) < 1:
        parser.error("way/shot/query 和 episode 数必须为正整数")
    transform = get_cub_transforms(training["image_size"], False)
    if args.dataset == "cub":
        data_root = args.data_root or PROJECT_ROOT / "data/raw/CUB_200_2011"
        for split in ("train", "val", "test"):
            if sha256_file(args.splits_dir / f"{split}_images.txt") != training["split_hashes"][split]:
                parser.error(f"CUB {split} 清单与训练检查点不一致")
        manifest = args.splits_dir / f"{args.split}_images.txt"
        dataset = CUBFewShotDataset(data_root, manifest, transform)
        metric_key = "cub_validation" if args.split == "val" else "cub_novel_test"
    else:
        data_root = args.data_root or PROJECT_ROOT / "data/raw/nabirds"
        manifest = args.manifest
        dataset = NABirdsDataset(data_root, manifest, transform,
                                 review_file=args.review,
                                 cub_train_classes=args.splits_dir / "train_classes.txt")
        if dataset.manifest_metadata["source_train_classes_sha256"] != training["train_classes_sha256"]:
            parser.error("NABirds 物种排除审核所用 CUB 训练类别与模型训练清单不一致")
        metric_key = "cross_dataset_test"
    manifest_hash = sha256_file(manifest)
    plan_file = project_output(args.output_dir / "episodes" /
                               f"{args.dataset}_{args.split}_{n_way}way_{k_shot}shot_{n_query}query_{episodes}ep_seed{seed}_{manifest_hash[:12]}.json")
    plan = load_or_create_plan(plan_file, dataset, n_way, k_shot, n_query, episodes, seed)
    metrics, predictions = run_episodes(
        model, episode_loader(dataset, plan, args.workers), n_way, k_shot, n_query,
        device, collect_predictions=args.export_errors,
    )
    checkpoint_hash = sha256_file(args.checkpoint)
    result = {
        "schema_version": 2, "seed": training["seed"],
        "model_config": checkpoint["model_config"], "training_config": training,
        "evaluation_protocol": {
            "source_dataset": "cub", "target_dataset": args.dataset,
            "split": args.split,
            "target_manifest_hash": manifest_hash,
            "episode_plan_hash": plan_hash(plan), "test_episode_seed": seed,
            "n_way": n_way, "k_shot": k_shot, "n_query": n_query,
            "episodes": episodes, "image_size": training["image_size"],
            "preprocessing": "imagenet-resize-center-v1",
        },
        metric_key: metrics, "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash, "test_plan": str(plan_file),
        "device": str(device), "torch_version": str(torch.__version__),
    }
    suffix = f"{args.dataset}_{args.split}_{n_way}way_{k_shot}shot_{n_query}query_{episodes}ep_seed{seed}_{manifest_hash[:12]}"
    output_dir = project_output(args.output_dir / "evaluations" / checkpoint_hash[:12])
    output = output_dir / f"{suffix}.json"
    if output.exists():
        parser.error(f"评估结果已存在: {output}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.split == "val":
        result["status"] = "validation_only"
    if args.export_errors:
        errors_path = output_dir / f"{suffix}_queries.jsonl"
        with errors_path.open("w", encoding="utf-8") as stream:
            for episode, batch in zip(predictions, plan["batches"]):
                support = batch[:n_way * k_shot]
                queries = batch[n_way * k_shot:]
                class_labels = [dataset.samples[support[i * k_shot]][1] for i in range(n_way)]
                for query_slot, query_index in enumerate(queries):
                    stream.write(json.dumps({
                        "episode_id": episode["episode_id"],
                        "query_image": str(dataset.samples[query_index][0]),
                        "true_class": dataset.classes[class_labels[episode["true_slots"][query_slot]]],
                        "predicted_class": dataset.classes[class_labels[episode["predicted_slots"][query_slot]]],
                        "confidence": episode["confidences"][query_slot],
                        "support_images": [str(dataset.samples[i][0]) for i in support],
                    }, ensure_ascii=False) + "\n")
        result["predictions"] = str(errors_path)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(output), "accuracy": metrics["accuracy"],
                      "scenario": args.dataset}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
