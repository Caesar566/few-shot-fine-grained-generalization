"""Compare matching rules on identical, frozen ImageNet features and CUB validation tasks.

This is a validation-only diagnostic, not a trained-model test result.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pattern_recognition_coursework.data.cub_dataset import CUBFewShotDataset, get_cub_transforms
from pattern_recognition_coursework.data.episodes import load_or_create_plan, plan_hash
from pattern_recognition_coursework.engine.experiment import PROJECT_ROOT, project_output, sha256_file
from pattern_recognition_coursework.models.backbones.resnet import make_backbone, spatial_features
from pattern_recognition_coursework.models.metrics.episode import prototype_scores, standardize_from_support
from pattern_recognition_coursework.models.metrics.local import local_descriptor_scores


METHODS = (
    "prototype_cosine", "prototype_euclidean", "prototype_mahalanobis",
    "prototype_cosine_standardized", "matching_attention", "dn4_local",
    "global_local_half",
)


def score_episode(features: torch.Tensor, n_way: int, k_shot: int, n_query: int) -> dict[str, float]:
    """Return episode accuracies; every rule receives the exact same embeddings."""
    support_count = n_way * k_shot
    local_support, local_query = features[:support_count], features[support_count:]
    support, query = local_support.mean((-1, -2)), local_query.mean((-1, -2))
    labels = torch.arange(n_way, device=features.device).repeat_interleave(n_query)
    global_cosine = prototype_scores(support, query, n_way, k_shot, "cosine")
    standardized = standardize_from_support(support, query)
    local = local_descriptor_scores(local_support, local_query, n_way, k_shot, 3)
    attention = (F.normalize(query, dim=-1) @ F.normalize(support, dim=-1).T).softmax(-1)
    support_labels = torch.arange(n_way, device=features.device).repeat_interleave(k_shot)
    class_probabilities = torch.zeros(query.shape[0], n_way, device=features.device)
    class_probabilities.scatter_add_(1, support_labels.expand(query.shape[0], -1), attention)
    scores = {
        "prototype_cosine": global_cosine,
        "prototype_euclidean": prototype_scores(support, query, n_way, k_shot, "euclidean"),
        "prototype_mahalanobis": prototype_scores(support, query, n_way, k_shot, "mahalanobis", 0.5),
        "prototype_cosine_standardized": prototype_scores(*standardized, n_way, k_shot, "cosine"),
        "matching_attention": class_probabilities,
        "dn4_local": local,
        "global_local_half": 0.5 * (global_cosine + local),
    }
    return {name: float((values.argmax(-1) == labels).float().mean()) for name, values in scores.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/CUB_200_2011")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/cub_fixed_feature_v1")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "reports/records/cub_fixed_feature_v1.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.batch_size < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("episode 数、batch size 必须为正，seeds 必须非空且互不重复")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    args.output_dir = project_output(args.output_dir)
    args.report = project_output(args.report)
    if args.report.exists():
        parser.error(f"已有报告，拒绝覆盖: {args.report}")
    torch.hub.set_dir(str(project_output(PROJECT_ROOT / "data/cache/torch")))
    device = torch.device("cuda" if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    dataset = CUBFewShotDataset(args.data_root, args.splits_dir / "val_images.txt", get_cub_transforms(224, False))
    plans = {}
    needed = set()
    manifest_hash = sha256_file(args.splits_dir / "val_images.txt")
    for shot in (1, 5):
        for seed in args.seeds:
            path = args.output_dir / "episodes" / f"cub_val_5way_{shot}shot_1query_{args.episodes}ep_seed{seed + 10000}_{manifest_hash[:12]}.json"
            plan = load_or_create_plan(path, dataset, 5, shot, 1, args.episodes, seed + 10000)
            plans[(shot, seed)] = (plan, path)
            needed.update(index for batch in plan["batches"] for index in batch)
    backbone = make_backbone(pretrained=True).to(device).eval()
    indices = sorted(needed)
    loader = DataLoader(torch.utils.data.Subset(dataset, indices), batch_size=args.batch_size, shuffle=False)
    embeddings = {}
    position = 0
    with torch.inference_mode():
        for images, _ in loader:
            feature_batch = spatial_features(backbone, images.to(device)).cpu()
            for feature in feature_batch:
                embeddings[indices[position]] = feature
                position += 1
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    results = {}
    with torch.inference_mode():
        for (shot, seed), (plan, path) in plans.items():
            episode_rows = []
            for batch in plan["batches"]:
                features = torch.stack([embeddings[index] for index in batch]).to(device)
                episode_rows.append(score_episode(features, 5, shot, 1))
            results[f"{shot}shot_seed{seed}"] = {
                "episode_plan": str(path), "episode_plan_hash": plan_hash(plan),
                "accuracy": {name: sum(row[name] for row in episode_rows) / len(episode_rows) for name in METHODS},
            }
            print(f"{shot}-shot seed {seed}: " + ", ".join(
                f"{name}={results[f'{shot}shot_seed{seed}']['accuracy'][name]:.3f}" for name in METHODS))
    output = {
        "schema_version": 1,
        "status": "validation_only_frozen_features",
        "dataset": "CUB-200-2011 validation classes",
        "val_manifest_sha256": manifest_hash,
        "preprocessing": "imagenet-resize-center-v1; 224px",
        "backbone": {"architecture": "ResNet-18", "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "sha256": sha256_file(PROJECT_ROOT / "data/cache/torch/checkpoints/resnet18-f37072fd.pth"),
            "trainable_parameters": 0},
        "n_way": 5, "n_query": 1, "episodes_per_seed": args.episodes,
        "seeds": args.seeds, "device": str(device), "torch_version": torch.__version__,
        "methods": list(METHODS), "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report), "encoded_images": len(embeddings)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
