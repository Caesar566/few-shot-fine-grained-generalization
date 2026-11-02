"""Train CUB few-shot models with validation-only selection and saved test tasks."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from pattern_recognition_coursework.analysis.metrics import aggregate_episode_metrics, evaluation_metrics
from pattern_recognition_coursework.data.cub_dataset import (
    CUBFewShotDataset, FewShotBatchSampler, get_cub_transforms,
)
from pattern_recognition_coursework.data.episodes import load_or_create_plan, plan_hash
from pattern_recognition_coursework.models.few_shot import FewShotModel


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_fingerprint() -> str:
    """Identify the exact local package source used by a new experiment."""
    root = PROJECT_ROOT / "src/pattern_recognition_coursework"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def project_output(path: Path) -> Path:
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("运行产物必须保存在工程目录内")
    return resolved


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def model_from_config(config: dict, *, load_initial_weights: bool = True) -> FewShotModel:
    options = dict(config)
    if not load_initial_weights:
        options["pretrained"] = False
        options["backbone_weights"] = None
        options["backbone_prefix"] = ""
    return FewShotModel(**options)


def configure_trainable_scope(model: FewShotModel, scope: str) -> list[str]:
    """Freeze the encoder as requested while training method-specific context."""
    if scope == "full":
        return [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if scope == "layer4":
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad_(True)
    elif scope == "adapter":
        if model.adapter is None:
            raise ValueError("adapter 范围需要线性适配层")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.adapter.parameters():
            parameter.requires_grad_(True)
        if model.context is not None:
            for parameter in model.context.parameters():
                parameter.requires_grad_(True)
        if isinstance(model.log_temperature, nn.Parameter):
            model.log_temperature.requires_grad_(True)
    else:
        raise ValueError(f"未知可训练范围: {scope}")
    model.freeze_backbone_running_stats = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def episode_loader(dataset, plan: dict, workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_sampler=plan["batches"], num_workers=workers)


def run_episodes(model, loader, n_way: int, k_shot: int, n_query: int,
                 device: torch.device, optimizer=None, *, collect_predictions: bool = False):
    model.train(optimizer is not None)
    if optimizer is not None and getattr(model, "freeze_backbone_running_stats", False):
        model.backbone.eval()
    labels = torch.arange(n_way, device=device).repeat_interleave(n_query)
    episode_metrics = []
    predictions = []
    for episode_id, (images, _) in enumerate(loader):
        images = images.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images, n_way, k_shot, n_query)
            loss = F.cross_entropy(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        episode_metrics.append(evaluation_metrics(logits, labels))
        if collect_predictions:
            probabilities = logits.detach().softmax(dim=-1).cpu()
            predictions.append({
                "episode_id": episode_id,
                "predicted_slots": probabilities.argmax(dim=-1).tolist(),
                "true_slots": labels.cpu().tolist(),
                "confidences": probabilities.max(dim=-1).values.tolist(),
            })
    result = aggregate_episode_metrics(episode_metrics)
    result["loss"] = result["nll"]
    return result, predictions


def supervised_epoch(model, head, dataset, batch_size: int, lr_optimizer,
                     device: torch.device, seed: int, workers: int):
    model.train()
    if getattr(model, "freeze_backbone_running_stats", False):
        model.backbone.eval()
    head.train()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=generator, num_workers=workers)
    losses = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        features = model.encode(images)
        if features.ndim == 4:
            features = features.mean(dim=(-1, -2))
        loss = F.cross_entropy(head(features), labels)
        lr_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        lr_optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def create_plan(dataset, args, split_name: str, seed: int):
    fingerprint = sha256_file(args.splits_dir / f"{split_name}_images.txt")[:12]
    name = f"cub_{split_name}_{args.n_way}way_{args.k_shot}shot_{args.n_query}query_{args.eval_episodes}ep_seed{seed}_{fingerprint}.json"
    path = project_output(args.output_dir / "episodes" / name)
    plan = load_or_create_plan(path, dataset, args.n_way, args.k_shot,
                               args.n_query, args.eval_episodes, seed)
    return plan, path


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data/raw/CUB_200_2011")
    parser.add_argument("--splits-dir", type=Path, default=PROJECT_ROOT / "data/splits")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--stage", choices=["episodic", "supervised", "meta_baseline"], default="episodic")
    parser.add_argument("--method", choices=["protonet", "matchingnet", "matchingnet_fce", "dn4", "global_local"], default="protonet")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--backbone-weights", type=Path)
    parser.add_argument("--backbone-prefix", default="")
    parser.add_argument("--weight-source", help="Source and training data of a local initialization checkpoint")
    parser.add_argument("--metric", choices=["euclidean", "cosine", "mahalanobis"], default="euclidean")
    parser.add_argument("--episode-standardization", action="store_true")
    parser.add_argument("--shrinkage", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--learnable-temperature", action="store_true")
    parser.add_argument("--local-k", type=int, default=3)
    parser.add_argument("--local-weight", type=float, default=0.5)
    parser.add_argument("--fce-steps", type=int, default=5)
    parser.add_argument("--adapter", action="store_true", help="Identity-initialized 512-d linear adapter")
    parser.add_argument("--trainable-scope", choices=["full", "layer4", "adapter"], default="full",
                        help="adapter 冻结骨干；若方法带 FCE，上下文 LSTM 仍参与训练")
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=1)
    parser.add_argument("--n-query", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--train-episodes", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10, help="Episodic epochs")
    parser.add_argument("--supervised-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--export-errors", action="store_true")
    parser.add_argument("--skip-test", action="store_true", help="Save validation-selected checkpoint without reading test episodes")
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    if min(args.n_way, args.k_shot, args.n_query, args.image_size,
           args.train_episodes, args.eval_episodes, args.batch_size) < 1:
        parser.error("way/shot/query、图像尺寸、episode 数和 batch size 必须为正整数")
    if args.epochs < 0 or args.supervised_epochs < 0 or args.lr <= 0 or args.workers < 0:
        parser.error("轮次和 workers 不得为负，学习率必须大于 0")
    if args.stage in {"episodic", "meta_baseline"} and args.epochs == 0:
        parser.error("episodic 与 meta_baseline 阶段需要 --epochs > 0")
    if args.stage in {"supervised", "meta_baseline"} and args.supervised_epochs == 0:
        parser.error("supervised 与 meta_baseline 阶段需要 --supervised-epochs > 0")
    if args.stage in {"supervised", "meta_baseline"} and (args.method != "protonet" or args.metric != "cosine"):
        parser.error("监督最近类中心强基线使用 --method protonet --metric cosine")
    if args.skip_test and args.export_errors:
        parser.error("--skip-test 与 --export-errors 不能同时使用")
    if args.backbone_weights and not args.weight_source:
        parser.error("本地初始化权重需要 --weight-source 记录算法、训练数据与来源")
    if args.trainable_scope == "adapter" and not args.adapter:
        parser.error("--trainable-scope adapter 需要 --adapter")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA 不可用")
    try:
        args.output_dir = project_output(args.output_dir)
    except ValueError as error:
        parser.error(str(error))
    # torchvision caches optional ImageNet weights inside the project.
    torch.hub.set_dir(str(project_output(PROJECT_ROOT / "data/cache/torch")))
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    train = CUBFewShotDataset(args.data_root, args.splits_dir / "train_images.txt",
                               get_cub_transforms(args.image_size, True))
    val = CUBFewShotDataset(args.data_root, args.splits_dir / "val_images.txt",
                             get_cub_transforms(args.image_size, False))
    if set(train.classes) & set(val.classes):
        raise ValueError("CUB train/val 类别存在交集")
    split_hashes = {split: sha256_file(args.splits_dir / f"{split}_images.txt")
                    for split in ("train", "val", "test")}
    val_plan, val_plan_path = create_plan(val, args, "val", args.seed + 10_000)
    model = FewShotModel(
        args.method, args.pretrained, args.metric, args.episode_standardization,
        args.shrinkage, args.temperature, args.learnable_temperature,
        args.backbone_weights, args.backbone_prefix, args.local_k, args.local_weight,
        args.fce_steps, args.adapter,
    ).to(device)
    trainable_names = configure_trainable_scope(model, args.trainable_scope)
    model_config = dict(model.model_config)
    model_config["backbone_weights"] = str(args.backbone_weights.resolve()) if args.backbone_weights else None
    training_config = {
        "stage": args.stage, "seed": args.seed, "n_way": args.n_way,
        "k_shot": args.k_shot, "n_query": args.n_query, "image_size": args.image_size,
        "train_episodes_per_epoch": args.train_episodes, "eval_episodes": args.eval_episodes,
        "epochs": args.epochs if args.stage != "supervised" else 0,
        "supervised_epochs": args.supervised_epochs if args.stage != "episodic" else 0,
        "batch_size": args.batch_size, "learning_rate": args.lr,
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "trainable_parameter_names": trainable_names,
        "split_hashes": split_hashes,
        "weight_source": args.weight_source or (model.backbone_load_info or {}).get("url"),
        "train_classes_sha256": sha256_file(args.splits_dir / "train_classes.txt"),
        "backbone_load_info": model.backbone_load_info,
        "code_fingerprint_sha256": code_fingerprint(),
    }
    run_id = hashlib.sha256(json.dumps({"model": model_config, "training": training_config},
                                       sort_keys=True, default=str).encode()).hexdigest()[:12]
    run_dir = project_output(args.output_dir / "runs" / f"{args.method}_{args.stage}_{args.n_way}way_{args.k_shot}shot_seed{args.seed}_{run_id}")
    if run_dir.exists():
        parser.error(f"运行目录已存在，请选择新配置或移走旧运行: {run_dir}")
    run_dir.mkdir(parents=True)
    best_path = run_dir / "best.pt"
    history = []

    def select_best(epoch, phase, best_value):
        validation, _ = run_episodes(model, episode_loader(val, val_plan, args.workers),
                                     args.n_way, args.k_shot, args.n_query, device)
        if validation["accuracy"] > best_value:
            best_value = validation["accuracy"]
            torch.save({"model": model.state_dict(), "model_config": model_config,
                        "training_config": training_config, "best_epoch": epoch,
                        "selected_phase": phase, "best_val_accuracy": best_value,
                        "validation_plan_hash": plan_hash(val_plan)}, best_path)
        return validation, best_value

    if args.stage in {"supervised", "meta_baseline"}:
        head = nn.Linear(512, len(train.classes)).to(device)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad]
            + list(head.parameters()), lr=args.lr)
        best_value = -1.0
        for epoch in range(1, args.supervised_epochs + 1):
            set_seed(args.seed + epoch)
            train_loss = supervised_epoch(model, head, train, args.batch_size, optimizer,
                                          device, args.seed + epoch, args.workers)
            validation, best_value = select_best(epoch, "supervised", best_value)
            history.append({"phase": "supervised", "epoch": epoch, "train_loss": train_loss,
                            "val_accuracy": validation["accuracy"]})
            print(f"supervised {epoch}: loss={train_loss:.4f} val={validation['accuracy']:.4f}")
        if args.stage == "meta_baseline":
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model"])

    if args.stage in {"episodic", "meta_baseline"}:
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr)
        best_value = -1.0
        for epoch in range(1, args.epochs + 1):
            set_seed(args.seed + 100_000 + epoch)
            sampler = FewShotBatchSampler(train, args.train_episodes, args.n_way,
                                          args.k_shot, args.n_query, seed=args.seed + epoch)
            loader = DataLoader(train, batch_sampler=sampler, num_workers=args.workers)
            training, _ = run_episodes(model, loader, args.n_way, args.k_shot, args.n_query,
                                       device, optimizer)
            validation, best_value = select_best(epoch, "episodic", best_value)
            history.append({"phase": "episodic", "epoch": epoch,
                            "train_accuracy": training["accuracy"],
                            "val_accuracy": validation["accuracy"]})
            print(f"episodic {epoch}: train={training['accuracy']:.4f} val={validation['accuracy']:.4f}")

    selected = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(selected["model"])
    if args.skip_test:
        result = {
            "schema_version": 2, "status": "validation_only", "seed": args.seed,
            "model_config": model_config, "training_config": training_config,
            "best_epoch": selected["best_epoch"], "selected_phase": selected["selected_phase"],
            "best_val_accuracy": selected["best_val_accuracy"],
            "checkpoint": str(best_path), "validation_plan": str(val_plan_path),
            "validation_plan_hash": plan_hash(val_plan),
            "device": str(device), "torch_version": str(torch.__version__),
        }
        (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "best_val_accuracy": selected["best_val_accuracy"],
                          "status": "validation_only"}, ensure_ascii=False))
        return result
    test = CUBFewShotDataset(args.data_root, args.splits_dir / "test_images.txt",
                              get_cub_transforms(args.image_size, False))
    if set(train.classes) & set(test.classes) or set(val.classes) & set(test.classes):
        raise ValueError("CUB test 类别与 train/val 存在交集")
    test_plan, test_plan_path = create_plan(test, args, "test", args.seed + 20_000)
    test_metrics, predictions = run_episodes(model, episode_loader(test, test_plan, args.workers),
                                             args.n_way, args.k_shot, args.n_query, device,
                                             collect_predictions=args.export_errors)
    protocol = {
        "source_dataset": "cub", "target_dataset": "cub",
        "target_manifest_hash": split_hashes["test"],
        "episode_plan_hash": plan_hash(test_plan), "test_episode_seed": test_plan["seed"],
        "n_way": args.n_way, "k_shot": args.k_shot, "n_query": args.n_query,
        "episodes": args.eval_episodes, "image_size": args.image_size,
        "preprocessing": "imagenet-resize-center-v1",
    }
    result = {
        "schema_version": 2, "seed": args.seed, "model_config": model_config,
        "training_config": training_config, "evaluation_protocol": protocol,
        "best_epoch": selected["best_epoch"], "selected_phase": selected["selected_phase"],
        "best_val_accuracy": selected["best_val_accuracy"],
        "cub_novel_test": test_metrics, "checkpoint": str(best_path),
        "validation_plan": str(val_plan_path), "test_plan": str(test_plan_path),
        "device": str(device), "torch_version": str(torch.__version__),
    }
    if args.export_errors:
        error_path = run_dir / "query_predictions.jsonl"
        with error_path.open("w", encoding="utf-8") as stream:
            for row, batch in zip(predictions, test_plan["batches"]):
                support_ids = batch[:args.n_way * args.k_shot]
                query_ids = batch[args.n_way * args.k_shot:]
                class_ids = [test.samples[support_ids[i * args.k_shot]][1] for i in range(args.n_way)]
                for index, image_index in enumerate(query_ids):
                    stream.write(json.dumps({
                        "episode_id": row["episode_id"],
                        "image": str(test.samples[image_index][0]),
                        "true_class": test.classes[class_ids[row["true_slots"][index]]],
                        "predicted_class": test.classes[class_ids[row["predicted_slots"][index]]],
                        "confidence": row["confidences"][index],
                        "support_images": [str(test.samples[i][0]) for i in support_ids],
                    }, ensure_ascii=False) + "\n")
        result["predictions"] = str(error_path)
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "cub_novel_accuracy": test_metrics["accuracy"],
                      "best_epoch": selected["best_epoch"]}, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
