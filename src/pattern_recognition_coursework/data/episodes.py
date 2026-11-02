"""Persisted few-shot episodes shared by methods and repeated evaluations."""

import hashlib
import json
from pathlib import Path

from pattern_recognition_coursework.data.cub_dataset import FewShotBatchSampler


def dataset_fingerprint(dataset) -> str:
    """Bind an index-based plan to the exact ordered image/class manifest."""
    root = Path(dataset.data_root).resolve()
    rows = [
        [str(Path(path).resolve().relative_to(root)).replace("\\", "/"), int(label)]
        for path, label in dataset.samples
    ]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def plan_hash(plan: dict) -> str:
    payload = json.dumps(plan, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_plan(plan: dict, dataset, n_way: int, k_shot: int, n_query: int, episodes: int) -> None:
    if plan.get("schema") != "few-shot-episodes-v1" or plan.get("dataset_fingerprint") != dataset_fingerprint(dataset):
        raise ValueError("episode 计划与当前数据清单不一致")
    if any(plan.get(key) != value for key, value in {
        "n_way": n_way, "k_shot": k_shot, "n_query": n_query, "episodes": episodes,
    }.items()):
        raise ValueError("episode 计划与当前 way/shot/query/episode 数不一致")
    batches = plan.get("batches")
    if not isinstance(batches, list) or len(batches) != episodes:
        raise ValueError("episode 计划数量不正确")
    support_count = n_way * k_shot
    for batch in batches:
        if not isinstance(batch, list) or len(batch) != n_way * (k_shot + n_query):
            raise ValueError("episode 样本数量不正确")
        if any(type(index) is not int or not 0 <= index < len(dataset) for index in batch):
            raise ValueError("episode 包含无效图像索引")
        if len(set(batch)) != len(batch):
            raise ValueError("episode 内图像重复")
        support_labels = [dataset.samples[index][1] for index in batch[:support_count]]
        query_labels = [dataset.samples[index][1] for index in batch[support_count:]]
        class_order = support_labels[::k_shot]
        if len(set(class_order)) != n_way:
            raise ValueError("episode 类别未达到指定 way")
        if support_labels != [label for label in class_order for _ in range(k_shot)]:
            raise ValueError("support 类别顺序错误")
        if query_labels != [label for label in class_order for _ in range(n_query)]:
            raise ValueError("query 类别顺序错误")


def load_or_create_plan(path: Path, dataset, n_way: int, k_shot: int, n_query: int,
                        episodes: int, seed: int) -> dict:
    path = Path(path)
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
        if plan.get("seed") != seed:
            raise ValueError("episode 计划种子与请求不一致")
    else:
        sampler = FewShotBatchSampler(dataset, episodes, n_way, k_shot, n_query, seed=seed)
        plan = {
            "schema": "few-shot-episodes-v1", "dataset_fingerprint": dataset_fingerprint(dataset),
            "n_way": n_way, "k_shot": k_shot, "n_query": n_query,
            "episodes": episodes, "seed": seed, "batches": list(sampler),
        }
        validate_plan(plan, dataset, n_way, k_shot, n_query, episodes)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    validate_plan(plan, dataset, n_way, k_shot, n_query, episodes)
    return plan
