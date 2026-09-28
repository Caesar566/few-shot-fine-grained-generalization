from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader
from pattern_recognition_coursework.data.cub_dataset import (
    CUBFewShotDataset,
    FewShotBatchSampler,
    get_cub_transforms,
)

@pytest.fixture
def project_root():
    return Path(__file__).resolve().parent.parent

def test_classes_disjoint(project_root):
    """验证：类别划分校验（train/val/test 严格无交集）"""
    splits_dir = project_root / "splits"
    def get_classes(split_name):
        with open(splits_dir / f"{split_name}_classes.txt", "r", encoding="utf-8") as f:
            return set(line.strip().split(" ")[0] for line in f if line.strip())

    train_c = get_classes("train")
    val_c = get_classes("val")
    test_c = get_classes("test")

    assert len(train_c) == 100, f"train 类别应为 100，实际为 {len(train_c)}"
    assert len(val_c) == 50, f"val 类别应为 50，实际为 {len(val_c)}"
    assert len(test_c) == 50, f"test 类别应为 50，实际为 {len(test_c)}"

    # 交集互斥验证
    assert len(train_c & val_c) == 0, "train 与 val 存在重叠类别！"
    assert len(train_c & test_c) == 0, "train 与 test 存在重叠类别！"
    assert len(val_c & test_c) == 0, "val 与 test 存在重叠类别！"

def test_sampler_support_query_disjoint(project_root):
    """验证：N-way K-shot 采样中，support 与 query 图像严格不重叠"""
    data_root = project_root / "data" / "raw" / "CUB_200_2011"
    split_file = project_root / "splits" / "train_images.txt"

    dataset = CUBFewShotDataset(
        data_root=str(data_root),
        split_file=str(split_file),
        transform=get_cub_transforms(image_size=84, is_train=True)
    )

    n_way, k_shot, n_query = 5, 2, 3
    sampler = FewShotBatchSampler(
        dataset=dataset,
        episodes_per_epoch=5,
        n_way=n_way,
        k_shot=k_shot,
        n_query=n_query
    )

    for batch_indices in sampler:
        support_idx = batch_indices[: n_way * k_shot]
        query_idx = batch_indices[n_way * k_shot :]

        # 1. 验证 support 与 query 长度
        assert len(support_idx) == n_way * k_shot
        assert len(query_idx) == n_way * n_query

        # 2. 验证 support 与 query 严格无交集 (图像不重叠)
        overlap = set(support_idx) & set(query_idx)
        assert len(overlap) == 0, f"检测到 support 与 query 重叠样本: {overlap}"

        # 3. 验证 batch 内部样本无重复
        assert len(set(batch_indices)) == len(batch_indices), "Episode 内存在重复采样图像！"