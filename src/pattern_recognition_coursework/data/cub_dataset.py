import random
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import torch
from torch.utils.data import Dataset, Sampler
import torchvision.transforms as transforms


def get_cub_transforms(image_size: int = 224, is_train: bool = True):
    """
    数据增强与预处理：训练阶段随机裁剪与翻转，评估阶段中心裁剪
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])


class CUBFewShotDataset(Dataset):
    """
    CUB-200-2011 元数据读取与图像加载
    """

    def __init__(self, data_root: str, split_file: str, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []  # (图片物理路径, 内部数字标签)
        self.class_to_indices: Dict[int, List[int]] = {}  # 类别标签 -> 样本索引列表
        self.classes: List[str] = []

        if not Path(split_file).exists():
            raise FileNotFoundError(f"划分清单文件不存在: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            raw_lines = [line.strip().split(" ") for line in f if line.strip()]

        # 类别名称与索引映射
        self.classes = sorted(list(set(parts[1] for parts in raw_lines)))
        self.class_to_idx = {cname: i for i, cname in enumerate(self.classes)}

        for parts in raw_lines:
            img_id, cname, rel_path = parts[0], parts[1], parts[2]
            img_path = self.data_root / "images" / rel_path

            label = self.class_to_idx[cname]
            idx = len(self.samples)
            self.samples.append((img_path, label))

            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class FewShotBatchSampler(Sampler):
    """
    Episodic Sampler：
    每次采样严格保证：
    1. 抽取 n_way 个互不相同的类别
    2. 对每个类别无放回采样 k_shot + n_query 个互不相同的样本
    3. 严格杜绝 support 与 query 样本重叠 (无交集)
    """

    def __init__(
            self,
            dataset: CUBFewShotDataset,
            episodes_per_epoch: int,
            n_way: int,
            k_shot: int,
            n_query: int
    ):
        self.dataset = dataset
        self.episodes_per_epoch = episodes_per_epoch
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.classes = list(self.dataset.class_to_indices.keys())

        # 校验：每个类别的样本数必须大于等于 k_shot + n_query
        for cls, indices in self.dataset.class_to_indices.items():
            if len(indices) < self.k_shot + self.n_query:
                raise ValueError(
                    f"类别 {cls} 的样本数 ({len(indices)}) 少于单轮采样所需 ({self.k_shot + self.n_query})"
                )

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __iter__(self):
        for _ in range(self.episodes_per_epoch):
            selected_classes = random.sample(self.classes, self.n_way)

            support_indices = []
            query_indices = []

            for cls in selected_classes:
                indices = self.dataset.class_to_indices[cls]
                # 无放回抽样 k_shot + n_query 个样本，从数学上保证互不重复
                sampled = random.sample(indices, self.k_shot + self.n_query)
                support_indices.extend(sampled[:self.k_shot])
                query_indices.extend(sampled[self.k_shot:])

            # 返回完整的 episode 批次索引：前部为 support，后部为 query
            yield support_indices + query_indices