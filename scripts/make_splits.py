import os
import random
from pathlib import Path


def find_cub_root():
    project_root = Path(__file__).resolve().parent.parent

    # 1. 优先在常见候选路径中查找
    candidates = [
        project_root / "data" / "raw" / "CUB_200_2011",
        project_root / "cub2002011" / "CUB_200_2011",
        project_root / "cub2002011",
        project_root / "data" / "raw",
    ]
    for p in candidates:
        if (p / "classes.txt").exists() and (p / "images.txt").exists():
            return p

    # 2. 如果没在上面列出的直接路径下，直接在整个项目根目录下全局搜索 classes.txt
    for txt_path in project_root.rglob("classes.txt"):
        parent = txt_path.parent
        if (parent / "images.txt").exists():
            return parent

    raise FileNotFoundError(f"在工程目录 {project_root} 下未检索到包含 classes.txt 和 images.txt 的数据集文件夹")


def make_cub_splits(
        output_dir: str = "splits",
        train_num: int = 100,
        val_num: int = 50,
        test_num: int = 50,
        seed: int = 42
):
    project_root = Path(__file__).resolve().parent.parent
    data_path = find_cub_root()
    print(f"成功定位到数据源目录: {data_path.resolve()}")

    classes_file = data_path / "classes.txt"
    images_file = data_path / "images.txt"

    # 读取全部 200 个类别
    with open(classes_file, "r", encoding="utf-8") as f:
        classes = [line.strip().split(" ") for line in f if line.strip()]
        classes = [(parts[0], parts[1]) for parts in classes]

    assert len(classes) == train_num + val_num + test_num, f"类别总数不为 200: {len(classes)}"

    # 固定种子划分 100 / 50 / 50
    rng = random.Random(seed)
    shuffled_classes = classes.copy()
    rng.shuffle(shuffled_classes)

    train_classes = sorted(shuffled_classes[:train_num], key=lambda x: int(x[0]))
    val_classes = sorted(shuffled_classes[train_num:train_num + val_num], key=lambda x: int(x[0]))
    test_classes = sorted(shuffled_classes[train_num + val_num:], key=lambda x: int(x[0]))

    # 读取全部图像对应关系
    image_dict = {}
    with open(images_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) >= 2:
                img_id, rel_path = parts[0], parts[1]
                class_name = rel_path.split("/")[0]
                image_dict.setdefault(class_name, []).append((img_id, rel_path))

    # 输出到规范目录 splits/
    out_path = project_root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": train_classes,
        "val": val_classes,
        "test": test_classes
    }

    for split_name, cls_list in splits.items():
        # 1. 类别清单文件
        split_classes_file = out_path / f"{split_name}_classes.txt"
        with open(split_classes_file, "w", encoding="utf-8") as f:
            for cid, cname in cls_list:
                f.write(f"{cid} {cname}\n")

        # 2. 图像清单文件
        split_images_file = out_path / f"{split_name}_images.txt"
        total_images = 0
        with open(split_images_file, "w", encoding="utf-8") as f:
            for _, cname in cls_list:
                imgs = image_dict.get(cname, [])
                total_images += len(imgs)
                for img_id, rel_path in imgs:
                    f.write(f"{img_id} {cname} {rel_path}\n")

        print(f"[{split_name.upper()}] 划分完成: {len(cls_list)} 类, 共 {total_images} 张图像")

    print(f"\n划分清单已成功写入规范目录: {out_path.resolve()}")


if __name__ == "__main__":
    make_cub_splits()