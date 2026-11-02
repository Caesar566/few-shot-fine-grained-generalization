# NABirds 跨数据集测试协议

本项目只把 NABirds 用于训练结束后的跨数据集评估。CUB 训练、验证和超参数选择不能读取 NABirds 测试成绩。NABirds 官方介绍约 400 个物种，并把雌雄、年龄或外观型细分成 550 多个视觉类别；本研究的 5-way 类别单位是**人工复核后的物种**，不是原始视觉类别 ID。原始数据及使用条件见 [Cornell NABirds 页面](https://dl.allaboutbirds.org/nabirds) 和 [官方使用条款](https://dl.allaboutbirds.org/merlin-computer-vision-terms-of-use)。

官方页面需要填写下载表单并确认使用条款，下载入口不在公开网页正文中。取得官方压缩包后运行：

```powershell
uv run python scripts/import_nabirds.py <官方压缩包路径>
```

工具只写入工程内 `data/raw/nabirds/`，检查压缩包路径、类别层级、图像 ID、索引图像和元数据哈希，拒绝覆盖已有数据。该目录中会直接存在 `classes.txt`、`images.txt`、`image_class_labels.txt`、`hierarchy.txt` 和 `images/`。`hierarchy.txt` 每行是 `<child_class_id> <parent_class_id>`。

先生成审核模板：

```powershell
uv run python scripts/prepare_nabirds.py review
```

这会在工程内写入 `data/splits/nabirds_review.json`。每个有图像的视觉类别初始为 `pending`。`candidate_cub_overlap` 只提示名称或祖先名称的明显重叠；没有候选**不表示**已排除同物异名、亚种或类别层级重叠。审核人需要按物种名称、科学名、同义名和层级逐行判断。对可用类别填 `decision: "non_overlap"`、`reviewer`、`evidence`，并将 `species_class_id` 设为该视觉类别自身或其物种祖先。对可能重叠或无法确定的类别填 `decision: "exclude"` 并说明依据。`pending` 类别不会进入测试。

审核完成后构建清单：

```powershell
uv run python scripts/prepare_nabirds.py build --min-images 20
```

产物是 `data/splits/nabirds_test.json`，包含审核记录、源数据哈希、每张图像的相对路径与物种 ID。脚本拒绝明显重叠候选、父子类别同时入选、无审核说明、图像不足或缺失的情况。每个物种至少需要 `shot + query` 张图像；默认 20 张只是建表门槛，评估时还会检查当前 episode 配置是否可采样。

同一被选物种的所有有图像视觉子类都必须明确审核并归入同一个物种；留下某个性别或年龄子类为 `pending` 会拒绝该物种，以免只评估容易的外观型。

只在人工复核与清单生成后运行评估：

```powershell
uv run pattern-recognition-coursework evaluate --checkpoint outputs/runs/<运行名>/best.pt --dataset nabirds --data-root data/raw/nabirds --manifest data/splits/nabirds_test.json --review data/splits/nabirds_review.json --export-errors
```

结果会记录原始清单和 episode 计划的哈希。若无法获得 NABirds 原始数据或可信物种审核，不应把 CUB 未见类别测试称为跨数据集结果。
