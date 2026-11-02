# A03 小样本细粒度鸟类分类的跨数据集泛化

工程以 [开题报告](docs/reference/A03_模式识别开题报告.pdf) 为研究协议。主线是 ResNet-18 + ProtoNet 全局原型、DN4 式局部描述子匹配及两者融合；Matching Networks、监督最近类中心、距离与预训练来源是对照。**5 个百分点是 NABirds 跨数据集测试的研究目标，当前没有正式实验结论。**

## 已实现

- CUB-200-2011 的 100/50/50 类清单、官方元数据校验、可重复且 support/query 无交集的 episode 计划。
- ResNet-18 全局原型、平方欧氏/余弦/收缩马氏距离、简化匹配注意力、带上下文编码的 Matching Networks、DN4 式局部匹配和全局局部融合；可选 support 统计量标准化、温度与线性适配器。
- episodic、监督特征学习和监督后 episodic 微调三种训练路径。仅用固定验证任务选最佳检查点；测试任务保存为 JSON，能在独立评估命令中复用。
- NABirds 的层级类别和物种审核工具。它先生成待人工复核清单，只有明确确认不重叠的物种才能生成目标域测试清单。评估记录准确率、NLL、macro-F1、ECE、局部任务槽位混淆矩阵和可选错误样本。
- 至少 3 个训练种子的配对置信区间与 CUB→NABirds 泛化差距汇总。比较前核对测试 episode 计划、清单哈希和实验协议。

已完成一轮 CUB 未见类 3 种子基线与固定融合权重比较，协议、原始结果和限制见 [CUB 第一轮实验](reports/cub_study_v1.md)。该轮 1-shot 未见提升，5-shot 的小幅差异也不能排除零差异。

第二轮 [CUB 验证集优化](reports/cub_optimization_v2.md) 扫描了 1-shot 融合权重，并分别训练了 5-shot ProtoNet 与固定权重融合模型。在当前短训练、冻结主干设置下，1-shot 验证集选择全局特征，5-shot 权重 0.5 的融合在三个种子上也未优于 ProtoNet。该轮没有新的测试结果；原先的 CUB 测试划分已被查看，后续复评需标为探索性。

[CUB 范围进展汇总](reports/cub_progress_2026-10-23.md) 集中列出匹配方式、距离、随机/监督/SimCLR 初始化、适配范围和第一轮错误审计的具体结果与未完成项。它区分验证结果、已查看的第一轮测试结果及仍需人工核验的错误因素。

[CUB 后续验证](reports/cub_followup_2026-10-23.md) 补齐可训练上下文 Matching Networks 的 1-shot/5-shot 三种子对照、监督最近类中心基线、较长训练预算及 8 张错误图像的人工观察。所有新增模型数字均为验证集结果；在当前配置下尚未看到相对全局余弦 ProtoNet 的稳定收益。

## 数据

Python 要求 3.11。CUB 解压后目录为 `data/raw/CUB_200_2011/`，该目录直接包含 `classes.txt`、`images.txt`、`image_class_labels.txt` 和 `images/`。版本化划分文件位于 `data/splits/`。NABirds 数据默认放在 `data/raw/nabirds/`，准备方法见 [NABirds 协议](docs/nabirds_protocol.md)。原始图像、下载缓存和实验权重不会纳入 Git。

下载和校验：

```powershell
uv sync
uv run python scripts/validate_cub.py --check-images
uv run python -m pytest -q
```

CUB 已存在，无需重复下载。`scripts/dataset_get.py` 仅在缺少 CUB 时从 KaggleHub 获取副本，并把缓存固定在工程内。NABirds 官方下载页面需要填写表单并确认使用条款；取得官方压缩包后运行 `uv run python scripts/import_nabirds.py <压缩包路径>`，程序会将其放入 `data/raw/nabirds/` 并校验。报告中的局部特征示意使用 224×224 输入，训练入口也以 224 为默认值。小显存设备可通过 `--image-size 84 --n-query 1` 做贯通检查；不同分辨率产生的成绩不得直接比较。

## 训练与复评

```powershell
uv run pattern-recognition-coursework train --method protonet --metric euclidean --k-shot 1 --seed 42
uv run pattern-recognition-coursework train --method dn4 --k-shot 1 --seed 42
uv run pattern-recognition-coursework train --method global_local --metric cosine --local-weight 0.5 --k-shot 1 --seed 42
uv run pattern-recognition-coursework train --stage meta_baseline --method protonet --metric cosine --k-shot 1 --seed 42
```

训练输出保存在 `outputs/runs/<运行名>/`，包括 `best.pt`、`metrics.json` 和逐轮记录；共享验证/测试任务在 `outputs/episodes/`。本地自监督权重可用 `--backbone-weights <文件> --backbone-prefix <明确前缀> --weight-source <算法、训练数据和来源>` 加载，编码器张量必须完整匹配。ImageNet 监督权重使用 `--pretrained`，其缓存也写入工程目录。

搜索融合权重等超参数时，在每个候选训练命令后加 `--skip-test`。所有候选以相同的随机种子和固定验证任务训练；将各运行的 `metrics.json` 传给 `pattern-recognition-coursework select --results <候选结果...> --output reports/fusion_selection.json`。该命令至少要求两个权重、每个权重三个相同种子，并拒绝读过测试集的结果。选定权重后，才使用对应 `best.pt` 进行独立测试。

保存的检查点可使用 `evaluate --dataset cub --split val` 在验证类上复评另一种 shot。输出明确标为 `validation_only`；默认 `--split test`，不能把验证集复评写成测试结果。自监督对照所用权重可用 `scripts/convert_simclr_resnet18.py` 从已下载的 Lightning 检查点严格提取，来源和哈希见 `reports/records/simclr_weight_provenance.json`。

使用保存的检查点独立复评，可将 1-shot 模型同时用于 5-shot 测试；报告时注明检查点依据哪种 shot 的验证集选出。以下路径由实际训练输出替换：

```powershell
uv run pattern-recognition-coursework evaluate --checkpoint outputs/runs/<运行名>/best.pt --dataset cub --k-shot 1 --export-errors
uv run pattern-recognition-coursework evaluate --checkpoint outputs/runs/<运行名>/best.pt --dataset nabirds --data-root data/raw/nabirds --manifest data/splits/nabirds_test.json --review data/splits/nabirds_review.json --k-shot 1 --export-errors
```

对两组方法各运行至少 3 个相同随机种子后，将对应 JSON 传给 `pattern-recognition-coursework summarize --baseline <基线结果...> --candidate <候选结果...> --output reports/paired_summary.json`。用 `--mode gap` 时，`--baseline` 输入 CUB 结果、`--candidate` 输入同一检查点的 NABirds 结果。CUB 内未见类别成绩只说明跨类别泛化；5 个百分点目标只在 NABirds 场景判断。

## 尚需数据与实验

本地没有 NABirds 原始图像，也没有完成人工核查的非重叠物种表；因此跨数据集结果和最终 5 个百分点结论尚不能给出。CUB 范围已完成一份可追溯 SimCLR ResNet-18 权重的短训验证对照，但其训练配方与监督权重不同，不能将差异仅归因于预训练方式。CUB 第一轮实验使用了固定的 1-shot/5-shot 测试任务、3 个训练随机种子和可追溯的 ImageNet 权重；本轮进一步完成了 ProtoNet 800-episode 长预算对照，但尚未进行大预算的骨干解冻训练。预训练数据可能与 CUB 图像来源重合，报告时应列为潜在偏差。
