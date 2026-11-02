# CUB 未见类第一轮实验（2026-10-22）

## 问题与范围

比较 ResNet-18 + ProtoNet 全局特征基线与 DN4 式局部描述子和全局特征融合，场景仅为 CUB-200-2011 **训练、验证、测试物种互斥**的未见类分类。这里没有 NABirds 原始图像，因此不是跨数据集结果，也不能判断开题报告中的跨数据集 5 个百分点目标。

## 固定协议

| 项目 | 设置 |
| --- | --- |
| CUB 划分 | train 100 类 / 5890 图，val 50 类 / 2947 图，test 50 类 / 2951 图；官方元数据和索引图像逐项校验 |
| 任务 | 5-way；每类 1 个 query；训练 1-shot；独立复评 1-shot 和 5-shot |
| 初始化 | torchvision `ResNet18_Weights.IMAGENET1K_V1`，下载地址 `https://download.pytorch.org/models/resnet18-f37072fd.pth`，SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec` |
| 输入与更新 | 224×224；冻结 ResNet-18，仅训练恒等初始化的 512×512 线性适配层；Adam，学习率 1e-4 |
| 方法 | ProtoNet 余弦原型；全局局部融合使用余弦原型 + 局部 top-3 描述子匹配，固定融合权重 0.5 |
| 训练和选择 | 种子 42、43、44；每种子 3 epoch，每 epoch 30 个训练 episode；每 epoch 在同一组 30 个验证 episode 上选最佳检查点，训练入口使用 `--skip-test` |
| 测试 | 每种子、每 shot 各 200 个固定 test episode，即 1000 个 query 决策；相同种子的两种方法使用完全相同的测试任务计划 |
| 不确定性 | 以 3 个配对训练种子为统计单位，使用双侧 95% Student t 区间；测试 episode 与图像可能重复出现，不能将 1000 个 query 当作独立种子 |

5-shot 是用**按 1-shot 验证任务选出的相同检查点**直接复评，不是单独进行 5-shot 训练。训练和验证任务的图像采样协议、测试任务哈希、权重来源及软件版本均保存在 `outputs/cub_study_v1/` 的各运行 JSON 中。

## 结果

准确率和差值均为百分比或百分点。原始结果位于 `outputs/cub_study_v1/evaluations/`，配对汇总为 [1-shot JSON](records/cub_study_v1_1shot_summary.json) 和 [5-shot JSON](records/cub_study_v1_5shot_summary.json)。

| 测试设置 | ProtoNet 平均准确率（95% CI） | 融合模型平均准确率（95% CI） | 配对差值：融合 − ProtoNet（95% CI） |
| --- | ---: | ---: | ---: |
| 5-way 1-shot | 80.67%（77.36–83.97） | 79.43%（76.82–82.04） | −1.23 pp（−3.03–0.56） |
| 5-way 5-shot | 91.20%（89.46–92.94） | 91.77%（90.83–92.71） | +0.57 pp（−0.23–1.37） |

逐种子准确率：

| 种子 | 1-shot ProtoNet | 1-shot 融合 | 5-shot ProtoNet | 5-shot 融合 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 80.0% | 78.4% | 90.9% | 91.6% |
| 43 | 79.8% | 79.4% | 92.0% | 92.2% |
| 44 | 82.2% | 80.5% | 90.7% | 91.5% |

1-shot 的固定融合方案在三个种子上均低于 ProtoNet；5-shot 的微小正差值不足以排除零差异。当前数据**不支持**“局部融合已提升 CUB 准确率”的结论。更不能据此宣称达到跨数据集目标。

## 错误与限制

- 每个独立评估结果都保存了 `*_queries.jsonl`，记录查询图像、真实/预测类别、置信度及 support 图像，可用于后续人工标注姿态、光照、遮挡等因素。现在只能定位错误，尚未做可核查的错误成因归类。
- 三个种子和每类一个 query 的协议可作第一轮比较，但区间仍宽；短训练（每种子仅 90 个训练 episode）也不足以代表充分收敛。不能把本轮数值当作不同论文或完整训练协议的直接可比成绩。
- 输入图像来自 CUB；ImageNet 预训练数据与 CUB 是否有源图像重合尚未核实。预训练对照还缺随机初始化与可追溯的 SimCLR 权重。
- 0.5 融合权重在测试前固定，没有用测试结果调参。今后若搜索权重，只能用验证任务选择，并需清楚区分这一轮已查看的测试结果与新的预注册评估。
- 原始 NABirds 图像和人工复核的非重叠物种清单仍缺，跨数据集实验尚未开始。

## 复现入口

在项目根目录、已安装依赖且 CUB 数据完整时，以下命令展示单个训练运行；改变 `--method` 和 `--seed` 即为其他五个运行。局部融合运行额外传 `--local-weight 0.5`。

```powershell
.\.venv\Scripts\pattern-recognition-coursework.exe train --method protonet --metric cosine --pretrained --adapter --trainable-scope adapter --n-way 5 --k-shot 1 --n-query 1 --image-size 224 --train-episodes 30 --eval-episodes 30 --epochs 3 --seed 42 --device cuda --skip-test --output-dir outputs/cub_study_v1
```

随后使用各运行的 `best.pt` 执行 `evaluate --dataset cub --k-shot 1` 和 `--k-shot 5`，两者均指定 `--n-query 1 --episodes 200 --output-dir outputs/cub_study_v1 --export-errors`。运行目录和评估 JSON 内包含检查点路径及哈希；配对比较由 `pattern-recognition-coursework summarize` 验证相同种子、划分、预处理和测试任务计划。
