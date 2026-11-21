# Stage 06：冻结配置的NABirds跨数据集评估

## 已确认的协议

明确选择自动保守筛选，而不是等待人工物种审核。本实现采用单独的`nabirds-auto-conservative-ebird2026-officialtest-v1`协议，`human_reviewed=false`，不伪造审核人，不更改原`NABirdsDataset`人工审核门槛。

- 使用NABirds官方test split（`train_test_split.txt`中0），不使用官方train图片。
- 官方原始数据48562张、555个视觉类别、1011个层级节点；不是将555个性别/年龄视觉类直接当成物种。
- 固定Cornell eBird taxonomy v2026快照，合并同一物种的视觉子类，排除与CUB Train语义重叠或无法可靠映射的目标物种。
- 当前池为317个物种、19307张官方test图片，每物种13～192张。555视觉类中90个因Train语义重叠排除，18个因未知/冲突旧分类名排除；其余447个视觉类合并成317个物种。
- 100个CUB Train类别全部解析到保守的科学分类排除集合，含171个全球eBird taxon code；这不表示171个NABirds物种被剔除。Sayornis、Geococcyx等属标签及历史拆分类用超集排除，名字不确定的目标类不自动批准。
- 检查全部候选JPEG SHA256；和任意CUB split完全相同的图片、跨物种内容重复均排除，同物种重复逻辑保留确定性一个文件。当前候选没有这些重复。原图和原始划分不修改。

当前种类映射、排除原因、Train语义解析、层级和物种样本数完整保存在`outputs/stage06_nabirds/protocol/manifest.json`、`visual_mapping_audit.csv`、`species_mapping.csv`。

自动名称/分类匹配不等于专家物种鉴定，仍可能有历史分类残余不确定性。结果应明确报告这个边界，也不应宣称相对于ImageNet外部预训练全部物种未见。

## 冻结内容与代码复用

- 沿用Stage04三个best.pth，SHA256必须与已完成CUB表示实验一致。
- Global：同一Layer4→GAP的原始512维，Support算术均值，再在Cosine公式内部除范数。
- Local：同一Layer4 49×512描述子；每个Query位置在候选类K×49描述子中取Top-3，49×3项取均值。
- Fusion：每Query五类分数的population z-score，eps=1e-6，固定0.5/0.5加权。没有Query标签参与参数选择。
- 沿用同一`representation_scores`函数及源码SHA256；既有归一化、Prototype、DN4和融合实现不改。
- 复用旧采样器/验证器生成一份NABirds 600对计划。三模型和三方法共享；两Shot的类别与Query完全一致，1-shot Support嵌套于5-shot Support。
- 冻结eval/inference_mode/FP32，关闭AMP和TF32；CUDA优先，不可用明确WARNING并记录CPU回退。
- CPU缓存按dataset、manifest、权重、预处理、模式、有序样本ID和软件版本绑定，绝不混用CUB或不同初始化；GPU只放图像Batch、Support及Query块，默认Batch8/chunk4。

新增`data/nabirds_automatic.py`与`engine/nabirds_transfer.py`；旧缓存收集函数只增加“新目标集尚无Global缓存”的可选分支，旧CUB调用行为保持不变。旧计划保存器按catalog实际类别数统计，不再硬编码50，原CUB的输出内容不变。

## 分类依据来源

Cornell官方eBird API固定v2026：

```text
https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=csv&locale=en&version=2026
```

CSV保存在`outputs/stage06_nabirds/preflight/ebird_taxonomy_v2026.csv`，SHA256为`93a8c9c94cfd541715d9847e38ec8e82afd29630ad21316bde55e6bd167a23ee`。版本列表也已保存。Cornell静态网页CSV下载曾返回403，实际成功来源为官方API，不是第三方改写表。

## 复现：先准备固定协议

已经准备好的目录不要重复覆盖。新机器需同步原NABirds、清洗CUB类别清单、固定taxonomy CSV及新代码。

```bash
uv run python scripts/prepare_stage06_nabirds.py \
  --data-root nabirds \
  --taxonomy outputs/stage06_nabirds/preflight/ebird_taxonomy_v2026.csv \
  --protocol-dir outputs/stage06_nabirds/protocol \
  --allow-automatic-screening
```

若原协议已存在，直接复用，不重新采样或覆盖。该工具记录每张图像UUID、原始视觉类、物种code、dataset局部标签及Episode 0～4映射、照片哈希与两Shot嵌套关系。所有规则和计划在首次NABirds模型前向之前固定。

## 先运行真实功能检查

```bash
uv run python -m pattern_recognition_coursework.engine.nabirds_transfer \
  --mode smoke --allow-automatic-screening \
  --protocol-dir outputs/stage06_nabirds/protocol \
  --cub-reference outputs/stage05_protonet/representation_full_seed42_600_cuda \
  --output-dir outputs/stage06_nabirds/smoke_imagenet_cuda \
  --device cuda --batch-size 8 --query-chunk 4 --workers 0
```

默认只执行ImageNet、每种Shot前3个原任务，不改任务清单。真实Layer4/GAP形状、匹配矩阵、Top-k、分数、预测和样本UUID都会保存。

## 再执行完整600任务

```bash
uv run python -m pattern_recognition_coursework.engine.nabirds_transfer \
  --mode full --allow-automatic-screening \
  --validation-report outputs/stage06_nabirds/smoke_imagenet_cuda/checks.json \
  --protocol-dir outputs/stage06_nabirds/protocol \
  --cub-reference outputs/stage05_protonet/representation_full_seed42_600_cuda \
  --output-dir outputs/stage06_nabirds/full_seed42_600_cuda \
  --device cuda --batch-size 8 --query-chunk 4 --workers 2
```

Stage06数据必须重新提取NABirds特征，不能复用CUB图像缓存；只复用原模型与算法。smoke子集缓存可以按ID与绑定安全复用到full，已有完整目标缓存也会复用。输出必须是全新目录；如目录已存在请换新名称，切勿删除旧证据来伪装首次实验。

默认三个Stage04路径沿用Stage05服务器Batch128实验；路径不同可传`--imagenet-checkpoint`、`--random-checkpoint`、`--simclr-checkpoint`，但文件SHA256必须相同。24GB服务器可提高仅控制提取吞吐的`--batch-size`，不影响冻结方法；现有2GB配置为8/4。

也可使用`uv run python scripts/run_stage06_nabirds.py`按顺序执行CUDA回归、smoke、通过后full。它用于本次固定目录，不覆盖已有运行。

## 输出与统计解释

- 每模型/表示/Shot 600个Accuracy，Mean、ddof=1样本标准差、Student-t 95%均值CI。
- 同一个NABirds Episode内对三种表示计算配对差和CI，保存`paired_comparison.csv`、逐Episode差值及配对图。
- `cub_vs_nabirds.csv`按同一Backbone/表示/Shot并列CUB和NABirds，给出NABirds−CUB均值差（百分点）。两个数据集任务不同，不能按episode_id跨数据集配对。类别池从CUB Val 50类变为NABirds 317类，不能将差值全部解释为纯视觉域偏移。
- 新数据的Global结果也必须计算，Stage06总共10800次新方法评分、270000条Query方法预测。
- `performance.json/status.json/failure.json`记录耗时、阶段、GPU峰值、缓存命中和异常；源Checkpoint和CUB旧结果哈希在前后核验。

所有结果独立保存在Stage06，Stage04/Stage05结果不覆盖。CUB Val曾用于开发配置和选Checkpoint；NABirds本阶段只做预先冻结配置的评估，不根据目标成绩搜索权重或选择训练轮次。
