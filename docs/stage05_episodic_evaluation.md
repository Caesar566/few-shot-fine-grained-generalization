# Stage 05：CUB Val 批量成对 ProtoNet 评估

本入口只评估，不训练。沿用阶段5的全局512维特征与`-平方欧氏距离/512`，温度1，无特征归一化、标准化或Adapter。阶段4训练代码、原ProtoNet和数据划分不改。

## 已有功能与本次扩展

| 能力 | 处理方式 |
| --- | --- |
| 无放回Episode采样 | 复用 `FewShotBatchSampler` |
| 清单数据指纹、类序、数量检查 | 复用 `dataset_fingerprint/validate_plan/plan_hash` |
| 冻结ResNet特征与原型计算 | 复用 `FewShotModel.encode/classify_embeddings` |
| Accuracy、NLL、样本标准差等 | 复用 `evaluation_metrics/aggregate_episode_metrics` |
| 两种Shot严格配对 | 新增 `data/paired_episodes.py` |
| Student-t Episode均值置信区间 | 新增 `analysis/episodic_statistics.py` |
| 监督Checkpoint加载、缓存、批量预测、图表 | 新增 `engine/episodic_evaluation.py` |

原 `engine/evaluate.py` 面向含 `model_config/training_config` 的旧Episodic Checkpoint，不能直接读取阶段4监督Checkpoint，因此本次使用独立入口，保持原入口行为不变。

## 任务协议

- 数据仅为清洗后的 CUB Val：50类、2947张；不读取 CUB Test 或 NABirds 图片。
- 固定5-way，每类Query为5张，共25张Query。
- 每种Shot默认600个Episode，采样seed默认42。
- 先由现有采样器生成5-shot任务，每类Support 5张、Query 5张；1-shot保留每类第1张Support和完全相同的Query。
- 两种Shot的类序完全相同，1-shot Support是5-shot Support子集；三种初始化共用同一份计划。
- 每个Episode内部索引无重复；所有Val实际图片SHA256与清洗索引一致，且Val内部内容哈希唯一，从而保证Support/Query内容互斥。不同Episode可以重复使用图片。

## 启动命令

在服务器或本地项目根目录执行，输出目录必须尚不存在：

```bash
uv run python -m pattern_recognition_coursework.engine.episodic_evaluation \
  --output-dir outputs/stage05_protonet/episodic_eval_seed42_600_server \
  --episodes 600 --seed 42 \
  --device cuda --batch-size 64 --workers 4
```

本地CPU可用`--device cpu --workers 0`。`--batch-size`控制特征提取，不改变5-way/shot/query协议，也不涉及训练Batch Size。第一次执行需要对每个模型提取全部Val特征；后续同配置可读取缓存。

默认Checkpoint：

```text
outputs/stage04_training/supervised_imagenet_seed42_server_bs128_fresh/checkpoints/best.pth
outputs/stage04_training/supervised_random_seed42_bs128/checkpoints/best.pth
outputs/stage04_training/supervised_simclr_seed42_bs128/checkpoints/best.pth
```

也可显式指定三份文件：

```bash
uv run python -m pattern_recognition_coursework.engine.episodic_evaluation \
  --output-dir outputs/stage05_protonet/episodic_eval_custom \
  --episodes 600 --seed 42 --device cuda --batch-size 64 --workers 4 \
  --imagenet-checkpoint outputs/stage04_training/supervised_imagenet_seed42_server_bs128_fresh/checkpoints/best.pth \
  --random-checkpoint outputs/stage04_training/supervised_random_seed42_bs128/checkpoints/best.pth \
  --simclr-checkpoint outputs/stage04_training/supervised_simclr_seed42_bs128/checkpoints/best.pth
```

权重只加载`model`下`backbone.`前缀，与Checkpoint内编码器state逐项精确核对，监督分类头明确排除。没有Git元信息也可以运行，源码版本通过SHA256记录。

## 缓存

默认缓存位置`data/cache/stage05_features/`，已被Git忽略。每个模型`[2947,512]` float32特征约5.76 MiB，三组共约17.27 MiB，不缓存全部原图或全部增强Tensor。

缓存绑定完整Checkpoint SHA256、编码器state、实际数据内容和顺序、Transform、软件版本、eval/frozen/float32、设备环境与编码批量大小。旧缓存必须通过binding、shape、dtype、finite、内容校验才能读取。

```bash
# 使用其他缓存目录（建议保持在工程目录中）
uv run python -m pattern_recognition_coursework.engine.episodic_evaluation \
  --output-dir outputs/stage05_protonet/episodic_eval_cache_reuse \
  --episodes 600 --seed 42 --device cpu --workers 0 \
  --cache-dir data/cache/stage05_features
```

`--no-cache`关闭磁盘缓存读写，但仍只提取一次全量冻结特征并在内存中复用。每个模型和Shot的前3个Episode通过实际图片直接前向及原`run_episodes`核验缓存预测；每个设置的全部600个Accuracy再重复计算验证一致。编码器保持eval，所有参数冻结，无优化器。

## 指标

令`a_e = correct_queries/25`，默认`n=600`：

```text
Mean Accuracy = mean(a_e)
Sample Std    = std(a_e, ddof=1)
SEM           = Sample Std / sqrt(n)
95% CI        = Mean Accuracy ± t.ppf(0.975, n-1) × SEM
```

JSON以0～1存储Accuracy，CSV与报告的均值、CI用百分数，标准差与半宽用百分点。CI统计单位为Episode，不将所有Query视为独立样本。

该CI只反映固定模型、固定CUB Val上采样任务的波动，不包含训练seed不确定性或跨数据集泛化不确定性。阶段4已用CUB Val选择Checkpoint，本次是验证集评估，不应称为独立最终测试，也不能与之前cosine评分的结果直接混合。

## 产物

```text
<output-dir>/
  report.md / comparison.csv / summary.json
  accuracy_distribution.svg / histogram_counts.csv
  config.json / environment.json / code_version.json / command.json
  data_audit.json / image_catalog.csv
  plan_1shot.json / plan_5shot.json / paired_episodes.jsonl / plan_checks.json
  checkpoints.json / checks.json / run.log
  <imagenet|random|simclr>/
    cache.json / direct_verification.json
    <1shot|5shot>/episodes.csv / metrics.json / predictions.jsonl
```

`paired_episodes.jsonl`包含每个任务的Support/Query ID、路径、内容哈希、原始标签与0～4局部映射。`predictions.jsonl`每行一个Episode，包含全部25张Query的真实/预测类别与5个logits、概率；通过episode_id连接其Support清单。

## 回归测试

```bash
uv run python -m pytest tests/test_paired_episodic_evaluation.py \
  tests/test_stage05_protonet.py tests/test_episode_plan_and_nabirds.py \
  tests/test_episode_metrics.py tests/test_few_shot_baselines.py -q
```

覆盖600对任务的一致性/子集关系、种子复现、Student-t和ddof=1、缓存创建与复用、训练模式拒绝、权重/预处理/数据/模式不匹配缓存拒绝、损坏特征拒绝及原有数学与指标回归。
