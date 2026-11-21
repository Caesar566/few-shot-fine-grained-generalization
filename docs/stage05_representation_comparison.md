# Stage 05：Global / Local / Fusion 表示对比

本次新增独立评估入口，保留训练、距离度量、原DN4和旧未校准Fusion模块。只做冻结前向，不创建优化器，不反向传播，不读NABirds或CUB Test图片。

## 文件与复用

- `models/metrics/representation.py`：复用原`prototype_scores`和`local_descriptor_scores`，添加Query分块、固定行内分数校准及0.5融合。
- `engine/representation_comparison.py`：原600任务/Global结果绑定、CPU缓存、CUDA-first推理、smoke→full门槛、统计与输出。
- `tests/test_representation_comparison.py`：Top-k手算、分块等价、校准、融合、CPU/CUDA一致性、缓存ID/权重隔离和显式回退。

## 固定定义

**Global**：原GAP512维、原始Support均值Prototype、Cosine公式内范数除法。直接复用距离对比中的Cosine结果，同时在当前GPU上重新计算分数以供Fusion使用并核对旧预测。完整600模式下表中Global统计直接引用旧结果。

**Local**：同一ResNet-18 Layer4 `[N,512,7,7]`，转为每图`[49,512]`。仅在余弦匹配公式内部归一化描述子。候选类集合包含该类K张Support的全部`K*49`个描述子，K=1时49个，K=5时245个。

```text
sim[q,h,c,j] = cosine(query_descriptor[q,h], support_descriptor[c,j])
local_score[q,c] = mean_h mean_top3_j sim[q,h,c,j]
```

Top-3从候选类所有Support描述子中选取，可来自同一张Support图片。对49个Query位置、每个位置的3个相似度求均值，不求和，避免分数随描述子数或Top-k改变而线性增大。默认Query chunk=4，并逐类匹配，最大相似度张量`[4,49,245]`。

**Fusion**：预先固定的无标签行内z-score，权重固定0.5，无权重调参、温度拟合或可学习模块。

```text
z(s) = (s - mean_across_5_classes(s)) / max(population_std_across_5_classes(s), 1e-6)
fusion_score = 0.5 * z(global_score) + 0.5 * z(local_score)
```

每张Query独立校准，不用其他Query，不用真实标签。常数分数行变成零向量。校准只作用于分数，不作用于图像特征或Prototype。旧`FewShotModel.global_local`没有这一校准，本次不改变它的行为。

## GPU 与显存

- 默认`--device auto`：优先CUDA。可指定`cuda/cpu`；CUDA不可用会在终端及run.log明确WARNING，记录请求与实际设备，绝不静默回退。
- 推理、Global均值/Cosine、DN4相似度/Top-k、校准、Fusion、Softmax与预测都在实际设备上执行PyTorch Tensor运算；没有用NumPy执行批量距离计算。CPU只做图像预处理、缓存、文件保存与标量统计聚合。
- 默认FP32、关闭AMP/TF32、eval/inference_mode；Batch Size=8，Query chunk=4，适配2GB显存。
- Local完整缓存只放CPU。每模型最多约282 MiB的Layer4特征，逐模型处理；GPU只放一个图像Batch或一个Episode Support和Query块，不上传全Val Local缓存。
- 每个缓存绑定权重SHA256、编码器state、预处理、实际图片内容指纹、有序样本ID、源码、dtype、模式、设备及版本。缓存命中后仍执行真实设备前向探针，核对Layer4→GAP与完整encoder输出。
- `performance.json`记录提取/分类耗时、命中情况和PyTorch峰值allocated/reserved显存。这两个峰值不包含驱动上下文或其他进程的GPU内存。

## 先运行最小功能验证

先同步两个新增src文件和本说明到服务器，确保原始CUB/清洗索引、三份best.pth、两个旧实验目录和三份原Global缓存仍在。缓存`.pt`不在Git内，跨机器需额外传输或使用服务器原有缓存。

```bash
uv run python -m pattern_recognition_coursework.engine.representation_comparison \
  --mode smoke \
  --output-dir outputs/stage05_protonet/representation_smoke_server \
  --episode-source-dir outputs/stage05_protonet/episodic_eval_seed42_600 \
  --global-baseline-dir outputs/stage05_protonet/distance_comparison_seed42_600 \
  --baseline-cache-dir data/cache/stage05_features \
  --device cuda --batch-size 8 --query-chunk 4 --smoke-episodes 3 --workers 0
```

默认ImageNet、1-shot/5-shot各原计划前3个Episode，不重新采样。保存完整600计划，同时用`used_episode_ids`明确本次只执行0、1、2。该模式额外用CPU重新前向每种Shot首个Episode，记录数值误差和预测是否一致。少量CPU诊断不是批量评估默认CPU计算。

GPU与CPU的FP32卷积不保证逐位一致：Layer4校验采用明确记录的`rtol=2e-4, atol=2e-4`；分数另做更严格核对，最终预测要求一致。原Global GPU与旧CPU结果逐Episode核对，若预测不一致就停止，不用旧准确率掩盖数值问题。

## 完整600-Episode命令

功能验证必须先通过，full入口要求指定通过的checks.json，且核心方法、校准参数、源码哈希及ImageNet权重必须匹配：

```bash
uv run python -m pattern_recognition_coursework.engine.representation_comparison \
  --mode full \
  --validation-report outputs/stage05_protonet/representation_smoke_server/checks.json \
  --output-dir outputs/stage05_protonet/representation_full_seed42_600_server \
  --episode-source-dir outputs/stage05_protonet/episodic_eval_seed42_600 \
  --global-baseline-dir outputs/stage05_protonet/distance_comparison_seed42_600 \
  --baseline-cache-dir data/cache/stage05_features \
  --device cuda --batch-size 8 --query-chunk 4 --workers 4
```

把输入目录换成实际旧实验目录；若服务器目录有`_server`后缀，请明确替换。不要将单Episode的早期演示目录当成600任务来源。

full固定三种Backbone，1-shot/5-shot各600任务。可通过`--imagenet-checkpoint/--random-checkpoint/--simclr-checkpoint`定位原best.pth，但SHA256必须与Global基线一致。`--cache-dir`改变新增Local缓存目录；`--batch-size`、`--query-chunk`仅调整吞吐/显存，不改变任务或Top-k公式。24GB服务器可以在验证后尝试更大编码Batch，如32；不需要改学习率，因为不训练。

所有输出目录必须不存在，不覆盖旧记录。运行中OOM时从新目录以更小Batch/chunk重跑，已完成且绑定匹配的缓存可复用。

## 输出与统计

- `comparison.csv/summary.json/report.md`：完整模式18个配置的Mean、ddof=1样本标准差、Student-t 95% CI和图表。
- `paired_comparison.csv`：同一Episode的Local−Global、Fusion−Global、Fusion−Local配对差及CI。
- 每模型/Shot：三路逐Episode Accuracy、统一完整预测JSONL（样本ID、映射、Support/Query、全部logits、概率与预测、校准分数）。
- 每种Shot首个Episode：`similarity_q0_class0.csv`、Top-k邻居索引/值、Tensor Shape与分数范围。
- 缓存绑定、全套旧计划副本、源码哈希、实际设备、耗时、显存、功能核验与日志。

smoke仅用于功能正确性；三次任务不能用于判断Fusion优劣。完整评估仍是曾用于选Checkpoint的CUB Val，不是独立最终测试或NABirds跨数据集结果。

## 当前验证证据

本地MX450 CUDA验证保存在：`outputs/stage05_protonet/representation_smoke_imagenet_cuda_v2/`。

- 实际FP32 CUDA前向：`[8,3,224,224] → [8,512,7,7] → [8,512]`。
- 两种Shot各3个真实任务通过，首个任务CPU/GPU三路预测相同。
- PyTorch峰值allocated约97.9 MiB，reserved约194 MiB；本次Local缓存命中，仍执行实际GPU前向。
- 真实对照Layer4最大绝对差约1.14e-4；分数最大绝对差约8.05e-6，未改变计算精度。
- 首轮较严格CPU/GPU激活容差失败保存在`representation_smoke_imagenet_cuda/failure.log`，未覆盖。合成近常数分数的校准测试使用FP32误差传播界限，仍严格要求raw score接近、最终预测相同，未改校准公式或参数。
- CUDA回归17项通过，含缓存身份隔离及显式回退；沙箱内CPU测试为15项通过、2项CUDA不可访问而跳过。

上述为前一轮小规模验证记录。后续完整实验现已执行并复核，见下节。

## 完整实验结果（2026-11-18）

结果目录：`outputs/stage05_protonet/representation_full_seed42_600_cuda/`，完整统计见该目录`report.md`和`comparison.csv`。

- 三种Backbone、两种Shot、每个方法600个Episode，共10800次方法评分、270000条Query方法预测；Global 3600项沿用旧Cosine结果，Local/Fusion新增7200项。
- 实际使用MX450 CUDA，FP32，Batch Size 8，Query chunk 4；方法源码哈希、校准eps、Top-k、0.5权重与smoke一致。
- 完整评估116.77秒，独立CUDA复核及图表导出另用约16.80秒。峰值PyTorch allocated约129.99 MiB、reserved约220 MiB，无OOM、无CPU回退。
- 新增部分缓存复用：按相同权重、模式、预处理和ID校验后，ImageNet复用smoke的150张Local特征，只提取其余2796张；Random/SimCLR各提取2946张任务实际使用的图片。未使用的1张Val图片仍保留原划分，只是没有被此次600任务抽中。
- 完整缓存为CPU Tensor；GPU仍只放图像Batch、Episode Support及Query块。原Global缓存、训练权重和旧实验目录逐文件哈希保持不变。
- 19项CUDA回归测试通过。独立进程复核全部预测、校准与统计，并在CUDA上对18个首/中/末Episode重新计算三路原始公式，逐位复现。
- 本次固定Fusion没有稳定超过Global；结果如实保留，没有用Val标签调权重或Top-k。

完整输出新增`accuracy_bars.svg/png`、`paired_lifts.svg/png`和`paired_episode_differences.csv`，以及`performance.json`、`status.json`、独立核验证据和实际日志。异常或中断会写`failure.json`，并保护已有输出目录不被改写。
