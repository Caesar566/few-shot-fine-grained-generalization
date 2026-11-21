# Stage 05：固定特征与Episode的距离对比

## 范围和实现

本次只改变距离。直接读取第四阶段三个最佳完整Checkpoint和上一轮欧氏基线的600对Episode、原始512维Val特征缓存。欧氏结果只引用旧文件，新增Cosine与Train共享对角Mahalanobis，绝不重新训练Backbone。

已有 `models/metrics/episode.py` 的原型与欧氏实现不改。该文件的Cosine分支已满足“原始Support均值→公式内范数除法”，直接复用。已有的Episode-Support收缩马氏分支保留不动，本实验另用Train类内共享对角版本，避免在1-shot/5-shot上估计高维协方差。

新增/扩展的文件：

- `engine/distance_comparison.py`：加载固定基线、旧缓存、Train特征提取、完整评估、配对比较和图表。
- `models/metrics/shared_covariance.py`：仅Train标签与特征的类内方差估计，正则化及逐元素马氏距离；无矩阵求逆。
- `analysis/episodic_statistics.py`：追加配对Accuracy差值的样本标准差与Student-t区间，不改变原统计函数。
- `tests/test_distance_comparison.py`：人工向量、原型顺序、类内/类间差异、零方差、禁止inverse、配对差值方向与统计测试。

## 输入必须完整

需要：

1. 第四阶段三个最佳 `best.pth`。
2. 上一轮完整欧氏评估目录（默认为 `outputs/stage05_protonet/episodic_eval_seed42_600/`），包括计划、预测、统计和每个模型的cache.json。
3. 上一轮三份Val特征缓存 `.pt`（默认 `data/cache/stage05_features/`，约17.3 MiB）。它们被Git忽略，迁移服务器时需要另外复制。
4. 原始CUB数据和原来的清洗索引。

旧cache.json里的绝对路径可能来自笔记本；程序优先在 `--baseline-cache-dir` 中按原cache key查找，再尝试旧绝对路径。找不到原缓存就报错，不在新硬件上悄悄重提Val特征。每个缓存都验证binding、SHA256、Tensor Shape、dtype和finite，编码器及预处理源码也要与基线一致。

## 命令

在项目根目录执行，替换为实际旧评估目录。若上一轮服务器输出目录带 `_server` 后缀，应在 `--baseline-dir` 中使用该路径。

```bash
uv run python -m pattern_recognition_coursework.engine.distance_comparison \
  --baseline-dir outputs/stage05_protonet/episodic_eval_seed42_600 \
  --baseline-cache-dir data/cache/stage05_features \
  --output-dir outputs/stage05_protonet/distance_comparison_server \
  --device cuda --batch-size 64 --workers 4
```

本地CPU可用 `--device cpu --workers 0 --threads 4`。输出目录必须不存在。默认使用之前三组服务器Batch Size128实验的best.pth；位置不同可传：

```text
--imagenet-checkpoint <ImageNet实验的best.pth>
--random-checkpoint <Random实验的best.pth>
--simclr-checkpoint <SimCLR实验的best.pth>
```

Checkpoint必须与旧欧氏缓存中的SHA256完全相同，不能换成last.pth或其他轮次。编码批量 `--batch-size` 仅控制本次Train特征提取，不改变Episode；`--cache-dir` 默认为 `data/cache/stage05_metric_train_features/`，该目录已由现有Git规则忽略。

## 完全固定的内容

- 原来的两个plan不重新采样，逐字节复制；还核对其原始配置中的plan hash。
- 三个Backbone共享同一组任务；Shot之间类别和Query相同，1-shot Support为5-shot子集。
- 原始Val特征逐位复用旧缓存，不做预先L2归一化、标准化、白化或Adapter。
- `prototype = support.reshape(5,K,512).mean(1)`，保持原始float32均值。
- Query也使用原始512维特征。余弦公式内的范数除法不是Support预归一化。
- RGB/Resize257/CenterCrop224/ToTensor/ImageNet像素Normalize与原实验相同。
- 每个模型/距离/Shot600个Episode，温度固定1。

## 距离与共享协方差

| 距离 | 计算 |
| --- | --- |
| Euclidean | 旧结果，`logit=-sum((q-p)^2)/512` |
| Cosine | `cos(q,p)=dot(q,p)/(norm(q)*norm(p))`，内部稳定eps为1e-12；`logit=cos`，等价于负`1-cos`距离加一个公共常数 |
| Mahalanobis | `logit=-sum((q-p)^2/v)/512`，其中v为该Backbone的Train共享正对角协方差 |

共享协方差对每个Backbone分别估计一次，同一组v用于两种Shot的全部Episode：

```text
Train原始特征与Train标签 → 每类均值mu_c
S_d = sum_i (z_i[d] - mu_yi[d])² / (N-C)
scale = mean_d(S_d)
jitter = max(absolute_jitter, relative_jitter * scale)
v_d = (1-alpha)*S_d + alpha*scale + jitter
```

本次固定默认 `alpha=0.1`、`relative_jitter=1e-4`、`absolute_jitter=1e-8`，不做参数搜索。N=5890、C=100，自由度5790。训练图片仍采用确定性eval预处理，冻结参数；标签仅用于估计类内残差，不涉及梯度更新。估计器不接收Val/Query特征或标签，不接收NABirds信息。

协方差统计和马氏距离内部累加用float64，原型仍先由原始float32特征求均值；最终logits为float32。通过正则化确保v有限且严格为正，只做逐元素除法，无少样本矩阵求逆。

## 统计与配对

每个配置保存600个Accuracy。均值、ddof=1样本标准差和Student-t 95% CI沿用既有统计口径。

距离之间的差值按同一个episode_id配对：`delta_e=Accuracy_second,e-Accuracy_first,e`；对600个delta计算均值、样本标准差和95% CI。三个距离两两比较，共18个配对设置。均值差以百分点报告，不把两组独立CI相减。结果为描述性对比，不做多重比较显著性结论。

总表18行：3初始化×3距离×2Shot。共10800次Episode评分，其中3600次旧欧氏结果直接引用，本次新增7200次、180000条Query预测。

## 产物和边界

新目录保存：

- `comparison.csv`、`summary.json`、`accuracy_comparison.svg`：完整18配置结果。
- `paired_comparison.csv`、`paired_episode_differences.csv`、`paired_differences.svg`：配对差值与CI。
- 复制的两个计划、配对样本ID/标签映射及图片catalog。
- 每个模型的Train缓存绑定、原始/正则化方差、样本计数、条件数、Checkpoint和Val特征绑定。
- 新增距离逐Episode Accuracy、完整Query预测（含ID、标签、类别、全部logits和概率）。
- `baseline_references.json`与`baseline_file_hashes.json`：旧欧氏预测位置及完整性证据；旧日志不覆盖。
- `config.json`、源码SHA256、实际运行日志、验收和report.md。

此实验仍是CUB Val验证集比较，阶段4已用这批Val选择Checkpoint；不能称为独立最终测试或NABirds跨数据集成绩。CI只反映固定模型和数据下的Episode采样波动。

```bash
uv run python -m pytest tests/test_distance_comparison.py \
  tests/test_paired_episodic_evaluation.py tests/test_stage05_protonet.py \
  tests/test_episode_metrics.py tests/test_few_shot_baselines.py \
  tests/test_advanced_models.py -q
```
