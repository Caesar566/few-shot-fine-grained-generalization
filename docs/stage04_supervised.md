# Stage 04：完整监督训练与未见类验证

## 实现范围和验证选择

按本阶段最新要求，主训练采用 **CUB基础类监督分类**：保留100类线性分类头，CrossEntropyLoss、SGD、CosineAnnealingLR，默认50 Epoch。不是之前Stage04草案里的ProtoNet episodic training；两种协议和成绩不能混合。

原始Train100类与Val50类互斥，不能将Val的0..49本地标签直接送进Train100类分类头。用户已明确选择：保留完整划分，训练使用全部5890张CUB Train图片；验证复用现有cosine原型少样本评分，在2947张CUB Val图片中构造固定5-way1-shot、每类5 Query、100个episode。**只在验证中复用原算法，没有新增或训练ProtoNet模块**。

因此Train Loss/Accuracy属于100类监督任务；Val Loss/Accuracy属于未见类少样本任务，曲线图例和报告明确标识，不能用两者的数值差直接判断普通分类过拟合。根据固定Val Accuracy选择best，并列保留最早Epoch。不读NABirds或CUB Test图片。

## 复用与新增文件

- `models/backbones/resnet.py`：复用已验证`make_backbone`、历史BN计数器兼容加载；新增`SupervisedResNet18`包装，在现有512维编码器后接分类头。`forward_features()`保留特征接口，原ResNet与few-shot结构未改变。
- `engine/supervised.py`：独立的监督训练入口，复用Dataset、Transform、清洗索引、episode计划和`run_episodes`验证函数；增加SGD、余弦调度、AMP、梯度裁剪、CSV/JSON/TensorBoard、完整checkpoint和epoch边界恢复。
- `__init__.py`：新增`supervised-train`和`export-backbone`命令，旧入口保留原行为。不要误用旧`train --stage supervised`（旧路径仍是Adam且检查点不完整）。
- `tests/test_supervised_training.py`：小型训练/调度/恢复/导出与CUDA AMP回归测试。

没有改动原图、清洗索引、Train/Val划分或少样本评分公式。

## 默认训练启动命令（本轮没有执行50 Epoch）

在项目根目录终端执行：

```bash
uv run python -m pattern_recognition_coursework.engine.supervised \
  --run-dir outputs/stage04_training/supervised_imagenet_seed42 \
  --data-root cub2002011/versions/7/CUB_200_2011 \
  --splits-dir data/splits/cub_clean_v1 \
  --weights resnet18-f37072fd.pth \
  --device cuda \
  --epochs 50 \
  --batch-size 8 \
  --lr 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0001 \
  --min-lr 0 \
  --amp \
  --grad-clip 5 \
  --seed 42 \
  --workers 0 \
  --val-episodes 100
```

也可将`uv run python -m pattern_recognition_coursework.engine.supervised`替换为`uv run pattern-recognition-coursework supervised-train`。

默认只读本地已验证ImageNet文件，不自动下载。默认输入224，训练增强与之前完全相同；验证固定Resize/CenterCrop。默认学习率0.01、momentum0.9、weight_decay1e-4、scheduler T_max50 eta_min0，均可参数化。`--no-amp`禁用AMP，`--grad-clip 0`禁用裁剪。默认确定性算法，`--no-deterministic`可显式关闭；不保证跨硬件位级复现。

MX450已执行的AMP验证是Batch Size2、4张真实图片、2个Batch，不是对Batch Size8的完整运行承诺。若OOM，程序记录failure并停止，不擅自缩小Batch或改变图像/Query设置。可先将`--batch-size`改为4或2，并为重跑使用全新run-dir。训练Dataset完整遍历，没有drop_last，也不只训练固定episode。

AMP使用torch.autocast及GradScaler；梯度先unscale再clip，日志grad_norm为裁剪前范数。若AMP梯度溢出，GradScaler跳过该步并降低scale，CSV记录optimizer_step_skipped和epoch的amp_skipped_steps，不伪造参数已更新。CPU执行自动关闭AMP和GPU显存记录；显式`--device cuda`不可用时会报错而非静默CPU回退。

## 数据检查

启动前核对：清洗标记、六份清单哈希、原始元信息、Train/Val类别交集、样本ID交集、解析路径交集、内容哈希唯一性，并重新计算Train/Val图片实际SHA256，避免使用过期审计。类别映射保存到command.json与checkpoint。检查结果写入data_audit.json。

为绑定数据版本可读取Test清单的字节哈希及清洗CSV，但不读取Test或NABirds图片，也不计算其结果。

## 日志与Checkpoint

每个run目录独立，已存在即拒绝覆盖。逐step记录loss、accuracy、lr、裁剪前梯度范数和AMP跳步；逐epoch记录Train/Val Loss/Accuracy、当轮使用的lr、下轮lr、梯度范数、样本数、训练耗时、CUDA显存及训练峰值显存。验证关闭梯度，Val Gradient Norm不适用，不填伪造零值。

```text
<run-dir>/
  config.json / command.json / environment.json / code_version.json
  data_audit.json / validation_plan.json
  history.json / result.json / report.md
  logs/steps.csv / epochs.csv / run.log / tensorboard/
  figures/loss.svg / accuracy.svg / learning_rate.svg / gradient_norm.svg
  checkpoints/best.pth / last.pth / backbone_best.pth
```

`best.pth`和`last.pth`含完整backbone+classifier、optimizer、scheduler、AMP scaler、当前epoch/step、训练配置、类别映射、验证指标、数据/任务哈希、Torch/Python/NumPy/CUDA RNG以及历史记录。最佳checkpoint训练后重新从文件加载、重跑同一固定验证任务并检查loss/accuracy等一致；这是复现检查，不是新的独立盲测。

查看TensorBoard：

```bash
uv run tensorboard --logdir outputs/stage04_training/supervised_imagenet_seed42/logs/tensorboard
```

## 断点恢复（保留旧实验，使用新输出目录）

```bash
uv run python -m pattern_recognition_coursework.engine.supervised \
  --resume outputs/stage04_training/supervised_imagenet_seed42/checkpoints/last.pth \
  --run-dir outputs/stage04_training/supervised_imagenet_seed42_resumed_bs4 \
  --device cuda --batch-size 4
```

恢复原模型/分类头、SGD动量、余弦调度位置、启用时的AMP scaler、RNG和验证任务。必须保持原总Epoch数与训练/验证协议，不能恢复时偷偷延长Cosine T_max。允许改batch size/device/workers及数据根目录映射，但清单内容与标签必须一致。改变batch size会改变BN和优化轨迹，需要如实记录。

恢复粒度为完整epoch边界，OOM发生于半个epoch时从上一个完整last.pth恢复。首epoch尚未存checkpoint就失败则需用更小batch从头在新目录启动。checkpoint已经达到最终epoch时拒绝继续训练。

CPU合成数据回归已比较连续两epoch与epoch1中断后恢复至epoch2，完整模型state_dict及scheduler逐项相等；这一证据不等于跨硬件/GPU或中途batch精确续训保证。

## 导出与第五阶段特征接口

训练完成会自动导出`backbone_best.pth`，只含encoder的原生state_dict，不包含监督分类头，兼容现有严格本地导入器。也可独立导出（目标存在时拒绝覆盖）：

```bash
uv run pattern-recognition-coursework export-backbone \
  --checkpoint outputs/stage04_training/supervised_imagenet_seed42/checkpoints/best.pth \
  --output outputs/stage04_training/exported_backbone_seed42.pth
```

```python
from pattern_recognition_coursework.models.backbones.resnet import make_backbone
import torch
encoder = make_backbone(backbone_weights="outputs/stage04_training/supervised_imagenet_seed42/checkpoints/backbone_best.pth")
encoder.eval()
with torch.inference_mode():
    features = encoder(images)  # [B, 512]
```

训练封装本身也提供`model.forward_features(images)`，无需删除或改写已保存的分类头。完整监督checkpoint的model键同时含classifier，不应未经导出就冒充纯encoder文件。

## 当前测试证据

测试与实施报告在`outputs/stage04_training/supervised_implementation/`。只完成CPU小数据回归和MX450上2个小Batch测试，没有启动50epoch全量训练，尚无正式最佳Epoch/Val Accuracy，不能把单元测试结果填入正式比较表。
