# Stage04监督训练实现与测试报告

完整使用说明：[docs/stage04_supervised.md](../../../docs/stage04_supervised.md)。本次只实现并执行小规模回归验证，**没有启动50epoch完整CUB训练，也没有正式最佳Epoch或Val Accuracy**。

## 现有能力与新增能力

旧项目已有清洗Dataset、Transform、ResNet-18、普通监督epoch、未见类少样本验证；旧监督入口使用Adam，未保存分类头/optimizer/scheduler/AMP完整状态，也没有本次所需完整日志。新增监督入口复用数据与验证评分，增加SGD+CosineAnnealingLR、保留分类头、AMP、unscale后裁剪、TensorBoard/CSV/JSON/SVG、best/last和epoch边界恢复。

用户已批准验证协议：Train100类全部图片做普通监督CrossEntropy，Val50个未见类仅复用现有cosine原型评分。不直接用Val本地标签计算100类头准确率，不新增ProtoNet/Matching/Local/Fusion算法，不进行目标域评估。

## 验证结果

| 检查 | 结果 | 证据 |
|---|---|---|
| CPU小型监督训练、特征接口与既有加载兼容回归 | 16 passed；1 GPU用例因沙箱不可见而skip | [CPU日志](final_cpu_tests.txt)、[JUnit](final_cpu_junit.xml) |
| MX450实际AMP+SGD+梯度裁剪 | 1 passed；4张真实Train图、224像素、batch2，共2 Batch | [GPU日志](final_gpu_tests.txt)、[实测数据](gpu_supervised_evidence.json) |
| 完整分类头checkpoint、scheduler、optimizer与TensorBoard | CPU两epoch夹具通过 | [测试源码](../../../tests/test_supervised_training.py) |
| epoch1中断后恢复与连续epoch2对照 | 完整模型state_dict逐项相等，scheduler相等 | [最终恢复回归](resume_final_tests.txt) |
| Backbone导出 | 去掉分类头，现有严格make_backbone可读；覆盖目标拒绝 | 同一CPU回归 |
| GPU AMP状态和SGD动量恢复 | 完全相等，裁剪后梯度L2≤1.001 | GPU回归 |
| 正式50epoch | 未执行；不可填写测试夹具的指标冒充全量成绩 | 新run-dir启动后生成实际记录 |

GPU测试使用独立测试GradScaler init_scale128，生产默认使用PyTorch动态初始scale；生产代码记录溢出跳步，不将跳步当作已更新。此项证明AMP路径及状态恢复可用，不是对所有Batch Size、显存占用或50epoch收敛的保证。

MX450小测试实测：loss=4.046875，accuracy=0.00%，裁剪前平均grad norm=21.062556，训练峰值allocated=219.7 MiB；这些仅为2 Batch功能测试，不是训练结论。

## 运行与恢复

默认启动：

```bash
uv run python -m pattern_recognition_coursework.engine.supervised \
  --run-dir outputs/stage04_training/supervised_imagenet_seed42 \
  --device cuda --weights resnet18-f37072fd.pth \
  --epochs 50 --batch-size 8 --lr 0.01 --momentum 0.9 \
  --weight-decay 0.0001 --amp --grad-clip 5 --seed 42
```

如显存不足，停止后用新run-dir和更小batch size。若已有last.pth：

```bash
uv run python -m pattern_recognition_coursework.engine.supervised \
  --resume outputs/stage04_training/supervised_imagenet_seed42/checkpoints/last.pth \
  --run-dir outputs/stage04_training/supervised_imagenet_seed42_resume_bs4 \
  --device cuda --batch-size 4
```

best/last包括模型及分类头、SGD动量、余弦调度、AMP scaler、epoch/step、验证结果、RNG、类别映射及数据哈希。恢复到新目录保留历史，改变batch size的轨迹差异会被配置记录，不能延长原T_max后冒称原协议续训。

本轮没有修改原始图像、清洗索引、训练/验证类别划分或既有few-shot评分公式。旧ProtoNet Stage04计划是历史草案，不能混作当前监督训练结果。源码与命令见使用文档；完整训练完成后会自动生成真实曲线、最佳epoch/验证分数和权重路径。
