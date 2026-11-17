# ResNet-18历史ImageNet本地加载兼容修复

已修复 `src/pattern_recognition_coursework/models/backbones/resnet.py:load_backbone_weights`，只放宽真实BatchNorm模块注册的缺失`num_batches_tracked`缓冲区：创建与目标shape/dtype/device一致的零张量。不会按名字后缀误补普通模块的同名缓冲区，已有计数器保持原值，已有但dtype错误的计数器仍拒绝。

卷积、BN weight/bias、running_mean、running_var及其他编码器状态仍要求完整。未知键、shape及dtype不一致都会在写入模型之前抛错，保持strict=True。先验证其余状态，再补计数器和加载，因此关键参数缺失时不会产生部分写入。原实现并没有显式拒绝dtype转换，本次补上dtype检查，避免load_state_dict静默转换。

全部补齐名称通过UserWarning输出，并保存在backbone_load_info['initialized_buffers']；loaded_keys表示从文件实际加载的编码器条目（100），不把20个补零条目当作文件加载成功。显式剔除的1000类fc头仍在ignored_keys记录。

## 真实本地权重与torchvision对比

- 原文件SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`，前后保持不变。
- 比较完整编码器state_dict的 **120** 个条目：键名、dtype、shape与数值均逐项完全相等。
- 同4张真实CUB验证图片：输入`[4, 3, 224, 224]`，全局`[4, 512]`、空间`[4, 512, 7, 7]`。
- 全局最大绝对差：**0.0**；空间最大绝对差：**0.0**，使用rtol=0、atol=0断言。
- 原生模型通过torchvision IMAGENET1K_V1读取已有本地缓存；禁止download_url_to_file执行。自定义模型直接读取用户原始pth文件。没有下载、反向传播或训练。

## 补零缓冲区完整清单

```text
bn1.num_batches_tracked
layer1.0.bn1.num_batches_tracked
layer1.0.bn2.num_batches_tracked
layer1.1.bn1.num_batches_tracked
layer1.1.bn2.num_batches_tracked
layer2.0.bn1.num_batches_tracked
layer2.0.bn2.num_batches_tracked
layer2.0.downsample.1.num_batches_tracked
layer2.1.bn1.num_batches_tracked
layer2.1.bn2.num_batches_tracked
layer3.0.bn1.num_batches_tracked
layer3.0.bn2.num_batches_tracked
layer3.0.downsample.1.num_batches_tracked
layer3.1.bn1.num_batches_tracked
layer3.1.bn2.num_batches_tracked
layer4.0.bn1.num_batches_tracked
layer4.0.bn2.num_batches_tracked
layer4.0.downsample.1.num_batches_tracked
layer4.1.bn1.num_batches_tracked
layer4.1.bn2.num_batches_tracked
```

## 回归测试

15 passed，6.24秒。覆盖：缺失BN计数器补零与日志；已有非零计数器保留；卷积、BN weight/bias、running_mean/running_var缺失仍拒绝；未知键、错误shape、错误dtype和错误计数器dtype拒绝；非BN同名缓冲区不放宽；显式前缀/容器兼容；真实官方完整状态及同批CUB前向精确一致；SimCLR原提取器完整性回归。

测试文件：`tests/test_backbone_legacy_loading.py`，另执行`tests/test_cub_analyses.py::test_simclr_extractor_requires_complete_torchvision_encoder`。原始日志：[pytest_stdout.txt](pytest_stdout.txt)；逐用例：[junit.xml](junit.xml)；真实状态、输入路径、完整warning及补齐名单：[real_checkpoint_comparison.json](real_checkpoint_comparison.json)。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -B -m pytest tests/test_backbone_legacy_loading.py tests/test_cub_analyses.py::test_simclr_extractor_requires_complete_torchvision_encoder -q -p no:cacheprovider -o addopts=''
```

现在可以直接使用`make_backbone(backbone_weights='resnet18-f37072fd.pth')`，或现有CLI的`--backbone-weights resnet18-f37072fd.pth --weight-source <来源说明>`。不需要先修改checkpoint。未修改网络结构、数据划分、训练或分类逻辑，未覆盖以前的验证记录。
