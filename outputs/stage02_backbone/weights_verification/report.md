# 本地ImageNet与SimCLR权重可用性补充检查

结论：**两个文件均可用且真实前向通过**。ImageNet必须区分两种加载路径：项目`make_backbone(pretrained=True)`的torchvision官方路径可加载，但当前`backbone_weights=<原始pth>`严格导入会拒绝缺失的20个BN批次计数器。没有修改模型源码来掩盖此差异。

## 文件身份与内容

- ImageNet SHA256：`f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`。与官方既有完整哈希一致。
- SimCLR SHA256：`4a50dfd4f78f83b8eba1a7c358cd0a476a9da1cf5579b7cf01a2051597ca217a`。与仓库历史provenance的源文件完整哈希一致。

SimCLR源epoch：99，global_step：4003600。完整哈希吻合证明是所记录的文件；本轮未重新审计其外部预训练过程，不能据此前向结果评价监督/自监督优劣。

## ImageNet加载兼容性

原始文件含卷积、BN参数和1000类分类头，缺20个`num_batches_tracked`缓冲区（非可训练参数）。项目本地严格导入预先检查全部state_dict键，因此拒绝；初次失败记录保存在`run.log`。torchvision原生兼容加载可以补齐这些历史计数器为0，其余所有源编码器张量均逐项与原文件完全相等。没有卷积权重/BN可训练参数缺失，没有形状不匹配。

本轮构建独立的`local_torch_cache/checkpoints/resnet18-f37072fd.pth`符号链接，指向用户原文件，设置进程内torch hub缓存后走`pretrained=True`，拦截下载函数，确认只读本地文件。该缓存仅用于本次检查，不等于训练入口默认`data/cache/torch`已配置好。以后训练若走`--pretrained`，应先将本地文件配置到入口实际缓存；若走`--backbone-weights`直接加载原pth，则需要单独处理此兼容性问题。没有启动训练，也未自动修复加载器。

## SimCLR编码器与投影头

严格使用现有转换脚本原始extract_encoder函数：backbone.0/1/4/5/6/7依次映射conv1/bn1/layer1/2/3/4。真实源文件中编码器120个state条目完整，键名、形状和dtype均匹配ResNet-18；只导出编码器，不把投影头作为512维图像特征。非编码器张量的真实名称与形状如下（不根据名字猜测器官或效果）：

```json
{
  "projection_head.layers.0.weight": {
    "shape": [
      512,
      512
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.1.weight": {
    "shape": [
      512
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.1.bias": {
    "shape": [
      512
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.1.running_mean": {
    "shape": [
      512
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.1.running_var": {
    "shape": [
      512
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.1.num_batches_tracked": {
    "shape": [],
    "dtype": "torch.int64"
  },
  "projection_head.layers.3.weight": {
    "shape": [
      128,
      512
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.4.weight": {
    "shape": [
      128
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.4.bias": {
    "shape": [
      128
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.4.running_mean": {
    "shape": [
      128
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.4.running_var": {
    "shape": [
      128
    ],
    "dtype": "torch.float32"
  },
  "projection_head.layers.4.num_batches_tracked": {
    "shape": [],
    "dtype": "torch.int64"
  }
}
```

导出文件：[simclr_resnet18_backbone.pt](simclr_resnet18_backbone.pt)，可作为项目`--backbone-weights`的输入，并配套记录`--weight-source`。其全部编码器张量已与真实源checkpoint逐项完全相等，原始checkpoint不适合未经转换直接作为torchvision格式state_dict导入。导出SHA256：`3874cdaee3dd3e7c02e53ca4ccee6c76e74c7700aeab189d58b80b20b31b48a9`。torch.save的序列化哈希可能因版本/文件名不同于旧导出哈希，因此以源哈希加逐张量一致性验证为依据。

## 同批图片实际前向

使用第一阶段选定的同4张CUB验证图片，清洗索引、224×224、确定性eval预处理。CPU、eval及inference_mode，参数和BN状态前后不变；没有optimizer、backward或训练。

| 初始化 | 空间特征 | 全局特征 | 均值 | 标准差 | 参数量 / requires_grad参数量 |
|---|---|---|---:|---:|---:|
| Random | `[4, 512, 7, 7]` | `[4, 512]` | 0.753609 | 0.764170 | 11176512 / 11176512 |
| ImageNet | `[4, 512, 7, 7]` | `[4, 512]` | 0.926160 | 0.892753 | 11176512 / 11176512 |
| SimCLR | `[4, 512, 7, 7]` | `[4, 512]` | 0.665929 | 0.715795 | 11176512 / 11176512 |

三组均有限、非异常常数；每组不同图片输出不同；三组完整state指纹互不相同，前向输出也不同。GAP空间均值与全局输出一致。requires_grad仍为True只是模型默认属性，不代表本轮进行了训练。

原始两个权重文件前后SHA256不变。完整缺键说明、源投影头、加载信息、张量统计、图片路径及三组指纹见[results.json](results.json)；成功原始日志见[run_complete.log](run_complete.log)。早期失败日志保留，不当作最终验证失败或权重损坏。运行脚本为[verify_weights.py](verify_weights.py)。
