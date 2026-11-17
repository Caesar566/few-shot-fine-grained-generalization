> 当前主训练实现已按用户最新要求改为普通监督训练（ImageNet + CrossEntropy + SGD + Cosine），验证方式经用户确认复用原未见类评分。见[实施与测试报告](supervised_implementation/report.md)。以下ProtoNet训练配置是此前未执行草案，不是当前监督训练命令或成绩。

> 显存实测更新：MX450上当前Random配置2 Step通过，无OOM；峰值allocated 904.9 MiB、reserved 1022 MiB。[显存报告](gpu_memory_probe/report.md)。仅批准并执行了探针，正式训练仍待批准；下文保留原计划检查记录。

> GPU环境更新：在沙箱外按用户相同命令`uv run python cuda_test.py`实测CUDA=True，MX450，报告总显存1.66 GiB。此前沙箱中的不可用结果保留为历史环境记录，不代表用户设备无GPU。代码未硬编码某个GPU型号；正式训练仍待批准，224像素、30图/episode、全量FP32训练尚未验证显存是否足够。见[更新证据](preflight/gpu_visibility_update.json)。

# Stage 04：完整训练运行计划与当前状态

## 当前结论

已完成数据、真实权重和环境检查，生成9份待批准配置（3初始化×seed42/43/44）。**当前未启动Stage04训练，未生成best.pth/last.pth或训练成绩。** 所有空白指标都表示未运行，不是0准确率。GPU预检查CUDA不可用、device_count=0，nvidia-smi退出9且无法连接驱动；详见[environment.json](preflight/environment.json)。本轮仅做CPU本地权重反序列化检查和运行器配置解析，没有optimizer step。

## 1. 主协议与报告对应

最终开题报告第三章3.1明确训练阶段按episode组织support/query，3.2建立ResNet-18原型基线，将Matching和监督最近类中心作为补充对照；第五章要求先做小任务过拟合再训练原型等基线。因此本轮主线为**ProtoNet Episodic Training**，不是在CUB上普通100类监督预训练。

ImageNet Supervised的“监督”描述的是外部初始化来源，不表示其下游实验改成普通监督Loss。三组均使用同一ProtoNet、同一query交叉熵。普通监督训练/Meta-Baseline的指标必须另表，不混入本比较。

## 2. 提议的统一配置（尚未批准）

| 项目 | 提议值 |
|---|---|
| 数据 | cub_clean_v1：train5890图/100类，val2947图/50类 |
| 网络/范围 | ResNet-18 + ProtoNet；full骨干；无adapter |
| Episode | 5-way 1-shot，每类5 query，30图/episode |
| 输入/增强 | 224；现有RandomResizedCrop/Flip/ColorJitter训练，固定Resize/CenterCrop验证 |
| Loss | 负平方欧氏距离/512作为logits，query CrossEntropy；固定温度1 |
| 优化器 | Adam lr1e-4；默认betas(.9,.999)、eps1e-8、weight_decay0 |
| Scheduler | None，恒定lr；沿用Stage03已验证配置，不擅自增加未验证调度 |
| 预算 | 10 epoch ×100训练episode =1000 step/实验；不能据此预先宣称充分收敛 |
| 验证 | 每epoch200个固定CUB val episode，种子training_seed+10000 |
| 训练随机性 | 采样seed+epoch，增强/网络随机数seed+100000+epoch；对应seed各初始化一致 |
| 选模 | val平均episode准确率严格提高才更新best；并列保留最早epoch |
| 重复种子 | 提议42/43/44；只按每次明确批准的单实验串行执行 |
| 终评 | 本阶段不读CUB test/NABirds图片，不给跨数据集结论 |

该设置验证了组成组件的CPU训练流程，不等于已完成GPU端验证或1000step收敛验证。GPU就绪后，首个批准实验的前2个step检查显存、finite loss/梯度及参数更新；失败停止，不自动缩小输入、query或更改预算。不擅自改CPU运行。

这是1-shot下游训练方案；不能把将来的5-shot复评说成专门5-shot训练。最终报告要求的5-shot专项及更多任务预算应单独批准。9项提议最多9000训练step，不是本轮已授权启动9000step。

## 3. 权重来源与比较限制

[weight_sources.json](preflight/weight_sources.json)逐项保存文件路径、SHA256、来源、数据声明及差异；本轮重新核对真实文件并成功执行两组严格本地加载。

- Random：torchvision weights=None，无外部预训练数据。
- ImageNet：IMAGENET1K_V1监督权重，已验证完整SHA256；仅20个历史BN计数器补零，编码器与原生加载精确一致。
- SimCLR：DM-Diaz公开ResNet-18来源记录，原Lightning文件源SHA256吻合，编码器120个state条目与源逐张量一致，投影头排除。源文件没有hyper_parameters字段；历史报告称100epoch/batch32，不能据此声称与监督权重训练配方完全相同。

三组可以统一**下游协议**，但外部预训练预算/数据配方不等；结果只能比较具体初始化，不能纯因果归因为“监督优于自监督”。预训练与CUB来源重叠风险也未因此消除。

## 4. 输出、记录及复验约定

每个批准实验独立`runs/<初始化_seed>/`，目录存在即拒绝重跑覆盖。Stage04锁阻止并发；不自动循环执行9项。记录config、实际argv、环境、git HEAD/dirty状态、源码及运行器SHA256、固定验证计划及其哈希。git工作区有未提交修复，不能只记HEAD冒充代码版本。

逐step CSV记录Train Loss/Accuracy/LR；逐epoch CSV记录Train/Validation Loss/Accuracy/LR及耗时，TensorBoard辅助记录；SVG曲线从真实epoch CSV生成。

`checkpoints/best.pth`与`last.pth`包含完整模型、单独backbone、model/training配置、epoch/step、optimizer、scheduler=None、对应验证指标、seed、RNG及验证计划hash；另导出`best_backbone.pth`。仅更新本次新运行目录内best/last，不覆盖任何旧实验。

训练后重新构造模型、加载best、在相同固定验证计划上独立重跑并逐指标精确核对。这是**复现性检查**，不是新的无偏独立测试；没有目标域调参。保存状态不代表已验证精确断点续训。

[train_one.py](train_one.py)已准备；目前仅配置解析检查通过，GPU端完整训练和新增best/last流程尚待批准后实跑，不能把准备好代码当成训练完成。

## 5. 逐实验状态比较

| 实验 | 状态 | 最佳Epoch | Val Loss | Val Accuracy |
|---|---|---:|---:|---:|
| imagenet_seed42 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| imagenet_seed43 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| imagenet_seed44 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| random_seed42 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| random_seed43 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| random_seed44 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| simclr_seed42 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| simclr_seed43 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |
| simclr_seed44 | NOT_RUN_PENDING_APPROVAL_AND_GPU |  |  |  |

[统一CSV](comparison.csv) · [具体运行计划](plan.json) · [配置目录](configs) · [环境证据](preflight/environment.json) · [数据检查](preflight/data.json) · [代码版本](preflight/code_version.json)

## 6. 待批准的实际运行命令

建议首先批准Random/seed42；以下命令**本轮没有执行**：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -B outputs/stage04_training/train_one.py --config outputs/stage04_training/configs/random_seed42.json --execute
```

当前命令会因CUDA不可用在建运行目录前停止，不会自动CPU回退。CPU训练部分按Stage03实测中位数粗估约38分钟/实验，尚未计每epoch200任务验证、I/O和checkpoint开销；不是GPU耗时估计。

用户要求“按明确批准实验逐项执行”，因此当前必须等待具体批准，且先解决GPU环境或明确另批CPU配置。没有任何已执行实验，自然也没有本阶段可加载训练checkpoint；已有Stage03或预训练权重不能冒充Stage04 best/last。
