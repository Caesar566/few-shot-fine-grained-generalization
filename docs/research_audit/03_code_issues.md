# 代码与实验有效性问题

范围、基线和证据等级见 [01](01_project_structure.md)。问题包括已复现的代码缺陷、环境阻塞及证据缺失；后两类不冒充算法错误。未发现足够证据指控历史指标伪造，也未证明发生过NABirds标签泄漏。

## Critical

### C01：不同索引的同一图像可同时进入support/query（已复现）

- 位置：`src/pattern_recognition_coursework/data/cub_dataset.py:140–156 FewShotBatchSampler.__iter__`；`data/episodes.py:39–42 validate_plan`。
- 机制：`rng.sample`及`len(set(batch))`只排除重复索引，不排除相同字节图像。
- 实查：CUB 11788条索引只有11787种SHA256。test内图像ID `3310` 与 `3340`、类别 `058.Pigeon_Guillemot` 的两张图片内容相同。无跨train/val/test的精确重复。
- 复现：真实test清单，seed=42，5-way 5-shot，每类5 query，本轮新采样第201个episode（零基200）出现这对图片分别处于support/query。见 [duplicate_content_episode_reproduction.json](evidence/duplicate_content_episode_reproduction.json) 和 [cub_exact_duplicates.json](evidence/cub_exact_duplicates.json)。未调用模型，静态采样缺陷已足够成立。
- 影响：该配置下可产生内容级泄漏，影响未来测试正确性。**不能据此推定旧成绩已受污染**：旧计划缺失；历史审计声称旧计划只曾在support中同时出现，本轮无法重核。
- 建议：保留原始数据，按内容哈希分组采样或版本化排除清单，冻结新任务计划；增加“同内容不同路径”测试。不要静默改旧实验。

## High

### H01：真实实验的证据链不完整

- 位置：`reports/records/*.json`、`reports/*.md`；`engine/experiment.py:279–282,331–332,378–379`；`engine/evaluate.py:88–103`。
- 实查：项目扫描没有训练checkpoint、ImageNet/SimCLR文件、metrics/history、逐query预测或episode计划；`outputs/`不存在。现存摘要无法重放模型推理。
- 结论：结果存在但证据不足；不能称已验证来自当前代码，也不能称结果虚构。
- 建议：先取回只读原始产物、各自SHA256、原命令、完整配置及源码版本。缺失时按新协议重跑并标明新实验。

### H02：当前环境不能执行PyTorch测试或训练

- 位置：`pyproject.toml:10–27`、`.venv/`；[commands.json](evidence/commands.json)。
- 虚拟环境3.11.18缺torch/torchvision/numpy/scipy/Pillow/pytest等；其pytest退出1。系统3.14不满足项目版本，测试收集10处导入错误退出2。
- 本轮未验证GPU可用性，也未执行前向、loss、backward、optimizer更新、checkpoint round-trip或完整episode推理。历史“32 passed”不是本次验证结果。
- 建议：在项目规定Python下按lock恢复依赖，再跑原测试及新增诊断。不要为了绕过导入失败伪造torch实现。

### H03：字节哈希使Windows结果在Linux清单上拒绝复评

- 位置：`engine/experiment.py:26–27 sha256_file`；`engine/evaluate.py:65–67`；`data/nabirds.py:125,141–143`。
- 复现：历史train/val/test清单哈希均不等当前LF文件；仅将当前内容按CRLF编码，即与历史三个哈希全部一致。见 [line_ending_hash_reproduction.json](evidence/line_ending_hash_reproduction.json)。不是据此发现类别或图像清单变更。
- 影响：若取回旧checkpoint，现有严格字节检查会因换行不同而拒绝；审核哈希同理需要检查。实际checkpoint加载受阻，本轮只重现哈希判断条件。
- 建议：记录“原字节哈希+规范化语义哈希”，固定.gitattributes；为旧产物设计显式兼容迁移及内容核验，不要直接关闭校验或改写旧结果。

### H04：泛化差距未验证同一个checkpoint

- 位置：`analysis/summarize.py:71–96 _paired`，特别89–91；`122–129 generalization_gap`。
- `same_model=True`只比较model_config/training_config，不比较 `checkpoint_sha256`。同配置、同seed不保证参数相同。
- 已复现：三个seed的CUB记录用`a*64`哈希，NABirds记录用`b*64`，配置一致时原 `_paired` 仍接受。见 [stdlib_checks.json](evidence/stdlib_checks.json) 中 `checkpoint_identity_gap_reproducer`。这是合成记录，不是实际泛化成绩。
- 建议：要求每seed两场景checkpoint哈希相等且非空；训练内测试结果也保存哈希；增加负向回归测试。

### H05：要求的episode“分数标准化”尚未实现

- 位置：`models/metrics/episode.py:9–15 standardize_from_support`；`models/few_shot.py:123–138`。
- 现有操作是在support的**特征维度**估计均值和方差，变换全局support/query，之后计算原型；局部分数在此前已计算。未对两个分支的score做episode级标准化。
- 影响：最新任务中的score normalization目标不能计为已实现。最终开题报告写“episode级标准化”较宽泛，已有feature normalization可作为一个独立方案，不能混称。
- 建议：先明确统计范围与不使用query真值的规则，再单独实现/测试score版本并与feature版本比较。本轮不改代码。

### H06：研究主结论所需实验不完整，且已有比较存在混杂

- 位置：`reports/cub_progress_2026-10-23.md`、`cub_followup_2026-10-23.md`、`cub_study_v1.md`。
- 现有三种初始化主要是短训验证，随机冻结是负对照；SimCLR与监督权重训练配方不同。完整FCE参数量不同；Meta-Baseline额外监督阶段预算不同。冻结特征实验的三个seed是任务抽样seed，不是三个训练模型。
- 旧CUB测试已查看；后续在同一测试上验证新候选不能自动成为新的独立盲测。NABirds正式人工审核与测试没有产物。
- 建议：先冻结独立协议、确定主要假设与公平预算；补跨数据集多种子和必要消融，不能把验证均值改名为测试成绩。

## Medium

### M01：默认路径与真实数据位置不一致，自动定位可误选NABirds

- `engine/experiment.py:162`、`evaluate.py:64,72` 默认data/raw，而实际CUB在cub2002011/versions/7，NABirds在根目录nabirds。
- `scripts/make_splits.py:20–24 find_cub_root`递归接受任意同时含classes.txt/images.txt的目录，没验证CUB身份。仓库现在两个数据集都有这两文件；遍历顺序变化可能选择NABirds。未实际运行会覆盖划分的脚本。
- 建议显式数据根路径及元信息身份校验，不执行盲目重建划分。

### M02：温度有可训练路径，但不能把单独后缩放当作准确率改进

- `few_shot.py:77–81,121–138`定义log-temperature参数/缓冲区并输出`s/T`；`experiment.py:85–86,303–304`将可训练温度纳入范围及optimizer。
- 固定其他参数、对ProtoNet/DN4/Fusion最终所有类别统一除以正T，argmax不变；会改变NLL/ECE。训练时T改变梯度，可间接影响准确率，不能断言训练可学习T永远无用。Matching按support注意力聚类时不适用简单等价结论。
- 缺温度optimizer-step测试、专项及组合消融。建议同时报告训练温度和仅后校准。

### M03：双分支尺度处理有限，且global_local默认不是余弦

- `few_shot.py:123–138`与`local.py:15–36`：局部top-k平均让分数不随patch数直接线性增长。余弦全局与局部均有界，但方差/动态范围不一定一致；没有分支特定校准。
- `experiment.py:171`默认euclidean；若只选global_local，分数会是负距离与余弦混合，不能称全局余弦方案。`few_shot.py:42–43`的DN4/Matching“euclidean配置、内部余弦”占位语义也容易误读。
- 建议显式配置metric，记录实际评分函数；先做验证集尺度诊断，不能无证据声称当前融合错误。

### M04：训练入口未执行全部数据完整性校验

- `experiment.py:231–238,336–339`只检查类别名称交集，未在入口调用官方metadata验证、内容哈希重复检查。独立工具验证通过不保证任意未来输入仍通过。
- `episodes.py:10–17`指纹绑定路径/标签顺序，不绑定图片字节。更换图像内容而路径不变不会使任务指纹变化。
- 建议持久化数据内容版本并把数据审计作为运行前置门槛；近重复、CUB↔NABirds及外部预训练重叠仍未审计。

### M05：复现记录与训练恢复能力不完整

- `experiment.py:48–54`设置seed但未启用严格确定性算法；GPU可重复性不能仅靠该函数保证。
- `279–282`checkpoint不含optimizer、RNG、监督头，不能精确续训；`314–316` episodic history只保存train/val accuracy，没有逐epoch loss；未记录统一耗时、峰值显存、完整命令、torchvision版本。
- 当前源码指纹包含src，不包含scripts/依赖版本；历史基线摘要未保存该指纹。建议记录完整运行清单，别把“可评估”当“可精确恢复训练”。

### M06：错误分析输出与统计粒度不足

- `evaluate.py:121–128`不导出原型向量、距离和分支分数；train内导出键为`image`而独立evaluate为`query_image`（`experiment.py:369–375`），`analyze_cub_errors.py:91`只支持后者。
- `analysis/metrics.py:72–103`等权平均episode F1/ECE，混淆矩阵是局部槽位，不能当物种矩阵。此处不是公式错误，需明确报告定义。
- 1708条错误审核主表各因素字段均空；8张独立抽样观察不代表完整因素归因。UMAP/t-SNE未找到实现。

### M07：跨数据集排重范围与直接Dataset API需明确

- `nabirds.py:99–125`及`evaluate.py:74–78`只把CUB训练类别作为排重来源。若报告要求排除所有CUB物种，包括val/test，当前工具覆盖不足；现阶段不能据此认定实际测试泄漏。
- `NABirdsDataset:216–224`允许review与cub_train_classes同时不传，此时不重新构建审核；CLI总传两者，因此主CLI受保护，但其他调用方需要同样约束。
- 建议固定“排除train还是全部CUB”的协议，对可公开构造Dataset收紧审核要求，并由人工处理同义名与层级。72个明显重叠候选不是全部重叠物种。

### M08：Git追踪规则不改变实际输出位置

- `.gitignore`忽略outputs、源码仍默认outputs；新output目录里的数据预览不是训练结果。reports现在被ignore但既有报告仍已跟踪，未来新增report可能遗漏。
- 建议为新运行明确output-dir，保存小型manifest到版本库、大模型到有哈希的外部存储。不能因gitignore变化推断模型产物已经迁移。

## Low

- L01：缺一份可执行的完整实验矩阵和统一指标字段schema；目前依赖长CLI和多份叙述报告。`pyproject.toml`、`experiment._parser`、`evaluate.main`。
- L02：历史数据根目录使用Windows绝对路径，跨机器无法直接定位；应另存根目录映射，不改原记录。`reports/records/simclr_weight_provenance.json:9`及审计CSV。
- L03：超参数合法性对NaN/类型不完全统一，例如`experiment.py:206`仅用lr<=0；不是当前观察到的实验错误，建议统一配置验证。

## 修复与验证顺序

先处理C01、H04和H03的正确性/复现门槛，再恢复依赖及证据，澄清标准化目标并冻结协议；之后才补实验。以上为建议，没有修改原实现。详细实验矩阵见 [05](05_missing_experiments.md)。
