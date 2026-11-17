# 缺失实验、优先级与资源方案

目前没有可用训练checkpoint，不能实际“直接复用已有ResNet checkpoint”。下表“可复用”均指**找回且校验通过后**的条件方案。没有测得GPU型号/速度，因此资源只给相对等级，不虚构小时数或显存保证。

资源记号：R0=CPU元数据/合成检查，通常秒至分钟；R1=冻结骨干编码或短小验证，需要torch环境，CPU可小规模运行、GPU更适合全图；R2=多seed adapter/上下文训练与评估；R3=多seed骨干微调/从零收敛训练，资源最高。正式运行前测一个小批次时间和峰值显存再估算。

| 优先级 | 缺失实验 | 实验目的 | 需要使用的现有模块 | 需要新增的代码 | 预期输出 | 是否需要重新训练Backbone |
|---|---|---|---|---|---|---|
| P0 | 重复内容隔离与新任务计划验证 | 防止support/query同图泄漏 | sampler、episodes、重复审计 | 内容分组/明确排除清单、回归测试、计划版本 | 内容互斥审计、新计划哈希；R0 | 否 |
| P0 | 同checkpoint的跨域配对负例 | 防止把不同模型算成Δgen | summarize._paired | checkpoint哈希强制匹配及测试 | 不同权重必须拒绝；R0 | 否 |
| P0 | Linux/Windows清单语义一致性 | 保留旧产物可复评能力 | 哈希与清单验证工具 | 原字节+规范化语义哈希及迁移验证 | 换行等价与真实内容变更负例；R0 | 否 |
| P0 | 原套件及数学/梯度检查 | 确认核心算法可执行 | tests全部、prototype_scores、local_descriptor_scores | 补手算数值、optimizer step、温度和冻结状态断言 | pytest原始日志、前向shape与参数差值；R0/R1 | 否，仅合成更新 |
| P0 | 取回原始训练与推理产物 | 核验旧结果而非重复训练 | evaluate、summarize、报告记录 | 产物manifest/来源校验器、相对路径映射 | checkpoint/plan/日志/预测及哈希；R0 | 找回即可，不需重训 |
| P0 | NABirds物种人工审核 | 确定目标域合法候选 | prepare_nabirds、nabirds.create_review/build_manifest | 可选审核辅助展示，不自动代替人工 | review含审核人证据、物种级manifest；CPU+人工 | 否 |
| P1 | 统一CUB/NABirds的5-way 1/5-shot主基线 | 给出真实跨数据集基准和Δgen | evaluate、metrics、summarize | 可追溯批量评估入口/汇总清单 | 每seed两域两shot JSON、预测、CI；R1 | 有合格源域checkpoint则否；缺失需源域训练 |
| P1 | Global / Local / Fusion | 区分局部信息与匹配贡献 | FewShotModel、global/local度量 | 明确同特征诊断与各自训练两套表 | 两域两shot三方法、配对差值CI；R1/R2 | 固定特征诊断否；各自训练可先冻结骨干 |
| P1 | 不同融合权重的源域验证选择 | 选λ且隔离目标测试 | select、train --skip-test | 候选批量清单、参数来源记录 | 各λ各seed验证记录、选择JSON、一次最终测试；R1/R2 | 固定特征重新评分否；分别优化adapter则需重训adapter |
| P1 | Random / ImageNet / SimCLR | 检验表征初始化影响 | make_backbone、严格权重导入、convert脚本 | 运行矩阵、真实加载hash检查 | 各自来源/预算/两域1/5-shot/seed/CI；R2/R3 | 冻结诊断否；要比较充分下游学习需分别训练 |
| P1 | 平方欧氏 / 余弦 / 收缩马氏 | 分离度量贡献 | prototype_scores、compare脚本 | 扩展到两域、独立评估及α验证扫描 | 相同特征和任务、三度量指标及CI；R1 | 固定特征可复用；比较各自训练度量则另训 |
| P1 | 多训练seed与95%CI | 判断稳定性和5pp目标 | summarize | checkpoint绑定、完整协议表 | 至少3共同训练seed；均值、std、差值CI；R2/R3 | seed对应模型不同，不能把换评估seed当重新训练 |
| P2 | 无标准化 / support特征标准化 | 验证当前已有标准化 | standardize_from_support | 两域统一评估配置 | 两shot配对结果及NLL/ECE；R1/R2 | 固定模型改评分可先复用；训练效果需新训 |
| P2 | 无标准化 / episode分数标准化 | 验证用户最新指定的score版本 | Fusion局部及全局分数 | 分支score标准化定义与实现、无真值泄漏测试 | feature与score版本分开标记；R1/R2 | 纯后处理否；训练内使用需重训adapter等 |
| P2 | 固定温度 / 可学习温度 / 后校准 | 分清训练收益与置信度校准 | log_temperature、NLL/ECE | 温度更新断言、源域val后校准、温度轨迹 | accuracy/NLL/ECE及learned T；R1/R2 | 后校准否；训练温度至少重训下游参数 |
| P2 | 无adapter / 有adapter | 隔离adapter贡献 | encode、configure_trainable_scope | 无adapter冻结只评估入口、严格公平配置 | 同初始化/任务的有无adapter表；R1/R2 | 否，可冻结骨干；不能简单删掉已适配层当公平训练对照 |
| P2 | 冻结 / layer4 / full | 判别训练范围与过拟合 | configure_trainable_scope | 预算/收敛控制、资源记录 | 曲线、可训练参数量、两域成绩；R2/R3 | layer4/full需，冻结组否 |
| P2 | ProtoNet / Matching / FCE / supervised / Meta-Baseline | 检验元学习与分类器收益 | 三stage、context、few_shot | 公平预算及跨shot标记 | 五类基线在两域两shot的对照；R2/R3 | 可先冻结骨干，FCE/adapter仍须训练 |
| P2 | 组件独立与组合消融 | 避免把组件存在当成有效 | normalization、T、adapter、Fusion | source-val实验矩阵，控制主干及预算 | 无组件、单组件、组合、完整模型的配对结果；R2 | 冻结主干可否；训练组件需独立checkpoint |
| P2 | 系统错误分析 | 验证姿态/背景/遮挡等假设 | query导出、error脚本、CSV | 统一导出schema、分支分数、原型诊断、盲法标注与重复图聚类 | 正确和错误样本分层标注、因素统计和不确定性；R0/R1+人工 | 否，有预测可复用 |
| P3 | 嵌入UMAP/t-SNE | 辅助查看表征结构 | encode | 嵌入导出与可视化脚本 | 标记数据来源/参数的图；R1 | 否，不替代准确率证据 |

## 最小可交付实验矩阵

先固定一个主假设，避免把三初始化×三距离×三特征×多组件×两shot全部笛卡尔积展开。

1. **主表**：ImageNet初始化、固定源域学习协议，Global/Local/Fusion × 1/5-shot × CUB/NABirds × 至少3训练seed；源域val选λ。共用评估任务，checkpoint绑定。
2. **表征表**：固定主要匹配规则，三初始化 × 两shot × 两域 × 相同seed。冻结特征诊断与充分下游训练分开；报告预训练数据与配方混杂。
3. **度量表**：在同一checkpoint/同一图像特征下比较三度量，明确收缩α的源域val选择；不要一边改度量一边换骨干。
4. **组件表**：对源域选定Fusion做feature标准化、score标准化、温度、adapter的单独及组合消融。先源域筛选，最终候选按预定协议测目标域，不用目标结果返调。
5. **统计与错误表**：报告每seed值、配对CI、Δgen、辅助指标以及正确/错误样本的盲法因素观察。相同图像多次出现需聚类或明确描述性范围。

## 复用原则

- 同一编码器、相同确定性预处理时可以缓存逐图空间特征；全局池化、不同距离、局部top-k和分数融合可复用。改骨干、adapter、输入分辨率/增强会使缓存失效。
- 若只找回全局方法且带adapter的checkpoint，线性无偏置adapter与GAP在代数上可交换，但这不证明其空间描述子经过局部训练；相应实验只能叫“同特征的局部评分诊断”。
- 1-shot选出的模型可以直接做5-shot推理；须标注跨shot复评，不能称专门5-shot训练。
- 可学习温度/adapter的**训练效果**不能通过修改checkpoint配置字段来补实验，需要真实优化并保存参数。
- 不建议为了对比SimCLR立即重新做ImageNet自监督预训练。优先恢复可追溯的现成权重并做严格导入，再判断其配方是否满足可比性。
- 目标域只用于最后评估；目标support构造原型是协议允许的少样本参考，当前实现没有目标域参数更新，不能称测试时adapter微调。
