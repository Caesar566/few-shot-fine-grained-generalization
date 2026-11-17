# 研究目标逐项审计表

依据用户附件七类研究目标逐项展开；以当前代码、数据与本轮检查为准，不把历史作者描述当本轮运行。状态“已实现且已验证”只对该行的具体范围成立，不表示整个研究目标完成。

“是否实际调用”严格区分：**主入口可达，未实跑**=静态确认调用；**原函数抽取执行**=不改变函数体的标准库验证，不是torch模型运行；**真实推理**=本轮没有成功执行。历史摘要统一标注证据不足。路径均相对仓库根目录，行号按当前源码；证据路径相对本目录，reports路径仍相对仓库根目录。

本轮34文件语法通过、CUB官方校验通过、7组辅助检查执行完成；第7组中的历史算术一致不证明模型真实性，checkpoint负例“复现成功”表示发现缺陷。PyTorch原套件、数学前向、反传、optimizer step、checkpoint round-trip全部阻塞，详见04及evidence/commands.json。

最新任务明确要求**episode分数标准化**，而现实现是support统计量的**特征标准化**；在6.1和6.2分别列出。NABirds已有真实数据，但没有人工review/正式manifest，不能沿用旧报告的“没有NABirds原图”状态。

子项数量：81。标签分布：实现有错误 2；已实现且已验证 6；已实现但未验证 36；无法判断 6；未实现 2；部分实现 29。这是审计标签计数，不是按工作量计算的项目完成率。

## 目标1：跨数据集小样本协议

| 编号 | 研究目标 | 对应代码文件与行号 | 实现状态 | 是否实际调用 | 是否通过测试 | 是否有实验结果 | 证据位置 | 存在的问题 | 需要补充的工作 |
|---|---|---|---|---|---|---|---|---|---|
| 1.1 | CUB源域类别划分与官方标签一致 | scripts/validate_cub.py:24–71；data/splits/* | 已实现且已验证 | 实际执行独立校验 | 本轮通过 | 数据质量结果 | evidence/commands.json:cub_official_validation | 默认数据根与实际位置不同 | 训练显式指定已验证的数据根 |
| 1.2 | NABirds元信息和图像索引完整 | src/pattern_recognition_coursework/data/nabirds.py:65–96 | 已实现且已验证 | 原标准库函数抽取执行 | 48562个路径均存在 | 数据质量结果 | evidence/nabirds_metadata.json | 未逐图全部解码；1011层级节点不等于物种数 | 补全解码审计并人工审核物种 |
| 1.3 | 源域仅CUB、目标域仅NABirds | src/pattern_recognition_coursework/engine/experiment.py:231–234；engine/evaluate.py:63–79 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | evidence/commands.json | 源码路径分离，但没有真实跨域运行 | 恢复依赖及checkpoint后核对运行manifest |
| 1.4 | CUB与NABirds排除重叠物种 | src/pattern_recognition_coursework/data/nabirds.py:99–202；scripts/prepare_nabirds.py:13–38 | 部分实现 | 元信息候选检查已执行；正式建表未执行 | 555行均pending，72明显候选 | 无对应真实结果 | evidence/nabirds_metadata.json；docs/nabirds_protocol.md | 正式review/manifest不存在；仅比较CUB train物种 | 明确排重范围并完成人工taxonomy复核 |
| 1.5 | 支持5-way 1-shot与5-shot任务 | src/pattern_recognition_coursework/data/cub_dataset.py:108–156；engine/experiment.py:182–184 | 已实现且已验证 | 原采样器抽取执行 | 两shot合计120任务通过 | 旧摘要存在，证据不足 | evidence/stdlib_checks.json | 只验证索引任务，不是模型推理 | 在模型环境下补两shot完整episode |
| 1.6 | 每episode类别顺序与局部标签重映射 | src/pattern_recognition_coursework/data/cub_dataset.py:148–156；engine/experiment.py:102,105；models/few_shot.py:117–120 | 部分实现 | 采样/plan已执行，torch标签链未执行 | 顺序验证通过；torch标签测试阻塞 | 旧摘要存在，证据不足 | evidence/stdlib_checks.json；tests/test_few_shot_baselines.py:28 | run_episodes丢弃Dataset标签，强依赖约定 | 真实DataLoader前向及打乱顺序负例 |
| 1.7 | support/query索引互斥及抽样可重复 | src/pattern_recognition_coursework/data/cub_dataset.py:140–156；data/episodes.py:25–51 | 已实现且已验证 | 原函数抽取执行 | 120任务、重复种子一致及篡改拒绝 | 无对应真实结果 | evidence/stdlib_checks.json | 索引互斥不能证明内容互斥 | 与内容级隔离共同验收 |
| 1.8 | support/query图片内容互斥 | src/pattern_recognition_coursework/data/cub_dataset.py:150–156；data/episodes.py:41–42 | 实现有错误 | 真实清单新采样已复现 | 发现第201个任务同内容跨support/query | 无对应真实结果 | evidence/duplicate_content_episode_reproduction.json | C01：不同索引相同字节仍可泄漏；旧任务情况未知 | 按内容分组/版本化过滤并冻结新计划 |
| 1.9 | 训练/验证/测试类别与图像划分无交集 | scripts/validate_cub.py:24–71；src/pattern_recognition_coursework/engine/experiment.py:235,338 | 已实现且已验证 | 独立官方校验与全CUB哈希已执行 | 类别/ID/路径通过；无跨split精确重复 | 数据质量结果 | evidence/cub_exact_duplicates.json | 未排除近重复和外部预训练重叠；入口非自动全校验 | 增加内容版本与运行前门槛 |
| 1.10 | 目标测试集不参与训练或超参选择 | src/pattern_recognition_coursework/engine/experiment.py:231–245；analysis/select.py:12–58 | 部分实现 | 静态路径+选择器合成检查 | 明确CUB测试记录拒绝；历史行为无法核验 | 旧摘要存在，证据不足 | evidence/stdlib_checks.json；reports/cub_study_v1.md | 旧CUB test已查看；缺完整运行记录证明过程 | 固定独立协议，目标域仅终评 |
| 1.11 | 多独立训练seed重复实验 | src/pattern_recognition_coursework/engine/experiment.py:48–54,307–309；analysis/summarize.py:20–59 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | reports/cub_study_v1.md；evidence/historical_result_consistency.json | 有42/43/44摘要，没有对应模型；固定特征seed非训练seed | 取回或重跑独立seed模型 |
| 1.12 | 平均准确率和95%置信区间 | src/pattern_recognition_coursework/analysis/metrics.py:72–103；analysis/summarize.py:62–68 | 部分实现 | 旧数字独立标准库重算 | n=3区间算术一致；原SciPy函数未执行 | 旧摘要存在，证据不足 | evidence/historical_result_consistency.json | 无原推理；没有NABirds真实CI | 补真实多seed评估再汇总 |
| 1.13 | 测试指标来自实际模型推理 | src/pattern_recognition_coursework/engine/evaluate.py:49–103；engine/experiment.py:97–125 | 无法判断 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | evidence/inventory.json；reports/records/* | 缺所有checkpoint、query预测和任务计划 | 取回原产物重放，不能凭合理数值判真 |

## 目标2：三种ResNet-18初始化

| 编号 | 研究目标 | 对应代码文件与行号 | 实现状态 | 是否实际调用 | 是否通过测试 | 是否有实验结果 | 证据位置 | 存在的问题 | 需要补充的工作 |
|---|---|---|---|---|---|---|---|---|---|
| 2.1 | 实际骨干架构为ResNet-18 | src/pattern_recognition_coursework/models/backbones/resnet.py:85–109 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | evidence/commands.json | 静态直接调用torchvision.resnet18，未实际实例化 | 导入并核对层结构与state_dict |
| 2.2 | 随机初始化路径 | src/pattern_recognition_coursework/models/backbones/resnet.py:95–96 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | reports/cub_progress_2026-10-23.md | weights=None路径明确，未验证真实模型初始化 | 固定seed比较初始张量及记录hash |
| 2.3 | ImageNet监督初始化路径 | src/pattern_recognition_coursework/models/backbones/resnet.py:95–105 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | reports/records/cub_study_v1_1shot_summary.json | IMAGENET1K_V1标识存在，权重文件不存在 | 取回指定权重，核验SHA256和加载张量 |
| 2.4 | SimCLR初始化接口与真实加载 | src/pattern_recognition_coursework/models/backbones/resnet.py:14–81,98–100 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | reports/records/simclr_weight_provenance.json | 仅通用state_dict导入器；不能识别权重是否真的SimCLR | 核验源权重、输出hash、参数与运行log |
| 2.5 | SimCLR来源与预训练数据可核验 | scripts/convert_simclr_resnet18.py:20–22；reports/records/simclr_weight_provenance.json | 部分实现 | 来源记录静态阅读 | 未联网核验模型卡，源ckpt缺失 | 来源声明存在 | reports/cub_progress_2026-10-23.md | 记录称DM-Diaz/ImageNet1K；训练配方不可等同监督权重 | 保存固定版本模型卡/训练配方及原权重 |
| 2.6 | SimCLR编码器键名转换及严格提取 | scripts/convert_simclr_resnet18.py:33–57 | 已实现但未验证 | convert入口可达，未执行 | 阻塞：缺torch等依赖 | 历史提取声明 | tests/test_cub_analyses.py:27；provenance JSON | 检查完整键/shape/dtype是实代码；测试未运行 | 运行完整与缺键/错形状负例 |
| 2.7 | 不将随机/ImageNet权重误称SimCLR | src/pattern_recognition_coursework/models/backbones/resnet.py:14–81；engine/experiment.py:216–217 | 无法判断 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | reports/records/simclr_weight_provenance.json | weight-source是用户字符串，不是算法真实性证明 | 逐实验验证源checkpoint SHA256而非标签 |
| 2.8 | 三初始化一致的下游协议 | src/pattern_recognition_coursework/engine/experiment.py:249–264；reports/cub_progress_2026-10-23.md | 部分实现 | 配置路径可达，原运行缺失 | 阻塞：缺torch等依赖 | 短训验证均值 | reports/cub_progress_2026-10-23.md | 预训练配方、随机冻结负对照与充分从零训练不同 | 控制下游预算并分开报告冻结/微调 |
| 2.9 | 分别保存三组权重、日志和测试 | src/pattern_recognition_coursework/engine/experiment.py:267–282,331–332,378–379 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | evidence/inventory.json | 保存代码存在，当前三组产物均缺 | 恢复完整运行目录及模型hash清单 |
| 2.10 | 两域两shot完整初始化比较 | reports/cub_progress_2026-10-23.md；src/pattern_recognition_coursework/engine/evaluate.py | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 仅CUB验证叙述 | reports/cub_progress_2026-10-23.md | 无可信NABirds初始化表及最终CI | 按05实验矩阵补全 |

## 目标3：三种距离度量

| 编号 | 研究目标 | 对应代码文件与行号 | 实现状态 | 是否实际调用 | 是否通过测试 | 是否有实验结果 | 证据位置 | 存在的问题 | 需要补充的工作 |
|---|---|---|---|---|---|---|---|---|---|
| 3.1 | 平方欧氏确实平方并用于原型分数 | src/pattern_recognition_coursework/models/metrics/episode.py:33–39；models/few_shot.py:133–135 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 固定特征验证摘要 | reports/records/cub_fixed_feature_v1.json | 返回负平方和/特征维数，非普通开根号距离 | 补手算输入及torch数值测试 |
| 3.2 | 余弦正确归一化query和原型 | src/pattern_recognition_coursework/models/metrics/episode.py:33–35 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 固定特征验证摘要 | tests/test_episode_metrics.py；cub_fixed_feature_v1.json | 先均值再归一化；不是逐support归一化后均值 | 测零向量、尺度不变性和手算原型 |
| 3.3 | 马氏距离是真正协方差逆度量 | src/pattern_recognition_coursework/models/metrics/episode.py:41–56 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 固定特征验证摘要 | tests/test_episode_metrics.py:21 | Woodbury低秩求解可静态识别，未执行数值对照 | 与直接收缩矩阵逆逐元素比较 |
| 3.4 | 协方差估计范围明确 | src/pattern_recognition_coursework/models/metrics/episode.py:43–46 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | models/metrics/episode.py | 所有support围绕总体均值，含类间散度；不是每类残差协方差 | 记录设计并诊断1-shot影响，别误称类内协方差 |
| 3.5 | 正则化处理病态矩阵 | src/pattern_recognition_coursework/models/metrics/episode.py:45–56 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | tests/test_episode_metrics.py:37 | trace clamp、αI项与1e-5可见；奇异梯度测试未跑 | 执行rank-deficient与极端输入梯度测试 |
| 3.6 | 正则化系数选择有源域验证依据 | src/pattern_recognition_coursework/engine/experiment.py:173；models/metrics/episode.py:24 | 部分实现 | CLI配置可达 | 仅语法配置检查 | α=0.5固定特征记录 | reports/records/cub_fixed_feature_v1.json | 未找到α专项扫描/选择结果 | 固定预注册α或源域val扫描，保留选择记录 |
| 3.7 | 三距离真正接入ProtoNet/Fusion | src/pattern_recognition_coursework/models/few_shot.py:132–138 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | scripts/compare_cub_matching.py:29–51 | 内部调用清晰，真实模型执行尚未验证 | 三个metric参数分别完整episode推理 |
| 3.8 | 同嵌入公平比较三度量 | scripts/compare_cub_matching.py:29–51,83–120 | 已实现但未验证 | 脚本入口可达，未实跑 | 阻塞：缺torch等依赖 | 明确固定特征验证JSON | reports/records/cub_fixed_feature_v1.json | 同缓存方案存在；原权重/计划缺失 | 取回并重算；seed是任务而非训练重复 |
| 3.9 | 跨数据集三距离最终比较 | src/pattern_recognition_coursework/engine/evaluate.py；analysis/summarize.py | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | evidence/inventory.json | 原型接口可用，尚无NABirds真实结果 | 同checkpoint两域1/5-shot逐metric测评 |

## 目标4：Global / Local / Fusion

| 编号 | 研究目标 | 对应代码文件与行号 | 实现状态 | 是否实际调用 | 是否通过测试 | 是否有实验结果 | 证据位置 | 存在的问题 | 需要补充的工作 |
|---|---|---|---|---|---|---|---|---|---|
| 4.1 | 提取ResNet空间特征图 | src/pattern_recognition_coursework/models/backbones/resnet.py:112–117；models/few_shot.py:96–100 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_advanced_models.py:21 | layer4前向保留空间可见，未实测224→7×7 | 补真实shape断言 |
| 4.2 | Global是GAP全局向量+原型 | src/pattern_recognition_coursework/models/backbones/resnet.py:95–97；models/few_shot.py:101,127,133 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | reports/records/cub_study_v1_1shot_summary.json | torchvision标准avgpool在全局路径；未数值跑通 | 前向与均值等价检查 |
| 4.3 | Local保留局部描述子而非拼接 | src/pattern_recognition_coursework/models/metrics/local.py:28–31 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | reports/records/cub_fixed_feature_v1.json | flatten空间并转置为每patch向量 | 补人工2通道patch测试 |
| 4.4 | 同类所有support描述子合并 | src/pattern_recognition_coursework/models/metrics/local.py:24,31 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_advanced_models.py:10 | reshape依赖support按类排列 | 验证多shot类库与错误顺序负例 |
| 4.5 | 真正DN4式类内top-k余弦近邻 | src/pattern_recognition_coursework/models/metrics/local.py:29–36 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_advanced_models.py:10 | normalize+矩阵乘+topk真实存在，非名字冒充 | 执行可手算相似度与预测对照 |
| 4.6 | Top-K范围和聚合方式正确 | src/pattern_recognition_coursework/models/metrics/local.py:24–27,33–36 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | models/metrics/local.py:14–18 | 取HW×k平均而非原论文求和，有尺度归一化差异 | 明确DN4-style并验证k=1及边界 |
| 4.7 | Global/Local共享同一Backbone | src/pattern_recognition_coursework/models/few_shot.py:62,96–100,123–138 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | models/few_shot.py | 融合只encode一次再分支，不是两网拼接 | 统计前向调用次数和共享梯度 |
| 4.8 | Fusion确实分数融合进入预测 | src/pattern_recognition_coursework/models/few_shot.py:136–138 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_advanced_models.py:21 | 代码融合两个[NQ,N]得分，梯度测试存在但未跑 | 验证λ=0/1端点及中间值 |
| 4.9 | Global/Local/Fusion分别评估 | scripts/compare_cub_matching.py:43–50；engine/evaluate.py | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | CUB验证3方法；第一轮test无独立Local组 | cub_progress_2026-10-23.md；cub_study_v1.md | 没有两域统一三方法最终测试表 | 补同任务三方法与独立seed |
| 4.10 | Fusion提升有独立证据支持 | src/pattern_recognition_coursework/analysis/summarize.py:99–119 | 无法判断 | 旧摘要可核算，推理不可重放 | 旧CI算术通过 | 1-shot负差；5-shot微小正差且CI含0 | evidence/historical_result_consistency.json | 现有证据不支持稳定提升，更无NABirds结论 | 预固定最终候选，做独立目标评估 |

## 目标5：全局余弦原型 + DN4双分支

| 编号 | 研究目标 | 对应代码文件与行号 | 实现状态 | 是否实际调用 | 是否通过测试 | 是否有实验结果 | 证据位置 | 存在的问题 | 需要补充的工作 |
|---|---|---|---|---|---|---|---|---|---|
| 5.1 | 指定模型全局分支确实余弦 | src/pattern_recognition_coursework/models/few_shot.py:133–135；engine/experiment.py:171 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧Fusion配置cosine | reports/records/cub_study_v1_1shot_summary.json | global_local允许euclidean/mahalanobis，默认euclidean | 必须显式metric=cosine并记录实际函数 |
| 5.2 | 全局/局部输入输出维度一致 | src/pattern_recognition_coursework/models/few_shot.py:117–138；models/metrics/local.py:33–36 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_advanced_models.py:21 | 静态均输出[NQ,N]；真实tensor未验证 | 224分辨率与两shot前向shape检查 |
| 5.3 | 分数尺度处理合理且有验证 | src/pattern_recognition_coursework/models/metrics/local.py:14–18,34；models/metrics/episode.py:35,39 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | models/metrics/local.py；reports/cub_optimization_v2.md | 局部平均、全局有界但方差不同；无分支score校准 | 源域记录各分支分布并做校准消融 |
| 5.4 | 分数级权重融合公式正确 | src/pattern_recognition_coursework/models/few_shot.py:136–138 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | models/few_shot.py | (1-λ)sg+λsl后共同温度，非拼接 | torch端点/梯度与手算测试 |
| 5.5 | λ通过源域固定验证任务选择 | src/pattern_recognition_coursework/analysis/select.py:12–58 | 已实现且已验证 | 原选择函数抽取执行 | 有效合成候选可选择；CUB测试输入拒绝 | 历史选λ=0摘要 | evidence/stdlib_checks.json；reports/records/cub_weight_sweep_v1_selection.json | 只验证选择器规则，不证明历史输入真实性 | 恢复候选运行与验证任务 |
| 5.6 | 候选λ的其余配置与seed一致 | src/pattern_recognition_coursework/analysis/select.py:25–49 | 已实现但未验证 | 选择入口可达 | 部分合成检查，完整负例未跑 | 历史摘要seed42/43/44 | tests/test_validation_selection.py:15 | 代码核对模型/训练/计划；原候选记录缺失 | 补不同计划/seed/预算拒绝测试 |
| 5.7 | λ不由NABirds测试调优 | src/pattern_recognition_coursework/analysis/select.py:17–18；engine/experiment.py:321–335 | 部分实现 | 源码选择路径只读验证摘要 | 没有实际目标域运行证据 | 旧摘要存在，证据不足 | reports/cub_optimization_v2.md | status和配置是输入声明，无法证明历史研究过程 | 保存预固定协议与目标终评时间顺序 |
| 5.8 | λ和验证结果被保存 | src/pattern_recognition_coursework/analysis/select.py:52–59,62–79 | 部分实现 | 旧文件可读取；原选择运行缺失 | 选择摘要JSON解析通过 | 五个λ均值、选0记录 | reports/records/cub_weight_sweep_v1_selection.json | 缺15个原候选metrics及其best.pt | 保存每候选与每seed原始记录 |

## 目标6：标准化、温度与adapter

| 编号 | 研究目标 | 对应代码文件与行号 | 实现状态 | 是否实际调用 | 是否通过测试 | 是否有实验结果 | 证据位置 | 存在的问题 | 需要补充的工作 |
|---|---|---|---|---|---|---|---|---|---|
| 6.1 | Episode级分数标准化 | src/pattern_recognition_coursework/models/few_shot.py:123–138；models/metrics/episode.py:9–15 | 未实现 | 主路径没有score标准化调用 | 静态确认不等于已有feature标准化 | 无对应真实结果 | 03_code_issues.md:H05 | 最新目标要求score normalization，当前只有feature版本 | 先定义score统计范围，再实现和消融 |
| 6.2 | 已有support-only特征标准化 | src/pattern_recognition_coursework/models/metrics/episode.py:9–15；models/few_shot.py:130–131 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 固定特征验证JSON | tests/test_episode_metrics.py:9 | 只用support特征统计，全局分支应用；DN4不支持该选项 | 执行query扰动不变与恒定维度测试 |
| 6.3 | 可学习温度为真实Parameter | src/pattern_recognition_coursework/models/few_shot.py:77–81,121 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | models/few_shot.py | log参数exp保证正值，clamp范围1e-3到1e3 | 实例化检查named_parameters及边界梯度 |
| 6.4 | 温度进入optimizer并实际更新 | src/pattern_recognition_coursework/engine/experiment.py:85–86,303–304；models/few_shot.py:138 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | tests中未找到温度step专项 | 训练路径会收集requires_grad参数；未证明非零梯度或参数变化 | 有label的CE backward和optimizer.step前后对比 |
| 6.5 | 温度对Accuracy与校准影响区分 | src/pattern_recognition_coursework/models/few_shot.py:121–138；analysis/metrics.py:16–69 | 部分实现 | 主入口可达，未实跑 | 固定模型s/T不变argmax属数学结论，未跑torch | 无对应真实结果 | 03_code_issues.md:M02 | 正T统一后缩放本身不改变三主方法accuracy；训练梯度可改变模型 | 分开训练温度与后校准，报告NLL/ECE |
| 6.6 | 线性adapter尺寸与参数量 | src/pattern_recognition_coursework/models/few_shot.py:65–67,98–102 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_advanced_models.py:38 | 512→512无偏置、262144参数、恒等初始化；未实例化核对 | 测试identity与named_parameter.numel |
| 6.7 | adapter在全局和局部编码中实际使用 | src/pattern_recognition_coursework/models/few_shot.py:96–102 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | models/few_shot.py | 局部按位置线性映射，全局映射向量 | 两路径反传及step验证 |
| 6.8 | 正确冻结骨干并维护BN状态 | src/pattern_recognition_coursework/engine/experiment.py:66–90,99–101,130–132 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_advanced_models.py:49 | adapter范围还训练FCE/温度，不是只adapter；冻结层用eval BN | optimizer前后骨干参数/BN buffer逐项比较 |
| 6.9 | layer4/full训练范围符合声明 | src/pattern_recognition_coursework/engine/experiment.py:68–74 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | CUB短训范围验证叙述 | reports/cub_progress_2026-10-23.md | layer4范围将整个backbone设eval以冻结BN统计，需明确语义 | 统计参数名、BN状态和公平训练预算 |
| 6.10 | 新增参数只用CUB训练学习 | src/pattern_recognition_coursework/engine/experiment.py:231–245,285–317 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | engine/experiment.py；reports/cub_study_v1.md | 主入口没有NABirds训练分支，但历史过程证据不全 | 恢复数据manifest和运行日志 |
| 6.11 | 不在目标测试时意外更新模型 | src/pattern_recognition_coursework/engine/evaluate.py:84–86；engine/experiment.py:99–113 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | tests/test_training_entry.py:30 | 不传optimizer、eval、关闭grad；未实测state_dict前后 | 真实episode评估前后参数和buffer一致检查 |
| 6.12 | 标准化独立有效性验证 | scripts/compare_cub_matching.py:36,46；reports/records/cub_fixed_feature_v1.json | 部分实现 | 诊断脚本可达；原运行缺失 | 阻塞：缺torch等依赖 | 仅feature版本CUB验证 | reports/cub_progress_2026-10-23.md | 无score版本、两域最终CI | 分别补feature与score两类对照 |
| 6.13 | 固定/可学习温度独立有效性 | src/pattern_recognition_coursework/models/few_shot.py:77–81 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | evidence/inventory.json | 代码存在，未找到专项训练/校准结果 | 配对源域val选择及两域终评 |
| 6.14 | 无/有adapter独立有效性 | src/pattern_recognition_coursework/models/few_shot.py:65–67；engine/experiment.py:66–90 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 有adapter及训练范围叙述；无严格两域有无表 | reports/cub_progress_2026-10-23.md | 不同预算固定特征诊断不能直接替代消融 | 同初始化同任务下无adapter基线对照 |
| 6.15 | 标准化/温度/adapter组合消融 | 现有CLI参数组合；未找到完整矩阵 | 部分实现 | 可配置部分组合；score版本缺失 | 阻塞：缺torch等依赖 | 无对应真实结果 | evidence/inventory.json | 可组合不等于进行过组合实验 | 源域筛选单组件与组合，保留独立目标终评 |

## 目标7：实验结果与可复现性

| 编号 | 研究目标 | 对应代码文件与行号 | 实现状态 | 是否实际调用 | 是否通过测试 | 是否有实验结果 | 证据位置 | 存在的问题 | 需要补充的工作 |
|---|---|---|---|---|---|---|---|---|---|
| 7.1 | 训练配置、代码版本及可训练参数记录 | src/pattern_recognition_coursework/engine/experiment.py:249–264 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | evidence/historical_result_consistency.json | 当前有src指纹，旧基线摘要没有；脚本/完整命令未指纹化 | 存完整manifest和代码提交/dirty状态 |
| 7.2 | 随机性记录与严格可重复性 | src/pattern_recognition_coursework/engine/experiment.py:48–54,307–309 | 部分实现 | sampler随机性已测；torch未测 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | evidence/stdlib_checks.json | 未启用严格确定性算法，不能保证GPU位级相同 | 补确定性/worker规则并记录平台 |
| 7.3 | Checkpoint保存加载与评估往返 | src/pattern_recognition_coursework/engine/experiment.py:279–282,319–320；engine/evaluate.py:49–53 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | tests/test_training_entry.py:30 | 当前无checkpoint，round-trip未执行 | 先合成小数据，再原权重重放 |
| 7.4 | Loss与Accuracy日志完整 | src/pattern_recognition_coursework/engine/experiment.py:296–297,314–316,331–332 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | evidence/inventory.json | episodic history无逐epoch loss；原history也缺 | 补loss/资源/命令记录并保存原始日志 |
| 7.5 | 验证集选best且不必读取test图片 | src/pattern_recognition_coursework/engine/experiment.py:274–282,321–335 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_training_entry.py:49 | skip-test仍读test清单hash，不读test图片；旧模型无法核验 | 运行隔离测试，别把清单hash读取当图像泄漏 |
| 7.6 | 评估JSON与逐query原始记录 | src/pattern_recognition_coursework/engine/evaluate.py:88–130 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | evidence/inventory.json | 缺原JSONL；train/eval字段image/query_image不同 | 统一schema并核对logits→预测→聚合 |
| 7.7 | 多seed均值和配对CI真实性 | src/pattern_recognition_coursework/analysis/summarize.py:20–119 | 部分实现 | 旧摘要标准库重算 | 算术一致但原统计函数依赖缺失 | 旧摘要存在，证据不足 | evidence/historical_result_consistency.json | 没有真实模型链路；不能把任务采样当训练seed | 恢复模型与计划后独立重推理 |
| 7.8 | 同模型两域泛化差距 | src/pattern_recognition_coursework/analysis/summarize.py:89–91,122–129 | 实现有错误 | 原配对函数合成负例执行 | 不同checkpoint哈希仍被接受 | 无对应真实结果 | evidence/stdlib_checks.json | H04：配置相同不代表同权重 | 要求每seed checkpoint_sha256非空相等 |
| 7.9 | 基线、消融和测试条件公平性 | reports/cub_progress_2026-10-23.md；cub_followup_2026-10-23.md | 部分实现 | 静态配置/报告核对 | 未复跑 | 旧摘要存在，证据不足 | reports/*.md | 额外监督预算、FCE参数量、不同shot选模和预训练配方混杂 | 分表报告并对齐主要控制变量 |
| 7.10 | 当前代码与已有结果逐项绑定 | src/pattern_recognition_coursework/engine/experiment.py:30–38,263；reports/records/* | 无法判断 | 当前src指纹已算 | 历史缺对应指纹/权重，不能验证 | 旧摘要存在，证据不足 | evidence/historical_result_consistency.json | 旧training_config schema较少；不能证明由当前代码生成 | 取回旧commit和产物而非覆盖摘要 |
| 7.11 | 排除硬编码、伪造和重复引用指标 | src/pattern_recognition_coursework/analysis/metrics.py:16–69；engine/experiment.py:108–114 | 无法判断 | 静态读真实路径；推理未运行 | 未发现正式路径硬编码成绩 | 旧摘要存在，证据不足 | reports/records/*；tests/test_result_summary.py | 合成测试手填值不是造假；摘要缺外部可核验链路 | 原logits/预测重算及checkpoint去重 |
| 7.12 | 准确率、NLL、macro-F1、ECE及混淆矩阵 | src/pattern_recognition_coursework/analysis/metrics.py:16–103 | 已实现但未验证 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 旧摘要存在，证据不足 | tests/test_episode_metrics.py；metrics.py | 混淆矩阵为局部槽位；F1/ECE是episode均值 | 手算指标边界与物种层级统计 |
| 7.13 | 错误归因和物种级诊断 | scripts/analyze_cub_errors.py:51–142；reports/records/*.csv | 部分实现 | 历史文件存在，原输入缺失 | 1708行主表0行填写因素 | 面积分组摘要+8图观察 | evidence/historical_result_consistency.json；cub_error_manual_sample_v1.csv | 未系统分析正确/错误样本；原型诊断缺失 | 盲法分层标注并处理重复图像 |
| 7.14 | 嵌入分布辅助可视化 | src/scripts未找到UMAP或t-SNE实现 | 未实现 | 无调用 | 未运行 | 无对应真实结果 | 本轮src/scripts静态扫描 | 仅有数据预览不能代替嵌入可视化 | 可选导出embedding和图，不替代定量证据 |
| 7.15 | 可精确恢复中断训练 | src/pattern_recognition_coursework/engine/experiment.py:279–282 | 部分实现 | 主入口可达，未实跑 | 阻塞：缺torch等依赖 | 无对应真实结果 | engine/experiment.py | checkpoint未存optimizer/RNG/监督头，只有推理恢复 | 若需要续训则另增完整训练状态，不改旧产物 |
| 7.16 | 跨数据集至少提升5个百分点 | src/pattern_recognition_coursework/analysis/summarize.py:117–118 | 无法判断 | 主入口可达，未实跑 | 无NABirds真实输入 | 无对应真实结果 | reports/records/cub_study_v1_*summary.json | 只有判定代码，CUB差值不能满足NABirds目标 | 正式目标域多seed配对CI；失败也如实报告 |

