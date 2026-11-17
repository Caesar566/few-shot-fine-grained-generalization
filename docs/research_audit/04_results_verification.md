# 结果真实性与可复现性核查

## 结论与证据等级

没有找到足以证明“现有模型成绩由当前工作区代码和指定权重真实推理产生”的完整证据链。存在历史Markdown、7份JSON与2份CSV；部分数值相互一致，数据质量审计也能独立复现，但这不等于重放过模型推理。不能认定这些成绩为伪造，也不能按充分验证结果对外发布新的结论。

| 分类 | 本轮判定 |
|---|---|
| 已成功运行且有充分证据 | CUB官方清单及图片路径验证；CUB字节级重复审计；NABirds元信息/全部索引路径核对；40张图片解码；标准库辅助函数检查；语法与TOML解析。仅针对各自范围，不是模型成绩 |
| 有结果但缺少完整证据 | 第一轮CUB准确率及CI、初始化/距离/匹配/适配消融、融合选择、历史错误和重复审计记录 |
| 代码存在但没有可验证运行记录 | 三初始化实际加载，模型前向/反传/参数更新，可学习温度，正式CUB→NABirds真实推理，score normalization（后者核心功能本身尚缺） |
| 运行失败 | 虚拟环境pytest启动失败；系统pytest收集失败；模型导入缺依赖。未进行训练，不能将环境错误写成“训练发散” |
| 未发现已开展实验的证据 | NABirds审核后1/5-shot多种子正式测试、温度专项、score标准化消融、融合组件完整组合消融、系统性错误归因、嵌入可视化 |

## 本轮实际执行记录

原始命令、退出码、stdout/stderr见 [commands.json](evidence/commands.json)。

| 检查 | 结果 | 限制 |
|---|---|---|
| 项目34个Python文件内存compile | 通过 | 无.pyc写入；只证明语法可编译 |
| pyproject和uv.lock解析 | 通过，lock有52包条目 | 未安装或同步依赖 |
| `.venv/bin/python -B -m pytest ... --collect-only` | 失败，退出1：No module named pytest | 不是32项通过 |
| 系统Python，PYTHONPATH=src，pytest collect-only | 失败，退出2，10个收集错误 | 缺torch/scipy；系统3.14也不满足项目Python约束 |
| `scripts/validate_cub.py --data-root cub2002011/versions/7/CUB_200_2011 --check-images` | 退出0 | 验证官方元数据与路径；不解码所有图片 |
| 原sampler/plan函数抽取执行 | 120个真实清单任务通过，5-way、shot1/5、query5、seed42/43/44 | 不执行torch Dataset/DataLoader，也不执行图像模型 |
| 原NABirds标准库元信息函数抽取执行 | 1011层级节点；555有图像视觉类；48562图片均存在；555行待审、72行明显重叠候选 | 不是555个已确认物种；未做人工taxonomy审核 |
| Pillow固定seed抽样解码 | CUB20+NABirds20张通过 | 未证明其余图片全部可解码；未运行torchvision transform |
| CUB所有索引图片SHA256 | 11788图、11787种内容；test内1对重复；跨划分0对 | 不覆盖近重复、NABirds全部内容、跨数据集、外部预训练 |
| 原融合选择函数抽取执行 | 合成有效记录可选权重，明确CUB测试记录被拒绝 | 不证明历史输入JSON真实；不等于真实模型验证 |
| 原泛化差距配对函数合成负例 | 接受不同checkpoint哈希，缺陷复现 | “复现成功”是发现bug，不是通过同模型校验 |
| 旧汇总算术 | 与错误审计JSON的逐seed准确率一致；n=3的t区间重算一致 | 两份JSON相互一致不能证明独立来源或真实推理 |

本轮**未执行**距离函数torch人工输入、原型torch构建、DN4 torch数值前向、反传、optimizer step、温度梯度、checkpoint保存加载或一个完整真实模型episode。没有伪装成通过；现有相关测试文件仅被静态阅读。

## 历史结果逐项核查

| 结果组 | 保存位置 | 命令/配置线索 | 模型/日志/原始推理 | 最终判断 |
|---|---|---|---|---|
| ProtoNet与融合第一轮CUB 1/5-shot | `reports/cub_study_v1.md`、`records/cub_study_v1_{1,5}shot_summary.json` | 报告有CLI示例；摘要有model/training配置、seed42/43/44、ImageNet标识、adapter范围 | 原best.pt、metrics/history、12份eval、query JSONL和test计划缺失 | 结果存在，但证据不足；可核算聚合，不能重推理 |
| 1-shot融合权重扫描 | `cub_optimization_v2.md`、`records/cub_weight_sweep_v1_selection.json` | λ=0/.25/.5/.75/1，3 seed验证均值 | 原15个运行与验证计划缺失 | 选λ=0的摘要存在，无法验证每候选训练及选择过程 |
| 固定特征距离/标准化/匹配 | `records/cub_fixed_feature_v1.json`、`scripts/compare_cub_matching.py` | 明确validation_only_frozen_features，两个shot，每seed100 episode，7规则 | 编码器权重及计划缺失；没有独立训练 | 有对照记录，不是3个训练模型的测试结论 |
| 三初始化与适配范围 | `cub_progress_2026-10-23.md` | 随机、监督、指定SimCLR；冻结/末层/全量；短预算 | 当前没有对应checkpoint或运行JSON | 主要为叙述性验证证据，无法核实实际权重加载和完整训练 |
| FCE、监督最近类中心、Meta-Baseline、长预算 | `cub_followup_2026-10-23.md` | 有seed、预算、均值和对照说明 | 运行目录缺失；历史32 passed无法重现于本机 | 不能当作当前已验证的训练或最终测试 |
| SimCLR来源 | `records/simclr_weight_provenance.json`、`scripts/convert_simclr_resnet18.py:20–22,33–87` | URL、源/输出SHA256、epoch99、120张量、严格提取说明 | 源ckpt与输出pt缺失，无法比对权重字节；本轮未联网核验模型卡 | 来源声明可追踪，真实性及加载状态无法验证 |
| 错误大小分组与人工观察 | `records/cub_error_audit_v1.json`、两个CSV | 分组及各seed accuracy、源摘要/预测哈希；8张人工抽样说明 | 原query JSONL缺失，无法重算全部query判断；主审核表1708行因素栏均空 | 自动统计摘要存在；系统人工归因未完成 |
| 精确重复审计 | `records/cub_duplicate_audit_v1.json` | 文件SHA256和历史计划共现描述 | 本轮对真实CUB复现重复对与跨split无精确重复；旧计划缺失 | 数据重复事实已验证，旧episode共现描述仍未核实 |
| CUB预览 | `outputs/cub_preview/summary.json`、report、8图 | 源目录、元信息与可视化 | 不是模型checkpoint或预测 | 不能计作few-shot实验结果 |
| NABirds正式结果 | 未找到 | 仅evaluate及summarize代码接口 | 无人工review/manifest/模型评估JSON | 尚无可信正式结果，不能计算实证Δgen |

## 数值一致性与统计定义

| 场景 | ProtoNet均值 | 融合均值 | 融合−基线的95%配对区间 |
|---|---:|---:|---:|
| CUB 1-shot | 80.67% | 79.43% | −1.23 pp，[-3.03,0.56] |
| CUB 5-shot | 91.20% | 91.77% | +0.57 pp，[-0.23,1.37] |

这些数值为**旧记录转述和算术核对**，不是本轮新推理。用错误审计JSON中的6组逐seed准确率重算，与两个汇总一致。n=3时95% Student t临界值通过df=2分布公式计算，未冒用缺失的SciPy。见 [historical_result_consistency.json](evidence/historical_result_consistency.json)。区间均含0，不能声称融合稳定改善，更不是NABirds提升5pp。

第一轮为224像素、每类1 query、3×30训练episode、固定30验证episode、冻结骨干训练adapter、每seed200测试episode。5-shot复用按1-shot选的checkpoint。不能把它与CLI默认query5、其他分辨率、专门5-shot训练或充分收敛模型直接排名。

`summarize._stats`使用训练seed间t区间，`metrics.aggregate_episode_metrics`保存episode准确率与std，但不自动输出同checkpoint的episode抽样CI。两种不确定性需分开。当前代码不使用bootstrap；仅3 seed的t区间也不代表所有数据、权重和训练选择不确定性。

## 历史来源与当前代码的对应

- 两个基线摘要的training_config没有当前源码中的`code_fingerprint_sha256`、可训练参数名/量字段，不能证明等于当前版本。当前src指纹为 `da42043e0122f3f06c815b801ab9966d6418045bee7dde1b43a739d2a02ec5dd`，保存在证据文件；未找到可比较的历史指纹。
- 历史清单字节哈希与当前不一致，但三个文件改用CRLF后完全匹配历史哈希。这里只需规范化换行就可解释，**不把这一点报告为实际类别划分变化**。现有evaluate的字节校验仍会拒绝，应通过可追踪兼容策略处理。
- 源码指标由logits及真值计算，没有发现正式路径以常数直接返回80.67等成绩。测试代码中的手写accuracy是明确的合成统计测试，不应作为造假线索。
- 摘要之间相同数字可能来自相同原实验的复用，例如1-shot检查点的5-shot复评；报告已说明部分复用。缺原checkpoint哈希，无法对全部实验做去重或确认不同初始化确实不同。
- 配置字段、哈希、时间、合理数值都不是签名认证。当前只能给“结果存在，但证据不足”，而不是“真实”或“伪造”的绝对判定。

## 原始文件保全

本轮未改写、删除或覆盖任何原实验文件，没有下载模型或数据，没有运行长训练。见受保护文件前后哈希清单。审计临时脚本在`/tmp/`，可复查副本保存在本目录evidence中；所有新增产物仅属于审计，不能混入正式实验统计。
