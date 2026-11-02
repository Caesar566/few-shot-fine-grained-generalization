# 工程文件与进度索引（2026-11-02）

## 文件分类

| 位置 | 用途 | 当前状态 |
| --- | --- | --- |
| `data/raw/CUB_200_2011/` | CUB 原始图像与官方元数据 | 已有；不要改动 |
| `data/splits/` | CUB 训练、验证、测试类别与图像清单 | 已生成并校验；NABirds 正式清单未生成 |
| `data/cache/` | 预训练权重缓存 | 运行依赖；不是实验结果 |
| `src/pattern_recognition_coursework/` | 数据读取、episode、模型、训练、评估及汇总实现 | 已实现 CUB 主流程和 NABirds 导入/评估接口 |
| `scripts/` | 数据校验、NABirds 导入准备、CUB 消融和审计脚本 | 可重复运行的工具 |
| `tests/` | 单元和流程测试 | 最近一次记录为 32 项通过（2026-10-23） |
| `outputs/cub_study_v1/` | 第一轮 CUB 测试、检查点与 episode 计划 | 正式保留的首轮结果 |
| `outputs/cub_*/` 其他目录 | 验证集消融、复评与检查点 | 探索性结果；不要混作独立测试 |
| `reports/` | 叙述性实验报告 | 主要阅读入口见下方 |
| `reports/records/` | JSON 实验汇总、审计和人工抽样 CSV | 保留机器可读原始记录 |
| `docs/reference/` | 开题报告 PDF | 研究依据 |
| `docs/nabirds_protocol.md` | NABirds 物种审核与评估协议 | 已准备；尚无原始图像和审核清单 |
| `README.md`、`pyproject.toml`、`uv.lock` | 使用说明和环境依赖 | 工程入口 |

`configs/`、`notebooks/` 当前为空目录；`.venv/`、`.pytest_cache/`、`.pytest_tmp/` 属于本地环境或测试临时目录。当前工作区有大量未提交或未跟踪的文件，因此不能把 Git 状态理解为已发布版本。已将原始 PDF 与机器可读报告记录移入上述子目录；`outputs/` 中的实验记录包含原始绝对路径，保持原位以供复评。

## 报告阅读顺序

1. `docs/reference/A03_模式识别开题报告.pdf`：研究目标与任务来源。
2. `reports/cub_study_v1.md`：第一轮 CUB 未见类别测试。
3. `reports/cub_progress_2026-10-23.md`：匹配、距离、初始化和适配范围的验证比较。
4. `reports/cub_followup_2026-10-23.md`：上下文匹配、监督基线、训练预算、错误抽样与重复图像审计。
5. `reports/records/*.json`、`reports/records/*.csv`：机器可读的汇总和审计明细；模型检查点与 episode 计划在 `outputs/`。

## 目前做到哪一步

- **完成：** CUB 图像与 100/50/50 类划分校验；5-way 1-shot/5-shot 的 ProtoNet 等方法实现；第一轮三种子 CUB 测试；验证集上的融合权重、距离、预训练来源、适配范围、完整上下文 Matching Networks、监督基线和长预算对照；精确重复图像审计。
- **阶段结论：** 当前验证证据没有显示候选方法稳定超过全局余弦 ProtoNet。第一轮 CUB 测试已查看，后续在同一测试划分上选择和比较新方法只能标为探索性。
- **未完成：** 系统化人工错误标注；外部预训练图像与 CUB 的重叠核查；新的独立评估协议；NABirds 官方图像、物种不重叠人工审核及跨数据集测试。跨数据集提高 5 个百分点的目标尚不能判断。

下一步应先固定独立评估方案并处理数据审核，再决定是否针对明确假设继续训练。单纯增加现有模型轮次没有得到清晰收益。
