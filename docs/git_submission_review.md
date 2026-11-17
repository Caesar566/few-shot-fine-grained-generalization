# 本次提交文件检查

本检查仅调整 `.gitignore` 并给出提交清单，没有执行 Git 暂存、提交、推送或删除磁盘文件。

## 检查结论

- `server_transfer/` 约 32 GB，包含迁移快照和压缩包，不应进入 Git。
- `outputs/` 约 2.9 GB，大部分是模型和测试临时产物；按新增规则过滤后，可提交的现存输出共 299 个文件、约 44.63 MiB（本清单生成时）。
- 暂存区已有 596 个测试临时文件和 1 个 44,784,766 字节的合成随机模型，需要从暂存区移除，磁盘文件继续保留。
- 原始 CUB、NABirds 和 NABirds 压缩包已有忽略规则。
- 没有整体忽略 `outputs/`，保留此前要求的报告、可视化与证据追踪。

## 应提交

| 类型 | 文件或目录 | 原因 |
| --- | --- | --- |
| 依赖与入口 | `.gitignore`、`pyproject.toml`、`uv.lock`、`src/pattern_recognition_coursework/__init__.py` | 固定环境与可执行入口 |
| 数据逻辑 | `data/cub_dataset.py`、`data/cub_dedup.py`（均位于 `src/pattern_recognition_coursework/`）、`scripts/audit_clean_cub.py` | 严格清洗索引加载与可重复去重 |
| 数据版本 | `data/splits/cub_clean_v1/` | 清洗后划分、内容哈希、去重策略与完整性标记；不包含原始图片 |
| Backbone | `src/pattern_recognition_coursework/models/backbones/resnet.py` | 历史 BN 缓冲区兼容及监督分类封装 |
| 训练 | `src/pattern_recognition_coursework/engine/supervised.py` | 监督训练、日志、恢复、导出与无 Git 环境版本记录 |
| 服务器补丁 | `scripts/patch_supervised_no_git.py`、`cuda_test.py` | 旧迁移包修补及设备检查 |
| 回归测试 | `tests/test_cub_dedup.py`、`tests/test_backbone_legacy_loading.py`、`tests/test_supervised_training.py`、`tests/test_supervised_code_version.py` | 清洗、权重严格性、恢复与迁移验证 |
| 文档 | `docs/README.md`、`docs/stage04_supervised.md`、`docs/research_audit/`、本文件、根目录最终开题报告 PDF | 协议、审计结论和运行说明 |
| 实验证据 | `outputs/` 中未被新规则排除的文件 | 保留配置、权重来源/哈希、划分统计、验证任务、CSV/JSON 日志、报告、SVG/PNG、测试结果和审计脚本 |

`outputs/stage01_data/evidence/image_sha256.csv` 约 10.05 MiB，是重复内容审计的直接证据，建议保留。12 组数据增强样例、联系图等体积适中，也可保留。

## 应忽略

- 原始数据和本地虚拟环境：已有规则保留。
- `server_transfer/`：迁移包和快照。
- `*.pth`、`*.pt`、`*.ckpt`、`*.onnx`、`*.safetensors`：初始化权重、训练 checkpoint、导出权重和合成测试模型。
- `pytest_tmp/`、`outputs/` 下以 `_tmp` 结尾的目录：自动生成的测试数据和中间状态。
- TensorBoard 事件文件和输出下的 `logs/tensorboard/`：可由 CSV/JSON 查看相同标量，事件文件在本地保留。
- `outputs/` 下的 `local_torch_cache/`：本机缓存与绝对路径链接。
- `*.tmp`、`*.before_no_git_fix`、Python 缓存、IDE 与机器工具目录。

Git 忽略规则不会移除已跟踪或已暂存文件。不能使用对全仓库所有忽略文件一概 `git rm --cached` 的方式，因为已有 `reports/` 中的历史报告本来就在 Git 中；它们本次不需要移除。

## 建议执行顺序

先移除明确属于生成物的暂存项，以下命令仅改变 Git 索引，不删除磁盘文件：

```bash
git rm -r --cached --ignore-unmatch -- \
  outputs/stage01_data/cub_dedup/evidence/pytest_tmp \
  outputs/stage01_data/evidence/pytest_tmp \
  outputs/stage02_backbone/evidence/synthetic_random_encoder.pt
```

再按具体路径暂存，避免将已有 `.idea/` 删除混入科研功能提交：

```bash
git add -- .gitignore pyproject.toml uv.lock src scripts tests \
  data/splits/cub_clean_v1 cuda_test.py \
  docs/README.md docs/stage04_supervised.md docs/research_audit \
  docs/git_submission_review.md outputs 'A03_模式识别开题报告.pdf'

git diff --cached --check
git diff --cached --stat
git status --short
```

有需要可分为两次提交：先提交代码、清洗索引、测试、依赖与运行说明，再提交阶段报告和实验记录。不要把旧 Batch Size 8 的中途记录与服务器 Batch Size 128 的完整运行合并为同一实验。

## 权重如何交付

权重不进入普通 Git；通过已有服务器迁移包或独立文件存储交付，仓库保留其来源、SHA256、转换方法与验证结果。SimCLR 编码器可以从原始已审核 checkpoint 重新转换：

```bash
uv run python scripts/convert_simclr_resnet18.py \
  --checkpoint checkpoint_100-resnet18-simclr-imagenet1k.ckpt \
  --output outputs/local_weights/simclr_resnet18_backbone.pt \
  --provenance outputs/local_weights/simclr_provenance.json
```

该命令拒绝覆盖现有目标；训练时传入实际导出文件路径。仅克隆 Git 仓库不会包含原始数据或模型二进制。
