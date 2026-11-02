# 运行脚本

`dataset_get.py`：把 KaggleHub 下载缓存固定在工程目录 `data/cache/kagglehub/`，并打印数据路径。

`make_splits.py`：按固定随机种子生成 CUB 100/50/50 类清单。重新运行会覆盖现有清单，正式实验中应保存清单版本。

`validate_cub.py`：将现有划分与 CUB 官方元数据逐项核对；`--check-images` 还检查每张图像是否存在。训练与评估入口见 `python -m pattern_recognition_coursework.engine.experiment --help`。

`audit_cub_duplicates.py`：对现有 CUB 图像计算 SHA256，报告划分内/跨划分的精确重复，并检查第一轮保存的测试 episode 是否让重复图像同时进入 support 和 query。输出位于 `reports/records/cub_duplicate_audit_v1.json`。

`prepare_nabirds.py`：生成待人工审核的物种表，再从明确确认不重叠的类别构建目标域清单。详见 `docs/nabirds_protocol.md`。

`import_nabirds.py`：把通过 Cornell 官方下载表单取得的 ZIP/TAR 压缩包安全解压到 `data/raw/nabirds/`，并校验元数据与索引图像；拒绝覆盖已有数据。
