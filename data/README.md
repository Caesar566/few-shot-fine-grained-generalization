# 数据与划分

`raw/CUB_200_2011/` 是已有的 CUB 原始图像和官方元数据；`raw/nabirds/` 是 NABirds 官方压缩包经校验后的目标目录。`splits/` 保存版本化的 CUB 类别/图像清单及后续人工审核的 NABirds 清单。`cache/` 放工程内下载缓存。原始图像和缓存不提交 Git。

目前 `raw/nabirds/` 尚不存在。NABirds 官方页面通过下载表单提供数据；取得官方压缩包后运行 `uv run python scripts/import_nabirds.py <压缩包路径>`，再按 `docs/nabirds_protocol.md` 审核物种重叠。不要用未经核实的第三方镜像替换官方数据。
