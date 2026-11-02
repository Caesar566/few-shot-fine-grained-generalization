# 数据加载与 episode 采样

`cub_dataset.py` 读取 CUB 清单并采样 support/query 无重叠任务；`episodes.py` 保存和复用完整任务索引。`nabirds.py` 解析 NABirds 类别层级、生成待人工审核清单，并只装载经物种审核的目标域图像。
