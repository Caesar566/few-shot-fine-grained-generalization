# 第一阶段：CUB重复样本审计与逻辑去重

原图与原始六份划分文件均未修改。扫描11796张图片（官方索引11788张、未索引8张），发现1组完全重复、1张冗余图片，无跨类别或跨划分重复。保留ID3310，派生索引排除ID3340，两张原图均保留。

| 划分 | 原始图数 | 清洗后 | 类别数 |
|---|---:|---:|---:|
| train | 5890 | 5890 | 100 |
| val | 2947 | 2947 | 50 |
| test | 2951 | 2950 | 50 |

清洗目录：`data/splits/cub_clean_v1/`。训练及正式评估需显式传入 `--splits-dir data/splits/cub_clean_v1`；原默认路径未切换。Dataset可用`require_clean_index=True`强制要求清洗标记。跨类别/划分重复全部保留待审，未解决时Dataset拒绝使用该派生索引。

测试：28 passed；三划分、两shot、query=1/5/15共18000个新episode均无内容重复，seed42可复现，seed43改变任务。11796张原图重新哈希一致。旧test任务指纹不适用于清洗索引，旧checkpoint也会因原清单哈希校验拒绝直接复评，不能静默沿用旧成绩。模型与训练代码未改动。

[重复明细CSV](duplicate_samples.csv) · [全量哈希CSV](full_inventory.csv) · [逐类对比](class_counts_comparison.csv) · [验证证据](evidence/real_data_verification.json) · [测试日志](evidence/pytest_stdout.txt)

目录由output/stage02_cub_dedup迁入第一阶段；旧测试日志内原绝对路径为历史记录，不改写。没有移动任何原始图片。
