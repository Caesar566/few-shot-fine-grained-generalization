# 运行脚本

`dataset_get.py`：现有 CUB 数据下载脚本，调用 KaggleHub 并输出实际下载路径。下载位置由 KaggleHub 决定，不保证写入根目录 `cub2002011/`。

`preview_cub.py`：检查 CUB-200-2011 的图像索引、类别标签、官方训练/测试划分、边界框和身体部位标注，并随机绘制预览图。需要 Pillow。在仓库根目录运行：

```bash
uv run python scripts/preview_cub.py
```

默认读取 `cub2002011/versions/7/CUB_200_2011/`，结果保存到 `output/cub_preview/`：`report.md` 是可阅读的报告，`class_counts.csv` 列出全部 200 类的样本数，`summary.json` 提供机器可读结果，`previews/` 存放标注图。可用 `--samples`、`--seed`、`--dataset-root` 和 `--output-dir` 调整运行参数。

待实现入口：生成类别划分、训练、评估、汇总实验结果。
