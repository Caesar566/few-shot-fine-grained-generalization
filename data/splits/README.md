# 类别划分清单

当前包含 CUB 训练、验证、测试的类别和图像清单。由 `scripts/make_splits.py` 使用随机种子 42 从 CUB-200-2011 官方 `classes.txt`、`images.txt` 生成，类别数为 100/50/50；可用 `scripts/validate_cub.py` 与官方元数据核对。NABirds 测试清单仍待物种名称复核后建立。

训练、验证、测试物种互不相交；跨 CUB/NABirds 需核对物种重叠，不能只比较类别 ID。数据集默认图像划分不能直接视为物种不相交划分。
