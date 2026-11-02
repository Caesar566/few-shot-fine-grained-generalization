# 训练与评估流程

`experiment.py` 在 CUB 上执行 episodic、监督最近类中心或监督后 episodic 微调，仅使用固定验证 episode 选最佳模型。`evaluate.py` 从检查点重建模型，复用保存的 CUB 测试任务，或读取人工审核的 NABirds 物种清单。两者都记录协议、设备、清单哈希和测试任务哈希。训练不支持中断后的精确续训；长实验应预留计算时间。
