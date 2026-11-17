# MX450显存实测：当前Stage04配置

**PASS：2 Step均完成，没有发生OOM。** 只执行用户批准的2步探针，没有启动1000步正式实验，没有修改配置。

- 设备：NVIDIA GeForce MX450，总显存1696.9 MiB。
- Random seed42，完整ResNet-18+ProtoNet，224×224，5-way 1-shot，每类5 Query，单episode30张图。
- FP32、Adam lr1e-4、全量骨干训练；未启用AMP、梯度累积或微批处理。复用现有run_episodes，首2步包含与Stage04相同的参数快照检查。
- 两步均完成前向、交叉熵、反向与Adam更新；梯度有限且参数确实变化。
- PyTorch实测峰值已分配显存：**904.9 MiB**；峰值缓存/保留显存：**1022.0 MiB**。第二步结束时CUDA报告空闲605.9 MiB；这是结束时测量，不是整个过程最小空闲量。

| Step | Loss | Accuracy | 耗时秒 | 峰值allocated MiB | 峰值reserved MiB |
|---|---:|---:|---:|---:|---:|
| 1 | 1.612425 | 16.00% | 0.845 | 795.3 | 1022.0 |
| 2 | 1.639032 | 16.00% | 0.350 | 904.9 | 1022.0 |

显存探针通过不证明收敛或泛化，也不保证任意后续后台显存占用下均不OOM。未测完整验证/最佳checkpoint重加载阶段的显存峰值；未单独运行ImageNet和SimCLR的GPU训练。正式实验仍需按此前要求明确批准，串行运行。探针模型未保存为正式checkpoint，两个训练点不混入Stage04主比较表。

[完整结果](result.json) · [原始日志](run.log) · [执行脚本](probe.py)
