# 模型

`few_shot.py` 统一全局 ProtoNet、简化 Matching Networks、带上下文编码的匹配网络、DN4 式局部描述子匹配和全局局部融合。episode 顺序是先按类排列 support，再按类排列 query。原型分支支持欧氏、余弦、收缩马氏距离和 support 统计量标准化。可选线性适配器、温度以及全量、ResNet layer4 或仅适配器训练。DN4 与 FCE 是报告中所述思想的工程实现，正式复现仍需核对论文所有训练细节。
