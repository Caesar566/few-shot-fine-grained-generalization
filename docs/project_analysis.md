# 小样本细粒度图像分类的跨数据集泛化研究项目分析

只给模型每个鸟类很少的样本，让它学会区分很相似的鸟；然后不只在原来的数据集上测，而是换一个来源不同的数据集，看它还能不能认出来。

主要问题：为什么小样本细粒度模型一换数据集就掉性能？到底是特征提取不够泛化，还是不同数据集之间的匹配/对齐方式有问题？

## 基本名词解释

### 小样本学习中的 5-way、1-shot 含义

- 5-way：一次任务包含 5 个类别
- 1-shot：每个类别只有 1 张已知样本

### 细粒度分类含义

样本仅在局部会有细微的差别，整体长得几乎一样

**细粒度分类要求模型学习非常细节的视觉特征。**

### 跨数据集泛化含义

测试的时候出现的是模型从来没有见过的新鸟类。

CUB训练 -> 模型 -> NABirds测试

## 参考文献指定的方法

**Prototypical Network (ProtoNet)**

### ProtoNet 做了什么

1. 图片经过CNN后计算特征向量作为类别原型(prototype)
2. Query 图片也提取特征得到 query feature
3. 计算Query 图片与之前 5个 prototype 的距离，离那个最近就认为是哪个

即不同于:
CNN -> Linear -> Softmax

而是：
CNN -> 特征空间 -> 距离分类

## 距离计算方法

### 欧氏距离

两点间的直线距离：

$$
d(x,y) = \sqrt{\sum_{i}(x_i - y_i)^2}
$$

### Cosine Similarity

更关注特征方向是否相似，而不是绝对的大小

$$
\cos(x, y) =\frac{x \cdot y}{\Vert x \Vert \Vert y \Vert}
$$

### Mahalanobis Distance

不同特征维度之间的相关性

观察三种距离计算方式，哪种度量对跨数据集泛化影响最大。

## 项目具体要完成的实验

1. 复现 ProtoNet，先证明 Few-Shot pipeline 是正确的，并将这个复现作为 baseline

2. 设计不相交类别划分$\ C_{train} \cap C_{test} = 0$，去测试模型能否对没见过的新类别进行 Few-Shot Classification。

3. 做跨数据集实验(**主实验**)$\ Generalization\ Gap = Acc_{in-domain}-Acc_{cross-domain}$

4. 研究特征骨干 Backbone。题目明确要求：ResNet-18，并比较 普通 ImageNet 预训练 与 自监督预训练。比如 ResNet18 ImageNet 与 ResNet18 SimCLR

5. 错误样本归因：将错误的样本拿出来对姿态、光照、部位遮挡做分类分析 **模型到底在哪些情况下容易跨域失败。**

## 对于 5% 提升的分析

1. 度量函数： 欧式距离 -> 余弦距离 -> 马氏距离
2. 预训练的模型改进
3. 数据增强
4. 特征归一化