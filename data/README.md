# 数据与划分

原始 CUB 数据暂保留在根目录 `cub2002011/`，避免影响已有下载。后续可将新增原始数据放在 `raw/`，处理结果放在 `processed/`；两者均不提交 Git。

`split` 文件统一放在 `splits/`，数据路径通过后续配置指定。

数据集结构

cub2002011/
├── 7.complete                 # 下载完成标记
└── versions/7/
    ├── CUB_200_2011/         # 原始图像及官方标注，约 1.3 GB
    │   ├── images/           # 按鸟类分成 200 个目录
    │   ├── images.txt        # 图像 ID → 相对路径
    │   ├── classes.txt       # 类别 ID → 鸟类名称
    │   ├── image_class_labels.txt
    │   ├── train_test_split.txt
    │   ├── bounding_boxes.txt
    │   ├── parts/            # 15 个身体部位的坐标标注
    │   └── attributes/       # 鸟类外观属性标注
    ├── segmentations/        # 按类别存放的 PNG 分割掩码，约 58 MB
    └── cvpr2016_cub/         # 文本描述、特征及类别划分，约 5.8 GB

核心数据怎么对应：[images.txt]给每张图像一个 image_id；image_class_labels.txt、train_test_split.txt 和 bounding_boxes.txt 都用这个 ID 关联类别、官方训练/测试划分和鸟的边界框。parts/ 记录部位坐标，attributes/ 包含 312 项属性名称、逐图属性标注及逐类属性值。原始数据的字段格式见 [README](/home/caesar/Projects/few-shot-fine-grained-generalization/cub2002011/versions/7/CUB_200_2011/README)。

cvpr2016_cub/ 另有一套按类别划分的 trainclasses.txt、valclasses.txt、testclasses.txt，分别包含 100、50、50 类，与原始数据的按图像训练/测试划分是两套不同索引。它还包含逐图文本描述（text_c10/ 下的 .txt）、对应 .h5 文件，以及 bow_c10/、w2v_c10/、word_c10/ 下的 .t7 文件。

数量核对：images.txt 收录 11,788 张图像；分割掩码和逐图文本各有 11,788 份，能与索引对应。images/ 实际有 11,796 个 JPG，多出的 8 个均以 _rgb.jpg 结尾，未列入 images.txt。因此按数据集样本读取时，应以索引中的 11,788 张为准。