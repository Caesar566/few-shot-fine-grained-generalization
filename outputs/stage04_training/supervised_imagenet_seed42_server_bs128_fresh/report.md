# Stage04监督训练结果

训练使用100类监督分类头；验证复用未见类cosine原型评分。Train/Val Loss与Accuracy属于不同任务，不可当作同一个分类头的训练/验证差距。验证不使用NABirds或CUB test。

```json
{
  "status": "completed",
  "best_epoch": 17,
  "best_validation_accuracy": 0.8783999999999998,
  "best_validation": {
    "accuracy": 0.8783999999999998,
    "nll": 1.4597597455947595,
    "macro_f1": 0.8728277500277499,
    "ece": 0.6441205981049927,
    "episode_accuracies": [
      0.92,
      1.0,
      0.8,
      0.88,
      0.76,
      0.68,
      0.96,
      0.92,
      0.76,
      0.96,
      0.96,
      0.88,
      1.0,
      0.84,
      0.96,
      0.48,
      0.84,
      0.72,
      0.92,
      0.88,
      0.92,
      0.88,
      0.96,
      0.88,
      0.96,
      1.0,
      0.8,
      0.84,
      1.0,
      1.0,
      0.96,
      0.88,
      0.88,
      0.68,
      1.0,
      0.84,
      0.96,
      0.64,
      0.92,
      0.96,
      0.8,
      1.0,
      0.96,
      0.96,
      0.68,
      0.92,
      0.84,
      0.88,
      0.96,
      1.0,
      0.92,
      0.92,
      0.92,
      0.92,
      0.64,
      0.96,
      0.76,
      0.84,
      0.84,
      0.96,
      1.0,
      0.96,
      0.76,
      0.72,
      0.88,
      0.64,
      0.84,
      0.96,
      0.92,
      0.76,
      0.88,
      0.76,
      0.96,
      0.96,
      0.8,
      0.92,
      0.96,
      0.8,
      0.96,
      0.84,
      0.88,
      0.92,
      1.0,
      0.84,
      0.8,
      0.88,
      0.96,
      0.76,
      0.92,
      0.96,
      0.84,
      0.84,
      0.96,
      1.0,
      0.72,
      0.84,
      0.88,
      0.92,
      0.92,
      0.96
    ],
    "episode_std": 0.10184876885447224,
    "episodes": 100,
    "samples": 2500,
    "confusion_matrix": [
      [
        442,
        14,
        12,
        9,
        23
      ],
      [
        8,
        451,
        15,
        12,
        14
      ],
      [
        17,
        19,
        436,
        15,
        13
      ],
      [
        10,
        12,
        19,
        432,
        27
      ],
      [
        21,
        18,
        16,
        10,
        435
      ]
    ],
    "confusion_matrix_labels": "episode-local class slots; rows=true, columns=predicted",
    "calibration_bins": 15,
    "metric_aggregation": "unweighted mean of episode accuracy, NLL, macro-F1 and ECE",
    "loss": 1.4597597455947595
  },
  "best_reload_validation_reproduced": true,
  "best_checkpoint": "/root/autodl-tmp/test/few-shot-fine-grained-generalization/outputs/stage04_training/supervised_imagenet_seed42_server_bs128_fresh/checkpoints/best.pth",
  "last_checkpoint": "/root/autodl-tmp/test/few-shot-fine-grained-generalization/outputs/stage04_training/supervised_imagenet_seed42_server_bs128_fresh/checkpoints/last.pth",
  "backbone_export": "/root/autodl-tmp/test/few-shot-fine-grained-generalization/outputs/stage04_training/supervised_imagenet_seed42_server_bs128_fresh/checkpoints/backbone_best.pth",
  "epochs_completed": 50,
  "validation_kind": "unseen-class episodic cosine prototypes",
  "train_kind": "supervised 100-base-class head (actual count in config)",
  "amp_effective": true
}
```

[配置](config.json) · [逐epoch CSV](logs/epochs.csv) · [逐step CSV](logs/steps.csv) · [运行日志](logs/run.log)

![loss](figures/loss.svg)

![accuracy](figures/accuracy.svg)

![learning_rate](figures/learning_rate.svg)

![gradient_norm](figures/gradient_norm.svg)

Checkpoint为epoch边界恢复，包含分类头、optimizer、scheduler和AMP scaler。修改batch size恢复会改变训练轨迹。未承诺跨硬件位级一致或中途batch精确续训。
