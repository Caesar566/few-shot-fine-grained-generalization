# 实验输出

`runs/` 保存每次训练的最佳模型、逐轮记录与 CUB 未见类指标；`episodes/` 保存固定验证与测试任务；`evaluations/` 保存独立复评及可选 query 预测记录。运行产物默认不提交 Git，需要提交的多种子汇总整理到 `reports/`。

正式报告需要多随机种子和足够数量的固定测试 episode；单个 episode 的贯通检查不构成实验结论。

现有 `cub_study_v1/` 含第一轮 CUB 测试与验证选模记录。`cub_weight_sweep_v1/`、`cub_5shot_selection_v1/`、`cub_fixed_feature_v1/`、`cub_simclr_v1/`、`cub_random_frozen_v1/`、`cub_method_ablation_v1/`、`cub_meta_baseline_v1/` 和 `cub_scope_v1/` 为后续验证集实验；它们的结果索引见 `reports/cub_progress_2026-10-23.md`。检查点和 episode 计划保留用于复核，不提交 Git。

`cub_fce_v1/`、`cub_fce_5shot_v1/` 为完整上下文 Matching Networks 的专门 1-shot/5-shot 训练；`cub_supervised_v1/` 为监督最近类中心对照；`cub_budget_v1/` 和 `cub_budget_comparison_v1/` 为长预算与原短训的同任务验证。结果和限制见 `reports/cub_followup_2026-10-23.md`。
