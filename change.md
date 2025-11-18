## 2025-11-09 更新摘要

- **模型框架**  
  - 为 `models/molopt_score_model.py` 注入全局/局部梯度融合、动态大步探索与精修采样方法，并新增 `GlintDM` 类作为 TargetDiff 的增强模型入口。  
  - 支持以 log-prob/logits 输入类别信息，新增分类稳定损失权重 `loss_v2_weight` 及若干梯度裁剪参数。
  - 通过 `_dynamic_diffusion` / `sample_diffusion_large_step` / `sample_diffusion_refinement` 实现 DiffDynamic 的核心两阶段调度，保持与 TargetDiff 原有 API 兼容。
  - `GlintDM.dynamic_sample_diffusion()` 暴露统一动态采样接口，封装大步探索与精修流程，供采样脚本直接驱动。
  - 动态步长默认遵循 README 中的 `n = a·λ_t + b` 公式：大步阶段 `a=80,b=20`，精修阶段 `a=40,b=5`，确保 1000→500→0 的分段采样策略。

- **训练脚本**  
  - `scripts/train_diffusion.py` 根据配置项 `model.name` 自动选择 `ScorePosNet3D` 或 `GlintDM`，日志中额外输出 `loss_v2`。

- **采样脚本**  
  - `scripts/sample_diffusion.py` 在保持 baseline 的前提下，利用 `sample_dynamic_diffusion_ligand()` 复刻 DiffDynamic 的「先大步探索 → 评分筛选 → 精修重复采样」流程。
  - 集成 QED/SA 评分与分子重建，新增 `meta` 字段记录候选筛选与耗时，使 TargetDiff 的采样结果结构与 DiffDynamic 对齐。
  - 自动识别 `GlintDM.dynamic_sample_diffusion()` 实现，当 `sample.dynamic.method` 为 `auto/unified` 时直接调用模型内部的统一动态采样逻辑，否则回退到原有两阶段候选筛选流程；统一入口 `_run_unified_dynamic()` / `_run_legacy_dynamic()` 消除了重复代码。

- **配置文件**  
  - `configs/training.yml` 与 `configs/sampling.yml` 增设 `model.name=glintdm`、动态步长、Top-N 筛选等参数，使 TargetDiff 可以按需切换到 DiffDynamic 行为，同时保留原配置默认值。
  - `configs/sampling.yml::sample.dynamic.method` 支持 `auto/unified/legacy`，便于在统一动态接口与旧版候选筛选之间切换。
  - `configs/sampling.yml::sample.dynamic.large_step/refine.schedule=lambda` 默认启用 λ_t 动态步长，`time_lower/time_upper` 分别限定 500→0 转换点，可按需覆盖。

- **文档**  
  - 新增《模型使用引导》文档，说明数据准备、配置方法与训练/采样命令。

