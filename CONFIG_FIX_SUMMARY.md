# sampling.yml 配置参数修复总结

## 修改日期
2024年

## 问题描述

在 `unified` 动态采样模式（`sample.dynamic.method = 'unified'` 或 `'auto'`）下，`sampling.yml` 中的以下配置参数未被使用：

- `sample.dynamic.large_step.*` 的所有参数
- `sample.dynamic.refine.*` 的所有参数

这些参数在 unified 模式下会使用训练时的默认值，而不是 `sampling.yml` 中指定的值。

## 根本原因

1. `_run_unified_dynamic` 函数调用 `model.dynamic_sample_diffusion()` 时，只传递了 `num_steps`, `center_pos_mode`, `pos_only` 参数
2. `dynamic_sample_diffusion` 方法内部使用 `self.dynamic_large_step_defaults` 和 `self.dynamic_refine_defaults` 来读取配置
3. 这些默认值是在模型初始化时从训练配置文件（`training.yml`）中读取的，而不是从 `sampling.yml` 中读取

## 解决方案

修改 `scripts/sample_diffusion.py` 中的 `_run_unified_dynamic` 函数，在采样循环开始之前，从 `sampling.yml` 读取配置并更新模型的默认值。

## 具体修改

### 文件: `scripts/sample_diffusion.py`

**位置**: `_run_unified_dynamic` 函数，在采样循环之前（第49-59行）

**修改内容**:
```python
# 从 sampling.yml 读取 dynamic 配置并更新模型的默认值，确保 unified 模式使用正确的配置
if 'large_step' in dynamic_cfg:
    # 保存原始默认值（如果需要恢复）
    original_large_step_defaults = getattr(model, 'dynamic_large_step_defaults', {})
    # 更新为 sampling.yml 中的配置，保留原始值中未覆盖的字段
    model.dynamic_large_step_defaults = {**original_large_step_defaults, **dynamic_cfg['large_step']}
if 'refine' in dynamic_cfg:
    # 保存原始默认值（如果需要恢复）
    original_refine_defaults = getattr(model, 'dynamic_refine_defaults', {})
    # 更新为 sampling.yml 中的配置，保留原始值中未覆盖的字段
    model.dynamic_refine_defaults = {**original_refine_defaults, **dynamic_cfg['refine']}
```

## 修改效果

修改后，以下 `sampling.yml` 中的参数在 unified 模式下会被正确使用：

### large_step 配置
- ✅ `sample.dynamic.large_step.schedule` - 调度类型（lambda/固定步长）
- ✅ `sample.dynamic.large_step.stride` - 时间步间隔
- ✅ `sample.dynamic.large_step.step_size` - 更新步幅
- ✅ `sample.dynamic.large_step.noise_scale` - 额外噪声强度
- ✅ `sample.dynamic.large_step.time_lower` - 大步阶段的最小时间步

### refine 配置
- ✅ `sample.dynamic.refine.schedule` - 调度类型
- ✅ `sample.dynamic.refine.stride` - 时间步间隔
- ✅ `sample.dynamic.refine.step_size` - 精炼步幅
- ✅ `sample.dynamic.refine.noise_scale` - 额外噪声
- ✅ `sample.dynamic.refine.time_upper` - 精炼起始时间步
- ✅ `sample.dynamic.refine.time_lower` - 精炼结束时间步
- ✅ `sample.dynamic.refine.cycles` - 重复精炼的轮数

## 工作原理

1. 在 `_run_unified_dynamic` 函数开始时，从 `config.sample.dynamic` 读取配置
2. 如果存在 `large_step` 或 `refine` 配置，则更新模型的 `dynamic_large_step_defaults` 和 `dynamic_refine_defaults` 属性
3. 使用字典合并（`{**original, **new}`）来保留原始默认值中未覆盖的字段
4. 当 `dynamic_sample_diffusion` 方法被调用时，它会从更新后的默认值中读取配置
5. 所有后续的采样操作都会使用 `sampling.yml` 中指定的配置

## 向后兼容性

- ✅ 如果 `sampling.yml` 中没有 `dynamic.large_step` 或 `dynamic.refine` 配置，代码会继续使用模型的原始默认值
- ✅ 如果 `sampling.yml` 中只提供了部分配置项，未提供的项会保留原始默认值
- ✅ Legacy 模式的代码未修改，继续正常工作

## 测试建议

1. 使用 unified 模式运行采样，验证配置参数是否生效
2. 修改 `sampling.yml` 中的 `dynamic.large_step.stride` 等参数，观察采样行为是否改变
3. 对比使用不同配置参数时的采样结果，确认参数确实生效

## 相关文件

- `scripts/sample_diffusion.py` - 修改的文件
- `models/molopt_score_model.py` - 使用配置的模型文件（未修改）
- `configs/sampling.yml` - 配置文件
- `sampling_config_parameter_check.md` - 详细的参数检查报告

