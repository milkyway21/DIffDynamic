# `_with_noise` 方法实现总结

## 修改概述

根据官方 [GlintDM 仓库](https://github.com/DMCB-GIST/GlintDM) 的实现，已在项目中集成了 `_with_noise` 方法的使用。

## 主要修改

### 1. `_dynamic_diffusion` 方法

**位置**: `models/molopt_score_model.py` 第 856-988 行

**修改内容**:
- 添加了 `use_with_noise` 参数（默认 `False`）
- 在类别更新时，根据 `use_with_noise` 标志选择使用：
  - `q_v_posterior_with_noise`: 带噪声的后验计算（更平滑）
  - `q_v_posterior`: 标准后验计算（更快）

**关键代码**:
```python
if use_with_noise:  # 使用带噪声的后验计算，提供更平滑的分布。
    log_model_prob = self.q_v_posterior_with_noise(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
else:  # 使用标准方法。
    log_model_prob = self.q_v_posterior(log_ligand_v_recon, log_ligand_v, t, batch_ligand)
```

### 2. `sample_diffusion_refinement` 方法

**位置**: `models/molopt_score_model.py` 第 1063-1160 行

**修改内容**:
1. **初始化阶段**: 使用 `q_v_pred_with_noise` 初始化 `log_ligand_v`
   - 提供更平滑的起点
   - 符合官方 GlintDM 的实现

2. **迭代阶段**: 传递 `use_with_noise=True` 给 `_dynamic_diffusion`
   - 默认启用带噪声方法
   - 可通过配置 `use_with_noise` 控制

**关键代码**:
```python
# 初始化时使用带噪声的预测
if use_with_noise and not pos_only:
    init_t = torch.full(size=(batch_ligand.max().item() + 1,), fill_value=time_upper, 
                       dtype=torch.long, device=protein_pos.device)
    log_ligand_v = self.q_v_pred_with_noise(log_ligand_v, init_t, batch_ligand)

# 迭代时使用带噪声的后验
ligand_pos_current, log_ligand_v_current, pos_traj, log_v_traj = self._dynamic_diffusion(
    ...,
    use_with_noise=use_with_noise  # 默认 True
)
```

### 3. `sample_diffusion_large_step` 方法

**位置**: `models/molopt_score_model.py` 第 990-1055 行

**修改内容**:
- 传递 `use_with_noise=False` 给 `_dynamic_diffusion`
- 大步阶段通常不使用带噪声方法，保持标准方法以获得更快的探索

**关键代码**:
```python
# 大步阶段通常不使用带噪声方法
use_with_noise = defaults.get('use_with_noise', False)  # 默认 False
ligand_pos, log_ligand_v, pos_traj, log_v_traj = self._dynamic_diffusion(
    ...,
    use_with_noise=use_with_noise  # 默认 False
)
```

## 配置选项

### 在配置文件中添加

可以在 `configs/sampling.yml` 或训练配置中添加：

```yaml
sample:
  dynamic:
    refine:
      use_with_noise: true  # 精炼阶段使用带噪声方法（默认 true）
    large_step:
      use_with_noise: false  # 大步阶段不使用带噪声方法（默认 false）
```

### 默认行为

- **精炼阶段 (`sample_diffusion_refinement`)**: 
  - `use_with_noise = True` (默认)
  - 使用 `q_v_pred_with_noise` 初始化
  - 使用 `q_v_posterior_with_noise` 计算后验

- **大步阶段 (`sample_diffusion_large_step`)**: 
  - `use_with_noise = False` (默认)
  - 使用标准方法以获得更快的探索

## 使用场景

### 何时使用带噪声方法？

✅ **推荐使用**:
- 精炼采样阶段（`sample_diffusion_refinement`）
- 需要平滑分布的场景
- 梯度更新可能不稳定的情况
- 需要探索-利用平衡的场景

❌ **不推荐使用**:
- 大步探索阶段（`sample_diffusion_large_step`）
- 训练阶段
- 标准扩散采样（`sample_diffusion`）

## 效果预期

使用 `_with_noise` 方法后，预期效果：

1. **更平滑的类别分布**: Gumbel 噪声软化离散分布
2. **更稳定的梯度更新**: 减少数值溢出/下溢
3. **更好的收敛性**: 更平滑的分布有助于更快、更稳定的收敛
4. **探索-利用平衡**: 在精炼阶段保持一定随机性

## 验证方法

可以通过以下方式验证实现：

1. **检查初始化**: 在 `sample_diffusion_refinement` 开始时，`log_ligand_v` 应该更平滑
2. **监控梯度**: 使用 `enable_monitoring=True` 观察梯度范数是否更稳定
3. **对比结果**: 对比使用/不使用 `_with_noise` 的采样质量

## 参考

- 官方 GlintDM 仓库: https://github.com/DMCB-GIST/GlintDM
- 详细说明文档: `WithNoise_Methods_Explanation.md`
- 对比报告: `GlintDM_Comparison_Report.md`

