# DiffDynamic → TargetDiff Migration Guide

本指南总结了当前 `DiffDynamic` 项目（集成 GlintDM + 动态跳步采样）与官方 [TargetDiff](https://github.com/guanjq/targetdiff) 基线之间的核心差异，并给出将这些增强能力迁移回 TargetDiff 代码库的操作步骤。阅读完本文件后，下一个 Cursor Agent 可以按部就班地在 TargetDiff 中复刻 GlintDM 与动态跳步采样/评分流程。

---

## 1. 项目现状概览

- **当前仓库 (`DiffDynamic`) 特性**
  - `models/GlintDM.py` 对扩散主干做了大幅改写，引入 **全局/局部梯度融合、动态步长、分类扩散稳定项**。
  - 采样脚本（`sample_CrossDocked_GlintDM.py`、`sample_MOAD_GlintDM.py`）实现 **先验原子数筛选 → 大步长探索 → Top-N 评分 → 精修采样** 的流水线，并内嵌 Vina/QED/SA 评分。
  - `utils/evaluation`、`utils/reconstruct`、`utils/transforms` 等工具配合评分/重建流程做了补强（重用 TargetDiff 原模块的基础上增加功能）。
  - `train_GlintDM.py`、`configs/*` 继承 TargetDiff 训练流程但含有多处硬编码（路径、设备）与日志增强，需要回落到通用写法。
  - `DiffDynamic help.md`、`DYNAMIC_STEP_README.md` 等文档记录了当前环境、运行命令、动态跳步细节。

- **TargetDiff 基线特性** [[1]](https://github.com/guanjq/targetdiff)
  - `models/molopt_score_model.py` 提供标准 ScoreNet（`ScorePosNet3D`），采样 `sample_diffusion()` 为 **固定步长、逐步退火**。
  - `scripts/sample_diffusion.py` 只负责单阶段采样与轨迹保存；Docking/QED 等评估通过独立脚本之后处理。
  - 配置、训练脚本与 README 针对原生流程设计，缺乏 GlintDM 和动态跳步相关的参数及说明。

---

## 2. 模块级对照（DiffDynamic vs TargetDiff）

| 模块 | TargetDiff 位置 | DiffDynamic 位置 | 主要差异 | 迁移动作 |
| --- | --- | --- | --- | --- |
| 扩散主干 | `models/molopt_score_model.py::ScorePosNet3D` | `models/GlintDM.py::GlintDM` | 输入处理方式（one-hot vs log prob）、采样算法、loss 组合、异常处理 | 用 `GlintDM` 替换/继承 `ScorePosNet3D`，并同步依赖函数 |
| 采样脚本 | `scripts/sample_diffusion.py` | `sample_CrossDocked_GlintDM.py`、`sample_MOAD_GlintDM.py` | 多阶段采样、Top-N 评分、动态跳步 | 重写采样脚本，抽象评分器，可配置化参数 |
| 训练脚本 | `scripts/train_diffusion.py` | `train_GlintDM.py` | 训练流程基本一致，DiffDynamic 有日志/路径硬编码与额外 loss 输出 | 合并差异，保留 GlintDM 所需日志，移除硬编码 |
| 工具/评估 | `utils/evaluation/*` `utils/reconstruct.py` | 同路径，但包含附加函数和修复 | DiffDynamic 延伸原模块（如 docking、评分、重建）；部分修复来自 `DiffDynamicUpdate.md` | 对比差异后挑选必要补丁，谨慎避免回滚 TargetDiff 原实现 |
| 配置 | `configs/training.yml` `configs/sampling.yml` | 同名文件 | 参数项类似，但 DiffDynamic 通过脚本内变量补充 TopN 等 | 将新增参数移入配置文件，保持默认行为兼容 |
| 文档 | `README.md` 等 | `DiffDynamic help.md`、`DYNAMIC_STEP_README.md` | DiffDynamic 记录动态跳步细节、环境说明 | 将关键段落迁移到 TargetDiff 文档或新附录 |

---

## 3. 重点差异详解

### 3.1 模型前向输入与损失

- **TargetDiff** 先对类别做 one-hot，再送入 Transformer：

```313:348:targetdiff_base/models/molopt_score_model.py
    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):

        batch_size = batch_protein.max().item() + 1
        init_ligand_v = F.one_hot(init_ligand_v, self.num_classes).float()
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
```

- **DiffDynamic** 直接传递 log-prob/logits，并允许时间嵌入拼接：

```305:338:models/GlintDM.py
    def forward(self, protein_pos, protein_v, batch_protein, init_ligand_pos, init_ligand_v, batch_ligand,
                time_step=None, return_all=False, fix_x=False):
        # time embedding
        if self.time_emb_dim > 0:
            if self.time_emb_mode == 'simple':
                input_ligand_feat = torch.cat([
                    init_ligand_v,
                    (time_step / self.num_timesteps)[batch_ligand].unsqueeze(-1)
                ], -1)
```

> **迁移提示**：在 TargetDiff 中接入 GlintDM 时，需要让模型同时兼容 one-hot 与 log-prob 两种输入（例如通过配置控制），确保向后兼容原 `ScorePosNet3D`.

### 3.2 动态步长与梯度融合采样

- TargetDiff 默认采样：

```634:703:targetdiff_base/models/molopt_score_model.py
        for i in tqdm(time_seq, desc='sampling', total=len(time_seq)):
            t = torch.full(size=(num_graphs,), fill_value=i, dtype=torch.long, device=protein_pos.device)
            preds = self(...)
            pos_model_mean = self.q_pos_posterior(x0=pos0_from_e, xt=ligand_pos, t=t, batch=batch_ligand)
            pos_log_variance = extract(self.posterior_logvar, t, batch_ligand)
            nonzero_mask = (1 - (t == 0).float())[batch_ligand].unsqueeze(-1)
            ligand_pos_next = pos_model_mean + nonzero_mask * (0.5 * pos_log_variance).exp() * torch.randn_like(ligand_pos)
```

- DiffDynamic 大步长阶段：

```767:855:models/GlintDM.py
        for i in tqdm(time_seq[::step_size], desc='sampling', total=len(time_seq)):
            ...
            local_pos_grad = pos_model_mean - ligand_pos
            global_pos_grad = pos0_from_e - ligand_pos
            combined_grad = lamb*global_pos_grad + (1-lamb)*local_pos_grad
            ligand_pos_next = ligand_pos + step_size*combined_grad
            ...
            global_v_grad = -cal_kl_gradient(log_ligand_v_recon, log_ligand_v)
            local_v_grad = -cal_kl_gradient(log_model_prob, log_ligand_v)
            combined_v_grad = lamb*global_v_grad + (1-lamb)*local_v_grad
            log_ligand_v_next = F.log_softmax(log_ligand_v + step_size*combined_v_grad, dim=-1)
```

- 精修阶段 `sample_diffusion_refinement` 会在局部窗口内重复上述融合，但以更小步长迭代。

> **迁移提示**：
> 1. 在 TargetDiff 中保留原 `sample_diffusion` 以兼容旧流程，再新增 `sample_diffusion_large_step`、`sample_diffusion_refinement`。
> 2. 设计统一入口（如 `sample_dynamic_diffusion`）根据配置切换算法。
> 3. 梯度融合使用的 `cal_kl_gradient`、`log_add_exp` 等函数请一并迁入。

#### 3.2.1 精修阶段及重复采样流程

- 模型内部的精修阶段 `sample_diffusion_refinement` 会在较小时间窗口内重复梯度融合，逐步收敛到高质量构象：

```873:972:models/GlintDM.py
    def sample_diffusion_refinement(...):
        ...
        for i in tqdm(time_seq[::step_size], desc='sampling', total=len(time_seq)):
            ...
            combined_grad = lamb*global_pos_grad + (1-lamb)*local_pos_grad
            ligand_pos_next = ligand_pos + step_size*combined_grad 
            ...
            combined_v_grad = lamb*global_v_grad + (1-lamb)*local_v_grad
            log_ligand_v_next = F.log_softmax(log_ligand_v + step_size*combined_v_grad, dim=-1)
```

- 采样脚本通过 `sampling()` 对 Top-N 候选执行多次精修，并将结果拼接成批次，便于后续评估：

```45:83:sample_CrossDocked_GlintDM.py
def sampling(...):
    batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
    ...
    for n in range(Nsampling):
        r = model.sample_diffusion_refinement(...)
        r_list.append(r)
    r['pos'] = torch.concat([r_list[i]['pos'] for i in range(N_r)])
    r['v'] = torch.concat([r_list[i]['v'] for i in range(N_r)])
```

- 主循环 `sample_diffusion_ligand` 先进行多轮大步探索并用评分函数挑选 Top-N，然后调用上述 `sampling()` 进行精修采样与打分：

```86:240:sample_CrossDocked_GlintDM.py
for re in range(Nrepeat):
    r = model.sample_diffusion_large_step(...)
...
best_indices = np.argsort(score_list)[:TopN]
...
r, ligand_num_atoms_list = sampling(model, data, ..., init_ligand_pos=ligand_pos, init_log_ligand_v=ligand_log_v, ...)
```

> **迁移提示**：
> - 在 TargetDiff 中实现动态模式时，建议复刻 `sampling()` 和 `sample_diffusion_ligand()` 的结构，确保 Top-N → 精修 → 打分 的闭环完整保留。
> - 注意 `Batch.from_data_list(..., follow_batch=FOLLOW_BATCH)`、`torch.repeat_interleave` 等 PyG 工具，保持 batch 对齐，避免在多分子精修时出现 batch 索引错乱。

### 3.3 采样脚本流水线差异

- TargetDiff 单阶段采样 → 保存结果：

```31:188:targetdiff_base/scripts/sample_diffusion.py
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms
    )
```

- DiffDynamic 采样包含：
  1. 原子数先验 (`atom_num.sample_atom_num`)。
  2. 多轮 `sample_diffusion_large_step` 迭代，间隔注入 Gumbel 噪声刷新 logits。
  3. 调 Vina/QED/SA 评分筛选 TopN，再调用 `sampling()` 执行精修（内部用 `sample_diffusion_refinement`）。

> **迁移提示**：在 TargetDiff 中新增 `--strategy {baseline,dynamic}` 或配置项，按选择走原流程或动态跳步流水线；评分模块要设成可选（无依赖时跳过）。

### 3.4 Loss 与训练差异

- TargetDiff `loss = loss_pos + loss_v * loss_v_weight`，仅包含 KL + reconstruction。
- DiffDynamic 在 `GlintDM.get_diffusion_loss` 中额外引入 `loss_v2 = F.nll_loss(log_ligand_v_recon, ligand_v)`，并把总损失设为 `loss_pos + loss_v + loss_v2`（无显式权重）。
- 训练脚本 `train_GlintDM.py` 基于 TargetDiff 但有硬编码：
  - `args.device = 'cuda:1'; args.device = 'cpu'`（需要删除）。
  - `vina = '/NAS_Storage4/...'` 未使用，可清理。

> **迁移提示**：统一损失组合方式或通过配置控制权重，避免破坏原权衡；改造训练脚本以便选择 `ScorePosNet3D` 或 `GlintDM`.

### 3.5 工具与修复

- DiffDynamic 累积了多项健壮性补丁（详见 `DiffDynamicUpdate.md`）：
  - `models/common.py`、`models/uni_transformer.py` 针对空张量、维度不匹配的修复。
  - `GlintDM` 中 `_validate_and_fix_batch_protein`（防止 batch id 异常）。
  - `utils/evaluation` 中 Vina/QVina、SA/QED 直接复用 TargetDiff 原实现，但部分函数参数有所扩展。

> **迁移提示**：在 TargetDiff 中引入 GlintDM 时务必同步这些修复，尤其是 transformer 维度展平、batch 校验，否则动态采样阶段会再次触发运行时错误。

---

## 4. 迁移操作手册

> 建议在 TargetDiff 中新建分支 `feature/glintdm_dynamic_sampling`，按顺序完成以下工作；每一步后运行基本脚本验证。

### 步骤 0：准备
1. Clone 官方 TargetDiff → `targetdiff_base/`（已下载）。
2. 对比 `targetdiff_base` 与当前仓库，确认依赖版本：Python 3.8、CUDA 11.6/11.8、`torch==2.0.x`、`torch-geometric==2.2.x`、RDKit、meeko、AutoDockTools、vina。
3. 确定规划：保留 TargetDiff 原 baseline（便于回归），新增 GlintDM 流程。

### 步骤 1：引入 GlintDM 模型
1. 在 TargetDiff `models/` 下新建 `glintdm.py`（或重命名原 ScoreNet），复制 `models/GlintDM.py` 并清理硬编码。
2. 合并 `ScorePosNet3D` 与 `GlintDM` 公共函数（beta 调度、log/kl 工具）以减少重复；可将公共函数抽到 `models/common.py`.
3. 在 `GlintDM` 中保留 `_validate_and_fix_batch_protein` 等安全检查，确保 forward 与采样稳定。
4. 在 TargetDiff 的 `__init__` 或工厂函数中根据配置加载 `ScorePosNet3D` 或 `GlintDM`。

### 步骤 2：实现动态采样接口
1. 在 TargetDiff 新脚本（建议命名 `scripts/sample_dynamic.py`）中复用当前 `sample_CrossDocked_GlintDM.py` 的逻辑：
   - `sample_diffusion_large_step`、`sample_diffusion_refinement` 直接调用模型方法。
   - 封装评分函数 `evaluate_candidates`，依赖 `VinaDockingTask`、`qed`、`compute_sa_score`。
   - 将 `TopN`、`Nrepeat`、`step_size`、`MAX_QED`、`MAX_SA` 等参数放入配置 `configs/sampling.yml.dynamic`.
2. 为兼容原脚本，可在 `scripts/sample_diffusion.py` 中添加 `--mode dynamic`，内部调用新接口。
3. 精修阶段的 `sampling()` 需保留批处理逻辑（`Batch.from_data_list` + `FOLLOW_BATCH`），注意异步评分带来的耗时。

### 步骤 3：整合评分/重建工具
1. DiffDynamic 已经使用 TargetDiff 原版 `utils/evaluation`，若当前仓库中有额外修复（查看 `DiffDynamicUpdate.md`），同步到 TargetDiff。
2. 确认 `utils/reconstruct.py`、`utils/transforms.py` 是否需要新增函数（例如 `get_atomic_number_from_index` 的 aromatic 模式）。
3. 在 TargetDiff `environment.yaml` / `requirements.txt` 补充 `meeko==0.1.dev3`、`vina==1.2.2`、`AutoDockTools_py3` 等。

### 步骤 4：训练脚本与配置
1. 修改 TargetDiff `scripts/train_diffusion.py`，允许通过配置选择模型类型（`model.name = score | glintdm`）。
2. 合并 `train_GlintDM.py` 的额外日志输出，但去掉硬编码设备/路径。
3. 在 `configs/training.yml` 增加 GlintDM 所需参数（如 `dynamic_loss_v2_weight`、`grad_fusion` 开关），默认关闭以保持 baseline。
4. 更新 `configs/sampling.yml` 或新增 `configs/sampling_dynamic.yml`，定义动态采样相关参数。
   - 建议新增字段示例：

```yaml
model:
  name: glintdm            # score | glintdm
  beta_schedule: sigmoid
  ...
  use_grad_fusion: true    # 是否启用全局/局部梯度融合
  loss_v2_weight: 1.0      # 分类额外损失权重

sample:
  mode: dynamic            # baseline | dynamic
  total_candidates: 100    # 初始采样数
  large_step:
    n_repeat: 10
    step_size: 50
  refine:
    start_t: 1000
    step_size: 50
    n_sampling: 5
  selector:
    top_n: 20
    max_qed: 0.5
    max_sa: 0.6
  docking:
    enabled: true          # 无 Vina 时可改为 false
    exhaustiveness: 1
```

### 步骤 5：文档与脚本清理
1. 在 TargetDiff README 或新建 `docs/GLINTDM_DYNAMIC.md`，整理：
   - 环境准备（参考 `DiffDynamic help.md` 中的关键步骤即可，WSL 特定内容可移入 FAQ）。
   - 训练/采样命令示例（含 baseline、dynamic 两套命令）。
   - 动态跳步核心公式（引用 `DYNAMIC_STEP_README.md` 中的 n(t)、梯度融合定义）。
2. 在 DiffDynamic 中仅保留迁移文档及必要说明，其余重复文档可归档。

### 步骤 6：验证 & 回归
1. **基础验证**：在 TargetDiff 中运行 baseline 采样，确保不受新代码影响。
2. **GlintDM 验证**：使用小批（例如 `num_samples=4`、`TopN=2`）跑一次动态采样，检查结果包是否包含评分信息。
3. **性能验证**：记录整体耗时，与 DiffDynamic 当前表现对比（`≤70s/complex` 为目标）。
4. **错误回归**：针对 `DiffDynamicUpdate.md` 中修复过的 bug，编写简短回归脚本（空 batch、极端 `batch_protein` 等）。

推荐的基本验证命令：

```bash
# Baseline 采样（确保旧路径仍可用）
python scripts/sample_diffusion.py configs/sampling.yml -i 0 --device cuda:0 --mode baseline

# 动态模式小批验证
python scripts/sample_diffusion.py configs/sampling_dynamic.yml -i 0 --device cuda:0 --mode dynamic \
  --batch_size 8 --result_path ./outputs/dynamic_smoke

# 训练端 smoke test（10 iter）
python scripts/train_diffusion.py configs/training.yml --device cuda:0 --max_iters 10 \
  --override "model.name=glintdm train.val_freq=10"
```

---

## 5. 迁移完成检查清单

- [ ] TargetDiff 代码库支持 `model.name=glintdm` 并成功训练单个 epoch。
- [ ] 新采样脚本支持 `--mode baseline|dynamic`，动态模式具备 Top-N 评分与精修。
- [ ] 评分依赖可在未安装时优雅降级（给出 warning，跳过 Vina）。
- [ ] 配置文件包含新增参数，默认配置仍与原版行为一致。
- [ ] README/文档更新涵盖环境、命令、动态跳步原理。
- [ ] 至少一次 CrossDocked 动态采样完成，输出包含位置/类别轨迹与评分。

---

## 6. 已知注意事项

1. **动态跳步调度器文档 vs 实际实现**  
   - `DYNAMIC_STEP_README.md` 提到的 `models/dynamic_step/*` 尚未真正实现，现有代码是通过 `GlintDM` 内的两个方法完成动态采样；迁移时以现有实现为准。

2. **评分耗时**  
   - Vina/QVina 依赖外部可执行，采样过程中要考虑并发或缓存机制；必要时提供 `--skip_docking` 开关。

3. **性能监控**  
   - `DiffDynamic help.md` 中的 `TimingStats` 代码示例目前并未接入主流程。若需要 GPU 监控，可另行集成。

4. **硬编码清理**  
   - 清除 `args.device = 'cuda:1'; args.device = 'cpu'`、`'/NAS_Storage4'` 等，改为配置或命令行参数。

5. **Vina/QVina 依赖**  
   - 未安装时应当捕获 ImportError 并给出清晰提示（例如 “未安装 vina，已跳过 docking 评分”），避免采样流程直接崩溃。
   - Windows 环境需确认 `vina` 可执行在 PATH 中，或者提供 `--vina_binary` 参数。

6. **PyG 版本兼容**  
   - 若 TargetDiff 当前环境为 torch>=2.1，需要重新确认 `torch-scatter`、`torch-sparse`、`torch-cluster` 的 wheel。建议保留 DiffDynamic 中的安装建议（conda 优先）。

7. **日志与持久化**  
   - DiffDynamic 会将 Top-N 结果的评分/SMILES 等保存在 result dict 中；迁移到 TargetDiff 后，确保 `.pt` 结果包含相同字段，方便评估脚本读取。

---

## 7. 附录

- **核心文件映射**
  | DiffDynamic | TargetDiff（建议位置） | 说明 |
  |-------------|-------------------------|------|
  | `models/GlintDM.py` | `models/glintdm.py`（新建） | 主模型实现 |
  | `sample_CrossDocked_GlintDM.py` | `scripts/sample_dynamic.py` | 动态、多阶段采样 |
  | `utils/evaluation/*` | 同路径 | 复用并补丁官方模块 |
  | `configs/training.yml` | 同名文件或 `configs/training_glintdm.yml` | 模型/训练参数 |
  | `train_GlintDM.py` | `scripts/train_diffusion.py`（合并逻辑） | 训练入口 |

- **参考资料**
  - TargetDiff 官方仓库 [[1]](https://github.com/guanjq/targetdiff)
  - GlintDM 论文与开源实现（参见 `DiffDynamic help.md` 引用）

---

> ⚠️ 迁移过程中建议按模块提交（模型 → 采样 → 评分 → 配置 → 文档），便于回滚。遇到 DiffDynamic 特有补丁时，可先查看 `DiffDynamicUpdate.md` 获取背景，再决定在 TargetDiff 中如何实现等价修复。完成后请在 TargetDiff 仓库记录新的变更日志，与原生 README 对应章节保持同步。


---

## 2. 关键代码差异

### 2.1 GlintDM 模型增强

- **全局/局部梯度融合**：结合 `q_pos_posterior`（局部梯度）与 `x₀` 预测（全局方向），自适应插值后更新位置与类别。
- **动态步长**：根据 `sigma_t²`、`λ_t = t/T` 设定大步或细化步长。
- **分类扩散双重损失**：在 `loss_v` 外增设 `loss_v2`，抑制类别漂移。
- **健壮性修复**：对空 batch、异常 `batch_protein` 值增加保护逻辑。

示例代码：

```827:849:models/GlintDM.py
local_pos_grad = pos_model_mean - ligand_pos
global_pos_grad = pos0_from_e - ligand_pos
combined_grad = lamb*global_pos_grad + (1-lamb)*local_pos_grad
ligand_pos_next = ligand_pos + step_size*combined_grad
// ... existing code ...
combined_v_grad = lamb*global_v_grad + (1-lamb)*local_v_grad
log_ligand_v_next = F.log_softmax(log_ligand_v + step_size*combined_v_grad, dim=-1)
```

### 2.2 多阶段采样与筛选

- **阶段划分**（`sample_CrossDocked_GlintDM.py`）：
  1. 依据口袋尺寸先从先验采样原子数。
  2. `sample_diffusion_large_step` 做粗粒度探索，多轮迭代更新初值。
  3. 按 Vina/QED/SA 等指标筛选 Top-N，再进入 `sample_diffusion_refinement` 做局部精修与多次重复。
- **目标驱动采样**：实时调用 Vina、QED、SA 估计，采样过程中保存最好分子。

### 2.3 动态跳步文档与实现差异

- `DYNAMIC_STEP_README.md` 描述了 `models/dynamic_step/` 与 `utils/dynamic_gradient/` 等模块，但代码库中尚未真正落地这些目录。当前可用的是 `GlintDM` 内置的 `sample_diffusion_large_step` / `sample_diffusion_refinement` 两阶段逻辑。
- 迁移时需 **基于现有可运行实现**（大步长 + 精修），而不是文档里尚未落地的调度器版本。

### 2.4 评估与工具链

- `utils/evaluation/` 增加了 QVina、Vina docking、SA/QED、Lipinski 等模块，并允许在采样时直接打分。
- `utils/transforms.py`、`utils/reconstruct.py` 中包含生成后重建分子的工具，需确保 TargetDiff 版本对齐。

---

## 3. 迁移步骤（建议顺序）

> 建议在 TargetDiff 仓库中新建特性分支，逐条完成以下任务并配套单元/集成测试。

### 步骤 0：基础准备
1. 将 TargetDiff 与 DiffDynamic 源码同时放置（例如 `targetdiff/`、`diffdynamic/` 两个目录）。
2. 确认 Python 3.8 + CUDA 11.6/11.8 环境，`torch==2.0.x`、`torch-geometric==2.2.x`，确保 RDKit、meeko、vina 等可用。
3. 统一配置文件格式：TargetDiff 使用 `configs/*.yml`，可直接复用 DiffDynamic 的配置项并补充注释。

### 步骤 1：模型层改造
1. 找到 TargetDiff 中负责扩散模型的类（通常在 `models/diffusion.py` 或 `models/model.py`）。
2. 引入或重写为 `GlintDM` 架构，使其包含：
   - 全局/局部梯度融合逻辑。
   - `sample_diffusion_large_step`、`sample_diffusion_refinement` 两阶段方法。
   - 多项 loss 组合（`loss_pos`、`loss_v`、`loss_v2`）。
   - 空 batch / 异常 batch 修复逻辑。
3. 若 TargetDiff 只有单文件模型，可将 `DiffDynamic` 的 `models/GlintDM.py` 迁移过去，并同步 `models/common.py`、`models/uni_transformer.py`、`models/egnn.py` 中的兼容性修复。
4. 清理硬编码路径或 Debug 行（如 `args.device = 'cuda:1'; args.device = 'cpu'` 等）。

### 步骤 2：采样流程升级
1. 以 TargetDiff 原 `scripts/sample_diffusion.py` 为基础，引入以下流程：
   - 原子数先验采样（`utils/evaluation/atom_num.py`）。
   - 多轮 `sample_diffusion_large_step` → 评分筛选 → `sample_diffusion_refinement`。
   - 对应的参数（`TopN`、`Nrepeat`、`step_size`、`total_n_samples` 等）接入 `configs/sampling.yml`。
2. 重构采样输出，保存：
   - 最终坐标/原子类型。
   - 中间轨迹（`pos_traj`、`v_traj` 等）以便可视化。
   - 评分统计（Vina、QED、SA）和耗时。
3. 抽象出评分器接口（如 `score_ligand` 函数），方便后续扩展其他目标。

### 步骤 3：评估/工具整合
1. 将 `utils/evaluation/` 中新增的 scoring、docking、similarity 模块迁移到 TargetDiff 对应目录。
2. 确保 Vina/QVina 的依赖被纳入 `environment.yml`/`requirements.txt`。
3. 检查 `utils/reconstruct.py`、`utils/transforms.py` 是否与 TargetDiff 兼容，必要时对照合并差异。

### 步骤 4：训练脚本同步
1. 合并 `train_GlintDM.py` 中的训练流程增强（特别是 `get_diffusion_loss` 的调用方式、日志、验证指标）。
2. 移除硬编码路径（如 `'/NAS_Storage4/...`），改为相对路径或配置项。
3. 确保模型保存时包含 `config`、`optimizer`、`scheduler` 等附加信息，以便在 TargetDiff 中保持兼容。

### 步骤 5：配置与文档
1. 更新 TargetDiff 的 `configs/training.yml`、`configs/sampling.yml`，加入新参数并保留默认值（向下兼容）。
2. 在 TargetDiff `README` 或新增文档中记录：
   - 环境依赖新增项。
   - 采样/评分流程。
   - 动态步长的调节说明。
3. 将 `DiffDynamic help.md` 中与环境/运行直接相关的章节裁剪整合至 TargetDiff 文档；将过时或项目环境特定（例如 WSL 配置）的部分移除或放入 FAQ。

### 步骤 6：测试与验证
1. **单测**：为梯度融合、动态步长函数编写最小单测，确认张量尺寸正确。
2. **集成测试**：在小批数据上运行采样脚本，确保：
   - 能生成分子且不会触发 `RuntimeError`。
   - 评分流程运行正常（即便没有安装 Vina，也要有 graceful fallback）。
3. **性能评估**：复现 `DYNAMIC_STEP_README.md` 中提到的效率目标（例：≤70秒/复合体），记录结果。

---

## 4. 迁移检查清单

- [ ] TargetDiff 仓库中存在 `GlintDM` 或等效模型实现，并通过基础训练用例。
- [ ] 新版采样脚本支持多阶段采样 & Top-N 筛选，参数可配置。
- [ ] Docking/QED/SA 评分在采样阶段可选调用，并提供开关或降级策略。
- [ ] 所有硬编码路径/设备号已清理或改为配置项。
- [ ] 文档更新覆盖环境依赖、运行命令、动态步长原理。
- [ ] 至少在 CrossDocked 测试集上复现一次采样流程并产出结果包。

---

## 5. 已知待办与注意事项

1. **动态跳步调度器未落地**  
   - 目前文档描述的 `models/dynamic_step/` 系列模块尚未实现；若后续需要真正的调度器式实现，可以以现有两阶段函数为起点迭代。

2. **评分依赖可选化**  
   - Vina/QVina 安装复杂，建议在迁移时提供 `--docking_mode none` 等选项，或在未安装依赖时给出清晰提示。

3. **性能监控与日志**  
   - `DiffDynamic help.md` 中的 `TimingStats` 示例并未在主代码中使用，如需保留性能监控，请决定是否迁移对应工具类。

4. **路径与硬件特定设置**  
   - 清理 `args.device = 'cuda:1'; args.device = 'cpu'`、`'/NAS_Storage4'` 等开发者私有配置，避免在 TargetDiff 引发混淆。

---

## 6. 附录

- **核心文件映射**
  | DiffDynamic | TargetDiff（目标位置建议） | 说明 |
  |-------------|-----------------------------|------|
  | `models/GlintDM.py` | `models/diffusion.py` 或新建 `models/glintdm.py` | 主模型实现 |
  | `sample_CrossDocked_GlintDM.py` | `scripts/sample_diffusion.py` | 采样流程与评分 |
  | `utils/evaluation/*` | `utils/evaluation/*` | 评估指标、对接工具 |
  | `configs/*.yml` | `configs/*.yml` | 训练/采样参数 |
  | `train_GlintDM.py` | `scripts/train_diffusion.py` | 训练主程序 |

- **参考资料**
  - TargetDiff 官方仓库与 README [[1]](https://github.com/guanjq/targetdiff)
  - GlintDM 研究与公共实现（若需要进一步查阅，可参考 `DiffDynamic help.md` 中的链接）

---

> ⚠️ 在迁移过程中，建议每完成一个大的功能块就提交一次 commit，便于回滚与审查。若遇到 DiffDynamic 特定的补丁无法直接套用，应优先理解其解决的问题（参阅 `DiffDynamicUpdate.md` 日志），再在 TargetDiff 中寻找等价实现路径。


