# 模型使用引导

本文档面向准备在 TargetDiff 环境下运行 DiffDynamic 改造版（含 GlintDM 与动态采样）的用户，涵盖数据准备、配置管理与模型启动流程。

---

## 1. 环境准备

1. **基础依赖**
   - Python 3.8+
   - PyTorch 2.0.x（匹配 CUDA 11.6/11.8）
   - Torch Geometric 2.2.x 及其伴随的 `torch-scatter`、`torch-sparse`、`torch-cluster`
   - RDKit、OpenBabel (`openbabel` + `pybel`)
   - Vina / QVina（可选，用于对接打分）
   - Meeko、AutoDockTools（若需真实 docking）

2. **推荐安装方式**
   - 先创建 Conda 环境并安装 PyTorch 与 PyG 官方二进制轮子；
   - 再通过 `pip install -r requirements.txt` 与 `conda install -c conda-forge rdkit openbabel`.

---

## 2. 数据准备

### 2.1 官方 CrossDocked 数据

1. 下载 CrossDocked v1.1 pocket10 数据集，并解压到 `./data/crossdocked_v1.1_rmsd1.0_pocket10`.
2. 下载官方划分文件 `crossdocked_pocket10_pose_split.pt` 放置于 `./data`.
3. 若需重新清洗，可使用：

```bash
python scripts/data_preparation/clean_crossdocked.py --input raw_dir --output ./data/crossdocked_clean
python scripts/data_preparation/split_pl_dataset.py --data ./data/crossdocked_clean --save ./data/crossdocked_split.pt
```

### 2.2 自定义口袋

1. 将蛋白口袋 PDB 文件放入 `examples/` 或自定义路径；
2. 使用 `scripts/sample_for_pocket.py` 进行采样（见后文）。

---

## 3. 配置文件说明

### 3.1 `configs/training.yml`

- `model.name`：`score`（原始 ScoreNet）或 `glintdm`（增强版）。
- `model.ligand_v_input`：`onehot`（默认）或 `log_prob`（动态采样推荐）。
- `model.use_grad_fusion`、`grad_fusion_lambda`：控制全局/局部梯度融合策略。
- `dynamic_*`：大步探索与精修默认超参；仅在动态采样时使用。

> 若要切换到 GlintDM，请至少设置：
> ```yaml
> model:
>   name: glintdm
>   ligand_v_input: log_prob
>   use_grad_fusion: true
> ```

### 3.2 `configs/sampling.yml`

- `sample.mode`：`baseline`（单阶段退火）或 `dynamic`。
- `sample.dynamic.large_step`：大步探索设置：批量大小、步长、重复次数等。
- `sample.dynamic.refine`：精修循环，包含 `time_upper`、`stride`、`n_sampling`。
- `sample.dynamic.selector`：Top-N 选择及 QED/SA 过滤阈值。

---

## 4. 训练流程

1. **启动命令（GlintDM）**
   ```bash
   python scripts/train_diffusion.py configs/training.yml \
     --device cuda:0 \
     --logdir ./logs_diffusion \
     --tag glintdm_run1 \
     --train_report_iter 200
   ```
2. **关键配置**
   - 若设备数量有限，可调整 `train.batch_size`、`n_acc_batch`；
   - 使用 `--tag` 标识实验，日志与模型权重保存于 `logs_diffusion/<tag>/`.
3. **验证**
   - 默认每 `train.val_freq` 次迭代执行一次验证；
   - `logs_diffusion/.../checkpoints/` 会存储最佳模型。

---

## 5. 采样流程

### 5.1 Baseline 模式

```bash
python scripts/sample_diffusion.py configs/sampling.yml \
  -i 0 --device cuda:0 --batch_size 64 \
  --mode baseline --result_path ./outputs/baseline_case0
```

输出文件 `result_<id>.pt` 包含：
- `pred_ligand_pos` / `pred_ligand_v`：每个样本的最终坐标与类别；
- 轨迹 `pos_traj`、`v_traj` 以及时间消耗列表 `time`.

### 5.2 动态模式

```bash
python scripts/sample_diffusion.py configs/sampling.yml \
  -i 0 --device cuda:0 --mode dynamic \
  --result_path ./outputs/dynamic_case0
```

动态模式结果新增：
- `meta.large_step_candidates`：大步探索阶段的候选及评分；
- `pred_ligand_log_v_traj`：类别 logits 轨迹；
- `meta.refined_candidates`：精修后分数、SMILES、筛选状态。

### 5.3 指定口袋文件

```bash
python scripts/sample_for_pocket.py configs/sampling.yml \
  --pdb_path ./examples/3ug2_protein.pdb \
  --device cuda:0 \
  --num_samples 128 \
  --result_path ./outputs/3ug2_dynamic
```

---

## 6. 评分与依赖

- 若未安装 Vina/QVina，动态模式会跳过 docking 评分并给出 warning；
- QED、SA 评分需要 RDKit 与 `utils/evaluation/sascorer.py`；
- `meta` 字段记录所有评分状态，可用于后处理或可视化。

---

## 7. 常见问题

1. **GPU 显存不足**  
   - 减少 `sample.dynamic.large_step.batch_size` 或 `refine.n_sampling`；
   - 降低训练的 `batch_size` 或启用梯度累积。

2. **OpenBabel/RDKit 相关报错**  
   - 确保 `openbabel`、`pybel` 已装；
   - Windows 平台需将 `OpenBabel` 目录加入 `PATH`。

3. **Vina 可执行找不到**  
   - 设置环境变量 `VINA_EXECUTABLE` 指向 Vina 可执行文件；
   - 或在配置 `sample.dynamic.selector` 中关闭 docking 相关评判。

---

## 8. 后续建议

- 将实验命令记录到 README 或实验日志，便于复现；
- 对 Top-N 精修结果进一步运行 docking/评分脚本 `scripts/dock_baseline.py`；
- 若需模型微调其他数据集，只需更新 `configs/training.yml::data` 中的路径与变换配置。

