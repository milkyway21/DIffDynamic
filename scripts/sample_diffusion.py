# 总结：
# - 根据配置执行扩散模型的统一动态或传统动态采样流程，生成配体 3D 结构及原子类型。
# - 支持温度控制、原子数量策略、候选筛选与精细化采样，最终保存生成结果和日志。
# - 输出包含轨迹、化学指标与采样耗时的详细记录，方便后续评估或对接。

import argparse  # 导入 argparse，解析命令行参数。
import os  # 导入 os，用于路径操作。
import shutil  # 导入 shutil，用于复制文件/目录。
import time  # 导入 time，记录采样耗时。

import numpy as np  # 导入 NumPy，处理数组。
import torch  # 导入 PyTorch，执行模型推断。
import torch.nn.functional as F  # 导入函数式接口，用于 softmax 等操作。
from rdkit import Chem  # 导入 RDKit，用于分子重建与处理。
from torch_geometric.data import Batch  # 导入 PyG Batch，构建图批次。
from torch_geometric.transforms import Compose  # 导入转换组合工具。
from torch_scatter import scatter_sum, scatter_mean  # 导入分散聚合函数。
from tqdm.auto import tqdm  # 导入 tqdm，显示进度条。

import utils.misc as misc  # 导入通用工具（日志、配置等）。
import utils.transforms as trans  # 导入特征转换工具。
from datasets import get_dataset  # 导入数据集工厂函数。
from datasets.pl_data import FOLLOW_BATCH  # 导入 PyG follow_batch 配置。
from models.molopt_score_model import ScorePosNet3D, GlintDM, log_sample_categorical  # 导入模型及辅助函数。
from utils.evaluation import atom_num, scoring_func  # 导入原子数量采样与化学评分函数。
import utils.reconstruct as reconstruct  # 导入分子重建工具。


def _run_unified_dynamic(model, data, config, device='cuda:0', logger=None):  # 统一动态采样流程。
    """按照统一动态策略执行扩散采样并收集轨迹。

    Args:
        model: 已加载权重的扩散模型，需实现 `dynamic_sample_diffusion`。
        data: 单个 `ProteinLigandData` 样本，作为蛋白口袋输入。
        config: 采样配置（通常来自 YAML），读取 `sample` 相关字段。
        device: 推理使用的设备字符串。
        logger: 可选日志记录器，用于输出采样进度。

    Returns:
        dict: 包含最终坐标/类型、完整轨迹、耗时以及元信息的字典。
    """
    dynamic_cfg = config.sample.get('dynamic', {})  # 读取动态采样配置。
    num_samples = config.sample.get('num_samples', 1)  # 生成样本数量。
    num_steps = config.sample.get('num_steps', 1000)  # 扩散步数。
    center_pos_mode = config.sample.get('center_pos_mode', 'protein')  # 坐标中心化模式。
    pos_only = config.sample.get('pos_only', False)  # 是否仅采样坐标。
    sample_num_atoms_mode = config.sample.get('sample_num_atoms', 'prior')  # 原子数量策略。

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

    pos_list, v_list = [], []  # 存储最终位置与类别。
    pos_traj_list, v_traj_list, log_v_traj_list = [], [], []  # 存储轨迹。
    time_list = []  # 记录耗时。
    meta_records = []  # 记录元数据。
    range_offset = 0  # 范围模式下的原子数量偏移。

    for sample_idx in range(num_samples):  # 逐个生成样本。
        batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)  # 构造单样本批次。
        batch_protein = batch.protein_element_batch  # 获取蛋白节点批索引。

        if sample_num_atoms_mode == 'prior':  # 依据空间大小采样原子数。
            pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
            # 验证 pocket_size
            if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
                pocket_size = 30.0
            try:
                sampled = atom_num.sample_atom_num(pocket_size)
                if isinstance(sampled, (np.ndarray, np.generic)):
                    sampled_val = float(sampled.item())
                else:
                    sampled_val = float(sampled)
                if np.isnan(sampled_val) or np.isinf(sampled_val):
                    sampled_val = 10.0
                ligand_num_atoms = max(1, int(abs(sampled_val)))
            except Exception:
                ligand_num_atoms = 10
            batch_ligand = torch.zeros(ligand_num_atoms, dtype=torch.long, device=device)  # 全部归属单个配体。
        elif sample_num_atoms_mode == 'range':  # 按序递增原子数量。
            ligand_num_atoms = max(1, range_offset + 1)  # 计算当前样本的原子数量（从1开始递增），确保至少为1。
            range_offset += 1  # 更新偏移量，为下一个样本做准备。
            batch_ligand = torch.zeros(ligand_num_atoms, dtype=torch.long, device=device)  # 创建配体批次索引（全部归属单个配体）。
        elif sample_num_atoms_mode == 'ref':  # 使用参考配体的原子数。
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = max(1, int((batch_ligand == 0).sum().item()))  # 确保至少为1
        else:
            raise ValueError(f'Unknown sample_num_atoms mode {sample_num_atoms_mode}')  # 未知模式报错。

        center = scatter_mean(batch.protein_pos, batch_protein, dim=0)  # 计算蛋白中心。
        init_ligand_pos = center[batch_ligand] + torch.randn((len(batch_ligand), 3), device=device)  # 初始化配体位置。

        init_log_ligand_v = torch.zeros(len(batch_ligand), model.num_classes, device=device)  # 均匀类别对数概率。
        init_log_ligand_v = F.log_softmax(init_log_ligand_v, dim=-1)  # 归一化。

        t_start = time.time()  # 记录开始时间。
        result = model.dynamic_sample_diffusion(  # 调用模型进行动态采样。
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=init_ligand_pos,
            init_log_ligand_v=init_log_ligand_v,
            batch_ligand=batch_ligand,
            num_steps=num_steps,
            center_pos_mode=center_pos_mode,
            pos_only=pos_only
        )
        t_end = time.time()  # 记录结束时间。

        pos = result['pred_ligand_pos'].detach().cpu().numpy().astype(np.float64)  # 获取最终坐标。
        log_v = result['pred_ligand_v'].detach().cpu().numpy().astype(np.float32)  # 获取最终类别对数概率。
        pos_traj = [p.detach().cpu().numpy().astype(np.float64) for p in result.get('pos_traj', [])]  # 坐标轨迹。
        log_v_traj = [lv.detach().cpu().numpy().astype(np.float32) for lv in result.get('log_v_traj', [])]  # 类别轨迹。
        v_traj = [np.argmax(lv, axis=-1).astype(np.int64) for lv in log_v_traj]  # 将轨迹转为类别索引。

        pos_list.append(pos)  # 累积最终坐标。
        v_list.append(np.argmax(log_v, axis=-1).astype(np.int64))  # 累积最终类别。
        pos_traj_list.append(pos_traj)  # 累积位置轨迹。
        v_traj_list.append(v_traj)  # 累积类别索引轨迹。
        log_v_traj_list.append(log_v_traj)  # 累积类别对数概率轨迹。
        time_list.append(t_end - t_start)  # 累积耗时。
        meta_records.append({  # 记录元信息。
            'method': 'unified_dynamic',
            'ligand_num_atoms': ligand_num_atoms,
            'time': t_end - t_start,
            'model_meta': result.get('meta')
        })

        if logger:
            logger.info(f'[Dynamic][Unified] Sample {sample_idx}: {len(pos)} atoms | time {t_end - t_start:.2f}s')

    return {
        'pos_list': pos_list,
        'v_list': v_list,
        'pos_traj': pos_traj_list,
        'v_traj': v_traj_list,
        'log_v_traj': log_v_traj_list,
        'time_list': time_list,
        'meta': {
            'method': 'unified',
            'records': meta_records
        }
    }  # 返回采样结果。


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):  # 根据累积原子数拆分类别轨迹。
    """按样本原子数量切分联合轨迹，用于还原逐分子的预测序列。

    Args:
        ligand_v_traj: 形状为 `[num_steps, total_atoms, num_types]` 的时间序列列表。
        n_data: 当前批次的样本个数。
        ligand_cum_atoms: 按样本累计的原子数边界（`[0, n1, n1+n2, ...]`）。

    Returns:
        list[list[np.ndarray]]: 长度为 `n_data` 的列表，每项为该样本的逐步类别轨迹。
    """
    all_step_v = [[] for _ in range(n_data)]  # 初始化每个样本的轨迹列表。
    for v in ligand_v_traj:  # 遍历每个时间步的类别分布。
        v_array = v.cpu().numpy()  # 转为 NumPy 数组。
        for k in range(n_data):  # 逐样本切片。
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v  # 返回拆分后的轨迹列表。


def split_tensor_by_counts(tensor, counts):  # 按原子数量拆分张量。
    """依据原子数统计将拼接张量拆分为若干子张量。

    Args:
        tensor: 已按所有样本拼接的张量（`[total_atoms, ...]`）。
        counts: 每个样本对应的原子数量列表。

    Returns:
        list[torch.Tensor]: 逐样本拆分后的张量列表。
    """
    splits = []  # 保存片段。
    start = 0  # 起始索引，用于标记当前样本在拼接张量中的起始位置。
    for count in counts:  # 遍历每个样本原子数。
        end = start + count
        splits.append(tensor[start:end])  # 提取对应片段。
        start = end
    return splits  # 返回拆分结果。


def evaluate_candidate(pos_array, v_array, ligand_atom_mode, selector_cfg):  # 评估候选分子的化学指标。
    """根据采样结果评估分子质量，计算化学指标并返回评分。

    Args:
        pos_array: 形状为 `[num_atoms, 3]` 的坐标数组。
        v_array: 原子类型索引数组。
        ligand_atom_mode: 配体原子编码模式（与训练配置保持一致）。
        selector_cfg: 筛选配置，包含权重及阈值。

    Returns:
        dict: 包含化学指标、SMILES、筛选状态及综合评分。
    """
    metrics = {}  # 初始化化学指标字典。
    smiles = None  # 初始 SMILES。
    status = 'ok'  # 状态标记。
    score_value = float('inf')  # 评分值（越小越好）。

    try:
        v_tensor = torch.tensor(v_array, dtype=torch.long)  # 将类别数组转为张量。
        atom_numbers = trans.get_atomic_number_from_index(v_tensor, mode=ligand_atom_mode)  # 获取原子序号。
        aromatic_flags = trans.is_aromatic_from_index(v_tensor, mode=ligand_atom_mode)  # 获取芳香标记。
        mol = reconstruct.reconstruct_from_generated(pos_array, atom_numbers, aromatic_flags)  # 重建分子。
        if mol is not None:
            smiles = Chem.MolToSmiles(mol)  # 转换为 SMILES。
            metrics = scoring_func.get_chem(mol)  # 计算化学指标（QED/SA）。
            qed = metrics.get('qed', float('nan'))  # 提取 QED。
            sa = metrics.get('sa', float('inf'))  # 提取 SA。
            qed_weight = selector_cfg.get('qed_weight', 1.0)  # QED 权重。
            sa_weight = selector_cfg.get('sa_weight', 1.0)  # SA 权重。
            if np.isnan(qed) or np.isnan(sa):  # 若指标非法，则标记无效。
                score_value = float('inf')
                status = 'metric_nan'
            else:
                score_value = sa_weight * sa - qed_weight * qed  # 根据权重计算综合分数。
        else:
            status = 'mol_none'  # 重建返回空。
            metrics = {'qed': float('nan'), 'sa': float('inf')}
    except reconstruct.MolReconsError:
        status = 'reconstruct_failed'  # 重建抛出特定错误。
        metrics = {'qed': float('nan'), 'sa': float('inf')}
        score_value = float('inf')
    except Exception as exc:
        status = f'error:{exc.__class__.__name__}'  # 捕获其他异常。
        metrics = {'qed': float('nan'), 'sa': float('inf')}
        score_value = float('inf')

    min_qed = selector_cfg.get('min_qed')  # QED 下限。
    max_sa = selector_cfg.get('max_sa')  # SA 上限。
    if metrics:
        qed = metrics.get('qed', float('nan'))
        sa = metrics.get('sa', float('inf'))
        if min_qed is not None and not np.isnan(qed) and qed < min_qed:  # 不满足 QED 下限则过滤。
            status = 'filtered_qed'
            score_value = float('inf')
        if max_sa is not None and not np.isnan(sa) and sa > max_sa:  # 不满足 SA 上限则过滤。
            status = 'filtered_sa'
            score_value = float('inf')

    return {
        'metrics': metrics,
        'smiles': smiles,
        'status': status,
        'score': score_value
    }  # 返回评估结果。


def select_top_candidates(candidates, top_n):  # 根据评分选出 top-N 候选。
    """按照综合得分排序候选分子，返回排名靠前的候选集合。

    Args:
        candidates: 由 `evaluate_candidate` 生成的候选列表。
        top_n: 需要保留的候选数量，若小于等于 0 则返回空列表。

    Returns:
        list[dict]: 经过排序与补充后的候选列表（长度不超过 `top_n`）。
    """
    if top_n <= 0:
        return []
    valid = [c for c in candidates if np.isfinite(c['score'])]  # 过滤有限分数的候选。
    if not valid:
        valid = candidates  # 若全部无效，则退回所有候选。
    sorted_candidates = sorted(valid, key=lambda x: x['score'])  # 按分数升序排序。
    if len(sorted_candidates) >= top_n:
        return sorted_candidates[:top_n]  # 返回前 top-N。
    # 补充剩余候选：如果有效候选不足 top_n，则从无效候选中选择补充。
    remaining = [c for c in candidates if c not in sorted_candidates]  # 找出未排序的候选。
    sorted_remaining = sorted(remaining, key=lambda x: x['score'])
    combined = sorted_candidates + sorted_remaining  # 合并两部分。
    return combined[:min(len(combined), top_n)]  # 返回最多 top-N 个候选。


def _run_legacy_dynamic(model, data, config, ligand_atom_mode, device='cuda:0', logger=None):
    """按照旧版两阶段策略执行动态采样。

    Args:
        model: 扩散模型，需实现 `sample_diffusion_large_step` 与 `sample_diffusion_refinement`。
        data: 单个蛋白口袋样本。
        config: 采样配置对象，含 `sample.dynamic` 字段。
        ligand_atom_mode: 配体原子编码模式。
        device: 推理设备。
        logger: 可选日志器。

    Returns:
        dict: 精炼后的分子列表、轨迹、耗时及候选元数据。
    """
    dynamic_cfg = config.sample.get('dynamic', {})  # 读取整体动态配置。
    large_cfg = dynamic_cfg.get('large_step', {})  # 大步探索阶段配置。
    refine_cfg = dynamic_cfg.get('refine', {})  # 精炼阶段配置。
    selector_cfg = dynamic_cfg.get('selector', {})  # 候选筛选配置。

    center_pos_mode = config.sample.get('center_pos_mode', 'protein')  # 坐标中心化策略。
    pos_only = config.sample.get('pos_only', False)  # 是否仅采样坐标。
    sample_num_atoms_mode = config.sample.get('sample_num_atoms', 'prior')  # 原子数量策略。

    large_batch_size = large_cfg.get('batch_size', config.sample.get('batch_size', 16))  # 大步批大小。
    n_repeat = large_cfg.get('n_repeat', 1)  # 重复次数。
    
    # 确保 batch_size 和 n_repeat 都是非负整数
    large_batch_size = max(1, int(large_batch_size))  # 确保至少为1
    n_repeat = max(0, int(n_repeat))  # 确保非负（0表示不执行，但不会报错）
    
    if logger:
        logger.info(f'[Dynamic] Large-step batch size: {large_batch_size} | repeats: {n_repeat}')
        if n_repeat == 0:
            logger.warning('n_repeat is 0, no large-step sampling will be performed')

    total_candidates = []  # 存储所有候选。
    range_offset = 0  # 范围模式的原子偏移。
    time_records = {'large_step': [], 'refine': []}  # 记录各阶段耗时。

    for repeat_idx in range(n_repeat):  # 逐次执行大步探索。
        batch = Batch.from_data_list([data.clone() for _ in range(large_batch_size)],
                                     follow_batch=FOLLOW_BATCH).to(device)  # 构建批次。
        batch_protein = batch.protein_element_batch  # 蛋白节点批索引。
        n_data = large_batch_size  # 当前批包含的样本数。

        if sample_num_atoms_mode == 'prior':  # 根据 pocket 尺寸采样原子数。
            pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
            # 验证 pocket_size 的有效性
            if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
                if logger:
                    logger.warning(f'Invalid pocket_size: {pocket_size}, using default value 30.0')
                pocket_size = 30.0
            ligand_num_atoms = []
            for i in range(n_data):
                try:
                    sampled = atom_num.sample_atom_num(pocket_size)
                    # 转换为Python原生类型，处理各种可能的numpy类型
                    if isinstance(sampled, (np.ndarray, np.generic)):
                        sampled_val = float(sampled.item())
                    else:
                        sampled_val = float(sampled)
                    # 检查是否为有效数字
                    if np.isnan(sampled_val) or np.isinf(sampled_val):
                        if logger:
                            logger.warning(f'Sampled invalid value: {sampled_val}, using default 10')
                        sampled_val = 10.0
                    # 确保为正整数，至少为1
                    atom_count = max(1, int(abs(sampled_val)))
                    ligand_num_atoms.append(atom_count)
                except Exception as e:
                    if logger:
                        logger.error(f'Error sampling atom num for sample {i}: {e}, using default 10')
                    ligand_num_atoms.append(10)  # 默认值
            # 最终验证：确保所有值都是正整数
            ligand_num_atoms = [max(1, abs(int(n))) for n in ligand_num_atoms]
            # 再次检查，确保没有任何值 <= 0
            for i, n in enumerate(ligand_num_atoms):
                if n <= 0:
                    if logger:
                        logger.error(f'Found non-positive atom count at index {i}: {n}, correcting to 1')
                    ligand_num_atoms[i] = 1
            # 创建tensor前最后一次验证
            try:
                repeats_tensor = torch.tensor(ligand_num_atoms, device=device, dtype=torch.long)
                # 使用clamp确保所有值都 >= 1
                repeats_tensor = torch.clamp(repeats_tensor, min=1)
                if logger and torch.any(repeats_tensor <= 0):
                    logger.error(f'Found non-positive values in repeats tensor after clamp: {repeats_tensor.tolist()}')
                    repeats_tensor = torch.ones_like(repeats_tensor)  # 如果还有问题，全部设为1
            except Exception as e:
                if logger:
                    logger.error(f'Error creating repeats tensor: {e}, ligand_num_atoms={ligand_num_atoms}')
                # 如果创建tensor失败，使用默认值
                repeats_tensor = torch.ones(n_data, device=device, dtype=torch.long) * 10
            batch_ligand = torch.repeat_interleave(
                torch.arange(n_data, device=device, dtype=torch.long),
                repeats_tensor
            )
        elif sample_num_atoms_mode == 'range':  # 使用连续范围。
            ligand_num_atoms = list(range(range_offset + 1, range_offset + n_data + 1))
            range_offset += n_data
            # 验证范围模式下的原子数都是正数（虽然应该总是正数，但为了安全起见）
            ligand_num_atoms = [max(1, abs(int(n))) for n in ligand_num_atoms]
            try:
                repeats_tensor = torch.tensor(ligand_num_atoms, device=device, dtype=torch.long)
                repeats_tensor = torch.clamp(repeats_tensor, min=1)  # 确保所有值 >= 1
            except Exception as e:
                if logger:
                    logger.error(f'Error creating repeats tensor in range mode: {e}, ligand_num_atoms={ligand_num_atoms}')
                repeats_tensor = torch.ones(n_data, device=device, dtype=torch.long) * 10
            batch_ligand = torch.repeat_interleave(
                torch.arange(n_data, device=device, dtype=torch.long),
                repeats_tensor
            )
        elif sample_num_atoms_mode == 'ref':  # 使用参考原子数。
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            ligand_num_atoms = [max(1, abs(int(n))) for n in ligand_num_atoms]  # 确保至少为1
            # 验证tensor创建前的值
            try:
                repeats_tensor = torch.tensor(ligand_num_atoms, device=device, dtype=torch.long)
                repeats_tensor = torch.clamp(repeats_tensor, min=1)  # 确保所有值 >= 1
            except Exception as e:
                if logger:
                    logger.error(f'Error creating repeats tensor in ref mode: {e}, ligand_num_atoms={ligand_num_atoms}')
                repeats_tensor = torch.ones(n_data, device=device, dtype=torch.long) * 10
            # 注意：ref模式下batch_ligand已经设置，不需要重新创建
        else:
            raise ValueError(f'Unknown sample_num_atoms mode {sample_num_atoms_mode}')

        center = scatter_mean(batch.protein_pos, batch_protein, dim=0)  # 计算蛋白中心。
        init_ligand_pos = center[batch_ligand] + torch.randn_like(center[batch_ligand])  # 初始化配体位置。

        total_atoms = len(batch_ligand)  # 当前批次总原子数。
        uniform_logits = torch.zeros(total_atoms, model.num_classes, device=device)  # 均匀类别 logits。
        if getattr(model, 'ligand_v_input', 'onehot') == 'log_prob':
            init_ligand_v_input = F.log_softmax(uniform_logits, dim=-1)
            log_mode = 'log_prob'
        else:
            init_ligand_v_input = log_sample_categorical(uniform_logits)
            log_mode = 'auto'  # 自动模式：根据模型配置自动选择输入格式。

        t_start = time.time()  # 记录大步采样开始时间。
        res = model.sample_diffusion_large_step(
            protein_pos=batch.protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch_protein,
            init_ligand_pos=init_ligand_pos,
            init_ligand_v=init_ligand_v_input,
            batch_ligand=batch_ligand,
            num_steps=large_cfg.get('num_steps'),
            center_pos_mode=center_pos_mode,
            pos_only=pos_only,
            step_stride=large_cfg.get('stride'),
            step_size=large_cfg.get('step_size'),
            add_noise=large_cfg.get('noise_scale'),
            pos_clip=large_cfg.get('pos_clip'),
            v_clip=large_cfg.get('v_clip'),
            log_ligand_input_mode='log_prob' if log_mode == 'log_prob' else 'auto'
        )
        t_end = time.time()  # 记录大步采样结束时间。
        time_records['large_step'].append(t_end - t_start)  # 记录大步采样耗时。

        # 提取大步采样结果并转换为 NumPy 数组。
        ligand_pos_array = res['pos'].detach().cpu().numpy().astype(np.float64)  # 配体位置数组。
        ligand_v_array = res['v'].detach().cpu().numpy()  # 配体类别索引数组。
        log_v_tensor = res['log_v'].detach().cpu()  # 配体类别对数概率张量。
        cum_atoms = np.cumsum([0] + ligand_num_atoms)  # 计算累积原子数边界，用于拆分批次结果。

        for idx in range(n_data):  # 拆分每个样本的结果。
            start, end = cum_atoms[idx], cum_atoms[idx + 1]
            pos_piece = ligand_pos_array[start:end]
            v_piece = ligand_v_array[start:end]
            log_v_piece = log_v_tensor[start:end]

            candidate = {
                'pos': pos_piece,
                'v': v_piece,
                'log_v': log_v_piece.numpy().astype(np.float32),
                'num_atoms': ligand_num_atoms[idx],
                'repeat': repeat_idx,
                'time_indices': res.get('time_indices'),
            }
            metric_info = evaluate_candidate(pos_piece, v_piece, ligand_atom_mode, selector_cfg)
            candidate.update(metric_info)  # 合并化学指标与评分。
            total_candidates.append(candidate)  # 收集候选。

    top_n = selector_cfg.get('top_n', len(total_candidates))  # 选择的候选数量。
    selected_candidates = select_top_candidates(total_candidates, top_n)  # 按评分选择 top-N。
    if logger:
        logger.info(f'[Dynamic] Total candidates: {len(total_candidates)} | Selected top-N: {len(selected_candidates)}')

    refined_records = []  # 存储精炼后的结果。
    n_sampling = max(refine_cfg.get('n_sampling', 1), 1)  # 精炼次数。

    for cand_idx, cand in enumerate(selected_candidates):  # 遍历候选集。
        for refine_idx in range(n_sampling):  # 对每个候选执行多次精炼采样。
            # 构建单样本批次用于精炼。
            batch = Batch.from_data_list([data.clone()], follow_batch=FOLLOW_BATCH).to(device)
            batch_protein = batch.protein_element_batch  # 获取蛋白批次索引。

            # 根据候选的原子数量创建配体批次索引。
            try:
                num_atoms_raw = cand.get('num_atoms', 10)
                # 确保是有效的正整数
                if isinstance(num_atoms_raw, (np.ndarray, np.generic)):
                    num_atoms = max(1, abs(int(num_atoms_raw.item())))
                else:
                    num_atoms = max(1, abs(int(float(num_atoms_raw))))
                # 再次验证
                if num_atoms <= 0:
                    if logger:
                        logger.warning(f'Invalid num_atoms: {num_atoms}, using default 10')
                    num_atoms = 10
                repeats_val = torch.tensor([num_atoms], device=device, dtype=torch.long)
                repeats_val = torch.clamp(repeats_val, min=1)
            except Exception as e:
                if logger:
                    logger.error(f'Error processing num_atoms: {e}, cand={cand.get("num_atoms", "N/A")}')
                repeats_val = torch.tensor([10], device=device, dtype=torch.long)
            batch_ligand = torch.repeat_interleave(
                torch.arange(1, device=device, dtype=torch.long),  # 创建单个样本的批次索引。
                repeats_val  # 重复原子数量次。
            )
            # 将候选的位置和类别转换为张量，作为精炼的初始状态。
            init_pos = torch.tensor(cand['pos'], dtype=torch.float32, device=device)  # 初始位置。
            init_log_v = torch.tensor(cand['log_v'], dtype=torch.float32, device=device)  # 初始类别对数概率。
            # 根据模型配置确定输入格式。
            if getattr(model, 'ligand_v_input', 'onehot') == 'log_prob':
                init_input = init_log_v  # 直接使用 log 概率。
                log_mode = 'log_prob'
            else:
                init_input = init_log_v.argmax(dim=-1)  # 转换为类别索引。
                log_mode = 'auto'

            t_start = time.time()  # 记录精炼采样开始时间。
            res = model.sample_diffusion_refinement(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,
                init_ligand_pos=init_pos,
                init_ligand_v=init_input,
                batch_ligand=batch_ligand,
                center_pos_mode=center_pos_mode,
                pos_only=pos_only,
                step_stride=refine_cfg.get('stride'),
                step_size=refine_cfg.get('step_size'),
                add_noise=refine_cfg.get('noise_scale'),
                pos_clip=refine_cfg.get('pos_clip'),
                v_clip=refine_cfg.get('v_clip'),
                time_upper=refine_cfg.get('time_upper'),
                time_lower=refine_cfg.get('time_lower', 0),
                num_cycles=refine_cfg.get('cycles', 1),
                log_ligand_input_mode='log_prob' if log_mode == 'log_prob' else 'auto'
            )
            t_end = time.time()  # 记录精炼采样结束时间。
            time_records['refine'].append(t_end - t_start)  # 记录精炼采样耗时。

            # 提取精炼后的最终结果并转换为 NumPy 数组。
            pos_final = res['pos'].detach().cpu().numpy().astype(np.float64)  # 最终配体位置。
            v_final = res['v'].detach().cpu().numpy()  # 最终配体类别索引。
            log_v_final = res['log_v'].detach().cpu().numpy().astype(np.float32)  # 最终类别对数概率。

            # 提取并转换轨迹数据。
            pos_traj = [traj.numpy().astype(np.float64) for traj in res.get('pos_traj', [])]  # 位置轨迹。
            log_v_traj = [traj.numpy().astype(np.float32) for traj in res.get('log_v_traj', [])]  # 类别对数概率轨迹。
            v_traj = [traj.argmax(axis=-1).astype(np.int64) for traj in log_v_traj]  # 将 log 概率轨迹转换为类别索引轨迹。

            metric_info = evaluate_candidate(pos_final, v_final, ligand_atom_mode, selector_cfg)  # 精炼后评估。
            refined_records.append({
                'pos': pos_final,
                'v': v_final,
                'log_v': log_v_final,
                'pos_traj': pos_traj,
                'v_traj': v_traj,
                'log_v_traj': log_v_traj,
                'num_atoms': num_atoms,
                'source_index': cand_idx,
                'repeat_index': refine_idx,
                'time_indices': res.get('time_indices'),
                **metric_info
            })  # 记录精炼分子。

    if logger:
        logger.info(f'[Dynamic] Refinement outputs: {len(refined_records)}')

    refined_pos_list = [rec['pos'] for rec in refined_records]  # 汇总精炼坐标。
    refined_v_list = [rec['v'] for rec in refined_records]  # 汇总精炼类别。
    refined_pos_traj = [rec['pos_traj'] for rec in refined_records]  # 汇总轨迹。
    refined_v_traj = [rec['v_traj'] for rec in refined_records]

    # Flatten time records to match baseline expectation
    time_list = time_records['large_step'] + time_records['refine']  # 合并耗时记录。

    return {
        'pos_list': refined_pos_list,
        'v_list': refined_v_list,
        'pos_traj': refined_pos_traj,
        'v_traj': refined_v_traj,
        'log_v_traj': [rec['log_v_traj'] for rec in refined_records],
        'time_list': time_list,
        'meta': {
            'large_step_candidates': total_candidates,
            'refined_candidates': refined_records,
            'time_records': time_records
        }
    }  # 返回传统动态采样结果。


def sample_dynamic_diffusion_ligand(model, data, config, ligand_atom_mode, device='cuda:0', logger=None):
    """封装动态采样入口，根据配置自动选择统一或旧版策略。

    Args:
        model: 扩散模型实例。
        data: 单个蛋白口袋样本。
        config: 采样配置，需包含 `sample.dynamic.method`。
        ligand_atom_mode: 配体原子编码模式。
        device: 运行设备。
        logger: 可选日志器。

    Returns:
        dict: 动态采样输出，结构与具体实现 `_run_*` 一致。
    """
    dynamic_cfg = config.sample.get('dynamic', {})  # 读取动态配置。
    dynamic_method = dynamic_cfg.get('method', 'auto')  # 指定方法。

    supports_unified = hasattr(model, 'dynamic_sample_diffusion')  # 检查模型是否实现统一动态接口。
    if dynamic_method == 'auto':
        dynamic_method = 'unified' if supports_unified else 'legacy'  # 自动选择。

    if dynamic_method == 'unified':
        if not supports_unified:
            raise RuntimeError('dynamic.method is set to "unified" but model does not implement dynamic_sample_diffusion().')
        return _run_unified_dynamic(model, data, config, device=device, logger=logger)  # 调用统一动态。

    return _run_legacy_dynamic(model, data, config, ligand_atom_mode, device=device, logger=logger)  # fallback。


def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior'):
    """批量运行标准扩散采样，返回位置/类型及轨迹列表。

    Args:
        model: 扩散模型实例，需实现 `sample_diffusion`。
        data: 作为模板的单个 `ProteinLigandData`。
        num_samples: 目标生成的样本数量。
        batch_size: 每批采样的数量。
        device: 运行设备。
        num_steps: 采样步数，默认为模型默认值。
        pos_only: 是否仅预测坐标而复用原始类型。
        center_pos_mode: 坐标中心化策略。
        sample_num_atoms: 原子数量选择策略（`prior/range/ref`）。

    Returns:
        tuple: 包含采样坐标、类别、完整轨迹及耗时的七元组。
    """
    # 验证 batch_size 和 num_samples 参数
    batch_size = max(1, int(batch_size))  # 确保至少为1
    num_samples = max(1, int(num_samples))  # 确保至少为1
    
    all_pred_pos, all_pred_v = [], []  # 累积最终坐标与类别。
    all_pred_pos_traj, all_pred_v_traj = [], []  # 累积轨迹。
    all_pred_v0_traj, all_pred_vt_traj = [], []  # 累积初/末时间轨迹。
    time_list = []  # 记录每批耗时。
    num_batch = int(np.ceil(num_samples / batch_size))  # 计算批次数。
    current_i = 0  # 范围模式偏移。
    for i in tqdm(range(num_batch)):  # 遍历每个批次，显示进度条。
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)  # 当前批大小（最后一批可能不满）。
        # 构建批次：克隆数据 n_data 次并组合为批次。
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)

        t1 = time.time()  # 记录批次起始时间。
        with torch.no_grad():
            batch_protein = batch.protein_element_batch  # 获取蛋白批索引。
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                # 验证 pocket_size
                if np.isnan(pocket_size) or np.isinf(pocket_size) or pocket_size <= 0:
                    pocket_size = 30.0
                ligand_num_atoms = []
                for _ in range(n_data):
                    try:
                        sampled = atom_num.sample_atom_num(pocket_size)
                        if isinstance(sampled, (np.ndarray, np.generic)):
                            sampled_val = float(sampled.item())
                        else:
                            sampled_val = float(sampled)
                        if np.isnan(sampled_val) or np.isinf(sampled_val):
                            sampled_val = 10.0
                        atom_count = max(1, int(abs(sampled_val)))
                        ligand_num_atoms.append(atom_count)
                    except Exception:
                        ligand_num_atoms.append(10)
                # 最终验证
                ligand_num_atoms = [max(1, abs(int(n))) for n in ligand_num_atoms]
                repeats_tensor = torch.tensor(ligand_num_atoms, dtype=torch.long)
                repeats_tensor = torch.clamp(repeats_tensor, min=1)
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), repeats_tensor).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                ligand_num_atoms = [max(1, abs(int(n))) for n in ligand_num_atoms]
                repeats_tensor = torch.tensor(ligand_num_atoms, dtype=torch.long)
                repeats_tensor = torch.clamp(repeats_tensor, min=1)
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), repeats_tensor).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
                ligand_num_atoms = [max(1, int(n)) for n in ligand_num_atoms]  # 确保至少为1
            else:
                raise ValueError  # 未知的原子数量采样模式。

            # 初始化配体位置：以蛋白中心为基准，添加随机噪声。
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)  # 计算每个样本的蛋白中心。
            batch_center_pos = center_pos[batch_ligand]  # 为每个配体原子分配对应的蛋白中心。
            init_ligand_pos = batch_center_pos + torch.randn_like(batch_center_pos)  # 在蛋白中心附近添加随机初始位置。

            # 初始化配体类别：根据 pos_only 标志选择策略。
            if pos_only:
                init_ligand_v = batch.ligand_atom_feature_full  # 如果仅采样位置，复用原始配体类别。
            else:
                uniform_logits = torch.zeros(len(batch_ligand), model.num_classes).to(device)  # 创建均匀的类别 logits。
                init_ligand_v = log_sample_categorical(uniform_logits)  # 从均匀分布中采样初始类别。

            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode
            )
            # 解包采样结果：提取位置、类别和轨迹。
            ligand_pos, ligand_v, ligand_pos_traj, ligand_v_traj = r['pos'], r['v'], r['pos_traj'], r['v_traj']
            ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']  # 提取 v0 和 vt 预测轨迹。
            # 拆分位置轨迹：将批次结果按样本原子数拆分。
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)  # 计算累积原子数边界。
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in
                             range(n_data)]  # num_samples * [num_atoms_i, 3]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:  # step_i
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in
                            all_step_pos]  # num_samples * [num_steps, num_atoms_i, 3]
            all_pred_pos_traj += [p for p in all_step_pos]  # 累积位置轨迹。

            # 拆分类别轨迹：将批次结果按样本原子数拆分。
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += [v for v in all_step_v]

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += [v for v in all_step_v0]
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += [v for v in all_step_vt]
        t2 = time.time()  # 记录结束时间。
        time_list.append(t2 - t1)  # 累计耗时。
        current_i += n_data  # 更新范围模式偏移。
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, all_pred_v0_traj, all_pred_vt_traj, time_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()  # 构建参数解析器。
    parser.add_argument('config', type=str)  # 配置文件路径。
    parser.add_argument('-i', '--data_id', type=int)  # 指定测试数据索引。
    parser.add_argument('--device', type=str, default='cuda:0')  # 指定运行设备。
    parser.add_argument('--batch_size', type=int, default=100)  # 批量大小。
    parser.add_argument('--result_path', type=str, default='./outputs')  # 结果输出目录。
    parser.add_argument('--mode', type=str, choices=['baseline', 'dynamic'], default=None)  # 采样模式。
    args = parser.parse_args()  # 解析命令行参数。
    
    # 验证 batch_size 参数
    args.batch_size = max(1, int(args.batch_size))  # 确保至少为1

    logger = misc.get_logger('sampling')  # 创建日志器。

    # 加载配置文件：从 YAML 文件读取采样配置。
    config = misc.load_config(args.config)  # 加载采样配置。
    logger.info(config)  # 记录配置信息到日志。
    misc.seed_all(config.sample.seed)  # 设置随机种子，确保可复现性。

    # 加载模型检查点：从检查点文件恢复模型权重和训练配置。
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)  # 加载检查点到指定设备。
    logger.info(f"Training Config: {ckpt['config']}")  # 记录训练配置信息。

    # 初始化特征转换器：创建蛋白和配体的特征提取管道。
    protein_featurizer = trans.FeaturizeProteinAtom()  # 创建蛋白原子特征化器。
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode  # 从检查点读取配体原子编码模式。
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)  # 创建配体原子特征化器。
    transform = Compose([  # 组合多个转换器为单一管道。
        protein_featurizer,  # 蛋白特征转换。
        ligand_featurizer,  # 配体特征转换。
        trans.FeaturizeLigandBond(),  # 配体键特征转换。
    ])

    # 加载数据集：根据检查点中的数据集配置加载数据。
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,  # 使用检查点中的数据集配置。
        transform=transform  # 应用特征转换管道。
    )
    train_set, test_set = subsets['train'], subsets['test']  # 提取训练集和测试集。
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')  # 记录数据集加载成功信息。

    # 加载模型：根据检查点配置实例化并加载模型权重。
    model_cfg = ckpt['config'].model  # 从检查点读取模型配置。
    model_name = getattr(model_cfg, 'name', 'score').lower()  # 获取模型名称，默认为 'score'。
    model_cls = GlintDM if model_name == 'glintdm' else ScorePosNet3D  # 根据名称选择模型类。
    model = model_cls(  # 实例化模型。
        model_cfg,  # 模型配置。
        protein_atom_feature_dim=protein_featurizer.feature_dim,  # 蛋白原子特征维度。
        ligand_atom_feature_dim=ligand_featurizer.feature_dim  # 配体原子特征维度。
    ).to(args.device)  # 将模型移动到指定设备。
    model.load_state_dict(ckpt['model'])  # 加载模型权重。
    logger.info(f'Successfully load the model! {config.model.checkpoint}')  # 记录模型加载成功信息。

    data = test_set[args.data_id]  # 选择待采样的测试样本。
    sampling_mode = args.mode or config.sample.get('mode', 'baseline')  # 决定采样模式：优先使用命令行参数，否则使用配置。
    if sampling_mode not in ['baseline', 'dynamic']:
        raise ValueError(f'Unsupported sampling mode: {sampling_mode}')

    if sampling_mode == 'dynamic':  # 动态采样模式。
        # 确保动态采样配置存在，并设置默认值。
        config.sample.setdefault('dynamic', {})  # 如果不存在则创建空字典。
        config.sample.dynamic.setdefault('large_step', {})  # 确保大步配置存在。
        config.sample.dynamic['large_step'].setdefault('batch_size', args.batch_size)  # 设置大步批量大小。
        # 执行动态采样。
        dynamic_output = sample_dynamic_diffusion_ligand(
            model=model,
            data=data,
            config=config,
            ligand_atom_mode=ligand_atom_mode,
            device=args.device,
            logger=logger
        )
        result = {
            'data': data,
            'pred_ligand_pos': dynamic_output['pos_list'],
            'pred_ligand_v': dynamic_output['v_list'],
            'pred_ligand_pos_traj': dynamic_output['pos_traj'],
            'pred_ligand_v_traj': dynamic_output['v_traj'],
            'pred_ligand_log_v_traj': dynamic_output['log_v_traj'],
            'time': dynamic_output['time_list'],
            'meta': dynamic_output['meta'],
            'mode': 'dynamic'  # 标记采样模式。
        }
    else:  # 基线采样模式。
        # 执行标准扩散采样。
        pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
            model, data, config.sample.num_samples,
            batch_size=args.batch_size, device=args.device,
            num_steps=config.sample.num_steps,
            pos_only=config.sample.pos_only,
            center_pos_mode=config.sample.center_pos_mode,
            sample_num_atoms=config.sample.sample_num_atoms
        )
        result = {
            'data': data,
            'pred_ligand_pos': pred_pos,
            'pred_ligand_v': pred_v,
            'pred_ligand_pos_traj': pred_pos_traj,
            'pred_ligand_v_traj': pred_v_traj,
            'pred_ligand_v0_traj': pred_v0_traj,
            'pred_ligand_vt_traj': pred_vt_traj,
            'time': time_list,  # 采样耗时列表。
            'mode': 'baseline'  # 标记采样模式。
        }
    logger.info('Sample done!')  # 记录采样完成信息。

    # 保存结果：将采样结果和配置保存到输出目录。
    result_path = args.result_path  # 获取结果输出目录路径。
    os.makedirs(result_path, exist_ok=True)  # 创建输出目录（如果不存在）。
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))  # 备份采样配置文件。
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))  # 保存采样结果为 PyTorch 文件。
