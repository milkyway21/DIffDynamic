#!/usr/bin/env python
"""
重新生成 LMDB 文件
添加调试信息以查看处理过程
"""

import os
import sys
import torch
from datasets import get_dataset
import utils.transforms as trans
from torch_geometric.transforms import Compose
from tqdm import tqdm

print("=" * 60)
print("重新生成 LMDB 文件")
print("=" * 60)

# 1. 删除旧的 LMDB 文件
lmdb_path = './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
lmdb_lock_path = lmdb_path + '-lock'

print(f"\n1. 删除旧的 LMDB 文件...")
if os.path.exists(lmdb_path):
    os.remove(lmdb_path)
    print(f"   ✅ 已删除: {lmdb_path}")
if os.path.exists(lmdb_lock_path):
    os.remove(lmdb_lock_path)
    print(f"   ✅ 已删除锁文件: {lmdb_lock_path}")

# 2. 加载检查点获取配置
print(f"\n2. 加载模型配置...")
try:
    ckpt = torch.load('pretrained_models/pretrained_diffusion.pt', map_location='cpu')
    config = ckpt['config']
    print(f"   ✅ 配置加载成功")
    print(f"   数据路径: {config.data.path}")
except Exception as e:
    print(f"   ❌ 加载配置失败: {e}")
    sys.exit(1)

# 3. 创建转换器
print(f"\n3. 创建数据转换器...")
try:
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = config.data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])
    print(f"   ✅ 转换器创建成功")
    print(f"   配体原子模式: {ligand_atom_mode}")
except Exception as e:
    print(f"   ❌ 创建转换器失败: {e}")
    sys.exit(1)

# 4. 生成 LMDB（这会触发 _process 方法）
print(f"\n4. 开始生成 LMDB...")
print(f"   这可能需要很长时间，请耐心等待...")
print(f"   数据路径: {config.data.path}")
print(f"   index.pkl 应该包含大量条目...")

try:
    # 这会触发 LMDB 生成
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    
    print(f"\n   ✅ LMDB 生成完成！")
    print(f"   数据集大小: {len(dataset)}")
    
    # 检查 LMDB 文件
    if os.path.exists(lmdb_path):
        size = os.path.getsize(lmdb_path)
        print(f"   LMDB 文件大小: {size / (1024*1024):.2f} MB")
        
        if size > 1024:  # 大于 1KB
            print(f"   ✅ LMDB 文件大小正常")
        else:
            print(f"   ⚠️  LMDB 文件仍然很小，可能有问题")
    
    # 检查数据集中的键
    if hasattr(dataset, 'keys') and dataset.keys:
        print(f"   LMDB 中的键数量: {len(dataset.keys)}")
        if len(dataset.keys) > 0:
            print(f"   前5个键: {[k.decode() if isinstance(k, bytes) else k for k in dataset.keys[:5]]}")
    
except Exception as e:
    print(f"\n   ❌ 生成 LMDB 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)

print("\n下一步:")
print("1. 如果 LMDB 生成成功，可以运行采样命令")
print("2. 如果仍然失败，检查错误信息并修复数据问题")

