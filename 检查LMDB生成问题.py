#!/usr/bin/env python
"""
检查为什么 LMDB 是空的
"""

import os
import pickle

print("=" * 60)
print("LMDB 生成问题检查")
print("=" * 60)

# 1. 检查数据路径配置
print("\n1. 检查数据路径配置...")
raw_path = './data/crossdocked_v1.1_rmsd1.0_pocket10'
print(f"   原始数据路径: {raw_path}")
print(f"   路径存在: {os.path.exists(raw_path)}")

if os.path.exists(raw_path):
    files = os.listdir(raw_path)
    print(f"   目录中的文件/文件夹数量: {len(files)}")
    print(f"   前10个文件/文件夹: {files[:10]}")

# 2. 检查 index.pkl
index_path = os.path.join(raw_path, 'index.pkl')
print(f"\n2. 检查 index.pkl: {index_path}")
if os.path.exists(index_path):
    print(f"   ✅ index.pkl 存在")
    try:
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        print(f"   ✅ index.pkl 包含 {len(index)} 个条目")
        if len(index) > 0:
            print(f"   前3个条目示例:")
            for i, item in enumerate(index[:3]):
                print(f"     [{i}]: {item}")
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pocket_fn, ligand_fn = item[0], item[1]
                    pocket_path = os.path.join(raw_path, pocket_fn) if pocket_fn else None
                    ligand_path = os.path.join(raw_path, ligand_fn) if ligand_fn else None
                    print(f"       口袋文件: {pocket_fn} -> {'存在' if pocket_path and os.path.exists(pocket_path) else '不存在'}")
                    print(f"       配体文件: {ligand_fn} -> {'存在' if ligand_path and os.path.exists(ligand_path) else '不存在'}")
    except Exception as e:
        print(f"   ❌ 读取 index.pkl 失败: {e}")
else:
    print(f"   ❌ index.pkl 不存在")
    print(f"   需要从原始数据生成 index.pkl")

# 3. 检查前几个文件是否存在
print(f"\n3. 检查前几个数据文件是否存在...")
if os.path.exists(index_path):
    try:
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        
        if len(index) > 0:
            print(f"   检查前5个条目...")
            for i, item in enumerate(index[:5]):
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    pocket_fn, ligand_fn = item[0], item[1]
                    
                    if pocket_fn is None:
                        print(f"   [{i}] 口袋文件为 None，会被跳过")
                        continue
                    
                    pocket_path = os.path.join(raw_path, pocket_fn)
                    ligand_path = os.path.join(raw_path, ligand_fn)
                    
                    pocket_exists = os.path.exists(pocket_path)
                    ligand_exists = os.path.exists(ligand_path)
                    
                    status = "✅" if (pocket_exists and ligand_exists) else "❌"
                    print(f"   {status} [{i}] 口袋: {pocket_fn[:50]}... ({'存在' if pocket_exists else '不存在'})")
                    print(f"       配体: {ligand_fn[:50]}... ({'存在' if ligand_exists else '不存在'})")
                    
                    if not pocket_exists:
                        print(f"       提示: 口袋文件路径可能不正确")
                    if not ligand_exists:
                        print(f"       提示: 配体文件路径可能不正确")
    except Exception as e:
        print(f"   ❌ 检查文件时出错: {e}")

# 4. 检查 LMDB 文件大小
lmdb_path = './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
print(f"\n4. 检查 LMDB 文件: {lmdb_path}")
if os.path.exists(lmdb_path):
    size = os.path.getsize(lmdb_path)
    print(f"   ✅ LMDB 文件存在")
    print(f"   文件大小: {size / (1024*1024):.2f} MB")
    if size < 1024:  # 小于 1KB，基本是空的
        print(f"   ⚠️  文件大小很小，可能是空的或不完整")
else:
    print(f"   ❌ LMDB 文件不存在")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)

print("\n建议:")
print("1. 如果 index.pkl 不存在，需要从原始数据生成")
print("2. 如果文件路径不正确，检查数据路径配置")
print("3. 如果所有文件都不存在，需要下载或准备数据文件")
print("4. 删除空的 LMDB 文件，重新运行采样命令让它自动重新生成")

