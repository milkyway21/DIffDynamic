#!/usr/bin/env python
"""
诊断数据集索引问题的脚本
用于检查 split 文件中的索引是否与 LMDB 中的实际键匹配
"""

import torch
import os
import lmdb
import pickle

print("=" * 60)
print("数据集索引问题诊断")
print("=" * 60)

# 1. 加载 split 文件
split_path = './data/crossdocked_pocket10_pose_split.pt'
print(f"\n1. 加载 split 文件: {split_path}")
if not os.path.exists(split_path):
    print(f"   ❌ Split 文件不存在: {split_path}")
    exit(1)

split = torch.load(split_path)
print(f"   ✅ Split 文件加载成功")
print(f"   Split 键: {list(split.keys())}")

for key, indices in split.items():
    print(f"   {key} 集大小: {len(indices)}")
    if len(indices) > 0:
        print(f"   {key} 集索引范围: [{min(indices)}, {max(indices)}]")
        print(f"   {key} 集前5个索引: {indices[:5].tolist() if hasattr(indices, 'tolist') else list(indices)[:5]}")

# 2. 检查 LMDB 文件
lmdb_path = './data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb'
print(f"\n2. 检查 LMDB 文件: {lmdb_path}")
if not os.path.exists(lmdb_path):
    print(f"   ❌ LMDB 文件不存在: {lmdb_path}")
    print(f"   提示: 可能需要从原始数据预处理生成 LMDB")
    exit(1)

print(f"   ✅ LMDB 文件存在")

# 3. 读取 LMDB 中的键
print(f"\n3. 读取 LMDB 中的键...")
try:
    db = lmdb.open(
        lmdb_path,
        map_size=10*(1024*1024*1024),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
        # 将字节键转换为整数
        if keys:
            key_ints = [int(k.decode()) for k in keys]
            key_ints.sort()
        else:
            key_ints = []
    
    db.close()
    
    if key_ints:
        print(f"   ✅ LMDB 中有 {len(key_ints)} 个键")
        print(f"   键的范围: [{min(key_ints)}, {max(key_ints)}]")
        print(f"   前10个键: {key_ints[:10]}")
        print(f"   后10个键: {key_ints[-10:]}")
    else:
        print(f"   ⚠️  LMDB 文件存在但是空的（0 个键）")
        print(f"   这意味着处理时没有成功写入任何数据")
        print(f"   可能的原因：")
        print(f"   1. index.pkl 文件不存在或为空")
        print(f"   2. 数据路径配置错误，找不到文件")
        print(f"   3. 所有文件都解析失败")
    
except Exception as e:
    print(f"   ❌ 读取 LMDB 失败: {e}")
    exit(1)

# 4. 检查索引匹配情况（如果 LMDB 不为空）
print(f"\n4. 检查索引匹配情况...")
if not key_ints:
    print(f"   ⚠️  由于 LMDB 为空，无法进行索引匹配检查")
    print(f"   需要先重新生成 LMDB 文件")
elif 'test' in split:
    test_indices = split['test']
    if hasattr(test_indices, 'tolist'):
        test_indices = test_indices.tolist()
    else:
        test_indices = list(test_indices)
    
    max_key = max(key_ints)
    min_key = min(key_ints)
    
    print(f"   LMDB 键范围: [{min_key}, {max_key}]")
    print(f"   测试集索引范围: [{min(test_indices)}, {max(test_indices)}]")
    
    # 检查超出范围的索引
    out_of_range = [idx for idx in test_indices if idx < min_key or idx > max_key]
    
    if out_of_range:
        print(f"\n   ⚠️  发现 {len(out_of_range)} 个超出范围的索引:")
        print(f"   前10个超出范围的索引: {out_of_range[:10]}")
        print(f"\n   问题分析:")
        print(f"   - Split 文件中的索引是基于原始 index.pkl 的")
        print(f"   - 但 LMDB 中只包含成功处理的样本")
        print(f"   - 如果处理时跳过了某些样本，索引就会不匹配")
    else:
        print(f"   ✅ 所有测试集索引都在 LMDB 键范围内")
    
    # 检查索引 5
    if 5 in test_indices:
        if 5 in key_ints:
            print(f"\n   ✅ 索引 5 在测试集中，且 LMDB 中存在对应的键")
        else:
            print(f"\n   ❌ 索引 5 在测试集中，但 LMDB 中不存在对应的键")
            print(f"   原因: 该样本在处理时可能被跳过了")
    else:
        print(f"\n   ⚠️  索引 5 不在测试集中")
        print(f"   检查索引 5 是否在其他集合中...")
        for key, indices in split.items():
            if key != 'test':
                if hasattr(indices, 'tolist'):
                    indices_list = indices.tolist()
                else:
                    indices_list = list(indices)
                if 5 in indices_list:
                    print(f"   索引 5 在 {key} 集中")

# 5. 检查原始 index.pkl
index_path = './data/crossdocked_v1.1_rmsd1.0_pocket10/index.pkl'
print(f"\n5. 检查原始 index.pkl: {index_path}")
if os.path.exists(index_path):
    try:
        with open(index_path, 'rb') as f:
            index = pickle.load(f)
        print(f"   ✅ index.pkl 存在，包含 {len(index)} 个条目")
        print(f"   前3个条目示例:")
        for i, item in enumerate(index[:3]):
            print(f"     [{i}]: {item}")
    except Exception as e:
        print(f"   ❌ 读取 index.pkl 失败: {e}")
else:
    print(f"   ⚠️  index.pkl 不存在")
    print(f"   提示: 可能需要从原始数据生成索引文件")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)

print("\n建议:")
print("1. 如果索引不匹配，可能需要重新生成 LMDB 文件")
print("2. 或者使用 LMDB 中实际存在的索引进行采样")
print("3. 检查数据路径是否正确，确保所有文件都能被正确读取")

