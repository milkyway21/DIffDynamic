# 快速重新生成 LMDB 指南

## 当前状态

根据检查结果：
- ✅ 数据路径存在：`./data/crossdocked_v1.1_rmsd1.0_pocket10`
- ✅ index.pkl 存在：包含 184087 个条目
- ✅ 数据文件存在：前5个文件都能找到
- ❌ LMDB 文件为空：只有 0.01 MB

## 快速解决方案

### 方法 1：使用自动重新生成脚本（推荐）

```bash
# 运行重新生成脚本
python 重新生成LMDB.py
```

这个脚本会：
1. 自动删除旧的 LMDB 文件
2. 重新生成 LMDB
3. 显示进度和调试信息

### 方法 2：手动删除后自动生成

```bash
# 1. 删除空的 LMDB 文件
rm -f ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
rm -f ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb-lock

# 2. 运行采样命令（会自动触发 LMDB 生成）
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0 --device cuda:0 --mode dynamic --result_path ./outputs/test_regenerate
```

## 注意事项

### 1. 生成时间

- 184087 个样本需要很长时间处理
- 预计需要几小时到十几小时（取决于硬件）
- 请耐心等待，不要中断

### 2. 监控进度

生成过程中会显示：
- 进度条（tqdm）
- 跳过的样本信息（如果有）

### 3. 如果所有样本都被跳过

如果看到大量 "Skipping" 消息，可能是：
- 文件路径问题
- 文件格式问题
- 依赖库问题（RDKit、OpenBabel 等）

检查第一个错误：
```bash
# 查看处理日志，找到第一个失败的原因
python 重新生成LMDB.py 2>&1 | head -50
```

## 验证生成成功

生成完成后，检查：

```bash
# 检查 LMDB 文件大小
ls -lh ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb

# 应该看到文件大小 > 100 MB（通常几 GB）

# 检查键数量
python -c "
import lmdb
db = lmdb.open('./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb', readonly=True, lock=False)
with db.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))
print(f'LMDB 中有 {len(keys)} 个键')
if len(keys) > 0:
    print(f'前5个键: {[k.decode() for k in keys[:5]]}')
"
```

## 如果生成失败

### 检查错误信息

```bash
python 重新生成LMDB.py 2>&1 | tee lmdb_generation.log
```

查看日志文件找出问题。

### 常见问题

1. **内存不足**
   - 减少批量处理大小
   - 使用更大的 swap 空间

2. **磁盘空间不足**
   - 确保至少有 10GB 可用空间
   - LMDB 文件可能很大

3. **文件权限问题**
   - 确保有写入权限
   - `chmod -R 755 ./data/`

4. **依赖库问题**
   - 确保 RDKit、OpenBabel 正确安装
   - 测试解析单个文件

## 测试单个文件解析

如果怀疑文件解析有问题，可以测试：

```python
from utils.data import PDBProtein, parse_sdf_file
import os

# 测试第一个文件
pocket_path = './data/crossdocked_v1.1_rmsd1.0_pocket10/1B57_HUMAN_25_300_0/3upr_C_rec.pdb'
ligand_path = './data/crossdocked_v1.1_rmsd1.0_pocket10/1B57_HUMAN_25_300_0/3upr_C_rec_3vri_1kx_lig_tt_min_0.sdf'

try:
    pocket_dict = PDBProtein(pocket_path).to_dict_atom()
    print(f"✅ 口袋文件解析成功: {len(pocket_dict['element'])} 个原子")
except Exception as e:
    print(f"❌ 口袋文件解析失败: {e}")

try:
    ligand_dict = parse_sdf_file(ligand_path)
    print(f"✅ 配体文件解析成功")
except Exception as e:
    print(f"❌ 配体文件解析失败: {e}")
```

## 推荐操作流程

```bash
# 1. 删除旧的 LMDB
rm -f ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb*

# 2. 运行重新生成脚本（会显示详细进度）
python 重新生成LMDB.py

# 3. 等待完成（可能需要几小时）

# 4. 验证生成成功
python -c "
import lmdb
db = lmdb.open('./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb', readonly=True, lock=False)
with db.begin() as txn:
    keys = list(txn.cursor().iternext(values=False))
print(f'✅ LMDB 生成成功！包含 {len(keys)} 个样本')
"

# 5. 如果成功，运行采样
python scripts/sample_diffusion.py configs/sampling.yml --data_id 5 --device cuda:0 --mode dynamic --result_path ./outputs/pocket5_samples
```

