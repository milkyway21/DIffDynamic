# LMDB 空文件问题解决方案

## 问题确认

根据诊断结果：
- ✅ Split 文件存在且正常（测试集有 100 个样本）
- ✅ LMDB 文件存在
- ❌ **LMDB 文件是空的（0 个键）**

这意味着处理时没有成功写入任何数据。

## 问题原因

LMDB 为空通常是因为：

1. **index.pkl 文件不存在或为空**
2. **数据路径配置错误**，找不到文件
3. **所有文件都解析失败**（文件损坏或格式错误）
4. **处理过程中断**，没有提交事务

## 解决步骤

### 步骤 1：检查数据文件

运行检查脚本：

```bash
python 检查LMDB生成问题.py
```

这会检查：
- 数据路径是否存在
- index.pkl 是否存在
- 前几个数据文件是否存在

### 步骤 2：删除空的 LMDB 文件

```bash
# 删除空的 LMDB 文件
rm -f ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb

# 同时删除锁文件（如果存在）
rm -f ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb-lock
```

### 步骤 3：检查数据路径配置

检查检查点中的配置：

```python
import torch
ckpt = torch.load('pretrained_models/pretrained_diffusion.pt', map_location='cpu')
print("数据路径配置:", ckpt['config'].data.path)
print("数据路径:", ckpt['config'].data.path)
```

确保路径指向正确的目录。

### 步骤 4：重新生成 LMDB

有两种方式：

#### 方式 A：自动重新生成（推荐）

删除 LMDB 文件后，运行采样命令会自动触发重新生成：

```bash
# 删除旧的 LMDB
rm -f ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb

# 运行采样（会自动重新生成 LMDB）
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0 --device cuda:0 --mode dynamic --result_path ./outputs/test_regenerate
```

**注意**：重新生成可能需要很长时间（取决于数据量），请耐心等待。

#### 方式 B：手动触发生成

创建一个简单的脚本来触发 LMDB 生成：

```python
# generate_lmdb.py
import torch
from datasets import get_dataset
import utils.transforms as trans
from torch_geometric.transforms import Compose

# 加载检查点获取配置
ckpt = torch.load('pretrained_models/pretrained_diffusion.pt', map_location='cpu')
config = ckpt['config']

# 创建转换器
protein_featurizer = trans.FeaturizeProteinAtom()
ligand_atom_mode = config.data.transform.ligand_atom_mode
ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
transform = Compose([
    protein_featurizer,
    ligand_featurizer,
    trans.FeaturizeLigandBond(),
])

# 这会触发 LMDB 生成
print("开始生成 LMDB...")
dataset, subsets = get_dataset(config=config.data, transform=transform)
print(f"✅ LMDB 生成完成！数据集大小: {len(dataset)}")
```

运行：
```bash
python generate_lmdb.py
```

## 如果数据文件不存在

如果您没有原始数据文件，有两个选择：

### 选择 1：下载预处理好的 LMDB（推荐）

从 Google Drive 下载：
- 链接：https://drive.google.com/drive/folders/1j21cc7-97TedKh_El5E34yI8o5ckI7eK?usp=share_link
- 文件：`crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`
- 放置位置：`./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb`

### 选择 2：从原始数据预处理

如果要从原始 CrossDocked2020 数据开始：

```bash
# 1. 下载 CrossDocked2020 v1.1
# 保存到 data/CrossDocked2020

# 2. 清洗数据
python scripts/data_preparation/clean_crossdocked.py \
  --source data/CrossDocked2020 \
  --dest data/crossdocked_v1.1_rmsd1.0 \
  --rmsd_thr 1.0

# 3. 提取口袋
python scripts/data_preparation/extract_pockets.py \
  --source data/crossdocked_v1.1_rmsd1.0 \
  --dest data/crossdocked_v1.1_rmsd1.0_pocket10

# 4. 运行采样会自动生成 LMDB
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0 --device cuda:0
```

## 验证 LMDB 生成成功

生成后，检查 LMDB：

```bash
python -c "
import lmdb
db = lmdb.open('./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb', readonly=True, lock=False)
with db.begin() as txn:
    keys = [int(k.decode()) for k in txn.cursor().iternext(values=False)]
print(f'LMDB 中有 {len(keys)} 个键')
print(f'键范围: [{min(keys)}, {max(keys)}]')
"
```

应该看到大量的键（例如 100000+）。

## 常见问题

### Q1: 重新生成时所有样本都被跳过

**原因**：数据路径不正确或文件不存在

**解决**：
1. 检查 `index.pkl` 中的路径是否正确
2. 检查实际文件是否存在
3. 可能需要修改 `index.pkl` 中的路径

### Q2: 重新生成很慢

**原因**：需要处理大量数据

**解决**：
- 这是正常的，请耐心等待
- 可以先用少量数据测试（修改 `index.pkl` 只保留前几个条目）

### Q3: 生成过程中断

**原因**：内存不足或磁盘空间不足

**解决**：
1. 删除不完整的 LMDB 文件
2. 确保有足够的磁盘空间（至少 10GB）
3. 确保有足够的内存
4. 重新运行生成

## 快速修复命令

```bash
# 1. 删除空的 LMDB
rm -f ./data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb*

# 2. 检查数据文件
python 检查LMDB生成问题.py

# 3. 如果数据文件存在，重新生成（会自动触发）
python scripts/sample_diffusion.py configs/sampling.yml --data_id 0 --device cuda:0 --mode dynamic --result_path ./outputs/test

# 4. 如果数据文件不存在，下载预处理好的 LMDB
# 从 Google Drive 下载并放置到 ./data/
```

