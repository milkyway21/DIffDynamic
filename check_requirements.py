#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查项目运行所需的所有依赖和文件
"""
import os
import sys
from pathlib import Path

# 修复Windows控制台编码问题
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def check_file(path, name, required=True):
    """检查文件是否存在"""
    exists = os.path.exists(path)
    status = "[OK]" if exists else ("[MISSING]" if required else "[OPTIONAL]")
    req_text = "必需" if required else "可选"
    print(f"{status} {name}: {path} ({req_text})")
    if not exists and required:
        print(f"   需要下载/准备此文件")
    return exists

def check_package(package_name, import_name=None):
    """检查Python包是否安装"""
    if import_name is None:
        import_name = package_name
    try:
        __import__(import_name)
        print(f"[OK] {package_name}: 已安装")
        return True
    except ImportError:
        print(f"[MISSING] {package_name}: 未安装")
        return False

def main():
    print("=" * 60)
    print("DiffDynamic 项目运行环境检查")
    print("=" * 60)
    print()
    
    # 检查Python版本
    print("【Python环境】")
    version = sys.version_info
    print(f"Python版本: {version.major}.{version.minor}.{version.micro}")
    if version.major != 3 or version.minor < 8:
        print("[WARNING] 建议使用 Python 3.8-3.10")
    print()
    
    # 检查必需包
    print("【Python依赖包】")
    packages = [
        ("torch", "torch"),
        ("torch_geometric", "torch_geometric"),
        ("rdkit", "rdkit"),
        ("openbabel", "openbabel"),
        ("numpy", "numpy"),
        ("yaml", "yaml"),
        ("lmdb", "lmdb"),
    ]
    all_packages_ok = True
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            all_packages_ok = False
    print()
    
    # 检查数据文件
    print("【数据文件】")
    data_files = [
        ("数据划分文件", "data/crossdocked_pocket10_pose_split.pt", True),
        ("训练数据 (LMDB)", "data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb", False),
        ("训练数据目录", "data/crossdocked_v1.1_rmsd1.0_pocket10", False),
        ("亲和力信息", "data/affinity_info.pkl", False),
    ]
    all_data_ok = True
    for name, path, required in data_files:
        if not check_file(path, name, required):
            if required:
                all_data_ok = False
    print()
    
    # 检查模型文件
    print("【模型文件】")
    model_files = [
        ("预训练扩散模型", "pretrained_models/pretrained_diffusion.pt", True),
    ]
    all_models_ok = True
    for name, path, required in model_files:
        if not check_file(path, name, required):
            if required:
                all_models_ok = False
    print()
    
    # 检查配置文件
    print("【配置文件】")
    config_files = [
        ("训练配置", "configs/training.yml", True),
        ("采样配置", "configs/sampling.yml", True),
    ]
    all_configs_ok = True
    for name, path, required in config_files:
        if not check_file(path, name, required):
            if required:
                all_configs_ok = False
    print()
    
    # 检查示例文件
    print("【示例文件】")
    example_files = [
        ("示例口袋", "examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb", False),
    ]
    for name, path, required in example_files:
        check_file(path, name, required)
    print()
    
    # 总结
    print("=" * 60)
    print("【检查总结】")
    if all_packages_ok and all_data_ok and all_models_ok and all_configs_ok:
        print("[SUCCESS] 所有必需项已就绪，可以开始运行项目！")
        print()
        print("快速开始：")
        print("1. 使用示例口袋采样：")
        print("   python scripts/sample_for_pocket.py configs/sampling.yml")
        print("     --pdb_path examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb")
        print()
        print("2. 从测试集采样：")
        print("   python scripts/sample_diffusion.py configs/sampling.yml --data_id 0")
    else:
        print("[WARNING] 还有必需项未准备完成，请参考 '运行准备清单.md'")
        if not all_packages_ok:
            print("   - 需要安装Python依赖包")
        if not all_data_ok:
            print("   - 需要下载数据文件")
        if not all_models_ok:
            print("   - 需要下载预训练模型")
    print("=" * 60)

if __name__ == "__main__":
    main()

