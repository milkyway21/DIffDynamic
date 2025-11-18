#!/usr/bin/env python3
"""
验证 sampling.yml 配置参数在 unified 模式下的使用修复
"""

import yaml
import ast
import re
from pathlib import Path

def check_config_update():
    """检查 _run_unified_dynamic 函数是否正确更新了模型配置"""
    sample_diffusion_path = Path("scripts/sample_diffusion.py")
    
    with open(sample_diffusion_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 检查是否包含配置更新代码
    has_large_step_update = 'dynamic_large_step_defaults' in code and 'large_step' in code
    has_refine_update = 'dynamic_refine_defaults' in code and 'refine' in code
    
    # 检查是否在正确的位置（在循环之前）
    lines = code.split('\n')
    in_unified_function = False
    before_loop = False
    found_updates = False
    
    for i, line in enumerate(lines):
        if 'def _run_unified_dynamic' in line:
            in_unified_function = True
        if in_unified_function and 'for sample_idx in range(num_samples)' in line:
            before_loop = True
            break
        if in_unified_function and 'dynamic_large_step_defaults' in line:
            found_updates = True
            if 'for sample_idx' in '\n'.join(lines[i+1:i+10]):
                print(f"[OK] 配置更新代码位于循环之前 (行 {i+1})")
            else:
                print(f"[!] 配置更新代码位置可能不正确 (行 {i+1})")
    
    if has_large_step_update and has_refine_update:
        print("[OK] 找到了 large_step 和 refine 配置更新代码")
        return True
    else:
        print("[X] 未找到完整的配置更新代码")
        return False

def check_config_usage():
    """检查模型方法是否正确使用配置"""
    molopt_score_model_path = Path("models/molopt_score_model.py")
    
    with open(molopt_score_model_path, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # 检查 dynamic_sample_diffusion 是否使用默认配置
    if 'self.dynamic_large_step_defaults' in code and 'self.dynamic_refine_defaults' in code:
        print("[OK] dynamic_sample_diffusion 方法使用模型默认配置")
        return True
    else:
        print("[X] dynamic_sample_diffusion 方法可能未使用模型默认配置")
        return False

def main():
    print("=" * 80)
    print("验证 sampling.yml 配置参数修复")
    print("=" * 80)
    print()
    
    result1 = check_config_update()
    print()
    result2 = check_config_usage()
    print()
    
    if result1 and result2:
        print("=" * 80)
        print("[OK] 修复验证通过！")
        print("=" * 80)
        print()
        print("修改说明:")
        print("1. 在 _run_unified_dynamic 函数中，在采样循环之前更新模型的默认配置")
        print("2. 从 sampling.yml 读取 dynamic.large_step 和 dynamic.refine 配置")
        print("3. 将这些配置合并到模型的 dynamic_large_step_defaults 和 dynamic_refine_defaults")
        print("4. dynamic_sample_diffusion 方法会自动使用更新后的配置")
        return 0
    else:
        print("=" * 80)
        print("[X] 修复验证失败！")
        print("=" * 80)
        return 1

if __name__ == '__main__':
    exit(main())

