#!/usr/bin/env python3
"""
检查 sampling.yml 配置文件中所有参数是否被正常使用
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

# 读取配置文件
config_path = Path("configs/sampling.yml")
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 读取相关代码文件
sample_diffusion_path = Path("scripts/sample_diffusion.py")
molopt_score_model_path = Path("models/molopt_score_model.py")

with open(sample_diffusion_path, 'r', encoding='utf-8') as f:
    sample_diffusion_code = f.read()

with open(molopt_score_model_path, 'r', encoding='utf-8') as f:
    molopt_score_model_code = f.read()

# 定义所有参数及其预期使用位置
params_to_check = {
    # model 部分
    'model.checkpoint': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.model\.checkpoint', r'config\[.model.\]\[.checkpoint.\]'],
    },
    
    # sample 部分
    'sample.seed': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.sample\.seed', r'config\[.sample.\]\[.seed.\]'],
    },
    'sample.num_samples': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.sample\.num_samples', r'config\[.sample.\]\[.num_samples.\]', r'num_samples\s*=\s*config'],
    },
    'sample.num_steps': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.sample\.num_steps', r'config\[.sample.\]\[.num_steps.\]', r'num_steps\s*=\s*config'],
    },
    'sample.pos_only': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.sample\.pos_only', r'config\[.sample.\]\[.pos_only.\]', r'pos_only\s*=\s*config'],
    },
    'sample.center_pos_mode': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.sample\.center_pos_mode', r'config\[.sample.\]\[.center_pos_mode.\]', r'center_pos_mode\s*=\s*config'],
    },
    'sample.sample_num_atoms': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.sample\.sample_num_atoms', r'config\[.sample.\]\[.sample_num_atoms.\]', r'sample_num_atoms\s*=\s*config'],
    },
    'sample.mode': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'config\.sample\.mode', r'config\[.sample.\]\[.mode.\]', r'config\.sample\.get\(.mode'],
    },
    
    # dynamic 部分
    'sample.dynamic.method': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'dynamic_cfg\.get\(.method', r'config\.sample\.dynamic\.method', r'dynamic_method'],
    },
    
    # large_step 部分
    'sample.dynamic.large_step.batch_size': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'large_cfg\.get\(.batch_size', r'large_step.*batch_size'],
    },
    'sample.dynamic.large_step.n_repeat': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'large_cfg\.get\(.n_repeat', r'n_repeat\s*=\s*large_cfg'],
    },
    'sample.dynamic.large_step.stride': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'large_cfg\.get\(.stride', r'step_stride\s*=\s*large_cfg', r'defaults\.get\(.stride'],
    },
    'sample.dynamic.large_step.step_size': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'large_cfg\.get\(.step_size', r'step_size\s*=\s*large_cfg', r'defaults\.get\(.step_size'],
    },
    'sample.dynamic.large_step.noise_scale': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'large_cfg\.get\(.noise_scale', r'add_noise\s*=\s*large_cfg', r'defaults\.get\(.noise_scale'],
    },
    'sample.dynamic.large_step.time_lower': {
        'expected_locations': ['molopt_score_model.py'],
        'search_patterns': [r'defaults\.get\(.time_lower', r'lambda_floor\s*=\s*defaults'],
    },
    'sample.dynamic.large_step.schedule': {
        'expected_locations': ['molopt_score_model.py'],
        'search_patterns': [r'defaults\.get\(.schedule', r'schedule_mode\s*=\s*defaults'],
    },
    
    # refine 部分
    'sample.dynamic.refine.stride': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'refine_cfg\.get\(.stride', r'step_stride\s*=\s*refine_cfg', r'defaults\.get\(.stride'],
    },
    'sample.dynamic.refine.step_size': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'refine_cfg\.get\(.step_size', r'step_size\s*=\s*refine_cfg', r'defaults\.get\(.step_size'],
    },
    'sample.dynamic.refine.noise_scale': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'refine_cfg\.get\(.noise_scale', r'add_noise\s*=\s*refine_cfg', r'defaults\.get\(.noise_scale'],
    },
    'sample.dynamic.refine.time_upper': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'refine_cfg\.get\(.time_upper', r'time_upper\s*=\s*refine_cfg', r'defaults\.get\(.time_upper'],
    },
    'sample.dynamic.refine.time_lower': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'refine_cfg\.get\(.time_lower', r'time_lower\s*=\s*refine_cfg', r'defaults\.get\(.time_lower'],
    },
    'sample.dynamic.refine.cycles': {
        'expected_locations': ['sample_diffusion.py', 'molopt_score_model.py'],
        'search_patterns': [r'refine_cfg\.get\(.cycles', r'num_cycles\s*=\s*refine_cfg', r'defaults\.get\(.cycles'],
    },
    'sample.dynamic.refine.n_sampling': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'refine_cfg\.get\(.n_sampling', r'n_sampling\s*=\s*refine_cfg'],
    },
    'sample.dynamic.refine.schedule': {
        'expected_locations': ['molopt_score_model.py'],
        'search_patterns': [r'defaults\.get\(.schedule', r'schedule_mode\s*=\s*defaults'],
    },
    
    # selector 部分
    'sample.dynamic.selector.top_n': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'selector_cfg\.get\(.top_n', r'top_n\s*=\s*selector_cfg'],
    },
    'sample.dynamic.selector.min_qed': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'selector_cfg\.get\(.min_qed', r'min_qed\s*=\s*selector_cfg'],
    },
    'sample.dynamic.selector.max_sa': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'selector_cfg\.get\(.max_sa', r'max_sa\s*=\s*selector_cfg'],
    },
    'sample.dynamic.selector.qed_weight': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'selector_cfg\.get\(.qed_weight', r'qed_weight\s*=\s*selector_cfg'],
    },
    'sample.dynamic.selector.sa_weight': {
        'expected_locations': ['sample_diffusion.py'],
        'search_patterns': [r'selector_cfg\.get\(.sa_weight', r'sa_weight\s*=\s*selector_cfg'],
    },
}

def check_parameter_usage(param_name: str, param_info: Dict) -> Tuple[bool, List[str]]:
    """检查参数是否在代码中被使用"""
    found_locations = []
    all_code = {
        'sample_diffusion.py': sample_diffusion_code,
        'molopt_score_model.py': molopt_score_model_code,
    }
    
    for file_name, code in all_code.items():
        if file_name not in param_info['expected_locations']:
            continue
            
        for pattern in param_info['search_patterns']:
            matches = re.finditer(pattern, code, re.MULTILINE)
            for match in matches:
                line_num = code[:match.start()].count('\n') + 1
                found_locations.append(f"{file_name}:{line_num}")
    
    return len(found_locations) > 0, found_locations

def get_nested_value(d: dict, keys: List[str]):
    """递归获取嵌套字典的值"""
    for key in keys:
        if isinstance(d, dict) and key in d:
            d = d[key]
        else:
            return None
    return d

def main():
    print("=" * 80)
    print("sampling.yml 参数使用情况检查")
    print("=" * 80)
    print()
    
    results = {
        'used': [],
        'unused': [],
        'partial': [],
    }
    
    for param_name, param_info in params_to_check.items():
        keys = param_name.split('.')
        config_value = get_nested_value(config, keys)
        
        if config_value is None:
            print(f"[!] {param_name}: 配置文件中不存在")
            continue
        
        is_used, locations = check_parameter_usage(param_name, param_info)
        
        if is_used:
            results['used'].append((param_name, locations))
            print(f"[OK] {param_name}")
            print(f"   使用位置: {', '.join(set(locations))}")
        else:
            results['unused'].append(param_name)
            print(f"[X] {param_name}: 未找到使用位置")
        print()
    
    print("=" * 80)
    print("总结")
    print("=" * 80)
    print(f"[OK] 正常使用: {len(results['used'])} 个参数")
    print(f"[X] 未使用: {len(results['unused'])} 个参数")
    print()
    
    if results['unused']:
        print("未使用的参数:")
        for param in results['unused']:
            print(f"  - {param}")
    
    print()
    print("注意: 此检查基于代码模式匹配，可能存在误报。")
    print("建议结合人工审查确认参数的实际使用情况。")

if __name__ == '__main__':
    main()

