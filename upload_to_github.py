#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将项目中所有小于10MB的文件上传到GitHub仓库
"""

import os
import subprocess
import sys
from pathlib import Path

# 配置
REPO_URL = "https://github.com/milkyway21/DIffDynamic.git"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB in bytes
PROJECT_ROOT = Path(__file__).parent.absolute()

def get_file_size(file_path):
    """获取文件大小（字节）"""
    try:
        return os.path.getsize(file_path)
    except (OSError, FileNotFoundError):
        return 0

def is_git_ignored(file_path, gitignore_patterns):
    """检查文件是否被.gitignore忽略"""
    file_path_str = str(file_path.relative_to(PROJECT_ROOT))
    # 检查是否匹配任何gitignore模式
    for pattern in gitignore_patterns:
        if pattern.startswith('/'):
            pattern = pattern[1:]
        if pattern.endswith('/'):
            if file_path_str.startswith(pattern):
                return True
        elif pattern in file_path_str or file_path_str.endswith(pattern):
            return True
    return False

def load_gitignore_patterns():
    """加载.gitignore中的模式"""
    gitignore_file = PROJECT_ROOT / '.gitignore'
    patterns = []
    if gitignore_file.exists():
        with open(gitignore_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    return patterns

def run_git_command(cmd, check=True):
    """执行Git命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if check and result.returncode != 0:
            print(f"错误: Git命令执行失败: {cmd}")
            print(f"错误信息: {result.stderr}")
            return False
        return result
    except Exception as e:
        print(f"执行Git命令时出错: {e}")
        return None

def init_git_repo():
    """初始化Git仓库"""
    if (PROJECT_ROOT / '.git').exists():
        print("Git仓库已存在")
        return True
    
    print("初始化Git仓库...")
    result = run_git_command('git init', check=False)
    if result and result.returncode == 0:
        print("[OK] Git仓库初始化成功")
        return True
    else:
        print("[ERROR] Git仓库初始化失败")
        return False

def setup_remote():
    """设置远程仓库"""
    print("检查远程仓库配置...")
    result = run_git_command('git remote -v', check=False)
    
    if result and 'origin' in result.stdout and REPO_URL in result.stdout:
        print("[OK] 远程仓库已配置")
        return True
    
    # 添加或更新远程仓库
    run_git_command('git remote remove origin', check=False)
    result = run_git_command(f'git remote add origin {REPO_URL}', check=False)
    if result and result.returncode == 0:
        print(f"[OK] 远程仓库已设置为: {REPO_URL}")
        return True
    else:
        print("[ERROR] 设置远程仓库失败")
        return False

def find_small_files():
    """查找所有小于10MB的文件"""
    print(f"\n扫描项目目录: {PROJECT_ROOT}")
    print(f"文件大小限制: {MAX_FILE_SIZE / (1024*1024):.1f} MB")
    
    gitignore_patterns = load_gitignore_patterns()
    small_files = []
    total_size = 0
    skipped_count = 0
    
    # 遍历所有文件
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # 跳过.git目录
        if '.git' in root:
            continue
        
        # 跳过一些明显的大文件目录
        skip_dirs = ['.git', '__pycache__', 'node_modules']
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        for file in files:
            file_path = Path(root) / file
            
            # 跳过.gitignore中的文件
            if is_git_ignored(file_path, gitignore_patterns):
                skipped_count += 1
                continue
            
            # 检查文件大小
            file_size = get_file_size(file_path)
            if file_size == 0:
                continue
            
            if file_size < MAX_FILE_SIZE:
                small_files.append(file_path)
                total_size += file_size
            else:
                skipped_count += 1
                print(f"跳过大文件: {file_path.relative_to(PROJECT_ROOT)} ({file_size / (1024*1024):.2f} MB)")
    
    print(f"\n找到 {len(small_files)} 个小文件 (总计 {total_size / (1024*1024):.2f} MB)")
    print(f"跳过 {skipped_count} 个大文件或忽略文件")
    
    return small_files

def add_files_to_git(files):
    """将文件添加到Git"""
    if not files:
        print("没有文件需要添加")
        return False
    
    print(f"\n添加 {len(files)} 个文件到Git...")
    
    # 分批添加文件，避免命令行过长
    batch_size = 100
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        file_paths = [str(f.relative_to(PROJECT_ROOT)) for f in batch]
        
        # 使用git add命令
        cmd = 'git add ' + ' '.join(f'"{f}"' for f in file_paths)
        result = run_git_command(cmd, check=False)
        
        if result and result.returncode == 0:
            print(f"[OK] 已添加 {min(i+batch_size, len(files))}/{len(files)} 个文件")
        else:
            print(f"[ERROR] 添加文件时出错 (批次 {i//batch_size + 1})")
            if result:
                print(f"错误信息: {result.stderr}")
    
    return True

def commit_and_push():
    """提交并推送到GitHub"""
    print("\n检查是否有更改需要提交...")
    result = run_git_command('git status --porcelain', check=False)
    
    if not result or not result.stdout.strip():
        print("没有更改需要提交")
        return False
    
    # 提交
    print("提交更改...")
    commit_msg = "Upload files smaller than 10MB"
    result = run_git_command(f'git commit -m "{commit_msg}"', check=False)
    
    if result and result.returncode == 0:
        print("[OK] 提交成功")
    else:
        print("[ERROR] 提交失败")
        if result:
            print(f"错误信息: {result.stderr}")
        return False
    
    # 推送到GitHub
    print("推送到GitHub...")
    result = run_git_command('git branch -M main', check=False)
    
    result = run_git_command('git push -u origin main', check=False)
    if result and result.returncode == 0:
        print("[OK] 推送成功")
        print(f"\n[OK] 所有文件已成功上传到: {REPO_URL}")
        return True
    else:
        print("[ERROR] 推送失败")
        if result:
            print(f"错误信息: {result.stderr}")
        print("\n提示: 如果这是第一次推送，可能需要:")
        print("  1. 确保GitHub仓库已创建")
        print("  2. 配置GitHub认证 (使用Personal Access Token)")
        print("  3. 手动执行: git push -u origin main")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("GitHub文件上传脚本")
    print("=" * 60)
    
    # 1. 初始化Git仓库
    if not init_git_repo():
        print("无法初始化Git仓库，退出")
        return 1
    
    # 2. 设置远程仓库
    if not setup_remote():
        print("无法设置远程仓库，退出")
        return 1
    
    # 3. 查找小文件
    small_files = find_small_files()
    
    if not small_files:
        print("没有找到需要上传的文件")
        return 0
    
    # 4. 添加文件到Git
    if not add_files_to_git(small_files):
        print("添加文件失败")
        return 1
    
    # 5. 提交并推送
    if not commit_and_push():
        print("提交或推送失败，但文件已添加到Git暂存区")
        print("您可以手动执行以下命令:")
        print("  git commit -m 'Upload files smaller than 10MB'")
        print("  git push -u origin main")
        return 1
    
    print("\n" + "=" * 60)
    print("[OK] 完成!")
    print("=" * 60)
    return 0

if __name__ == '__main__':
    sys.exit(main())

