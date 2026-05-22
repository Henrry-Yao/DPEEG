#!/usr/bin/env python3

# 读取文件内容
with open(main.py, r) as f:
    lines = f.readlines()

# 修复diffusion模型初始化的缩进问题
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # 处理diffusion模型初始化部分
    if self.match_similarities
