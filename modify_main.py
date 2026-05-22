#!/usr/bin/env python3

# 读取原始文件内容
with open(/home/henrry/UBP/Uncertainty-aware-Blur-Prior-main/main.py, r) as f:
    content = f.read()

# 添加diffusion模型的导入
content = content.replace(
    from
