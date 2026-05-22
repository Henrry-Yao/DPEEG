#!/usr/bin/env python3

# 读取原始文件内容
with open(main.py, r) as f:
    lines = f.readlines()

# 查找需要修改的位置
import_line_idx = -1
init_end_idx = -1
training_step_idx = -1
test_step_idx = -1
parser_end_idx = -1
config_end_idx = -1
logger_line_idx = -1

for i, line in enumerate(lines):
    if from
