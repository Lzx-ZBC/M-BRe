#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按 label 去重并同步过滤 entity/sent
条件：
  1. 每个唯一 label 最多留 20 个（随机）
  2. 只考虑 entity 行非空的索引
"""

import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple

# ========================== 路径配置 ==========================
label_path = "/data/zxli/M-BRe/NER/bbn_afet/test_data/test_label.txt"
entity_path = "/data/zxli/M-BRe/NER/bbn_afet/test_data/test_entity.txt"
sent_path  = "/data/zxli/M-BRe/NER/bbn_afet/test_data/test_sent.txt"

# 输出路径（如想自定义，可修改这里）
out_label  = label_path.replace(".txt", "_filtered.txt")
out_entity = entity_path.replace(".txt", "_filtered.txt")
out_sent   = sent_path.replace(".txt", "_filtered.txt")

# ========================== 读文件 ==========================
with open(label_path, 'r', encoding='utf-8') as f:
    labels = [json.loads(line.strip()) for line in f]

with open(entity_path, 'r', encoding='utf-8') as f:
    entities = [line.rstrip('\n') for line in f]

with open(sent_path, 'r', encoding='utf-8') as f:
    sents = [line.rstrip('\n') for line in f]

assert len(labels) == len(entities) == len(sents), "三文件行数不一致！"

# ========================== 建索引 ==========================
# key: label 元组，value: List[该行非空且 label 匹配的索引]
label2valid: Dict[Tuple[str, ...], List[int]] = defaultdict(list)

for idx, (lab, ent) in enumerate(zip(labels, entities)):
    if ent.strip():                       # 只保留 entity 非空
        label2valid[tuple(lab)].append(idx)

# ========================== 随机采样 ==========================
keep_idxs: List[int] = []
for tup, idxs in label2valid.items():
    # 随机选 10 个（或全部）
    chosen = random.sample(idxs, min(10, len(idxs))) if len(idxs) > 10 else idxs[:]
    keep_idxs.extend(chosen)

# 按原顺序排序
keep_idxs.sort()

# ========================== 写过滤后文件 ==========================
with open(out_label, 'w', encoding='utf-8') as f_l, \
     open(out_entity, 'w', encoding='utf-8') as f_e, \
     open(out_sent, 'w', encoding='utf-8') as f_s:

    for k in keep_idxs:
        f_l.write(json.dumps(labels[k], ensure_ascii=False) + '\n')
        f_e.write(entities[k] + '\n')
        f_s.write(sents[k] + '\n')

print(f"处理完成！保留 {len(keep_idxs)} 行，已写入：")
print(out_label)
print(out_entity)
print(out_sent)