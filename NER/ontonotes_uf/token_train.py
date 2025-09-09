#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import re

# ---------- 配置 ----------
INPUT_FILE = '/data/zxli/M-BRe/NER/ontonotes_uf/train_data/g_train.json'
OUT_SENT   = '/data/zxli/M-BRe/NER/ontonotes_uf/train_data/train_sent.txt'
OUT_LABEL  = '/data/zxli/M-BRe/NER/ontonotes_uf/train_data/train_label.txt'
OUT_ENTITY = '/data/zxli/M-BRe/NER/ontonotes_uf/train_data/train_entity.txt'
# --------------------------

def detokenize(tokens):
    """反 token 化规则"""
    sent = " ".join(tokens)
    sent = re.sub(r"\s+([',.!?;:%])", r"\1", sent)
    sent = re.sub(r"\s+([)\]\"”])", r"\1", sent)
    sent = re.sub(r"([(\[\"“])\s+", r"\1", sent)
    sent = re.sub(r"([$£€])\s+", r"\1", sent)
    sent = re.sub(r"(\d)\s+%", r"\1%", sent)
    sent = re.sub(r"\b([A-Z])\s+\.\s+([A-Z])\s+\.", r"\1.\2.", sent)
    sent = re.sub(r"\s{2,}", " ", sent).strip()
    return sent

def main():
    # 清理旧文件
    for f in (OUT_SENT, OUT_LABEL, OUT_ENTITY):
        if os.path.exists(f):
            os.remove(f)

    with open(INPUT_FILE, 'r', encoding='utf-8') as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            # 1. 拼接并反 token 化
            tokens = data['left_context_token'] + \
                     [data['mention_span']] + \
                     data['right_context_token']
            sent = detokenize(tokens)

            # 2. 确定要写的标签行
            raw_label = data['y_str']          # 已经是 list
            if isinstance(raw_label, list) and raw_label:
                labels_to_write = [str(lbl) for lbl in raw_label]
            else:
                labels_to_write = [str(raw_label)]

            n = len(labels_to_write)           # 需要重复写的次数

            # 3. 实体
            entity = data['mention_span']

            # 4. 批量写入（保证三者行数一致）
            with open(OUT_SENT, 'a', encoding='utf-8') as fs, \
                 open(OUT_LABEL, 'a', encoding='utf-8') as fl, \
                 open(OUT_ENTITY, 'a', encoding='utf-8') as fe:
                for lbl in labels_to_write:
                    fs.write(sent + '\n')
                    fl.write(lbl + '\n')
                    fe.write(entity + '\n')

    print('Finished! 输出：', OUT_SENT, OUT_LABEL, OUT_ENTITY)

if __name__ == '__main__':
    main()