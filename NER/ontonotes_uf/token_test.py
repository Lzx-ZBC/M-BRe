#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import re

# ---------- 配置 ----------
INPUT_FILE = '/data/zxli/M-BRe/NER/ontonotes_uf/test_data/g_test.json'
OUT_SENT   = '/data/zxli/M-BRe/NER/ontonotes_uf/test_data/test_sent.txt'
OUT_LABEL  = '/data/zxli/M-BRe/NER/ontonotes_uf/test_data/test_label.txt'
OUT_ENTITY = '/data/zxli/M-BRe/NER/ontonotes_uf/test_data/test_entity.txt'
# --------------------------

def detokenize(tokens):
    """你给出的反 token 化规则"""
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

            # 2. 标签列表直接转字符串
            label = str(data['y_str'])      # 如 "['/organization', '/organization/company']"

            # 3. 实体
            entity = data['mention_span']

            # 追加写入
            with open(OUT_SENT, 'a', encoding='utf-8') as fs:
                fs.write(sent + '\n')
            with open(OUT_LABEL, 'a', encoding='utf-8') as fl:
                fl.write(label + '\n')
            with open(OUT_ENTITY, 'a', encoding='utf-8') as fe:
                fe.write(entity + '\n')

    print('Finished! 输出：', OUT_SENT, OUT_LABEL, OUT_ENTITY)

if __name__ == '__main__':
    main()