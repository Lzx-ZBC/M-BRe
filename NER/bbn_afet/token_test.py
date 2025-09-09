import json
import os

import re

def detokenize(tokens):
    """
    把 BERT-style 的 token 列表还原成正常句子，同时处理常见标点空格。
    """
    sent = " ".join(tokens)
    # 1. 去掉英文撇号、逗号、句号、百分号、问号、感叹号等前面的空格
    sent = re.sub(r"\s+([',.!?;:%])", r"\1", sent)
    # 2. 去掉括号、引号前面的空格
    sent = re.sub(r"\s+([)\]\"”])", r"\1", sent)
    # 3. 去掉括号、引号后面的空格
    sent = re.sub(r"([(\[\"“])\s+", r"\1", sent)
    # 4. 去掉 $、£ 等货币符号后面的空格
    sent = re.sub(r"([$£€])\s+", r"\1", sent)
    # 5. 去掉数字与 % 之间的空格
    sent = re.sub(r"(\d)\s+%", r"\1%", sent)
    # 6. 去掉缩写中的空格，如 "U . S ." -> "U.S."
    sent = re.sub(r"\b([A-Z])\s+\.\s+([A-Z])\s+\.", r"\1.\2.", sent)
    # 7. 去掉多余空格
    sent = re.sub(r"\s{2,}", " ", sent).strip()
    return sent

# 1. 路径
json_path = r"/data/zxli/M-BRe/NER/bbn_afet/test_data/test.json"
out_dir   = r"/data/zxli/M-BRe/NER/bbn_afet/test_data"

sent_file   = os.path.join(out_dir, "test_sent.txt")
entity_file = os.path.join(out_dir, "test_entity.txt")
label_file  = os.path.join(out_dir, "test_label.txt")

# 2. 读文件并处理
all_lines = []
with open(json_path, "r", encoding="utf-8") as f:
    for line in f:
        sample = json.loads(line.strip())
        tokens   = sample["tokens"]
        sentence = detokenize(tokens)

        for mention in sample.get("mentions", []):
            start, end = mention["start"], mention["end"]
            entity = " ".join(tokens[start:end])
            all_lines.append((sentence, entity, mention["labels"]))   # 直接保存整

# 3. 写三个 txt
with open(sent_file,   "w", encoding="utf-8") as fs, \
     open(entity_file, "w", encoding="utf-8") as fe, \
     open(label_file,  "w", encoding="utf-8") as fl:
    for sent, ent, lab in all_lines:
        fs.write(sent + "\n")
        fe.write(ent  + "\n")
        fl.write(json.dumps(lab, ensure_ascii=False) + "\n")

print("处理完成，共写入", len(all_lines), "行")