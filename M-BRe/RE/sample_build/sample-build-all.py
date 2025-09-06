# import json
# import random

# random.seed(42)

# # 定义文件路径
# file_path1 = "/data/zxli/kp/a-wiki-extract/samplebuild1.txt"
# file_path2 = "/data/zxli/kp/a-qwen2.5-instruct/relation_pred_frame-class-qwen2.5-wiki-relation-group4.txt"
# output_file_path = "/data/zxli/kp/a-qwen2.5-instruct/output_data_group4.txt"  # 输出文件路径

# # 读取 file_path2 的前2000行，切分并去重
# with open(file_path2, 'r', encoding='utf-8') as file2:
#     lines2 = [next(file2) for _ in range(4401)] 
#     relations_list = [list(set(line.strip().split())) for line in lines2]  # 每行切分为单词列表并去重

# # 读取 file_path1 的前2000行，解析 JSON 数据，并为每个数据添加 relation 键值
# new_data_list = []
# with open(file_path1, 'r', encoding='utf-8') as file1:
#     lines1 = [next(file1) for _ in range(4401)]  
#     for line1, relations in zip(lines1, relations_list):  # 一一对应处理
#         try:
#             data = json.loads(line1.strip())  # 解析 JSON 格式
#             # 根据 relations 列表中的单词数量生成样本
#             for relation in relations:
#                 new_data = data.copy()  # 创建一个副本
#                 new_data["relation"] = relation  # 添加 relation 键值
#                 new_data_list.append(new_data)
#         except json.JSONDecodeError as e:
#             print(f"解析 JSON 时出错：内容：{line1.strip()}")
#             print(f"错误信息：{e}")

# # 将新的数据保存到新的 txt 文件中
# with open(output_file_path, 'w', encoding='utf-8') as output_file:
#     for new_data in new_data_list:
#         json.dump(new_data, output_file, ensure_ascii=False)  # 以 JSON 格式写入文件
#         output_file.write("\n")  # 每个数据占一行

# print(f"新数据已保存到文件：{output_file_path}")

# # ---------------------------------------------------------------------------------------------------------
# # 打开文件并读取内容
# file_path = "/data/zxli/kp/a-qwen2.5-instruct/output_data_group4.txt"
# relation_counts = {}  # 用于存储每个relation类别的数量
# na_relations = []  # 用于存储所有NA类别的条目

# with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         # 去掉行首行尾的空格和换行符
#         line = line.strip()
#         # 找到relation键值
#         start = line.find('"relation": "') + len('"relation": "')
#         end = line.find('"', start)
#         relation_value = line[start:end]
#         # 统计每个relation类别的数量
#         if relation_value == "NA":
#             na_relations.append(line)  # 将NA条目存储起来
#             relation_counts["NA"] = relation_counts.get("NA", 0) + 1
#         else:
#             if relation_value in relation_counts:
#                 relation_counts[relation_value] += 1
#             else:
#                 relation_counts[relation_value] = 1

# # 输出各类别的数量
# print("relation类别及其数量：")
# for relation, count in relation_counts.items():
#     print(f"{relation}: {count}")

# # 计算非NA类别的平均数量（四舍五入）
# non_na_total = sum(count for relation, count in relation_counts.items() if relation != "NA")
# non_na_count = len([relation for relation in relation_counts if relation != "NA"])
# if non_na_count > 0:
#     avg_non_na_count = round(non_na_total / non_na_count)
# else:
#     avg_non_na_count = 0

# print(f"\n非NA类别的平均数量（四舍五入）: {avg_non_na_count}")

# # --------------------------------------------------------------------------------
# # 定义文件路径
# file_path = "/data/zxli/kp/a-qwen2.5-instruct/output_data_group4.txt"
# sample_file_path = "/data/zxli/kp/a-qwen2.5-instruct/sample_data_all_group4.txt"  # 输出样本文件路径

# relation_counts = {}  # 用于存储每个relation类别的数量
# relation_data = {}  # 用于存储每个relation类别的数据

# # 读取文件并统计每个relation类别的数量和数据
# with open(file_path, 'r', encoding='utf-8') as file:
#     for line in file:
#         try:
#             data = json.loads(line.strip())  # 解析 JSON 格式
#             relation_value = data.get("relation")
#             if relation_value:
#                 if relation_value in relation_counts:
#                     relation_counts[relation_value] += 1
#                     relation_data[relation_value].append(data)
#                 else:
#                     relation_counts[relation_value] = 1
#                     relation_data[relation_value] = [data]
#         except json.JSONDecodeError as e:
#             print(f"解析 JSON 时出错：内容：{line.strip()}")
#             print(f"错误信息：{e}")

# # 计算非NA类别的平均数量（四舍五入）
# non_na_total = sum(count for relation, count in relation_counts.items() if relation != "NA")
# non_na_count = len([relation for relation in relation_counts if relation != "NA"])
# if non_na_count > 0:
#     max_count = round(non_na_total / non_na_count)  # 作为最大数量
# else:
#     max_count = 0

# # 对每种关系随机获取min(某个关系的全部，最大数量)个样本
# sample_data_list = []
# sample_relation_counts = {}  # 用于统计样本中每个关系类别的数量
# for relation, data_list in relation_data.items():
#     sample_count = min(len(data_list), max_count)  # 获取样本数量
#     random_samples = random.sample(data_list, sample_count)
#     sample_data_list.extend(random_samples)
#     sample_relation_counts[relation] = sample_count  # 统计样本中每个关系类别的数量

# # 将抽取的样本输出到sample_data.txt中
# with open(sample_file_path, 'w', encoding='utf-8') as sample_file:
#     for sample_data in sample_data_list:
#         json.dump(sample_data, sample_file, ensure_ascii=False)  # 以 JSON 格式写入文件
#         sample_file.write("\n")  # 每个数据占一行

# print(f"抽取的样本已保存到文件：{sample_file_path}")

# # 统计sample_data.txt里各个关系类别的数量
# print("\n样本文件中各个关系类别的数量：")
# for relation, count in sample_relation_counts.items():
#     print(f"{relation}: {count}")

import json
import random

random.seed(42)

# 定义文件路径
file_path1 = "/data/zxli/kp/a-wiki-extract/samplebuild1.txt"
file_path2 = "/data/zxli/kp/a-qwen2.5-instruct/relation_pred_frame-class-qwen2.5-14B-wiki-relation-group4.txt"
output_file_path = "/data/zxli/kp/a-qwen2.5-instruct/qwen2.5-14B/output_data_group4.txt"  # 输出文件路径

# 读取 file_path2 的前2000行，切分并去重
with open(file_path2, 'r', encoding='utf-8') as file2:
    lines2 = [next(file2) for _ in range(4401)] 
    relations_list = [list(set(line.strip().split())) for line in lines2]  # 每行切分为单词列表并去重

# 读取 file_path1 的前2000行，解析 JSON 数据，并为每个数据添加 relation 键值
new_data_list = []
with open(file_path1, 'r', encoding='utf-8') as file1:
    lines1 = [next(file1) for _ in range(4401)]  
    for line1, relations in zip(lines1, relations_list):  # 一一对应处理
        try:
            data = json.loads(line1.strip())  # 解析 JSON 格式
            # 根据 relations 列表中的单词数量生成样本
            for relation in relations:
                new_data = data.copy()  # 创建一个副本
                new_data["relation"] = relation  # 添加 relation 键值
                new_data_list.append(new_data)
        except json.JSONDecodeError as e:
            print(f"解析 JSON 时出错：内容：{line1.strip()}")
            print(f"错误信息：{e}")

# 将新的数据保存到新的 txt 文件中
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for new_data in new_data_list:
        json.dump(new_data, output_file, ensure_ascii=False)  # 以 JSON 格式写入文件
        output_file.write("\n")  # 每个数据占一行

print(f"新数据已保存到文件：{output_file_path}")

# ---------------------------------------------------------------------------------------------------------
# 打开文件并读取内容
file_path = "/data/zxli/kp/a-qwen2.5-instruct/qwen2.5-14B/output_data_group4.txt"
relation_counts = {}  # 用于存储每个relation类别的数量
na_relations = []  # 用于存储所有NA类别的条目

with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 去掉行首行尾的空格和换行符
        line = line.strip()
        # 找到relation键值
        start = line.find('"relation": "') + len('"relation": "')
        end = line.find('"', start)
        relation_value = line[start:end]
        # 统计每个relation类别的数量
        if relation_value == "NA":
            na_relations.append(line)  # 将NA条目存储起来
            relation_counts["NA"] = relation_counts.get("NA", 0) + 1
        else:
            if relation_value in relation_counts:
                relation_counts[relation_value] += 1
            else:
                relation_counts[relation_value] = 1

# 输出各类别的数量
print("relation类别及其数量：")
for relation, count in relation_counts.items():
    print(f"{relation}: {count}")

# 计算非NA类别的平均数量（四舍五入）
non_na_total = sum(count for relation, count in relation_counts.items() if relation != "NA")
non_na_count = len([relation for relation in relation_counts if relation != "NA"])
if non_na_count > 0:
    avg_non_na_count = round(non_na_total / non_na_count)
else:
    avg_non_na_count = 0

print(f"\n非NA类别的平均数量（四舍五入）: {avg_non_na_count}")

# --------------------------------------------------------------------------------
# 定义文件路径
file_path = "/data/zxli/kp/a-qwen2.5-instruct/qwen2.5-14B/output_data_group4.txt"
sample_file_path = "/data/zxli/kp/a-qwen2.5-instruct/qwen2.5-14B/sample_data_all_group4.txt"  # 输出样本文件路径

relation_counts = {}  # 用于存储每个relation类别的数量
relation_data = {}  # 用于存储每个relation类别的数据

# 读取文件并统计每个relation类别的数量和数据
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line.strip())  # 解析 JSON 格式
            relation_value = data.get("relation")
            if relation_value:
                if relation_value in relation_counts:
                    relation_counts[relation_value] += 1
                    relation_data[relation_value].append(data)
                else:
                    relation_counts[relation_value] = 1
                    relation_data[relation_value] = [data]
        except json.JSONDecodeError as e:
            print(f"解析 JSON 时出错：内容：{line.strip()}")
            print(f"错误信息：{e}")

# 计算非NA类别的平均数量（四舍五入）
non_na_total = sum(count for relation, count in relation_counts.items() if relation != "NA")
non_na_count = len([relation for relation in relation_counts if relation != "NA"])
if non_na_count > 0:
    max_count = round(non_na_total / non_na_count)  # 作为最大数量
else:
    max_count = 0

# 对每种关系随机获取min(某个关系的全部，最大数量)个样本
sample_data_list = []
sample_relation_counts = {}  # 用于统计样本中每个关系类别的数量
for relation, data_list in relation_data.items():
    sample_count = min(len(data_list), max_count)  # 获取样本数量
    random_samples = random.sample(data_list, sample_count)
    sample_data_list.extend(random_samples)
    sample_relation_counts[relation] = sample_count  # 统计样本中每个关系类别的数量

# 将抽取的样本输出到sample_data.txt中
with open(sample_file_path, 'w', encoding='utf-8') as sample_file:
    for sample_data in sample_data_list:
        json.dump(sample_data, sample_file, ensure_ascii=False)  # 以 JSON 格式写入文件
        sample_file.write("\n")  # 每个数据占一行

print(f"抽取的样本已保存到文件：{sample_file_path}")

# 统计sample_data.txt里各个关系类别的数量
print("\n样本文件中各个关系类别的数量：")
for relation, count in sample_relation_counts.items():
    print(f"{relation}: {count}")