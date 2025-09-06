import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re 
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

grained_type = [
    '/WORK_OF_ART',
    '/EVENT',
    '/LOCATION',
    '/DISEASE',
    '/FACILITY/BUILDING',
    '/GPE',
    '/GPE/STATE_PROVINCE',
    '/FACILITY/BRIDGE',
    '/ORGANIZATION/HOTEL',
    '/FAC/AIRPORT',
    '/FAC/BRIDGE',
    '/WORK_OF_ART/PAINTING',
    '/CONTACT_INFO',
    '/ORGANIZATION/HOSPITAL',
    '/FACILITY/ATTRACTION',
    '/SUBSTANCE/FOOD',
    '/ORGANIZATION/GOVERNMENT',
    '/LOCATION/LAKE_SEA_OCEAN',
    '/WORK_OF_ART/SONG',
    '/WORK_OF_ART/PLAY',
    '/LOCATION/REGION',
    '/GAME',
    '/GPE/COUNTRY',
    '/CONTACT_INFO/url',
    '/LAW',
    '/PRODUCT/WEAPON',
    '/SUBSTANCE/CHEMICAL',
    '/LOCATION/RIVER',
    '/ANIMAL',
    '/ORGANIZATION',
    '/LANGUAGE',
    '/FAC/ATTRACTION',
    '/PRODUCT/VEHICLE',
    '/GPE/CITY',
    '/ORGANIZATION/POLITICAL',
    '/FACILITY/AIRPORT',
    '/CONTACT_INFO/PHONE',
    '/PLANT',
    '/LOCATION/CONTINENT',
    '/SUBSTANCE/DRUG',
    '/PERSON',
    '/CONTACT_INFO/ADDRESS',
    '/SUBSTANCE',
    '/ORGANIZATION/CORPORATION',
    '/WORK_OF_ART/BOOK',
    '/ORGANIZATION/RELIGIOUS',
    '/EVENT/WAR',
    '/FAC/BUILDING',
    '/FAC/HIGHWAY_STREET',
    '/FACILITY',
    '/ORGANIZATION/EDUCATIONAL',
    '/PRODUCT',
    '/FAC',
    '/ORGANIZATION/MUSEUM',
    '/FACILITY/HIGHWAY_STREET',
    '/EVENT/HURRICANE'
    ]

# 提取描述文本并向量化
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(grained_type)

# # 构建相似度矩阵
sim_matrix = cosine_similarity(X)
diff_matrix = 1 - sim_matrix  # 将相似度转换为差异度

# 初始化参数
n_groups = 9
group_size = len(grained_type) // n_groups
remaining = len(grained_type) % n_groups

# colors = ["#ffcc80", "#ff9800", "#ffa726"] # 从浅橘色到深橘色
# cmap = mcolors.LinearSegmentedColormap.from_list("custom_orange", colors)

# # 绘制相似度矩阵热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(sim_matrix, annot=False, cmap=cmap, xticklabels=False, yticklabels=False)
# plt.title('Similarity Matrix Heatmap')
# # 保存热力图为图片文件
# heatmap_filename = "/data/zxli/kp/a-qwen2.5-instruct/similarity_matrix_heatmap.png"
# plt.savefig(heatmap_filename)
# print(f"Heatmap saved as {heatmap_filename}")

# # 绘制差异度矩阵热力图
# plt.figure(figsize=(12, 10))
# sns.heatmap(diff_matrix, annot=False, cmap=cmap, xticklabels=False, yticklabels=False)
# plt.title('Difference Matrix Heatmap')
# # 保存热力图为图片文件
# heatmap_filename = "/data/zxli/kp/a-qwen2.5-instruct/difference_matrix_heatmap.png"
# plt.savefig(heatmap_filename)
# print(f"Heatmap saved as {heatmap_filename}")

# 创建分组容器
groups = [[] for _ in range(n_groups)]
assigned = set()

# 第一阶段：选取种子元素
# 找到差异最大的两个元素作为初始种子
max_diff = np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)
seed_indices = list(max_diff)
groups[0].append(seed_indices[0])
groups[1].append(seed_indices[1])
assigned.update(seed_indices)

# 第二阶段：贪心分配剩余元素
for _ in range(len(grained_type)-2):
    best_group = -1
    best_element = -1
    min_similarity = float('inf')
    
    # 遍历未分配元素
    for elem in range(len(grained_type)):
        if elem in assigned:
            continue
            
        # 计算该元素对各组的适配度
        for gid, group in enumerate(groups):
            if len(group) >= group_size + (1 if gid < remaining else 0):
                continue
                
            # 计算与组内现有元素的最小相似度
            if not group:
                current_sim = 0
            else:
                current_sim = np.max([sim_matrix[elem, m] for m in group])
                
            if current_sim < min_similarity:
                min_similarity = current_sim
                best_element = elem
                best_group = gid

    # 执行分配
    groups[best_group].append(best_element)
    assigned.add(best_element)

# 将索引转换为原始关系
result_groups = []
for group in groups:
    sorted_group = sorted(group)
    result_groups.append([grained_type[i] for i in sorted_group])

# 打印分组结果
for i, group in enumerate(result_groups):
    print(f"============ Group {i+1} ============")
    for rel in group:
        print(f"• {rel}")
    print("\n")

for i, group in enumerate(result_groups):
    prompt_begin = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
Given a sentence, identify the fine-grained label of entity within it.

Here is a list of fine-grained label of entities:"""

    prompt_end = """
Below is the target sentence. Please identify the fine-grained label of entity within the target sentence. Just output the fine-grained label of entity with double quotes.

Target Sentence: (TEXT)

Target Answer:
"""
    print(group)
    prompt_mid="\n"
    for j in range(0,len(group)):
        prompt_mid = prompt_mid + f"({j+1}) {group[j]}" + "\n"
    # print(prompt_mid)

    prompt_begin = prompt_begin + prompt_mid
    prompt_final = prompt_begin + prompt_end

    output_dir = f"/data/zxli/kp/appendix_ner/mult-prompt-grainedner-{n_groups}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"/data/zxli/kp/appendix_ner/mult-prompt-grainedner-{n_groups}/{i+1}.txt"

    with open(file_name, "w", encoding="utf-8") as file:
        file.write(prompt_final)