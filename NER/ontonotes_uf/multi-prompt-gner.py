import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re 
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

grained_type = [
    '/location',
    '/organization',
    '/other',
    '/person',
    '/location/celestial',
    '/location/city',
    '/location/country',
    '/location/geography',
    '/location/geograpy',
    '/location/park',
    '/location/structure',
    '/location/transit',
    '/organization/company',
    '/organization/education',
    '/organization/government',
    '/organization/military',
    '/organization/music',
    '/organization/political_party',
    '/organization/sports_league',
    '/organization/sports_team',
    '/organization/stock_exchange',
    '/organization/transit',
    '/other/art',
    '/other/award',
    '/other/body_part',
    '/other/currency',
    '/other/event',
    '/other/food',
    '/other/health',
    '/other/heritage',
    '/other/internet',
    '/other/language',
    '/other/legal',
    '/other/living_thing',
    '/other/product',
    '/other/religion',
    '/other/scientific',
    '/other/sports_and_leisure',
    '/other/supernatural',
    '/person/artist',
    '/person/athlete',
    '/person/coach',
    '/person/doctor',
    '/person/legal',
    '/person/military',
    '/person/political_figure',
    '/person/religious_leader',
    '/person/title',
    '/location/geography/body_of_water',
    '/location/geography/island',
    '/location/geography/mountain',
    '/location/geograpy/island',
    '/location/structure/airport',
    '/location/structure/government',
    '/location/structure/hospital',
    '/location/structure/hotel',
    '/location/structure/restaurant',
    '/location/structure/sports_facility',
    '/location/structure/theater',
    '/location/transit/bridge',
    '/location/transit/railway',
    '/location/transit/road',
    '/organization/company/broadcast',
    '/organization/company/news',
    '/other/art/broadcast',
    '/other/art/film',
    '/other/art/music',
    '/other/art/stage',
    '/other/art/writing',
    '/other/event/accident',
    '/other/event/election',
    '/other/event/holiday',
    '/other/event/natural_disaster',
    '/other/event/protest',
    '/other/event/sports_event',
    '/other/event/violent_conflict',
    '/other/health/malady',
    '/other/health/treatment',
    '/other/language/programming_language',
    '/other/living_thing/animal',
    '/other/product/car',
    '/other/product/computer',
    '/other/product/mobile_phone',
    '/other/product/software',
    '/other/product/weapon',
    '/person/artist/actor',
    '/person/artist/author',
    '/person/artist/director',
    '/person/artist/music'
    ]

# 提取描述文本并向量化
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(grained_type)

# # 构建相似度矩阵
sim_matrix = cosine_similarity(X)
diff_matrix = 1 - sim_matrix  # 将相似度转换为差异度

# 初始化参数
n_groups = 14
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

    output_dir = f"/data/zxli/M-BRe/NER/ontonotes_uf/mult-prompt-grainedner-{n_groups}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"/data/zxli/M-BRe/NER/ontonotes_uf/mult-prompt-grainedner-{n_groups}/{i+1}.txt"

    with open(file_name, "w", encoding="utf-8") as file:
        file.write(prompt_final)