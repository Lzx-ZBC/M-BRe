import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re 
import os
import matplotlib.pyplot as plt
# import seaborn as sns

relations = [
    '"Component-Whole(e2,e1)": Tail entity e2 is the component of head entity e1, and head entity e1 is the whole of tail entity e2.',
    '"Instrument-Agency(e2,e1)": Tail entity e2 is the instrument of head entity e1, and head entity e1 is the agency of tail entity e2.',
    '"Member-Collection(e1,e2)": Head entity e1 is the member of tail entity e2, and tail entity e2 is the collection of head entity e1.',
    '"Cause-Effect(e2,e1)": Tail entity e2 is the cause of head entity e1, and head entity e1 is the effect of tail entity e2.',
    '"Entity-Destination(e1,e2)": Head entity e1 is the entity of tail entity e2, and tail entity e2 is the destination of head entity e1.',
    '"Content-Container(e1,e2)": Head entity e1 is the content of tail entity e2, and tail entity e2 is the container of head entity e1.',
    '"Message-Topic(e1,e2)": Head entity e1 is the message of tail entity e2, and tail entity e2 is the topic of head entity e1.',
    '"Product-Producer(e2,e1)": Tail entity e2 is the product of head entity e1, and head entity e1 is the producer of tail entity e2.',
    '"Member-Collection(e2,e1)": Tail entity e2 is the member of head entity e1, and head entity e1 is the collection of tail entity e2.',
    '"Entity-Origin(e1,e2)": Head entity e1 is the entity of tail entity e2, and tail entity e2 is the origin of head entity e1.',
    '"Cause-Effect(e1,e2)": Head entity e1 is the cause of tail entity e2, and tail entity e2 is the effect of head entity e1.',
    '"Component-Whole(e1,e2)": Head entity e1 is the component of tail entity e2, and tail entity e2 is the whole of head entity e1.',
    '"Message-Topic(e2,e1)": Tail entity e2 is the message of head entity e1, and head entity e1 is the topic of tail entity e2.',
    '"Product-Producer(e1,e2)": Head entity e1 is the product of tail entity e2, and tail entity e2 is the producer of head entity e1.',
    '"Entity-Origin(e2,e1)": Tail entity e2 is the entity of head entity e1, and head entity e1 is the origin of tail entity e2.',
    '"Content-Container(e2,e1)": Tail entity e2 is the content of head entity e1, and head entity e1 is the container of tail entity e2.',
    '"Instrument-Agency(e1,e2)": Head entity e1 is the instrument of tail entity e2, and tail entity e2 is the agency of head entity e1.',
    '"Entity-Destination(e2,e1)": Tail entity e2 is the entity of head entity e1, and head entity e1 is the destination of tail entity e2.',
    '"Other": There is no relationship or unrecognized relationship between the head and tail entities.'
    ]
examples = [
    "the original play was filled with very topical humor , so the director felt free to add current topical humor to the script.",
    "employees increasingly are turning to medication - and away from therapy - to treat depression.",
    "the head of the team has gathered a very dynamic and productive team.",
    "there is relatively little discomfort from this surgery and most individuals rarely take more than tylenol for their discomfort.",
    "he easily put the syringe in a beaker of water.",
    "the clay model was in a jar wrapped in a daily mirror from 1947.",
    "this is supplemented by columns and articles reflecting on fantasy literature 's past as well as the occasional interview.",
    "the baha ' u'llah religion has liberated human minds by a prohibition within his faith against any caste with ecclesiastical prerogatives.",
    "the most famous unkindness of six ravens at the tower of london are employees , kept on staff at the expense of the british government.",
    "my animation assignment from last term made everyone smile.",
    "signs placed at the flagpole island at the village square caused a stir on the linglestown forum on pennlive.com.",
    "now , your earlobe can never escape because this brass knuckle stud is attached to a chain attached to a cuff that affixes to the helix of the ear.",
    "the battle has been analysed in various publications.",
    "the three prints are a great example of a rare vintage photograph by an artist who had an influence on later 20th-century photographers.",
    "the stone ginger beer has a ginger 'bite ' without the 'ginger burn ' of more peppery jamaican ginger beers.",
    "a pack of the most popular cigarettes in the seychelles in 2008 cost $ 15 at purchasing-power parity ( ppp ).",
    "planetcad offers the best tools for engineers and now the whole industry is following.",
    "an unnamed lifeguard was pushed into the pool.",
    "manufacturers have traditionally been more concerned about factors like price , quality , or cycle time , and not as concerned over how much energy their manufacturing processes use."
]
# 提取描述文本并向量化
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(relations)

# # 构建相似度矩阵
sim_matrix = cosine_similarity(X)
diff_matrix = 1 - sim_matrix  # 将相似度转换为差异度

# 初始化参数
n_groups = 9
group_size = len(relations) // n_groups
remaining = len(relations) % n_groups

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
for _ in range(len(relations)-2):
    best_group = -1
    best_element = -1
    min_similarity = float('inf')
    
    # 遍历未分配元素
    for elem in range(len(relations)):
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
    result_groups.append([relations[i] for i in sorted_group])

# 打印分组结果
for i, group in enumerate(result_groups):
    print(f"============ Group {i+1} ============")
    for rel in group:
        print(f"• {rel}")
    print("\n")

for i, group in enumerate(result_groups):
    prompt_begin = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
Given a sentence, identify the relation category within it.

Here are definitions of relation categories and some examples:"""

    prompt_end = """
Below is the target sentence and the provided head and tail entities. Please identify the relation category within the target sentence. Just output the relation category with double quotes.

Target Sentence: (TEXT)
provided head entity: (HE)
provided tail entity: (TE)

Target Answer:
"""
    for j, rel in enumerate(group):
        double_quote_content = re.search(r'"(.*?)"', rel)
        if double_quote_content:
            double_quote_content = double_quote_content.group(1)

        colon_content = re.search(r'":\s*(.*)', rel)
        if colon_content:
            colon_content = colon_content.group(1)

        index = relations.index(rel)

        prompt_mid = f"""
({j+1}) "{double_quote_content}"
definition: {colon_content}.
example1: {examples[index]}
"""
        prompt_begin = prompt_begin + prompt_mid
    prompt_final = prompt_begin + prompt_end

    output_dir = f"/data/zxli/kp/mult-prompt-semeval-cos/mult-prompt-semeval{n_groups}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"/data/zxli/kp/mult-prompt-semeval-cos/mult-prompt-semeval{n_groups}/{i+1}.txt"

    with open(file_name, "w", encoding="utf-8") as file:
        file.write(prompt_final)

# ## random 
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import re 
# import os

# relations = [
#     '"Component-Whole(e2,e1)": Tail entity e2 is the component of head entity e1, and head entity e1 is the whole of tail entity e2.',
#     '"Instrument-Agency(e2,e1)": Tail entity e2 is the instrument of head entity e1, and head entity e1 is the agency of tail entity e2.',
#     '"Member-Collection(e1,e2)": Head entity e1 is the member of tail entity e2, and tail entity e2 is the collection of head entity e1.',
#     '"Cause-Effect(e2,e1)": Tail entity e2 is the cause of head entity e1, and head entity e1 is the effect of tail entity e2.',
#     '"Entity-Destination(e1,e2)": Head entity e1 is the entity of tail entity e2, and tail entity e2 is the destination of head entity e1.',
#     '"Content-Container(e1,e2)": Head entity e1 is the content of tail entity e2, and tail entity e2 is the container of head entity e1.',
#     '"Message-Topic(e1,e2)": Head entity e1 is the message of tail entity e2, and tail entity e2 is the topic of head entity e1.',
#     '"Product-Producer(e2,e1)": Tail entity e2 is the product of head entity e1, and head entity e1 is the producer of tail entity e2.',
#     '"Member-Collection(e2,e1)": Tail entity e2 is the member of head entity e1, and head entity e1 is the collection of tail entity e2.',
#     '"Entity-Origin(e1,e2)": Head entity e1 is the entity of tail entity e2, and tail entity e2 is the origin of head entity e1.',
#     '"Cause-Effect(e1,e2)": Head entity e1 is the cause of tail entity e2, and tail entity e2 is the effect of head entity e1.',
#     '"Component-Whole(e1,e2)": Head entity e1 is the component of tail entity e2, and tail entity e2 is the whole of head entity e1.',
#     '"Message-Topic(e2,e1)": Tail entity e2 is the message of head entity e1, and head entity e1 is the topic of tail entity e2.',
#     '"Product-Producer(e1,e2)": Head entity e1 is the product of tail entity e2, and tail entity e2 is the producer of head entity e1.',
#     '"Entity-Origin(e2,e1)": Tail entity e2 is the entity of head entity e1, and head entity e1 is the origin of tail entity e2.',
#     '"Content-Container(e2,e1)": Tail entity e2 is the content of head entity e1, and head entity e1 is the container of tail entity e2.',
#     '"Instrument-Agency(e1,e2)": Head entity e1 is the instrument of tail entity e2, and tail entity e2 is the agency of head entity e1.',
#     '"Entity-Destination(e2,e1)": Tail entity e2 is the entity of head entity e1, and head entity e1 is the destination of tail entity e2.',
#     '"Other": There is no relationship or unrecognized relationship between the head and tail entities.'
#     ]
# examples = [
#     "the original play was filled with very topical humor , so the director felt free to add current topical humor to the script.",
#     "employees increasingly are turning to medication - and away from therapy - to treat depression.",
#     "the head of the team has gathered a very dynamic and productive team.",
#     "there is relatively little discomfort from this surgery and most individuals rarely take more than tylenol for their discomfort.",
#     "he easily put the syringe in a beaker of water.",
#     "the clay model was in a jar wrapped in a daily mirror from 1947.",
#     "this is supplemented by columns and articles reflecting on fantasy literature 's past as well as the occasional interview.",
#     "the baha ' u'llah religion has liberated human minds by a prohibition within his faith against any caste with ecclesiastical prerogatives.",
#     "the most famous unkindness of six ravens at the tower of london are employees , kept on staff at the expense of the british government.",
#     "my animation assignment from last term made everyone smile.",
#     "signs placed at the flagpole island at the village square caused a stir on the linglestown forum on pennlive.com.",
#     "now , your earlobe can never escape because this brass knuckle stud is attached to a chain attached to a cuff that affixes to the helix of the ear.",
#     "the battle has been analysed in various publications.",
#     "the three prints are a great example of a rare vintage photograph by an artist who had an influence on later 20th-century photographers.",
#     "the stone ginger beer has a ginger 'bite ' without the 'ginger burn ' of more peppery jamaican ginger beers.",
#     "a pack of the most popular cigarettes in the seychelles in 2008 cost $ 15 at purchasing-power parity ( ppp ).",
#     "planetcad offers the best tools for engineers and now the whole industry is following.",
#     "an unnamed lifeguard was pushed into the pool.",
#     "manufacturers have traditionally been more concerned about factors like price , quality , or cycle time , and not as concerned over how much energy their manufacturing processes use."
# ]

# # 初始化参数
# n_groups = 4
# total_relations = len(relations)
# group_size = total_relations // n_groups
# remaining = total_relations % n_groups

# # 生成随机排列的索引
# indices = np.arange(total_relations)
# np.random.shuffle(indices)

# # 创建分组容器
# groups = []
# start = 0
# for i in range(n_groups):
#     # 计算当前组大小
#     current_group_size = group_size + 1 if i < remaining else group_size
#     end = start + current_group_size
#     # 提取当前组的索引并排序（保持原始顺序）
#     group_indices = sorted(indices[start:end].tolist())
#     groups.append(group_indices)
#     start = end

# # 将索引转换为原始关系
# result_groups = []
# for group in groups:
#     result_groups.append([relations[i] for i in group])

# # 打印分组结果
# for i, group in enumerate(result_groups):
#     print(f"============ Group {i+1} ============")
#     for rel in group:
#         print(f"• {rel}")
#     print("\n")

# # 生成提示模板文件（保持原有代码逻辑不变）
# for i, group in enumerate(result_groups):
#     prompt_begin = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

# Instruction:
# Given a sentence, identify the relation category within it.

# Here are definitions of relation categories and some examples:"""

#     prompt_end = """
# Below is the target sentence and the provided head and tail entities. Please identify the relation category within the target sentence. Just output the relation category with double quotes.

# Target Sentence: (TEXT)
# provided head entity: (HE)
# provided tail entity: (TE)

# Target Answer:
# """
#     for j, rel in enumerate(group):
#         double_quote_content = re.search(r'"(.*?)"', rel)
#         if double_quote_content:
#             double_quote_content = double_quote_content.group(1)

#         colon_content = re.search(r'":\s*(.*)', rel)
#         if colon_content:
#             colon_content = colon_content.group(1)

#         index = relations.index(rel)

#         prompt_mid = f"""
# ({j+1}) "{double_quote_content}"
# definition: {colon_content}.
# example1: {examples[index]}
# """
#         prompt_begin = prompt_begin + prompt_mid
#     prompt_final = prompt_begin + prompt_end
    
#     output_dir = f"/data/zxli/kp/mult-prompt-semeval-cos/mult-prompt-semeval-random{n_groups}"
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     file_name = f"{output_dir}/{i+1}.txt"

#     with open(file_name, "w", encoding="utf-8") as file:
#         file.write(prompt_final)