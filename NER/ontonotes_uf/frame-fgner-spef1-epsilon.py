import os
os.environ["CUDA_VISIBLE_DEVICES"]='4,5'
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import json
import numpy as np
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

epsilon = 0.01 #0.001 0.01 0.02 0.05 0.1
print(epsilon)
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
Type_dic = {grained_type[i-1]: i for i in range(1,len(grained_type)+1)}

multi_prompt_list=[]
for i in range(0,9):
    file_path = '/data/zxli/M-BRe/NER/ontonotes_uf/mult-prompt-grainedner-14/{}.txt'.format(i+1)
    print(file_path)
    with open(file_path, 'r', encoding='utf-8') as file:
        multi_prompt_list.append(file.read())

binary_prompt_list=[]
for i in range(0,len(grained_type)):
    file_path = '/data/zxli/M-BRe/NER/ontonotes_uf/binary-prompt/{}.txt'.format(i+1)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        binary_prompt_list.append(file.read())

def read_file():
    file_path = '/data/zxli/M-BRe/NER/ontonotes_uf/test_data/test_sent_filtered.txt'
    lines = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            lines.append(line.strip())
    return lines

def read_entity():
    file_path = '/data/zxli/M-BRe/NER/ontonotes_uf/test_data/test_entity_filtered.txt'
    entitys = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            entitys.append(line.strip())
    return entitys

def read_label():
    file_path = '/data/zxli/M-BRe/NER/ontonotes_uf/test_data/test_label_filtered.txt'
    labels = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            labels.append(line.strip())
    return labels

def make_prompt_multi(prompt, data):
    prompt = prompt.replace("(TEXT)", data)
    return prompt

def make_prompt_binary(prompt, data, provided_entity):
    prompt = prompt.replace("(TEXT)", data)
    prompt = prompt.replace("(PE)", provided_entity)
    return prompt

def multi_generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
            return_legacy_cache=True,
            temperature=0.6,
            # temperature=1e-2,
            do_sample=True,
            # top_k=50,
            # top_p=0.75,
        )

        response = tokenizer.decode(
            outputs.sequences[0][input_len:], skip_special_tokens=False
        )
    matches = re.findall(r'"(.*?)"', response)
    
    if matches:
        return matches[0]
    else:
        return "NA"
    
def binary_generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=100,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=inputs.attention_mask,
            return_legacy_cache=True,
            temperature=0.6,
            # temperature=1e-2,
            do_sample=True,
            # top_k=50,
            # top_p=0.75,
        )

        # 获取生成的 token 序列
        generated_tokens = outputs.sequences[0][input_len:].tolist()
        # target_word = "</think>" 
        # target_tokens = tokenizer(target_word, add_special_tokens=False)["input_ids"]
        # target_token_id = target_tokens[0]
        # word_index = -1
        # for i in range(len(generated_tokens)):
        #     if generated_tokens[i] == target_token_id:
        #         word_index = i 
        #         break
        
        response = tokenizer.decode(
                    generated_tokens[0:], skip_special_tokens=True
                )

        generation_scores = outputs.scores[0:]
        logits = torch.stack(generation_scores, dim=0)
        logits = logits.squeeze(dim=1).to('cpu')
        logits = torch.nn.functional.softmax(logits, dim=1)

        max_probabilities, _ = torch.max(logits, dim=1)
        confidence = torch.mean(max_probabilities).item()
        
    torch.cuda.empty_cache()
    del inputs, outputs, generation_scores, logits, max_probabilities

    return response, confidence

def last_result(last_judge, confidence, labels):
    pred_label = []
    label_candidate = [index for index, value in enumerate(last_judge) if value != 'No']
    beyond_conf_indices = [i for i in label_candidate if confidence[i] >= 1-epsilon]
    for i in range(0,len(beyond_conf_indices)):
        pred_label.append(last_judge[beyond_conf_indices[i]])

    # print(type(pred_label))
    # print(pred_label)
    # print(type(labels))
    # print(labels)

    intersection_size = len(set(pred_label) & set(labels))
    if intersection_size == 0:
        return pred_label, 1e-10, 1e-10
    else:
        precision = intersection_size/len(pred_label)
        recall = intersection_size/len(labels)
        return pred_label, precision, recall

# def accuracy2(relation_pred):
#     file_path = '/data/zxli/kp/a-deepseek-distill/tacred_test_relation.txt'
#     relation_label = []
    
#     with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
#         for line in file1:
#             relation_label.append(line.strip())
            
#     acc = accuracy_score(relation_label, relation_pred)
#     precision_micro = precision_score(relation_label, relation_pred, average='micro')
#     recall_micro = recall_score(relation_label, relation_pred, average='micro')
#     f1_micro = f1_score(relation_label, relation_pred, average='micro')
#     precision_macro = precision_score(relation_label, relation_pred, average='macro')
#     recall_macro = recall_score(relation_label, relation_pred, average='macro')
#     f1_macro = f1_score(relation_label, relation_pred, average='macro')

#     return acc, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro

def special_avg_f1(precisions, recalls):
    sum = 0
    
    for i in range(0,len(precisions)):
        sum = sum + (2*precisions[i]*recalls[i])/(precisions[i]+recalls[i])

    return sum/len(precisions)

if __name__ == '__main__':
    last_judge = []
    pred_labels = []
    confidences = []
    precisions = []
    recalls = []
    Con=[]
    lines = read_file()
    entitys = read_entity()
    labels = read_label()

    start_time = time.time()
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-7B-Instruct-1M'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_MB400_random'
    model_path = '/data/zxli/te-gen/models/Qwen2.5-14B-Instruct-1M'
    print(model_path)

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='auto')  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(0, len(lines)):
        for j in range(0, len(multi_prompt_list)):
            print(f"{i+1}/{len(lines)} | mult-prompt{j+1}")
            ## mult
            prompt_text1 = make_prompt_multi(multi_prompt_list[j], lines[i])
            generated_text1 = multi_generate_text(prompt_text1)
            print(generated_text1 + '\n')

            ## binary
            if generated_text1 not in grained_type:
                continue
            else:
                prompt_text2 = make_prompt_binary(binary_prompt_list[Type_dic[generated_text1]-1], lines[i], entitys[i])
                # print(prompt_text2)
                generated_text2, confidence = binary_generate_text(prompt_text2)
                confidences.append(confidence)
                if generated_text2 == 'Yes.':
                    print('Yes\n')
                    last_judge.append(generated_text1)
                else:
                    print('No\n')
                    last_judge.append('No')
            
        print(last_judge)
        print(confidences)
        Con.append(confidences)
        
        pred_label, precision, recall = last_result(last_judge, confidences, json.loads(labels[i]))

        pred_labels.append(pred_label)
        precisions.append(precision)
        recalls.append(recall)
        print(precisions, recalls)
        last_judge = []
        confidences = []
    
    with open('/data/zxli/M-BRe/NER/ontonotes_uf/frame_pred/pred_labels.txt', 'w', encoding='utf-8') as file:
        for item in pred_labels:
            file.write("%s\n" % item) 

    with open('/data/zxli/M-BRe/NER/ontonotes_uf/frame_pred/confidences-14B.txt', 'w', encoding='utf-8') as file:
        for item in Con:
            file.write("%s\n" % item) 

    # acc, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = accuracy2(relation_preds)
    end_time = time.time()

    elapsed_time = end_time - start_time

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    sa_f1 = special_avg_f1(precisions, recalls)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    # print(f"Accuracy: {acc:.8f}")
    # print(f"Precision_Micro: {precision_micro:.8f}")
    # print(f"Recall_Micro: {recall_micro:.8f}")
    # print(f"F1_Micro: {f1_micro:.8f}")
    # print(f"Precision_Macro: {precision_macro:.8f}")
    # print(f"Recall_Macro: {recall_macro:.8f}")
    # print(f"F1_Macro: {f1_macro:.8f}")
    print(f"Average_Precision: {mean_precision:.8f}")
    print(f"Average_Recall: {mean_recall:.8f}")
    print(f"Specical_Avg_F1: {sa_f1:.8f}")