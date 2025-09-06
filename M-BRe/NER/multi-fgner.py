import os
os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
import json
import time
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import torch.nn as nn
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

def read_file():
    file_path = '/data/zxli/kp/appendix_ner/test_sent.txt'
    lines = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            lines.append(line.strip())
    return lines

def read_entity():
    file_path = '/data/zxli/kp/appendix_ner/test_label.txt'
    entitys = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            entitys.append(line.strip())
    return entitys

def read_label():
    file_path = '/data/zxli/kp/appendix_ner/test_label.txt'
    labels = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            labels.append(line.strip())
    return labels

def make_prompt(data):
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
Given a sentence, identify the fine-grained label of entity within it.

Here is a list of fine-grained label of entities:
(1) /WORK_OF_ART
(2) /EVENT
(3) /LOCATION
(4) /DISEASE
(5) /FACILITY/BUILDING
(6) /GPE
(7) /GPE/STATE_PROVINCE
(8) /FACILITY/BRIDGE
(9) /ORGANIZATION/HOTEL
(10) /FAC/AIRPORT
(11) /FAC/BRIDGE
(12) /WORK_OF_ART/PAINTING
(13) /CONTACT_INFO
(14) /ORGANIZATION/HOSPITAL
(15) /FACILITY/ATTRACTION
(16) /SUBSTANCE/FOOD
(17) /ORGANIZATION/GOVERNMENT
(18) /LOCATION/LAKE_SEA_OCEAN
(19) /WORK_OF_ART/SONG
(20) /WORK_OF_ART/PLAY
(21) /LOCATION/REGION
(22) /GAME
(23) /GPE/COUNTRY
(24) /CONTACT_INFO/url
(25) /LAW
(26) /PRODUCT/WEAPON
(27) /SUBSTANCE/CHEMICAL
(28) /LOCATION/RIVER
(29) /ANIMAL
(30) /ORGANIZATION
(31) /LANGUAGE
(32) /FAC/ATTRACTION
(33) /PRODUCT/VEHICLE
(34) /GPE/CITY
(35) /ORGANIZATION/POLITICAL
(36) /FACILITY/AIRPORT
(37) /CONTACT_INFO/PHONE
(38) /PLANT
(39) /LOCATION/CONTINENT
(40) /SUBSTANCE/DRUG
(41) /PERSON
(42) /CONTACT_INFO/ADDRESS
(43) /SUBSTANCE
(44) /ORGANIZATION/CORPORATION
(45) /WORK_OF_ART/BOOK
(46) /ORGANIZATION/RELIGIOUS
(47) /EVENT/WAR
(48) /FAC/BUILDING
(49) /FAC/HIGHWAY_STREET
(50) /FACILITY
(51) /ORGANIZATION/EDUCATIONAL
(52) /PRODUCT
(53) /FAC
(54) /ORGANIZATION/MUSEUM
(55) /FACILITY/HIGHWAY_STREET
(56) /EVENT/HURRICANE

Below is the target sentence. Please identify the fine-grained label of entity within the target sentence. Just output the fine-grained label of entity with double quotes.

Target Sentence: (TEXT)

Target Answer:
"""
    prompt = prompt.replace("(TEXT)", data)
    # print(prompt)
    return prompt
    
def generate_text(prompt):
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
            # outputs.sequences[0][:], skip_special_tokens=False
        )
    print(response)
    # with open('a-sample.txt', 'w', encoding='utf-8') as file:
    #     file.write("%s\n" % response)     
    # pattern = r'Target Answer:\s*"\s*(.*?)\s*"\s*<\|im_end\|>'
    # matches = re.findall(pattern, response, re.DOTALL)            
    matches = re.findall(r'"(.*?)"', response)
    
    if matches:
        return matches[0]
    else:
        return "NA"
    # if matches:
    #     print(matches[0])
    # else:
    #     print("NA")

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
    lines = read_file()
    entitys = read_entity()
    labels = read_label()
    pred_label = []
    precisions = []
    recalls = []

    start_time = time.time()
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-7B-Instruct-1M'
    model_path = '/data/zxli/te-gen/models/Qwen2.5-14B-Instruct-1M'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi2'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi2'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi2_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi2_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi1_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi1_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi1_mannully8_generate'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi1_mannully8_generate'
    # model_path = '/data/zxli/te-gen/models/vicuna-13b-v1.5'
    # model_path = '/data/zxli/te-gen/models/Qwen3-14B'

    print(model_path)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='auto')
    #model.load_state_dict(torch.load(fine_tuned_weights_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lines = read_file()
    for i in range(0, len(lines)):
    # for i in range(15, 16):
        print(f"{i+1}/{len(lines)}")
        prompt_text = make_prompt(lines[i])
        generated_text = generate_text(prompt_text)
        print(generated_text + '\n')
        pred_label.append(generated_text.strip())

        with open('/data/zxli/kp/appendix_ner/multi-pred/qwen14b.txt', 'w', encoding='utf-8') as file:
            for item in pred_label:
                if item in grained_type:
                    file.write("%s\n" % item)   
                else:
                    file.write("NA\n")                

    # for i in range(0, len(relation_pred)):
    #     if relation_pred[i] not in Relation_list:
    #         relation_pred[i] = "NA"
               
    print(pred_label)
    end_time = time.time()

    for i in range(0, len(pred_label)):
        intersection_size = len(set(pred_label[i]) & set(labels[i]))
        if intersection_size == 0:
            precisions.append(1e-10)
            recalls.append(1e-10)
        else:
            precisions.append(intersection_size/len(pred_label[i]))
            recalls.append(intersection_size/len(labels[i]))
    
    # acc, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = accuracy2(relation_pred)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    sa_f1 = special_avg_f1(precisions, recalls)

    elapsed_time = end_time - start_time
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