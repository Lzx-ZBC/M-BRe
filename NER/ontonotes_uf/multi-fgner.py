import os
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
import time
import re
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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

def make_prompt(data):
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
Given a sentence, identify the fine-grained label of entity within it.

Here is a list of fine-grained label of entities:
(1) /location
(2) /organization
(3) /other
(4) /person
(5) /location/celestial
(6) /location/city
(7) /location/country
(8) /location/geography
(9) /location/geograpy
(10) /location/park
(11) /location/structure
(12) /location/transit
(13) /organization/company
(14) /organization/education
(15) /organization/government
(16) /organization/military
(17) /organization/music
(18) /organization/political_party
(19) /organization/sports_league
(20) /organization/sports_team
(21) /organization/stock_exchange
(22) /organization/transit
(23) /other/art
(24) /other/award
(25) /other/body_part
(26) /other/currency
(27) /other/event
(28) /other/food
(29) /other/health
(30) /other/heritage
(31) /other/internet
(32) /other/language
(33) /other/legal
(34) /other/living_thing
(35) /other/product
(36) /other/religion
(37) /other/scientific
(38) /other/sports_and_leisure
(39) /other/supernatural
(40) /person/artist
(41) /person/athlete
(42) /person/coach
(43) /person/doctor
(44) /person/legal
(45) /person/military
(46) /person/political_figure
(47) /person/religious_leader
(48) /person/title
(49) /location/geography/body_of_water
(50) /location/geography/island
(51) /location/geography/mountain
(52) /location/geograpy/island
(53) /location/structure/airport
(54) /location/structure/government
(55) /location/structure/hospital
(56) /location/structure/hotel
(57) /location/structure/restaurant
(58) /location/structure/sports_facility
(59) /location/structure/theater
(60) /location/transit/bridge
(61) /location/transit/railway
(62) /location/transit/road
(63) /organization/company/broadcast
(64) /organization/company/news
(65) /other/art/broadcast
(66) /other/art/film
(67) /other/art/music
(68) /other/art/stage
(69) /other/art/writing
(70) /other/event/accident
(71) /other/event/election
(72) /other/event/holiday
(73) /other/event/natural_disaster
(74) /other/event/protest
(75) /other/event/sports_event
(76) /other/event/violent_conflict
(77) /other/health/malady
(78) /other/health/treatment
(79) /other/language/programming_language
(80) /other/living_thing/animal
(81) /other/product/car
(82) /other/product/computer
(83) /other/product/mobile_phone
(84) /other/product/software
(85) /other/product/weapon
(86) /person/artist/actor
(87) /person/artist/author
(88) /person/artist/director
(89) /person/artist/music

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

        with open('/data/zxli/M-BRe/NER/ontonotes_uf/multi_pred/qwen14b.txt', 'w', encoding='utf-8') as file:
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
        if pred_label[i] not in labels[i]:
            precisions.append(1e-10)
            recalls.append(1e-10)
        else:
            precisions.append(1)
            recalls.append(1/len(labels[i]))
    
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