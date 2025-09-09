import os
os.environ["CUDA_VISIBLE_DEVICES"]='6,7'
import time
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import json
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

epsilon = 0.01 #0.001 0.01 0.02 0.05 0.1
print(epsilon)

prompt_list=[]
for i in range(0,len(grained_type)):
    file_path = '/data/zxli/M-BRe/NER/ontonotes_uf/binary-prompt/{}.txt'.format(i+1)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_list.append(file.read())

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

def make_prompt(prompt, data, provided_entity):
    prompt = prompt.replace("(TEXT)", data)
    prompt = prompt.replace("(PE)", provided_entity)
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
            # temperature=1e-2,
            temperature=0.6,
            do_sample=True,
            # top_k=50,
            # top_p=0.75,
        )

        # 获取生成的 token 序列
        generated_tokens = outputs.sequences[0][input_len:].tolist()
        # generated_tokens = outputs.sequences[0].tolist()

        # target_word = "Yes." 
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

    print(type(pred_label))
    print(pred_label)
    print(type(labels))
    print(labels)

    intersection_size = len(set(pred_label) & set(labels))
    # print(intersection_size)
    if intersection_size == 0:
        return pred_label, 1e-10, 1e-10
    else:
        precision = intersection_size/len(pred_label)
        recall = intersection_size/len(labels)
        return pred_label, precision, recall

def special_avg_f1(precisions, recalls):
    sum = 0
    
    for i in range(0,len(precisions)):
        sum = sum + (2*precisions[i]*recalls[i])/(precisions[i]+recalls[i])

    return sum/len(precisions)
  
if __name__ == '__main__':
    pred_labels = []
    precisions = []
    recalls = []
    confidences = []
    last_judge = [] 
    lines = read_file()
    entitys = read_entity()
    labels = read_label()
    start_time = time.time()

    # model_path = '/data/zxli/te-gen/models/Qwen2.5-7B-Instruct-1M'
    model_path = '/data/zxli/te-gen/models/Qwen2.5-14B-Instruct-1M'
    # model_path = '/data/zxli/te-gen/models/vicuna-13b-v1.5'
    print(model_path)
    
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='auto')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0, len(lines)):
        for j in range(0, len(prompt_list)):
        # for j in range(0, 1):
            print(f"{i+1}/{len(lines)} | relation-prompt{j+1}")
            prompt_text = make_prompt(prompt_list[j], lines[i], entitys[i])

            generated_text, confidence = generate_text(prompt_text)
            print(confidence)
            confidences.append(confidence)
            
            if generated_text == 'Yes.':
                print('Yes\n')
                last_judge.append(grained_type[j])
            else:
                print('No\n')
                last_judge.append('No')

        print(last_judge)
        print(confidences)
        pred_label, precision, recall = last_result(last_judge, confidences, json.loads(labels[i]))
        pred_labels.append(pred_label)
        precisions.append(precision)
        recalls.append(recall)
        print(precisions)
        print(recalls)
        last_judge = []
        confidences = []

    with open('/data/zxli/M-BRe/NER/ontonotes_uf/binary_pred/qwen14B.txt', 'w', encoding='utf-8') as file:
        for item in pred_labels:
            file.write("%s\n" % item) 
    print(pred_labels)

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