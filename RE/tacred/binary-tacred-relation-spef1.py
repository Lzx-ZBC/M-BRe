import os
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
import time
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import random
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)   
   
Relation_list = ["org:founded", "org:subsidiaries", "per:date_of_birth", "per:cause_of_death", "per:age", "per:stateorprovince_of_birth", "per:countries_of_residence", "per:country_of_birth", "per:stateorprovinces_of_residence", "org:website", "per:cities_of_residence", "per:parents", "per:employee_of", "NA", "per:city_of_birth", "org:parents", "org:political/religious_affiliation", "per:schools_attended", "per:country_of_death", "per:children", "org:top_members/employees", "per:date_of_death", "org:members", "org:alternate_names", "per:religion", "org:member_of", "org:city_of_headquarters", "per:origin", "org:shareholders", "per:charges", "per:title", "org:number_of_employees/members", "org:dissolved", "org:country_of_headquarters", "per:alternate_names", "per:siblings", "org:stateorprovince_of_headquarters", "per:spouse", "per:other_family", "per:city_of_death", "per:stateorprovince_of_death", "org:founded_by"]
prompt_list=[]
for i in range(0,len(Relation_list)):
    file_path = '/data/zxli/kp/binary-prompt/tacred/{}.txt'.format(i+1)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        prompt_list.append(file.read())

def read_file():
    file_path = '/data/zxli/kp/a-deepseek-distill/tacred_test_sentence.txt'
    lines = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            lines.append(line.strip())
    return lines

def read_head():
    file_path = '/data/zxli/kp/a-deepseek-distill/tacred_test_head.txt'
    heads = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            heads.append(line.strip())
    return heads

def read_tail():
    file_path = '/data/zxli/kp/a-deepseek-distill/tacred_test_tail.txt'
    tails = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            tails.append(line.strip())
    return tails

def read_relation():
    file_path = '/data/zxli/kp/a-deepseek-distill/tacred_test_relation.txt'
    tails = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            tails.append(line.strip())
    return tails

def make_prompt(prompt, data, head, tail):
    prompt = prompt.replace("(TEXT)", data)
    prompt = prompt.replace("(HE)", head)
    prompt = prompt.replace("(TE)", tail)
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

def element_extract(generated_text):
    match = re.search(r'\("([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)', generated_text)
    if match:
        h, r, t = match.groups()
        return h, r, t
    else:
        return None,None,None

def last_result(last_judge, confidence, test_relation):
    if all(x == 'No' for x in last_judge) == True:
        if test_relation == 'NA':
            return 'NA', 1, 1
        else:
            return 'NA', 1e-10, 1e-10
    relation_candidate = [index for index, value in enumerate(last_judge) if value != 'No']
    if(len(relation_candidate)==1):
        if test_relation == last_judge[relation_candidate[0]]:
            return last_judge[relation_candidate[0]], 1, 1
        else:
            return last_judge[relation_candidate[0]], 1e-10, 1e-10
    else:
        max_confidence = max([confidence[i] for i in relation_candidate])
        max_conf_indices = [i for i in relation_candidate if confidence[i] == max_confidence]
        if test_relation in [last_judge[i] for i in max_conf_indices]:
            return test_relation, 1/len(max_conf_indices), 1
        else:
            return last_judge[random.choice(max_conf_indices)], 1e-10, 1e-10

def accuracy2(relation_pred):
    file_path = '/data/zxli/kp/a-deepseek-distill/tacred_test_relation.txt'
    relation_label = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            relation_label.append(line.strip())
            
    acc = accuracy_score(relation_label, relation_pred)
    precision_micro = precision_score(relation_label, relation_pred, average='micro')
    recall_micro = recall_score(relation_label, relation_pred, average='micro')
    f1_micro = f1_score(relation_label, relation_pred, average='micro')
    precision_macro = precision_score(relation_label, relation_pred, average='macro')
    recall_macro = recall_score(relation_label, relation_pred, average='macro')
    f1_macro = f1_score(relation_label, relation_pred, average='macro')

    return acc, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro

def special_avg_f1(precisions, recalls):
    sum = 0
    
    for i in range(0,len(precisions)):
        sum = sum + (2*precisions[i]*recalls[i])/(precisions[i]+recalls[i])

    return sum/len(precisions)

if __name__ == '__main__':
    relation_preds = []
    precisions = []
    recalls = []
    confidences = []
    test_relations = []
    last_judge = [] 
    lines = read_file()
    heads = read_head()
    tails = read_tail()
    test_relations = read_relation()
    start_time = time.time()

    # model_path = '/data/zxli/te-gen/models/Qwen2.5-7B-Instruct-1M'
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-14B-Instruct-1M'
    model_path = '/data/zxli/te-gen/models/vicuna-13b-v1.5'
    print(model_path)
    
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='auto')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(0, len(lines)):
        for j in range(0, len(prompt_list)):
        # for j in range(0, 1):
            print(f"{i+1}/{len(lines)} | relation-prompt{j+1}")
            prompt_text = make_prompt(prompt_list[j], lines[i], heads[i], tails[i])

            generated_text, confidence = generate_text(prompt_text)
            print(confidence)
            confidences.append(confidence)
            
            h_generate,r_generate,t_generate=element_extract(generated_text)
            print(h_generate,r_generate,t_generate)
            if(h_generate == None or h_generate not in lines[i] or t_generate not in lines[i] or r_generate != Relation_list[j]):
                print('No' + '\n')
                last_judge.append('No')
            else:
                print('Yes' + '\n')
                last_judge.append(r_generate)
        print(last_judge)
        print(confidences)
        relation_pred, precision, recall = last_result(last_judge, confidences, test_relations[i])
        relation_preds.append(relation_pred)
        precisions.append(precision)
        recalls.append(recall)

        with open('relation_pred_binary-class-qwen2.5-tacred-relation.txt', 'w', encoding='utf-8') as file:
            for item in relation_preds:
                file.write("%s\n" % item) 
        last_judge = []
        confidences = []
        
    print(relation_preds)
    acc, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = accuracy2(relation_preds)
    end_time = time.time()

    elapsed_time = end_time - start_time
    sa_f1 = special_avg_f1(precisions, recalls)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Accuracy: {acc:.8f}")
    print(f"Precision_Micro: {precision_micro:.8f}")
    print(f"Recall_Micro: {recall_micro:.8f}")
    print(f"F1_Micro: {f1_micro:.8f}")
    print(f"Precision_Macro: {precision_macro:.8f}")
    print(f"Recall_Macro: {recall_macro:.8f}")
    print(f"F1_Macro: {f1_macro:.8f}")
    print(f"specical_avg_F1: {sa_f1:.8f}")