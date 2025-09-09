import os
os.environ["CUDA_VISIBLE_DEVICES"]='2,3'
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
torch.cuda.empty_cache()
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

Relation_list = ["Component-Whole(e2,e1)", "Instrument-Agency(e2,e1)", "Member-Collection(e1,e2)", "Cause-Effect(e2,e1)", "Entity-Destination(e1,e2)", "Content-Container(e1,e2)", "Message-Topic(e1,e2)", "Product-Producer(e2,e1)", "Member-Collection(e2,e1)", "Entity-Origin(e1,e2)", "Cause-Effect(e1,e2)", "Component-Whole(e1,e2)", "Message-Topic(e2,e1)", "Product-Producer(e1,e2)", "Entity-Origin(e2,e1)", "Content-Container(e2,e1)", "Instrument-Agency(e1,e2)", "Entity-Destination(e2,e1)", "Other"]
Relation_dic = {Relation_list[i-1]: i for i in range(1,len(Relation_list)+1)}

multi_prompt_list=[]
for i in range(0,4):
    file_path = '/data/zxli/kp/mult-prompt-semeval-cos/mult-prompt-semeval4/{}.txt'.format(i+1)
    with open(file_path, 'r', encoding='utf-8') as file:
        multi_prompt_list.append(file.read())

binary_prompt_list=[]
for i in range(0,len(Relation_list)):
    file_path = '/data/zxli/kp/binary-prompt/semeval/{}.txt'.format(i+1)
    
    with open(file_path, 'r', encoding='utf-8') as file:
        binary_prompt_list.append(file.read())

def read_file():
    file_path = '/data/zxli/kp/a-wiki-extract/wikineural-sentence.txt'
    lines = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            lines.append(line.strip())
    return lines

def read_head():
    file_path = '/data/zxli/kp/a-wiki-extract/wikineural-head_entities.txt'
    heads = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            heads.append(line.strip())
    return heads

def read_tail():
    file_path = '/data/zxli/kp/a-wiki-extract/wikineural-tail_entities.txt'
    tails = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            tails.append(line.strip())
    return tails

def make_prompt_multi(prompt, data, head, tail):
    prompt = prompt.replace("(TEXT)", data)
    prompt = prompt.replace("(HE)", head)
    prompt = prompt.replace("(TE)", tail)
    return prompt

def make_prompt_binary(prompt, data, head, tail):
    prompt = prompt.replace("(TEXT)", data)
    prompt = prompt.replace("(HE)", head)
    prompt = prompt.replace("(TE)", tail)
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
            # outputs.sequences[0][:], skip_special_tokens=False
        )
    matches = re.findall(r'"(.*?)"', response)
    # pattern = r'Target Answer:\s*"\s*(.*?)\s*"\s*<\|im_end\|>'
    # matches = re.findall(pattern, response, re.DOTALL)     

    print(response)
    if matches:
        return matches[0]
    else:
        return "Other"
    
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

def element_extract(generated_text):
    match = re.search(r'\("([^"]*)",\s*"([^"]*)",\s*"([^"]*)"\)', generated_text)
    if match:
        h, r, t = match.groups()
        return h, r, t
    else:
        return None,None,None

def last_result(last_judge, confidence):
    all_relation = ''
    if all(x == 'No' for x in last_judge) == True:
        all_relation = all_relation + 'Other' + ' '
    else:
        relation_candidate = [index for index, value in enumerate(last_judge) if value != 'No']
        if(len(relation_candidate)==1):
            if last_judge[relation_candidate[0]] in Relation_list:
                all_relation = all_relation + last_judge[relation_candidate[0]] + ' '
            else:
                all_relation = all_relation + 'Other' + ' '
        else:
            max_confidence = max([confidence[i] for i in relation_candidate])
            max_conf_indices = [i for i in relation_candidate if confidence[i] == max_confidence]
            for i in range(0,len(max_conf_indices)):
                if last_judge[i] in Relation_list:
                    all_relation = all_relation + last_judge[i] + ' '
                else:
                    all_relation = all_relation + 'Other' + ' '
    return all_relation

if __name__ == '__main__':
    last_judge = []
    relation_preds = []
    confidences = []
    lines = read_file()
    heads = read_head()
    tails = read_tail()

    start_time = time.time()
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-7B-Instruct-1M'
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-14B-Instruct-1M'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi1_mannully8_semeval'
    model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi1_mannully8_semeval'
    print(model_path)

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='auto')  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for i in range(0, len(lines)):
        for j in range(0, len(multi_prompt_list)):
            print(f"{i+1}/{len(lines)} | mult-prompt{j+1}")
            ## mult
            prompt_text1 = make_prompt_multi(multi_prompt_list[j], lines[i], heads[i], tails[i])
            generated_text1 = multi_generate_text(prompt_text1)
            print(generated_text1 + '\n')

            ## binary
            if generated_text1 not in Relation_list:
                generated_text1 = 'Other'
            prompt_text2 = make_prompt_binary(binary_prompt_list[Relation_dic[generated_text1]-1], lines[i], heads[i], tails[i])
            generated_text2, confidence = binary_generate_text(prompt_text2)
            confidences.append(confidence)
            print(generated_text2 + '\n')
            h_generate,r_generate,t_generate=element_extract(generated_text2)
            if(h_generate == None or h_generate not in lines[i] or t_generate not in lines[i] or r_generate != generated_text1):
                print('No' + '\n')
                last_judge.append('No')
            else:
                print('Yes' + '\n')
                last_judge.append(r_generate)
        print(last_judge)
        print(confidences)

        relation_candidate = last_result(last_judge, confidences)
        relation_preds.append(relation_candidate)

        with open('qwen2.5-14B-wiki-relation-semeval-group4.txt', 'w', encoding='utf-8') as file:
            for item in relation_preds:
                file.write("%s\n" % item)

        last_judge = []
        confidences = []
                         
    print(relation_preds)
    end_time = time.time()

    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time:.2f} seconds")