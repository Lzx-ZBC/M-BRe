import os
os.environ["CUDA_VISIBLE_DEVICES"]='4,5'
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

Relation_list = ["Component-Whole(e2,e1)", "Instrument-Agency(e2,e1)", "Member-Collection(e1,e2)", "Cause-Effect(e2,e1)", "Entity-Destination(e1,e2)", "Content-Container(e1,e2)", "Message-Topic(e1,e2)", "Product-Producer(e2,e1)", "Member-Collection(e2,e1)", "Entity-Origin(e1,e2)", "Cause-Effect(e1,e2)", "Component-Whole(e1,e2)", "Message-Topic(e2,e1)", "Product-Producer(e1,e2)", "Entity-Origin(e2,e1)", "Content-Container(e2,e1)", "Instrument-Agency(e1,e2)", "Entity-Destination(e2,e1)", "Other"]

def read_file():
    file_path = '/data/zxli/kp/a-deepseek-distill/semeval_test_sentence.txt'
    
    lines = []
        
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            lines.append(line.strip())
    return lines

def read_head():
    file_path = '/data/zxli/kp/a-deepseek-distill/semeval_test_head.txt'
    heads = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            heads.append(line.strip())
    return heads

def read_tail():
    file_path = '/data/zxli/kp/a-deepseek-distill/semeval_test_tail.txt'
    tails = []

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file1:
        for line in file1:
            tails.append(line.strip())
    return tails

def make_prompt(data, head, tail):
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Here are definitions of relation categories and some examples:

(1) "Component-Whole(e2,e1)"
definition: Tail entity e2 is the component of head entity e1, and head entity e1 is the whole of tail entity e2.
example: jeans have topstitching that is almost impossible to match.

(2) "Instrument-Agency(e2,e1)"
definition: Tail entity e2 is the instrument of head entity e1, and head entity e1 is the agency of tail entity e2.
example: a multi-purpose electrician pliers tool is crucial for severing any gauge wire with which the electrician is working.

(3) "Member-Collection(e1,e2)"
definition: Head entity e1 is the member of tail entity e2, and tail entity e2 is the collection of head entity e1.
example: jeff ma, who was a key member of the infamous mit blackjack team, notes the turn around of the oakland a's and the reversal of criticism directed toward gm billy beane.

(4) "Cause-Effect(e2,e1)"
definition: Tail entity e2 is the cause of head entity e1, and head entity e1 is the effect of tail entity e2.
example: the election was caused by the appointment of donald sumner, formerly conservative mp for orpington, to be a county court judge.

(5) "Entity-Destination(e1,e2)"
definition: Head entity e1 is the entity of tail entity e2, and tail entity e2 is the destination of head entity e1.
example: the celebrity poured money into modern artwork.

(6) "Content-Container(e1,e2)"
definition: Head entity e1 is the content of tail entity e2, and tail entity e2 is the container of head entity e1.
example: the tape recorder was in a cabinet outside the cab of the engine.

(7) "Message-Topic(e1,e2)"
definition: Head entity e1 is the message of tail entity e2, and tail entity e2 is the topic of head entity e1.
example: the lyrics point toward sexual politics, and the lack of understanding between boys and girls.

(8) "Product-Producer(e2,e1)"
definition: Tail entity e2 is the product of head entity e1, and head entity e1 is the producer of tail entity e2.
example: tributes have been paid to the writer who created goodness gracious me, the hit bbc television series.

(9) "Member-Collection(e2,e1)"
definition: Tail entity e2 is the member of head entity e1, and head entity e1 is the collection of tail entity e2.
example: in this way he gathered materials for weekly epistles destined to enlighten some county town or some bench of rustic magistrates.

(10) "Entity-Origin(e1,e2)"
definition: Head entity e1 is the entity of tail entity e2, and tail entity e2 is the origin of head entity e1.
example: the waverider design was evolved from work done in the u.k. in the 1950's and early 1960's on winged atmosphere re-entry vehicles.

(11) "Cause-Effect(e1,e2)"
definition: Head entity e1 is the cause of tail entity e2, and tail entity e2 is the effect of head entity e1.
example: the distraction caused by the students, coupled with limited vision down the track, caused the incident to occur.

(12) "Component-Whole(e1,e2)"
definition: Head entity e1 is the component of tail entity e2, and tail entity e2 is the whole of head entity e1.
example: four people died after their vehicle crashed into the support pillar of an overhead bridge in cheras.

(13) "Message-Topic(e2,e1)"
definition: Tail entity e2 is the message of head entity e1, and head entity e1 is the topic of tail entity e2.
example: inside of it, the first details about the game were revealed through an interview with series director masahiro yasuma.

(14) "Product-Producer(e1,e2)"
definition: Head entity e1 is the product of tail entity e2, and tail entity e2 is the producer of head entity e1.
example: mileson has sold his humble abode to a housing developer.

(15) "Entity-Origin(e2,e1)"
definition: Tail entity e2 is the entity of head entity e1, and head entity e1 is the origin of tail entity e2.
example: red grape wine is an alcoholic fruit drink of between 10 and 14 % alcoholic strength.

(16) "Content-Container(e2,e1)"
definition: Tail entity e2 is the content of head entity e1, and head entity e1 is the container of tail entity e2.
example: but the man had a bottle with water in it inside the bag, and it was not accidental.

(17) "Instrument-Agency(e1,e2)"
definition: Head entity e1 is the instrument of tail entity e2, and tail entity e2 is the agency of head entity e1.
example: guru jakob nielsen gives his advice on best practices for programmers.

(18) "Entity-Destination(e2,e1)"
definition: Tail entity e2 is the entity of head entity e1, and head entity e1 is the destination of tail entity e2.
example: a few days before the service, tom burris had thrown into karen's casket his wedding ring.

(19) "Other"
definition: There is no relationship or unrecognized relationship between the head and tail entities.
example: a group of women from curves timaru won best-dressed prize with their ensemble of purple clothes, colourful wigs and sparkly tiaras.

Below is the target sentence and the provided head and tail entities. Please identify the relation category within the target sentence. Just output the relation category with double quotes.

Target Sentence: (TEXT)
provided head entity: (HE)
provided tail entity: (TE)

Target Answer:
"""
    prompt = prompt.replace("(TEXT)", data)
    prompt = prompt.replace("(HE)", head)
    prompt = prompt.replace("(TE)", tail)
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
    matches = re.findall(r'"(.*?)"', response)
    # pattern = r'Target Answer:\s*"\s*(.*?)\s*"\s*<\|im_end\|>'
    # matches = re.findall(pattern, response, re.DOTALL)     
    
    if matches:
        return matches[0]
    else:
        return "Other"
    
def accuracy2(relation_pred):
    file_path = '/data/zxli/kp/a-deepseek-distill/semeval_test_relation.txt'

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
       
if __name__ == '__main__':
    relation_pred = []
    heads = read_head()
    tails = read_tail()

    start_time = time.time()
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-7B-Instruct-1M'
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-14B-Instruct-1M'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi1_mannully8_semeval'
    model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi1_mannully8_semeval'
    # model_path = '/data/zxli/te-gen/models/Qwen3-14B'

    print(model_path)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,device_map='auto')
    #model.load_state_dict(torch.load(fine_tuned_weights_path))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    lines = read_file()
    for i in range(0, len(lines)):
        print(f"{i+1}/{len(lines)}")
        prompt_text = make_prompt(lines[i], heads[i], tails[i])
        generated_text = generate_text(prompt_text)
        print(generated_text + '\n')
        relation_pred.append(generated_text.strip())

        # with open('relation_pred_mult-class-qwen2.5-semeval.txt', 'w', encoding='utf-8') as file:
        #     for item in relation_pred:
        #         if item in Relation_list:
        #             file.write("%s\n" % item)   
        #         else:
        #             file.write("Other\n")                

    # for i in range(0, len(relation_pred)):
    #     if relation_pred[i] not in Relation_list:
    #         relation_pred[i] = "NA"
               
    print(relation_pred)
    acc, precision_micro, recall_micro, f1_micro, precision_macro, recall_macro, f1_macro = accuracy2(relation_pred)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Accuracy: {acc:.8f}")
    print(f"Precision_Micro: {precision_micro:.8f}")
    print(f"Recall_Micro: {recall_micro:.8f}")
    print(f"F1_Micro: {f1_micro:.8f}")
    print(f"Precision_Macro: {precision_macro:.8f}")
    print(f"Recall_Macro: {recall_macro:.8f}")
    print(f"F1_Macro: {f1_macro:.8f}")