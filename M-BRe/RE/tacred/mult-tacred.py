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

Relation_list = ["org:founded", "org:subsidiaries", "per:date_of_birth", "per:cause_of_death", "per:age", "per:stateorprovince_of_birth", "per:countries_of_residence", "per:country_of_birth", "per:stateorprovinces_of_residence", "org:website", "per:cities_of_residence", "per:parents", "per:employee_of", "NA", "per:city_of_birth", "org:parents", "org:political/religious_affiliation", "per:schools_attended", "per:country_of_death", "per:children", "org:top_members/employees", "per:date_of_death", "org:members", "org:alternate_names", "per:religion", "org:member_of", "org:city_of_headquarters", "per:origin", "org:shareholders", "per:charges", "per:title", "org:number_of_employees/members", "org:dissolved", "org:country_of_headquarters", "per:alternate_names", "per:siblings", "org:stateorprovince_of_headquarters", "per:spouse", "per:other_family", "per:city_of_death", "per:stateorprovince_of_death", "org:founded_by"]

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

def make_prompt(data, head, tail):
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
Given a sentence, identify the relation category within it.

Here are definitions of relation categories and some examples:
(1) "org:founded"
definition: The founding time of an organization.
example1: Founded in 1919, the National Restaurant Association is the leading business association for the restaurant industry, which comprises 945,000 restaurant and food service outlets and a work force of nearly 13 million employees.

(2) "org:subsidiaries"
definition: The subsidiaries of an organization.
example1: The commercial market for feature documentaries has crashed after briefly flourishing when `` the Michael Moores of the world'' were seen to have breakout potential, said Geoffrey Gilmore, chief creative officer of Tribeca Enterprises and the former director of the Sundance Film Festival.

(3) "per:date_of_birth"
definition: The date of birth of a person.
example1: Ble Goude was born in 1972 in Gbagbo's centre west home region, Guiberoua, and rose to become secretary general of the powerful and aggressive Students' Federation of Ivory Coast -LRB- FESCI -RRB-.

(4) "per:cause_of_death"
definition: The cause of death of a person.
example1: Bruno turns himself in to Police If found guilty of participating in the alleged kidnapping and homicide of Samudio, Bruno's professional career will come to a screeching halt.

(5) "per:age"
definition: The age of a person.
example1: The judge had said earlier that testimony from three witnesses about the missionaries' efforts to set up an orphanage in the neighboring Dominican Republic would allow him to free Laura Silsby, 40, and Charisa Coulter, 24.  

(6) "per:stateorprovince_of_birth"
definition: The state or province of birth of a person.
example1: Steele, according to his online biography, grew up in rural Pennsylvania, obtained a master's degree in psychology, and became a marriage and family counselor in 1976.

(7) "per:countries_of_residence"
definition: The countries where a person resides.
example1: The higher court of southwestern China's Chongqing Municipality Friday upheld the sentences on female gang boss Xie Caiping and her 21 accomplices.

(8) "per:country_of_birth"
definition: The country of birth of a person.
example1: Francesco Introna, taking the stand at the murder trial of U.S. student Amanda Knox and Italian co-defendant Raffaele Sollecito, also said that no more than a single attacker could have assaulted the victim on the night of the 2007 slaying.

(9) "per:stateorprovinces_of_residence"
definition: The states or provinces where a person resides.
example1: The IG investigation was initiated by a request from congressional members concerned about allegations such as those by a Texas woman, Jamie Leigh Jones.

(10) "org:website"
definition: The website of an organization.
example1: Go to the National Restaurant Association's Web site -LRB- http://wwwrestaurantorg/ -RRB- for additional tips that can help you make eating out part of a healthy lifestyle.

(11) "per:cities_of_residence"
definition: The cities where a person resides.
example1: Two other missionaries -- group leader Laura Silsby and her confidante Charisa Coulter -- remained behind in detention in Port-au-Prince because Saint-Vil wants to determine their motives for an earlier trip to Haiti before the quake, Fleurant said.

(12) "per:parents"
definition: The parents of a person.
example1: Survivors include his wife, Sandra; four sons, Jeff, James, Douglas and Harris; a daughter, Leslie; his mother, Sally; and two brothers, Guy and Paul.

(13) "per:employee_of"
definition: The organization where a person is employed.
example1: They cited the case of Agency for International Development subcontractor Alan Gross, who was working in Cuba on a tourist visa and possessed satellite communications equipment, who has been held in a maximum security prison since his arrest Dec 3.

(14) "NA"
definition: Unknown or non-existent relation.
example1: `` He never gave the outward appearance that there was anything bothering him,'' Edwards said.

(15) "per:city_of_birth"
definition: The city of birth of a person.
example1: Gross, a native of Potomac, Maryland, was working for a firm contracted by USAID when he was arrested Dec 3, 2009.

(16) "org:parents"
definition: The parent company of an organization.
example1: A demonstration was scheduled for Thursday at the Westwood headquarters of KB Home to protest loans with five-year initial rates that the home builder issued in a partnership with Countrywide Financial Corp., now part of Bank of America Corp..

(17) "org:political/religious_affiliation"
definition: The political or religious affiliation of an organization.
example1: Suicide vest is vital clue after Uganda blasts While the bombers' actions appeared to support the Shebab's claim of responsibility, the police chief pointed a finger at a homegrown Muslim rebel group known as the Allied Democratic Forces -LRB- ADF -RRB-.

(18) "per:schools_attended"
definition: The schools attended by a person.
example1: While he was at Berkeley as a Packard fellow, Lange met another Packard fellow, Frances Arnold, a chemical engineer, who had attended Princeton and Berkeley when he did, but whom he had never met before then.

(19) "per:country_of_death"
definition: The country where a person died.
example1: They say Vladimir Ladyzhenskiy died late Saturday during the Sauna World Championships in southern Finland, while his Finnish rival Timo Kaukonen was rushed to a hospital.

(20) "per:children"
definition: The children of a person.
example1: Kercher's mother, Arline Kercher, tells court in emotional testimony that she will never get over her daughter's brutal death.

(21) "org:top_members/employees"
definition: The top members/employees of an organization.
example1: Earlier this year, we reported on the testimony of an anonymous EMT named Mike who told Loose Change producer Dylan Avery that hundreds of emergency rescue personnel were told over bullhorns that Building 7, a 47 story skyscraper adjacent the twin towers that was not hit by a plane yet imploded symmetrically later in the afternoon on 9/11, was about to be `` pulled'' and that a 20 second radio countdown preceded its collapse.

(22) "per:date_of_death"
definition: The date of death of a person.
example1: They say Vladimir Ladyzhenskiy died late Saturday during the Sauna World Championships in southern Finland, while his Finnish rival Timo Kaukonen was rushed to a hospital.

(23) "org:members"
definition: The members of an organization.
example1: OANA members include, to name just a few, Australia's AAP, China's Xinhua, India's PTI, Indonesia's ANTARA, Iran's IRNA, Japan's Kyodo and Jiji Press, Pakistan's PPI and APP, Kazakhstan's Kazinform, Kuwait's KUNA, Mongolia's MONTSAME, the Philippines' PNA, Russia's Itar-Tass and RIA, Saudi Arabia's SPA, the Republic of Korea's Yonhap, and Turkey's Anadolu.

(24) "org:alternate_names"
definition: The alternate names of an organization.
example1: A 2005 study by the Associatio of University Women -LRB- AAUW -RRB-, called `` The -LRB- Un -RRB- Changing Face of the Ivy League,'' showed that from 1993 to 2003, the number of female professors rose from 14 to 20 percent of tenured faculty.

(25) "per:religion"
definition: The religion of a person.
example1: He closed out the quarter making seven payments to Scientology groups totaling $ 13,500.

(26) "org:member_of"
definition: The organization to which a member belongs.
example1: The National Development and Reform Commission, China's main planning agency, also expects by 2015 to have 30 ethylene factories, each with an annual output of 1 million tons a year, the report said, citing unnamed officials.

(27) "org:city_of_headquarters"
definition: The city where the headquarters of an organization is located.
example1: The portfolios of two other major option ARM lenders overseen by OTS, Golden West Financial Corp of Oakland, Calif, and Countrywide Financial Corp of Calabasas, Calif, also have racked up huge losses and have been swallowed by other companies.

(28) "per:origin"
definition: The origin of a person.
example1: U.S. contractor Alan Gross, who was arrested for alleged espionage activities in December, remains under investigation, said Cuban Foreign Minister Bruno Rodriguez Wednesday.

(29) "org:shareholders"
definition: The shareholders of an organization.
example1: Financials were the weakest as investors continue to question what the impact will be over reports that the New York Federal Reserve will join institutional bond holders in an effort to force Bank of America Corp to repurchase billions of dollars in mortgage bonds issued by Countrywide Financial, which BofA purchased in 2008.

(30) "per:charges"
definition: The charges against a person.
example1: Rashid advocated a similar assault in the tribal belt to the one in Swat earlier this year, and said there was strong US pressure for such action.

(31) "per:title"
definition: The occupation of a person.
example1: SAN FRANCISCO, CA July 16, 2007 -- San Francisco architect Richard Gage, AIA, founder of the group, ` Architects & Engineers for 9/11 Truth,' announced today the statement of support from J. Marx Ayres, former member of the California Seismic Safety Commission and former member of the National Institute of Sciences Building Safety Council.

(32) "org:number_of_employees/members"
definition: The number of employees/members in an organization.
example1: After the staffing firm Hollister Inc. lost 20 of its 85 employees, it gave up nearly a third of its 3,750-square-foot Burlington office, allowing the property owner to put up a dividing wall to create a space for another tenant.

(33) "org:dissolved"
definition: The date of dissolution of the organization.
example1: At Countrywide, which is finishing up a round of 12,000 job cuts, Chief Executive Angelo Mozilo said in announcing the Bank of America takeover last week that the housing and mortgage sectors were being strained `` as never seen since the Great Depression.''

(34) "org:country_of_headquarters"
definition: The country where the headquarters of an organization is located.
example1: Outside the court, a US military spokesman said Budd had been in Australia to take part in the exercise codenamed `` Talisman Sabre'', which involves 7,500 Australian Defence Force personnel and 20,000 US troops.

(35) "per:alternate_names"
definition: The alternate names of a person.
example1: The boy, identified by the Dutch foreign ministry as Ruben but more fully by Dutch media as Ruben van Assouw, was found alive, strapped into his seat at the accident site.

(36) "per:siblings"
definition: The siblings of a person.
example1: He is also survived by his parents and a sister, Karen Lange, of Washington, and a brother, Adam Lange, of St. Louis.

(37) "org:stateorprovince_of_headquarters"
definition: The state or province where the headquarters of an organization is located.
example1: `` We're writing history here,'' said Nell Minow, cofounder of the Corporate Library, a corporate governance research firm in Portland, Maine.

(38) "per:spouse"
definition: The spouse of a person.
example1: CHONGQING, May 21 -LRB- Xinhua -RRB- Wen's wife Zhou Xiaoya was jailed for eight years after being convicted of taking advantage of her husband's official position and taking bribes totaling 449 million yuan with Wen.       

(39) "per:other_family"
definition: Other family members of a person.
example1: `` He's a humanitarian, an idealist, and probably was naive and maybe not understanding enough of what he was getting himself into... that he could be arrested,'' she said.

(40) "per:city_of_death"
definition: The city where a person died.
example1: OKLAHOMA CITY 2009-08-28 17:30:13 UTC Investigators also have said Daniels' body was `` staged,'' or moved into an unnatural position, after she was killed Sunday at the church in Anadarko.

(41) "per:stateorprovince_of_death"
definition: The state or province where a person died.
example1: Police have released scant information about the killing of 61-year-old Carol Daniels, whose body was found Sunday inside the Christ Holy Sanctified Church, a weather-beaten building on a rundown block near downtown Anadarko in southwest Oklahoma.

(42) "org:founded_by"
definition: The founder of an organization.
example1: `` It's a very small step in a very long journey,'' said Nell Minow, co-founder of the Corporate Library, an independent research company specializing in executive compensation.

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
       
if __name__ == '__main__':
    relation_pred = []
    heads = read_head()
    tails = read_tail()

    start_time = time.time()
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-7B-Instruct-1M'
    # model_path = '/data/zxli/te-gen/models/Qwen2.5-14B-Instruct-1M'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi2'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi2'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi2_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi2_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi1_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi1_mannully8'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-7B-Instruct-1M_lora_sft_multi1_mannully8_generate'
    # model_path = '/data/zxli/LLaMA-Factory-main/output/Qwen2.5-14B-Instruct-1M_lora_sft_multi1_mannully8_generate'
    model_path = '/data/zxli/te-gen/models/vicuna-13b-v1.5'
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
        prompt_text = make_prompt(lines[i], heads[i], tails[i])
        generated_text = generate_text(prompt_text)
        print(generated_text + '\n')
        relation_pred.append(generated_text.strip())

        with open('relation_pred_mult-class-qwen2.5-tacred.txt', 'w', encoding='utf-8') as file:
            for item in relation_pred:
                if item in Relation_list:
                    file.write("%s\n" % item)   
                else:
                    file.write("NA\n")                

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