import re
import json
import random
random.seed(42)

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

def json_make_yes_or_no(): 
    with open('/data/zxli/kp/appendix_ner/train_sent.txt', 'r', encoding='utf-8') as f:
        sentences = [sentences.rstrip('\n') for sentences in f]

    with open('/data/zxli/kp/appendix_ner/train_entity.txt', 'r', encoding='utf-8') as f:
        fg_ner = [fg_ner.rstrip('\n') for fg_ner in f]

    with open('/data/zxli/kp/appendix_ner/train_label.txt', 'r', encoding='utf-8') as f:
        label = [label.rstrip('\n') for label in f]

    return sentences, fg_ner, label

for i in range(0,len(grained_type)):
    prompt = """Given a sentence, determine whether it describes the fine-grained label "(Label)" of target entity in the sentence. If it does, simply output "Yes". Otherwise, simply output "No".

Here are several examples.

"""
    prompt_tail = """Below is the target sentence and the provided entity. Please determine if there is a "(Label)" label of target entity in the sentence.

Target Sentence: (TEXT)
provided entity: (PE)

Target Answer:
"""
    prompt = prompt.replace("(Label)", grained_type[i])
    prompt_tail = prompt_tail.replace("(Label)", grained_type[i])

    for j in range(0,7):
        if j == 0 or j == 2 or j == 4:
            sentences, fg_ner, label=json_make_yes_or_no()
            indexes = [index for index, value in enumerate(label) if value == grained_type[i]]
            if indexes:
                random_index = random.choice(indexes)
            example = f"""Sentence: {sentences[random_index]}
Entity: {fg_ner[random_index]}
Answer: Yes."""
            prompt = prompt + example + '\n\n'
        else:
            sentences, fg_ner, label=json_make_yes_or_no()
            indexes = [index for index, value in enumerate(label) if value != grained_type[i]]
            if indexes:
                random_index = random.choice(indexes)
            example = f"""Sentence: {sentences[random_index]}
Entity: {fg_ner[random_index]}
Answer: No."""
            prompt = prompt + example + '\n\n'

    prompt = prompt + prompt_tail
    # print(prompt)
    filename = '/data/zxli/kp/appendix_ner/binary-prompt/{}.txt'.format(i+1)
    with open(filename, 'w', encoding='utf-8') as output_file:
        output_file.write(prompt)