import re
import json
import random
random.seed(42)

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

def json_make_yes_or_no(): 
    with open('/data/zxli/M-BRe/NER/ontonotes_uf/train_data/train_sent.txt', 'r', encoding='utf-8') as f:
        sentences = [sentences.rstrip('\n') for sentences in f]

    with open('/data/zxli/M-BRe/NER/ontonotes_uf/train_data/train_entity.txt', 'r', encoding='utf-8') as f:
        fg_ner = [fg_ner.rstrip('\n') for fg_ner in f]

    with open('/data/zxli/M-BRe/NER/ontonotes_uf/train_data/train_label.txt', 'r', encoding='utf-8') as f:
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
    filename = '/data/zxli/M-BRe/NER/ontonotes_uf/binary-prompt/{}.txt'.format(i+1)
    with open(filename, 'w', encoding='utf-8') as output_file:
        output_file.write(prompt)