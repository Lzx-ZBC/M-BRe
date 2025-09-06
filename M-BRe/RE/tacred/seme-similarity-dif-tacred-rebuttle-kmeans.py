import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re 
import os

relations = [
    '"org:founded": The founding time of an organization', 
    '"org:subsidiaries": The subsidiaries of an organization', 
    '"per:date_of_birth": The date of birth of a person', 
    '"per:cause_of_death": The cause of death of a person', 
    '"per:age": The age of a person', 
    '"per:stateorprovince_of_birth": The state or province of birth of a person', 
    '"per:countries_of_residence": The countries where a person resides', 
    '"per:country_of_birth": The country of birth of a person', 
    '"per:stateorprovinces_of_residence": The states or provinces where a person resides', 
    '"org:website": The website of an organization',
    '"per:cities_of_residence": The cities where a person resides', 
    '"per:parents": The parents of a person', 
    '"per:employee_of": The organization where a person is employed', 
    '"NA": Unknown or non-existent relation', 
    '"per:city_of_birth": The city of birth of a person', 
    '"org:parents": The parent company of an organization', 
    '"org:political/religious_affiliation": The political or religious affiliation of an organization', 
    '"per:schools_attended": The schools attended by a person', 
    '"per:country_of_death": The country where a person died', 
    '"per:children": The children of a person', 
    '"org:top_members/employees": The top members/employees of an organization', 
    '"per:date_of_death": The date of death of a person', 
    '"org:members": The members of an organization', 
    '"org:alternate_names": The alternate names of an organization', 
    '"per:religion": The religion of a person', 
    '"org:member_of": The organization to which a member belongs', 
    '"org:city_of_headquarters": The city where the headquarters of an organization is located', 
    '"per:origin": The origin of a person', 
    '"org:shareholders": The shareholders of an organization', 
    '"per:charges": The charges against a person', 
    '"per:title": The occupation of a person', 
    '"org:number_of_employees/members": The number of employees/members in an organization', 
    '"org:dissolved": The date of dissolution of the organization', 
    '"org:country_of_headquarters": The country where the headquarters of an organization is located', 
    '"per:alternate_names": The alternate names of a person', 
    '"per:siblings": The siblings of a person', 
    '"org:stateorprovince_of_headquarters": The state or province where the headquarters of an organization is located', 
    '"per:spouse": The spouse of a person', 
    '"per:other_family": Other family members of a person', 
    '"per:city_of_death": The city where a person died', 
    '"per:stateorprovince_of_death": The state or province where a person died', 
    '"org:founded_by": The founder of an organization'
]
examples = [
    "Founded in 1919, the National Restaurant Association is the leading business association for the restaurant industry, which comprises 945,000 restaurant and food service outlets and a work force of nearly 13 million employees.",
    "The commercial market for feature documentaries has crashed after briefly flourishing when `` the Michael Moores of the world'' were seen to have breakout potential, said Geoffrey Gilmore, chief creative officer of Tribeca Enterprises and the former director of the Sundance Film Festival.",
    "Ble Goude was born in 1972 in Gbagbo's centre west home region, Guiberoua, and rose to become secretary general of the powerful and aggressive Students' Federation of Ivory Coast -LRB- FESCI -RRB-.",
    "Bruno turns himself in to Police If found guilty of participating in the alleged kidnapping and homicide of Samudio, Bruno's professional career will come to a screeching halt.",
    "The judge had said earlier that testimony from three witnesses about the missionaries' efforts to set up an orphanage in the neighboring Dominican Republic would allow him to free Laura Silsby, 40, and Charisa Coulter, 24.", 
    "Steele, according to his online biography, grew up in rural Pennsylvania, obtained a master's degree in psychology, and became a marriage and family counselor in 1976.",
    "The higher court of southwestern China's Chongqing Municipality Friday upheld the sentences on female gang boss Xie Caiping and her 21 accomplices.",
    "Francesco Introna, taking the stand at the murder trial of U.S. student Amanda Knox and Italian co-defendant Raffaele Sollecito, also said that no more than a single attacker could have assaulted the victim on the night of the 2007 slaying.",
    "The IG investigation was initiated by a request from congressional members concerned about allegations such as those by a Texas woman, Jamie Leigh Jones.",
    "Go to the National Restaurant Association's Web site -LRB- http://wwwrestaurantorg/ -RRB- for additional tips that can help you make eating out part of a healthy lifestyle.",
    "Two other missionaries -- group leader Laura Silsby and her confidante Charisa Coulter -- remained behind in detention in Port-au-Prince because Saint-Vil wants to determine their motives for an earlier trip to Haiti before the quake, Fleurant said.",
    "Survivors include his wife, Sandra; four sons, Jeff, James, Douglas and Harris; a daughter, Leslie; his mother, Sally; and two brothers, Guy and Paul.",
    "They cited the case of Agency for International Development subcontractor Alan Gross, who was working in Cuba on a tourist visa and possessed satellite communications equipment, who has been held in a maximum security prison since his arrest Dec 3.",
    "He never gave the outward appearance that there was anything bothering him,'' Edwards said.",
    "Gross, a native of Potomac, Maryland, was working for a firm contracted by USAID when he was arrested Dec 3, 2009.",
    "A demonstration was scheduled for Thursday at the Westwood headquarters of KB Home to protest loans with five-year initial rates that the home builder issued in a partnership with Countrywide Financial Corp., now part of Bank of America Corp..",
    "Suicide vest is vital clue after Uganda blasts While the bombers' actions appeared to support the Shebab's claim of responsibility, the police chief pointed a finger at a homegrown Muslim rebel group known as the Allied Democratic Forces -LRB- ADF -RRB-.",
    "While he was at Berkeley as a Packard fellow, Lange met another Packard fellow, Frances Arnold, a chemical engineer, who had attended Princeton and Berkeley when he did, but whom he had never met before then.",
    "They say Vladimir Ladyzhenskiy died late Saturday during the Sauna World Championships in southern Finland, while his Finnish rival Timo Kaukonen was rushed to a hospital.",
    "Kercher's mother, Arline Kercher, tells court in emotional testimony that she will never get over her daughter's brutal death.",
    "Earlier this year, we reported on the testimony of an anonymous EMT named Mike who told Loose Change producer Dylan Avery that hundreds of emergency rescue personnel were told over bullhorns that Building 7, a 47 story skyscraper adjacent the twin towers that was not hit by a plane yet imploded symmetrically later in the afternoon on 9/11, was about to be `` pulled'' and that a 20 second radio countdown preceded its collapse.",
    "They say Vladimir Ladyzhenskiy died late Saturday during the Sauna World Championships in southern Finland, while his Finnish rival Timo Kaukonen was rushed to a hospital.",
    "OANA members include, to name just a few, Australia's AAP, China's Xinhua, India's PTI, Indonesia's ANTARA, Iran's IRNA, Japan's Kyodo and Jiji Press, Pakistan's PPI and APP, Kazakhstan's Kazinform, Kuwait's KUNA, Mongolia's MONTSAME, the Philippines' PNA, Russia's Itar-Tass and RIA, Saudi Arabia's SPA, the Republic of Korea's Yonhap, and Turkey's Anadolu.",
    "A 2005 study by the Associatio of University Women -LRB- AAUW -RRB-, called `` The -LRB- Un -RRB- Changing Face of the Ivy League,'' showed that from 1993 to 2003, the number of female professors rose from 14 to 20 percent of tenured faculty.",
    "He closed out the quarter making seven payments to Scientology groups totaling $ 13,500.",
    "The National Development and Reform Commission, China's main planning agency, also expects by 2015 to have 30 ethylene factories, each with an annual output of 1 million tons a year, the report said, citing unnamed officials.",
    "The portfolios of two other major option ARM lenders overseen by OTS, Golden West Financial Corp of Oakland, Calif, and Countrywide Financial Corp of Calabasas, Calif, also have racked up huge losses and have been swallowed by other companies.",
    "U.S. contractor Alan Gross, who was arrested for alleged espionage activities in December, remains under investigation, said Cuban Foreign Minister Bruno Rodriguez Wednesday.",
    "Financials were the weakest as investors continue to question what the impact will be over reports that the New York Federal Reserve will join institutional bond holders in an effort to force Bank of America Corp to repurchase billions of dollars in mortgage bonds issued by Countrywide Financial, which BofA purchased in 2008.",
    "Rashid advocated a similar assault in the tribal belt to the one in Swat earlier this year, and said there was strong US pressure for such action.",
    "SAN FRANCISCO, CA July 16, 2007 -- San Francisco architect Richard Gage, AIA, founder of the group, ` Architects & Engineers for 9/11 Truth,' announced today the statement of support from J. Marx Ayres, former member of the California Seismic Safety Commission and former member of the National Institute of Sciences Building Safety Council.",
    "After the staffing firm Hollister Inc. lost 20 of its 85 employees, it gave up nearly a third of its 3,750-square-foot Burlington office, allowing the property owner to put up a dividing wall to create a space for another tenant.",
    "At Countrywide, which is finishing up a round of 12,000 job cuts, Chief Executive Angelo Mozilo said in announcing the Bank of America takeover last week that the housing and mortgage sectors were being strained `` as never seen since the Great Depression.''",
    "Outside the court, a US military spokesman said Budd had been in Australia to take part in the exercise codenamed `` Talisman Sabre'', which involves 7,500 Australian Defence Force personnel and 20,000 US troops.",
    "The boy, identified by the Dutch foreign ministry as Ruben but more fully by Dutch media as Ruben van Assouw, was found alive, strapped into his seat at the accident site.",
    "He is also survived by his parents and a sister, Karen Lange, of Washington, and a brother, Adam Lange, of St. Louis.",
    "`` We're writing history here,'' said Nell Minow, cofounder of the Corporate Library, a corporate governance research firm in Portland, Maine.",
    "CHONGQING, May 21 -LRB- Xinhua -RRB- Wen's wife Zhou Xiaoya was jailed for eight years after being convicted of taking advantage of her husband's official position and taking bribes totaling 449 million yuan with Wen.",
    "`` He's a humanitarian, an idealist, and probably was naive and maybe not understanding enough of what he was getting himself into... that he could be arrested,'' she said.",
    "OKLAHOMA CITY 2009-08-28 17:30:13 UTC Investigators also have said Daniels' body was `` staged,'' or moved into an unnatural position, after she was killed Sunday at the church in Anadarko.",
    "Police have released scant information about the killing of 61-year-old Carol Daniels, whose body was found Sunday inside the Christ Holy Sanctified Church, a weather-beaten building on a rundown block near downtown Anadarko in southwest Oklahoma.",
    "`` It's a very small step in a very long journey,'' said Nell Minow, co-founder of the Corporate Library, an independent research company specializing in executive compensation."
]

# 提取描述文本并向量化
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(relations)

n_groups = 4

# 使用K-means聚类
kmeans = KMeans(n_clusters=n_groups, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)

# 构建分组
groups = [[] for _ in range(n_groups)]
for i, label in enumerate(cluster_labels):
    groups[label].append(i)

# 将索引转换为原始关系并排序
result_groups = []
for group in groups:
    sorted_group = sorted(group)
    result_groups.append([relations[i] for i in sorted_group])

# 打印分组结果
for i, group in enumerate(result_groups):
    print(f"============ K-means Group {i+1} ============")
    for rel in group:
        print(f"• {rel}")
    print("\n")

# 生成提示文件
for i, group in enumerate(result_groups):
    prompt_begin = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

Instruction:
Given a sentence, identify the relation category within it.

Here are definitions of relation categories and some examples:"""

    prompt_end = """
Below is the target sentence and the provided head and tail entities. Please identify the relation category within the target sentence. Just output the relation category with double quotes.

Target Sentence: (TEXT)
provided head entity: (HE)
provided tail entity: (TE)

Target Answer:
"""
    for j, rel in enumerate(group):
        double_quote_content = re.search(r'"(.*?)"', rel)
        if double_quote_content:
            double_quote_content = double_quote_content.group(1)

        colon_content = re.search(r'":\s*(.*)', rel)
        if colon_content:
            colon_content = colon_content.group(1)

        index = relations.index(rel)

        prompt_mid = f"""
({j+1}) "{double_quote_content}"
definition: {colon_content}.
example1: {examples[index]}
"""
        prompt_begin = prompt_begin + prompt_mid
    prompt_final = prompt_begin + prompt_end

    output_dir = f"/data/zxli/kp/mult-prompt-tacred-cos/mult-prompt-tacred_kmeans{n_groups}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_name = f"/data/zxli/kp/mult-prompt-tacred-cos/mult-prompt-tacred_kmeans{n_groups}/{i+1}.txt"

    with open(file_name, "w", encoding="utf-8") as file:
        file.write(prompt_final)