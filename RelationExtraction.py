import nltk
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from spacy.matcher import DependencyMatcher
import numpy as np

from DependencyParsingPatterns import *
from NBA_Scraper import *
from utils import *
import gc

"""

Relation Extraction Reference
https://spacy.io/api/annotation#dependency-parsing
https://spacy.io/usage/rule-based-matching#phrasematcher
https://www.analyticsvidhya.com/blog/2019/09/introduction-information-extraction-python-spacy/
https://towardsdatascience.com/how-to-train-a-joint-entities-and-relation-extraction-classifier-using-bert-transformer-with-spacy-49eb08d91b5c
"""


"""Global variables and pre-request settings"""
# STRUCTURED_FILE = '/content/structured'
REL_BATCH_SIZE = 100
pd.set_option('display.max_colwidth', 200)

"""Load fundamental models and corpus"""
WordNetLemmatizer = WordNetLemmatizer()
SnowballStemmer = SnowballStemmer("english")
stopwords = set(stopwords.words('english'))
ner_path_1 = os.getcwd() + '/ner_model_trf/output/model-best/'
# ner_path_1 = os.getcwd() + '/ner_model_lg/'
nlp = spacy.load(ner_path_1)
# nlp.enable_pipe('parser')
# nlp.enable_pipe('lemmatizer')
# nlp.enable_pipe('transformer')
# nlp.add_pipe('tagger', before='ner')
# nlp.add_pipe('parser', before='ner')
# nlp.add_pipe('attribute_ruler')
# nlp.add_pipe('lemmatizer')

"""Load corpus"""
total_corpus = corpus_loader(ARTICLE_PATH)
# total_corpus = corpus_loader(STRUCTURED_FILE)


def preprocessing(text):
    """Tokenization, Pos_tagging, Lemmatization, Stemming"""
    # v_sen_token = [nltk.word_tokenize(s) for s in nltk.sent_tokenize(text)]
    v_sen_token = nltk.sent_tokenize(text)
    v_token = nltk.word_tokenize(text)
    v_pos_tag = nltk.pos_tag(v_token)  # Penn Treebank Tag Set
    v_token_lemm = [WordNetLemmatizer.lemmatize(tag[0], pos=lemmatization_tag_map[tag[1]]) for tag in v_pos_tag]  # add pos tag to imporove performance
    v_token_stem = [SnowballStemmer.stem(tk) for tk in v_token]
    return v_sen_token, v_token, v_pos_tag, v_token_lemm, v_token_stem


def save_unannotated_sen(unannotated_file_path, num_articles):
    with open(unannotated_file_path, 'w') as f_:
        for atc_idx in range(num_articles):
            sen_l = nltk.sent_tokenize(total_corpus[atc_idx]['content'])
            for sen in sen_l:
                f_.write(sen + '\n\r')


def ner(nlp_obj, corpus):
    """NER for a single sentence."""
    doc = nlp_obj(corpus)
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]


def ner_sen_batch(nlp_obj, sen_l):
    """
    NER for a list of sentence.
    :param nlp_obj: nlp handler
    :param sen_l: sentence list
    :return: list<list> of named entities /  list of docs
    """
    sen_ne = []
    doc_l = []
    for sen in sen_l:
        doc = nlp_obj(sen)
        sen_ne.append([(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents])
        doc_l.append(doc)
    return sen_ne, doc_l


def ner_article_batch(nlp_obj, corpus):
    """NER for a list of articles."""
    for a_idx in range(len(corpus)):
        doc = nlp_obj(corpus[a_idx]['content'])
        corpus[a_idx]['entities'] = [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]
        if a_idx % 10 == 0:
            print('NER index: ', a_idx)


def relation_extraction_person_team(doc, ne):
    person_team_relation = []
    matches = matcher_1(doc)
    if matches:
        if matches[0][0] == 1:
            # print(doc)
            ne_person_org_df = pd.DataFrame(columns=['index', 'person', 'org'])
            ne_person_date_df = pd.DataFrame(columns=['index', 'date'])
            for rule, m_list in matches:
                if rule == 1:
                    ne_person_org_df.loc[ne_person_org_df.shape[0]] = [m_list[0], m_list[-2], m_list[-1]]
                elif rule == 2:
                    ne_person_date_df.loc[ne_person_date_df.shape[0]] = [m_list[0], m_list[-1]]
            rel_df = pd.merge(ne_person_org_df, ne_person_date_df, on='index', how='left')
            # print(rel_df)

            for i in rel_df.iterrows():
                ne_person = [e[0] for e in ne if i[1][1] >= e[4] and i[1][1] < e[5]]
                ne_org = [e[0] for e in ne if i[1][2] >= e[4] and i[1][2] < e[5]]
                ne_date = [e[0] for e in ne if i[1][3] >= e[4] and i[1][3] < e[5]]
                if len(matches[0][1]) > 2:
                    predicate = doc[i[1][0]].lemma_
                else:
                    predicate = None
                triple = (ne_person[0], predicate, ne_org[0], ne_date[0] if ne_date else None)
                person_team_relation.append(triple)
                # print(triple)
            person_team_relation = list(set(person_team_relation))
            # print(relation_list_1)
    return person_team_relation


def relation_extraction_team_team(doc, ne):
    team_team_relation = []
    matches = matcher_2(doc)
    if matches:
        if matches[0][0] == 1:
            # print(doc)
            ne_org_org_df = pd.DataFrame(columns=['index', 'org1', 'org2'])
            ne_org_date_df = pd.DataFrame(columns=['index', 'date'])
            for rule, m_list in matches:
                if rule == 1 and m_list[-1] != m_list[-2]:
                    ne_org_org_df.loc[ne_org_org_df.shape[0]] = [m_list[0], m_list[-2], m_list[-1]]
                elif rule == 2:
                    ne_org_date_df.loc[ne_org_date_df.shape[0]] = [m_list[0], m_list[-1]]
            rel_df = pd.merge(ne_org_org_df, ne_org_date_df, on='index', how='left')
            # print(rel_df)

            for i in rel_df.iterrows():
                if abs(i[1][1] - i[1][2]) >= 1:
                    ne_org_1 = [e[0] for e in ne if i[1][1] >= e[4] and i[1][1] < e[5]]
                    ne_org_2 = [e[0] for e in ne if i[1][2] >= e[4] and i[1][2] < e[5]]
                    ne_date = [e[0] for e in ne if i[1][3] >= e[4] and i[1][3] < e[5]]
                    predicate_idx = [e[4] for e in ne if i[1][2] >= e[4] and i[1][2] < e[5]]
                    predicate = doc[predicate_idx[0] - 1].lemma_
                    if ne_org_1[0].lower() != ne_org_2[0].lower():
                        triple = (ne_org_1[0], predicate, ne_org_2[0], ne_date[0] if ne_date else None)
                        team_team_relation.append(triple)
                    # print(triple)
            team_team_relation = list(set(team_team_relation))
            # print(relation_list_1)
    return team_team_relation


def relation_extraction_person_score(doc, ne):
    person_score_relation = []
    matches = matcher_3(doc)
    if matches:
        if matches[0][0] == 1:
            # print(doc)
            ne_org_org_df = pd.DataFrame(columns=['index', 'person', 'object', 'score'])
            ne_org_date_df = pd.DataFrame(columns=['index', 'date'])
            for rule, m_list in matches:
                if rule == 1:
                    ne_org_org_df.loc[ne_org_org_df.shape[0]] = [m_list[0], m_list[-3], m_list[-2], m_list[-1]]
                elif rule == 2:
                    ne_org_date_df.loc[ne_org_date_df.shape[0]] = [m_list[0], m_list[-1]]
            rel_df = pd.merge(ne_org_org_df, ne_org_date_df, on='index', how='left')
            # print(rel_df)

            for i in rel_df.iterrows():
                ne_object_index = i[1][2]
                ne_score_index = i[1][3]
                if ne_score_index < ne_object_index:
                    ne_person = [e[0] for e in ne if i[1][1] >= e[4] and i[1][1] < e[5]]
                    ne_object = [e[0] for e in ne if i[1][2] >= e[4] and i[1][2] < e[5]]
                    ne_score = [e[0] for e in ne if i[1][3] >= e[4] and i[1][3] < e[5]]
                    ne_date = [e[0] for e in ne if i[1][4] >= e[4] and i[1][4] < e[5]]
                    if not ne_object:
                        ne_object = [doc[i[1][2]].orth_]
                    if not ne_score:
                        ne_score = [doc[i[1][3]].orth_]
                    predicate = doc[i[1][0]].lemma_
                    if ne_person[0].lower() != ne_score[0].lower():
                        triple = (ne_person[0], predicate, ne_object[0], ne_score[0], ne_date[0] if ne_date else None)
                        person_score_relation.append(triple)
            person_score_relation = list(set(person_score_relation))
    return person_score_relation


def relation_extraction_team_score(doc, ne):
    team_score_relation = []
    matches = matcher_4(doc)
    if matches:
        if matches[0][0] == 1:
            # print(doc)
            ne_org_org_df = pd.DataFrame(columns=['index', 'team', 'score'])
            ne_org_date_df = pd.DataFrame(columns=['index', 'date'])
            for rule, m_list in matches:
                if rule == 1:
                    ne_org_org_df.loc[ne_org_org_df.shape[0]] = [m_list[0], m_list[-2], m_list[-1]]
                elif rule == 2:
                    ne_org_date_df.loc[ne_org_date_df.shape[0]] = [m_list[0], m_list[-1]]
            rel_df = pd.merge(ne_org_org_df, ne_org_date_df, on='index', how='left')
            # print(rel_df)

            for i in rel_df.iterrows():
                ne_team = [e[0] for e in ne if i[1][1] >= e[4] and i[1][1] < e[5]]
                ne_score = [e[0] for e in ne if i[1][2] >= e[4] and i[1][2] < e[5]]
                ne_date = [e[0] for e in ne if i[1][3] >= e[4] and i[1][3] < e[5]]
                predicate = doc[i[1][0]].lemma_

                triple = (ne_team[0], predicate, ne_score[0], ne_date[0] if ne_date else None)
                team_score_relation.append(triple)
            team_score_relation = list(set(team_score_relation))
    return team_score_relation


def relation_extraction_person_injury(doc, ne):
    person_injury_relation = []
    matches = matcher_5(doc)
    if matches:
        if matches[0][0] == 1:
            # print(doc)
            ne_person_injury_df = pd.DataFrame(columns=['index', 'person', 'injury'])
            ne_person_injury_df_2 = pd.DataFrame(columns=['index', 'person', 'injury', 'complain1'])
            ne_person_injury_df_3 = pd.DataFrame(columns=['index', 'person', 'injury', 'adp', 'complain2'])
            ne_org_date_df = pd.DataFrame(columns=['index', 'date'])
            for rule, m_list in matches:
                if rule == 1:
                    ne_person_injury_df.loc[ne_person_injury_df.shape[0]] = [m_list[0], m_list[-2], m_list[-1]]
                elif rule == 2:
                    ne_person_injury_df_2.loc[ne_person_injury_df_2.shape[0]] = [m_list[0], m_list[-3], m_list[-2], m_list[-1]]
                elif rule == 3:
                    ne_person_injury_df_3.loc[ne_person_injury_df_3.shape[0]] = [m_list[0], m_list[-4], m_list[-3], m_list[-2], m_list[-1]]
                elif rule == 4:
                    ne_org_date_df.loc[ne_org_date_df.shape[0]] = [m_list[0], m_list[-1]]
            rel_df = pd.merge(ne_person_injury_df, ne_org_date_df, on='index', how='left')
            rel_df = pd.merge(rel_df, ne_person_injury_df_2, on=['index', 'person', 'injury'], how='left')
            rel_df = pd.merge(rel_df, ne_person_injury_df_3, on=['index', 'person', 'injury'], how='left')
            # print(rel_df)

            for i in rel_df.iterrows():
                ne_person = [e[0] for e in ne if i[1][1] >= e[4] and i[1][1] < e[5]]
                ne_injury = doc[i[1][2]].orth_
                if doc[i[1][2] - 1].dep_ == 'amod':
                    ne_injury_adj = doc[i[1][2] - 1].orth_
                else:
                    ne_injury_adj = None
                if i[1][4] >= 0:
                    ne_injury_complain = doc[i[1][4]].orth_
                elif i[1][6] >= 0:
                    ne_injury_complain = doc[i[1][6]].orth_
                elif i[1][2] + 2 <= len(doc):
                    if doc[i[1][2] + 1].pos_ == 'NOUN':
                        ne_injury_complain = doc[i[1][2] + 1].orth_
                    else:
                        ne_injury_complain = None
                else:
                    ne_injury_complain = None

                ne_date = [e[0] for e in ne if i[1][3] >= e[4] and i[1][3] < e[5]]
                predicate = doc[i[1][0]].lemma_

                triple = (ne_person[0], predicate, ne_injury, ne_injury_adj, ne_injury_complain, ne_date[0] if ne_date else None)
                person_injury_relation.append(triple)
            person_injury_relation = list(set(person_injury_relation))
    return person_injury_relation


def relation_extraction(doc):
    """

    :param doc:
    :return: ne (list), relation (list)
    """
    print(doc)
    person_team_relation = []
    team_team_relation = []
    person_score_relation = []
    team_score_relation = []
    person_injury_relation = []
    ne = [(ent.text, ent.label_, ent.start_char, ent.end_char, ent.start, ent.end) for ent in doc.ents]
    if ne:
        ne_df = pd.DataFrame(ne)

        person_flg = (ne_df[1] == 'PERSON').sum()
        org_flg = (ne_df[1].isin(['GPE', 'ORG'])).sum()
        score_flg = (ne_df[1].isin(['SCORE', 'CARDINAL', 'PERCENT'])).sum()

        # Extract relation: Person in Org/GPE
        if person_flg and org_flg:
            person_team_relation = relation_extraction_person_team(doc, ne)

        # Extract relation: team-team
        if org_flg >= 2:
            team_team_relation = relation_extraction_team_team(doc, ne)

        # Extract relation: score
        if score_flg:
            person_score_relation = relation_extraction_person_score(doc, ne)
            team_score_relation = relation_extraction_team_score(doc, ne)

        # Extract relation: injury
        if person_flg:
            person_injury_relation = relation_extraction_person_injury(doc, ne)

    return ne, [person_team_relation, team_team_relation, person_score_relation, team_score_relation, person_injury_relation]


"""Preprocessing to save some unannotated file in to file after sentence tokenization"""
# unannot_file_path = 'annotation_pre_sentence.txt'
# num_sens = 100
# save_unannotated_sen(save_unannotated_sen, num_sens)

"""Preprocessing: tokenization, POS tagging, lemmatization and stemming"""
# example_text = total_corpus[400]['content']
# sen_token, token, pos_tag, token_lemm, token_stem = preprocessing(example_text)

"""Named Entity Recognition by pre-trained transformer-based model"""
# ne_list, doc_list = ner_sen_batch(nlp, sen_token)

"""Relation Extraction by dependency parsing"""
# for tok in doc_list[1]:
#     print(tok.text, "-->", tok.dep_, "-->", tok.pos_, "-->", tok.tag_, "-->", tok.is_stop)


# Matcher: Person in Org/GPE
pattern_1 = [pattern_person_team_rel_1, pattern_person_team_rel_2, pattern_person_team_rel_3, pattern_team_person_rel_1,
             pattern_team_person_rel_2, pattern_team_person_rel_3, pattern_team_person_rel_4, pattern_team_person_rel_5]
pattern_2 = [pattern_team_team_rel_1]
pattern_3 = [pattern_person_score_rel_1, pattern_person_score_rel_2]
pattern_4 = [pattern_team_score_rel_1]
pattern_5_1 = [pattern_person_injury_rel_1]
pattern_5_2 = [pattern_person_injury_rel_2]
pattern_5_3 = [pattern_person_injury_rel_3]

matcher_1 = DependencyMatcher(nlp.vocab)
matcher_1.add(1, pattern_1)
matcher_1.add(2, [pattern_time_1, pattern_time_2])

matcher_2 = DependencyMatcher(nlp.vocab)
matcher_2.add(1, pattern_2)
matcher_2.add(2, [pattern_time_1, pattern_time_2])

matcher_3 = DependencyMatcher(nlp.vocab)
matcher_3.add(1, pattern_3)
matcher_3.add(2, [pattern_time_1, pattern_time_2])

matcher_4 = DependencyMatcher(nlp.vocab)
matcher_4.add(1, pattern_4)
matcher_4.add(2, [pattern_time_1, pattern_time_2])

matcher_5 = DependencyMatcher(nlp.vocab)
matcher_5.add(1, pattern_5_1)
matcher_5.add(2, pattern_5_2)
matcher_5.add(3, pattern_5_3)
matcher_5.add(4, [pattern_time_1, pattern_time_2])


batch_cnt = 0
num_batches = np.int(len(total_corpus) / REL_BATCH_SIZE)
start_batch_index = 99
end_batch_index = 99

for b_idx in np.linspace(start_batch_index, end_batch_index, end_batch_index - start_batch_index + 1, dtype=np.int):
    text_l = []
    for c_idx in range(REL_BATCH_SIZE):
        idx = b_idx * REL_BATCH_SIZE + c_idx
        if idx < len(total_corpus):
            m = re.search('FanDuel', total_corpus[idx]['title'], re.IGNORECASE)  # Delete FanDuel related info
            if not m:
                text_l.append(total_corpus[idx]['title'] + '.' + total_corpus[idx]['content'])
            else:
                text_l.append('')

    doc_generator = nlp.pipe(text_l)
    d_idx = 0
    for doc in doc_generator:
        total_d_index = b_idx * REL_BATCH_SIZE + d_idx
        print("Index:", total_d_index)
        ne, ne_rel = relation_extraction(doc)
        total_corpus[total_d_index]['ne'] = ne
        total_corpus[total_d_index]['ne_rel'] = ne_rel
        d_idx += 1

    del doc_generator
    gc.collect()

corpus_saver(STRUCTURED_FILE, total_corpus, batch_size=1000)


