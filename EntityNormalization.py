import numpy as np
import pandas as pd
from py_stringmatching import MongeElkan
# from py_stringmatching import Levenshtein
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from strsimpy.cosine import Cosine
from strsimpy.jaccard import Jaccard
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
from strsimpy.overlap_coefficient import OverlapCoefficient
from strsimpy.sorensen_dice import SorensenDice
import networkx as nx
from utils import *

"""
Reference: 
similarity 
https://awesomeopensource.com/project/luozhouyang/python-string-similarity
https://anhaidgroup.github.io/py_stringmatching/v0.3.x/MongeElkan.html

clustering
https://scikit-learn.org/stable/modules/clustering.html#overview-of-clustering-methods

"""
NUM_ENTITIES = 500
ENTITY_CLUSTERING_THRESHOLD = 0.81
NUM_NODES = 100

def ne_similarity(ne_list, limit):

    ne_cnt = pd.value_counts(ne_list)
    ne_top = ne_cnt.keys()[:limit]
    ne_sim_jaccard = np.zeros((limit, limit))
    ne_sim_overlap = np.zeros((limit, limit))
    ne_sim_dice = np.zeros((limit, limit))
    ne_sim_cosine = np.zeros((limit, limit))
    ne_sim_monge_elkan = np.zeros((limit, limit))
    ne_sim_normalized_levenshtein = np.zeros((limit, limit))
    idx = 0
    for i in range(limit):
        for j in range(i+1, limit):
            idx += 1
            e_i = ne_top[i].replace("The ", '').replace("the ", '')
            e_j = ne_top[j].replace("The ", '').replace("the ", '')
            # ne_sim_jaccard[i, j] = jaccard.similarity(e_i, e_j)
            # ne_sim_overlap[i, j] = overlap.similarity(e_i, e_j)
            # ne_sim_dice[i, j] = dice.similarity(e_i, e_j)
            # ne_sim_cosine[i, j] = cosine.similarity(e_i, e_j)
            ne_sim_monge_elkan[i, j] = monge_elkan.get_raw_score([e_i], [e_j])
            ne_sim_normalized_levenshtein[i, j] = normalized_levenshtein.similarity(e_i, e_j)

    return ne_top, ne_cnt, [ne_sim_jaccard, ne_sim_overlap, ne_sim_dice, ne_sim_cosine, ne_sim_monge_elkan, ne_sim_normalized_levenshtein]


def ne_clustering(ne, ne_cnt, similarity_matrix, threshold):
    """NE clustering"""
    clusters = []
    clusters_map = {}
    reverse_ordering = reverse_cuthill_mckee(csr_matrix(similarity_matrix >= threshold), symmetric_mode=False)

    cluster_start_index = 0
    cluster_end_index = 0

    # cluster all similar entities when the similarity is larger than threshold
    for i in range(len(reverse_ordering) - 1):
        sim = np.max([similarity_matrix[reverse_ordering[i], reverse_ordering[i + 1]],
                      similarity_matrix[reverse_ordering[i + 1], reverse_ordering[i]]])
        if sim >= threshold:
            cluster_end_index += 1
        else:
            cluster_index = reverse_ordering[cluster_start_index: cluster_end_index + 1]
            clusters.append(cluster_index)
            cluster_end_index += 1
            cluster_start_index = cluster_end_index
        # print(ne[reverse_ordering[i]], ne[reverse_ordering[i + 1]], sim)

    # find the cluster representation for each cluster
    # symmetric_similarity_matrix = similarity_matrix = similarity_matrix.T
    for c in clusters:
        if len(c) > 1:
            c_ele_cnt = []
            for c_ele in c:
                c_ele_cnt.append(ne_cnt[c_ele])
            repr_index = c[np.argmax(c_ele_cnt)]
            for c_ele in c:
                clusters_map[ne[c_ele]] = ne[repr_index]
        else:
            clusters_map[ne[c[0]]] = ne[c[0]]
    return clusters, clusters_map


"""Load corpus"""
total_corpus = corpus_loader(STRUCTURED_FILE)
# corpus_saver(STRUCTURED_FILE, total_corpus, batch_size=1000)

"""Initialize similarity objects"""
jaccard = Jaccard(2)
overlap = OverlapCoefficient()
dice = SorensenDice()
cosine = Cosine(2)
monge_elkan = MongeElkan()
normalized_levenshtein = NormalizedLevenshtein()

"""Initialize ne list"""
total_ne_person = []
total_ne_org = []
total_ne_gpe = []
total_ne_product = []

"""Put all named entities into list"""
i_doc = 0
for doc in total_corpus:
    ne = doc['ne']
    print('Loading article named entities: ', i_doc)
    for i_ne in ne:
        if len(i_ne[0]) > 3:
            if i_ne[1] == 'PERSON':
                total_ne_person.append(i_ne[0])
            elif i_ne[1] == 'ORG':
                total_ne_org.append(i_ne[0])
            elif i_ne[1] == 'GPE':
                total_ne_gpe.append(i_ne[0])
            elif i_ne[1] == 'PRODUCT':
                total_ne_product.append(i_ne[0])
                # ne_df.loc[ne_df.shape[0]] = [i_doc, i_ne[0], i_ne[1], i_ne[4], i_ne[5]]
    i_doc += 1

"""Calculate the similarity matrix and additional statistical information"""
ne_person, person_cnt, person_sim = ne_similarity(total_ne_person, NUM_ENTITIES)
ne_org, org_cnt, org_sim = ne_similarity(total_ne_org, NUM_ENTITIES)
ne_gpe, gpe_cnt, gpe_sim = ne_similarity(total_ne_gpe, NUM_ENTITIES)
ne_product, product_cnt, product_sim = ne_similarity(total_ne_product, NUM_ENTITIES)

org_cluster_l, org_cluster_map = ne_clustering(ne_org, org_cnt, org_sim[4], ENTITY_CLUSTERING_THRESHOLD)
gpe_cluster_l, gpe_cluster_map = ne_clustering(ne_gpe, gpe_cnt, gpe_sim[4], ENTITY_CLUSTERING_THRESHOLD)
person_cluster_l, person_cluster_map = ne_clustering(ne_person, person_cnt, person_sim[5], ENTITY_CLUSTERING_THRESHOLD)
product_cluster_l, product_cluster_map = ne_clustering(ne_product, product_cnt, product_sim[5], ENTITY_CLUSTERING_THRESHOLD)


"""Find the Top N most similar entities"""
# for i in range(NUM_ENTITIES):
#     top_n = np.argsort(- org_sim[4][i, :])[:3]
#     top_n_p = - np.sort(- org_sim[4][i, :])[:3]
#     if top_n_p[0] > 0.5:
#         print(ne_org[i], '|', ne_org[top_n[0]], ' - ', top_n_p[0], '|', ne_org[top_n[1]], ' - ', top_n_p[1], '|',
#               ne_org[top_n[2]], ' - ', top_n_p[2])
#     else:
#         print(ne_org[i])


"""Relations Normalization"""
# 0-person_team_relation
# 1-team_team_relation
# 2-person_score_relation
# 3-team_score_relation
# 4-person_injury_relation

person_team_rel = []
team_team_rel = []
person_score_rel = []
team_score_rel = []
person_injury_rel = []

print('Extract relations in to list')
for i in range(len(total_corpus)):
    person_team_rel.extend([e + [total_corpus[i]['date']] for e in total_corpus[i]['ne_rel'][0]])
    team_team_rel.extend([e + [total_corpus[i]['date']] for e in total_corpus[i]['ne_rel'][1]])
    person_score_rel.extend([e + [total_corpus[i]['date']] for e in total_corpus[i]['ne_rel'][2]])
    team_score_rel.extend([e + [total_corpus[i]['date']] for e in total_corpus[i]['ne_rel'][3]])
    person_injury_rel.extend([e + [total_corpus[i]['date']] for e in total_corpus[i]['ne_rel'][4]])

"""Replace the entity with normalized entity"""
print('Replace the entity with normalized entity')

for i in person_team_rel:
    if i[0] in person_cluster_map:
        i[0] = person_cluster_map[i[0]]
    if i[2] in gpe_cluster_map:
        i[2] = gpe_cluster_map[i[2]]
    if i[2] in org_cluster_map:
        i[2] = org_cluster_map[i[2]]

for i in team_team_rel:
    if i[0] in gpe_cluster_map:
        i[0] = gpe_cluster_map[i[0]]
    if i[0] in org_cluster_map:
        i[0] = org_cluster_map[i[0]]
    if i[2] in gpe_cluster_map:
        i[2] = gpe_cluster_map[i[2]]
    if i[2] in org_cluster_map:
        i[2] = org_cluster_map[i[2]]

for i in person_score_rel:
    if i[0] in person_cluster_map:
        i[0] = person_cluster_map[i[0]]
    if i[2] in product_cluster_map:
        i[2] = product_cluster_map[i[2]]

for i in team_score_rel:
    if i[0] in person_cluster_map:
        i[0] = person_cluster_map[i[0]]

for i in person_injury_rel:
    if i[0] in person_cluster_map:
        i[0] = person_cluster_map[i[0]]

for a_index in range(len(total_corpus)):

    ne_l = []
    for i in total_corpus[a_index]['ne']:
        if i[0] in person_cluster_map:
            ne = person_cluster_map[i[0]]
        elif i[0] in org_cluster_map:
            ne = org_cluster_map[i[0]]
        elif i[0] in gpe_cluster_map:
            ne = gpe_cluster_map[i[0]]
        elif i[0] in product_cluster_map:
            ne = product_cluster_map[i[0]]
        else:
            ne = i[0]
        ne_l.append([ne, i[1], i[2], i[3], i[4], i[5]])
    total_corpus[a_index]['ne_norm'] = ne_l

    for ne_rel_idx, ne_rel in enumerate(total_corpus[a_index]['ne_rel']):
        if ne_rel_idx == 0:
            for i in ne_rel:
                if i[0] in person_cluster_map:
                    i[0] = person_cluster_map[i[0]]
                if i[2] in gpe_cluster_map:
                    i[2] = gpe_cluster_map[i[2]]
                if i[2] in org_cluster_map:
                    i[2] = org_cluster_map[i[2]]

        if ne_rel_idx == 1:
            for i in ne_rel:
                if i[0] in gpe_cluster_map:
                    i[0] = gpe_cluster_map[i[0]]
                if i[0] in org_cluster_map:
                    i[0] = org_cluster_map[i[0]]
                if i[2] in gpe_cluster_map:
                    i[2] = gpe_cluster_map[i[2]]
                if i[2] in org_cluster_map:
                    i[2] = org_cluster_map[i[2]]

        if ne_rel_idx == 2:
            for i in ne_rel:
                if i[0] in person_cluster_map:
                    i[0] = person_cluster_map[i[0]]
                if i[2] in product_cluster_map:
                    i[2] = product_cluster_map[i[2]]

        if ne_rel_idx == 3:
            for i in ne_rel:
                if i[0] in person_cluster_map:
                    i[0] = person_cluster_map[i[0]]

        if ne_rel_idx == 4:
            for i in ne_rel:
                if i[0] in person_cluster_map:
                    i[0] = person_cluster_map[i[0]]

print('Load relations into dataframe')
person_team_df = pd.DataFrame(person_team_rel, columns=['person', 'verb', 'team', 'time', 'date'])
team_team_df = pd.DataFrame(team_team_rel, columns=['team', 'verb', 'team2', 'time', 'date'])
person_score_df = pd.DataFrame(person_score_rel, columns=['person', 'verb', 'object', 'score', 'time', 'date'])
team_score_df = pd.DataFrame(team_score_rel, columns=['person', 'verb', 'score', 'time', 'date'])
person_injury_df = pd.DataFrame(person_injury_rel, columns=['person', 'verb', 'injury', 'adj', 'complain', 'time', 'date'])

# person_team_df.groupby(by=['team', 'person']).agg({'verb': 'count'}).sort_values(by=['verb'], ascending=False)
print('Save file into disk.')
corpus_saver(NORMALIZED_FILE, total_corpus, batch_size=1000)


"""Prepare visualization data"""
org_blacklist = ['NBPA', 'NBA', 'ESPN', 'Portland First Citizen Award']
org_blacklist_str = "'" + "','".join(org_blacklist) + "'"
date_point = ['2018-01-01', '2018-04-01', '2018-07-01', '2018-10-01', '2019-01-01', '2019-04-01', '2019-07-01', '2019-10-01', '2020-01-01', '2020-04-01', '2020-07-01', '2020-10-01', '2021-01-01, 2021-04-01']
start_date = []
end_date = []
for i_date in range(len(date_point) - 1):
    start_date.append(date_point[i_date])
    end_date.append(date_point[i_date + 1])

p_t_stat = []
for i in range(len(start_date)):
    dfpt = person_team_df.query("'{0}' <= date < '{1}' and team not in ({2})".format(start_date[i], end_date[i], org_blacklist_str)).filter(
        items=['person', 'team', 'date']).groupby(['person', 'team']).count().sort_values(by=['date'], ascending=False)
    p_t_stat.append(dfpt)

node_list = []
edge_list = []
for sub_set in p_t_stat:
    element = []
    person_set = []
    team_set = []
    total_set = []

    edges = []
    i_record = 0

    for record in sub_set.iterrows():
        if record[0][0] not in person_set:
            person_set.append(record[0][0])
            total_set.append(record[0][0])
            # element.append({'data': {'id': person_set.index(record[0][0]), 'label': record[0][0]}})
        if record[0][1] not in team_set:
            team_set.append(record[0][1])
            total_set.append(record[0][1])
            # element.append({'data': {'id': team_set.index(record[0][1]), 'label': record[0][1]}})

        # node_index and edge_cnt
        # node_row = person_set.index(record[0][0])
        # node_col = team_set.index(record[0][1])
        node_row = total_set.index(record[0][0])
        node_col = total_set.index(record[0][1])
        edge_cnt = record[1][0]
        edges.append((node_row, node_col, edge_cnt))
        # element.append({'data': {'source': node_row, 'target': node_col}})
        i_record += 1
        if i_record == NUM_NODES:
            break
    # element_list.append(element)
    edge_list.append(edges)
    node_list.append(total_set)
    print(i_record)
