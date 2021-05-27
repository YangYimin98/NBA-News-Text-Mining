from utils import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
import os
from gensim import corpora, models, similarities
from gensim.models.coherencemodel import CoherenceModel
import numpy as np
import gc

"""
Reference
https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html
https://radimrehurek.com/gensim/models/tfidfmodel.html
https://radimrehurek.com/gensim_3.8.3/models/coherencemodel.html
https://stackoverflow.com/questions/54762690/what-is-the-meaning-of-coherence-score-0-4-is-it-good-or-bad
https://towardsdatascience.com/evaluate-topic-model-in-python-latent-dirichlet-allocation-lda-7d57484bb5d0
"""

DICTIONARY_PATH = 'total.dict'
INDEX_PATH = 'total.index'
TFIDF_PATH = 'total.tfidf'
LDA_PATH = 'total.lda'
STOPWORDS_PATH = 'stopwords.txt'


def get_lemma(text):
    v_token = nltk.word_tokenize(text)
    v_pos_tag = nltk.pos_tag(v_token)  # Penn Treebank Tag Set
    v_token_lemm = [WordNetLemmatizer.lemmatize(tag[0], pos=lemmatization_tag_map[tag[1]]) for tag in v_pos_tag if
                    tag[0].isalpha()]
    v_token_lemm = [w for w in v_token_lemm if w.lower() not in stop_words]
    return v_token_lemm


def query(keyword):
    query_lemma = get_lemma(keyword)
    query_vec = dictionary.doc2bow(query_lemma)
    sim = index[tfidf[query_vec]]
    res = sorted(enumerate(sim), key=lambda item: -item[1])
    return res


WordNetLemmatizer = WordNetLemmatizer()
SnowballStemmer = SnowballStemmer("english")
stop_words = []
with open(STOPWORDS_PATH, 'r') as f:
    for line in f.readlines():
        stop_words.append(line.strip('\n'))
# stop_words = set(stopwords.words('english'))


"""TF-IDF search enhancement"""

# Load corpus
total_corpus = corpus_loader(NORMALIZED_FILE)

# preprocessing
content_word = []
for c_index, c in enumerate(total_corpus):
    print("Lemmatization index: ", c_index)
    content_word.append(get_lemma(c['title'] + '.' + c['content']))

# make bag-of-words

if os.path.exists(DICTIONARY_PATH):
    dictionary = corpora.Dictionary.load(DICTIONARY_PATH)
else:
    dictionary = corpora.Dictionary(content_word)
    dictionary.save(DICTIONARY_PATH)

corpus = [dictionary.doc2bow(doc) for doc in content_word]

if os.path.exists(TFIDF_PATH):
    tfidf = models.TfidfModel.load(TFIDF_PATH)
else:
    tfidf = models.TfidfModel(corpus)
    tfidf.save(TFIDF_PATH)

corpus_tfidf = tfidf[corpus]

if os.path.exists(INDEX_PATH):
    index = similarities.SparseMatrixSimilarity.load(INDEX_PATH)
else:
    index = similarities.SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary.keys()))
    index.save(INDEX_PATH)

# query_keyword = 'Clippers, Kabengele, Kings'
# res = query(query_keyword)
# res[:10]


"""Topic modeling"""
# preprocessing
title_word = []
for c_index, c in enumerate(total_corpus):
    print("Lemmatization title index: ", c_index)
    title_word.append(get_lemma(c['title']))

dictionary2 = corpora.Dictionary(title_word)
corpus2 = [dictionary2.doc2bow(doc) for doc in title_word]
tfidf2 = models.TfidfModel(corpus2)
corpus_tfidf2 = tfidf2[corpus2]
index2 = similarities.SparseMatrixSimilarity(corpus_tfidf2, num_features=len(dictionary2.keys()))

lda = models.LdaModel(corpus_tfidf2, id2word=dictionary2, num_topics=30, iterations=2000)
# lda = models.LdaModel(corpus2, id2word=dictionary2, num_topics=30, iterations=100)
lda.print_topics(30)

cm = CoherenceModel(model=lda, corpus=corpus_tfidf2, coherence='u_mass')
coherence = cm.get_coherence()

# test the corpus
test_index = 1000
test_title = title_word[test_index]
corpus_test = [dictionary2.doc2bow(doc) for doc in [test_title]]
corpus_test_tfidf2 = tfidf2[corpus_test]
test_topic = lda.get_document_topics(corpus_test_tfidf2)
print(total_corpus[test_index]['title'])
print(test_topic[0])


for i in range(10):
    num_topics = np.int((i + 1) * 1)
    lda = models.LdaModel(corpus_tfidf2, id2word=dictionary2, num_topics=num_topics, iterations=100)
    # lda.print_topics(30)
    cm = CoherenceModel(model=lda, corpus=corpus_tfidf2, coherence='u_mass')
    cm = CoherenceModel(model=lda, texts=title_word, dictionary=dictionary2, coherence='c_v')
    coherence = cm.get_coherence()
    print('Topic number', num_topics, 'Coherence:', coherence)

gc.collect()