from NBA_Scraper import *
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tag.stanford import StanfordNERTagger
from nltk.sem import relextract
from nltk.corpus import stopwords
from nltk.chunk import tree2conlltags


word_net_lemmatizer = WordNetLemmatizer()
snowballstemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))


class InformationExtraction(object):
    def __init__(self, corpus):
        self.corpus = corpus

    def sentences_cleaner(self, corpus):
        """Data Cleaner to substitute some useless words"""
        for c_idx in corpus:
            c_idx['content'] = re.sub(r'\s+', ' ', c_idx['content'])
            c_idx['content'] = re.sub(r'\s\.', '.', c_idx['content'])
        corpus_using = [str(i) for i in corpus]
        corpus_final = ''.join(corpus_using)
        return corpus_final

    def tokenlization_and_stemming(self, corpus):
        """Tokenization for NER and Stemming for Sentiment Analysis and Topic Modelling"""
        tokens = word_tokenize(corpus)
        filter_tokens = []
        snowball_stemmer_tokens = []
        wordnet_tokens = []
        for token in tokens:
            snowball_stemmer_token = snowballstemmer.stem(token)
            # not necessary, but we want to see what happens there
            wordnet_token = word_net_lemmatizer.lemmatize(token)
            if token not in stop_words:
                snowball_stemmer_tokens.append(snowball_stemmer_token)
                wordnet_tokens.append(wordnet_token)
                filter_tokens.append(token)

        return filter_tokens, snowball_stemmer_tokens, wordnet_tokens

    def pos_tagging(self, corpus):
        """pos tagging for every token"""
        filter_tokens, snowball_stemmer_tokens, wordnet_tokens = self.tokenlization_and_stemming(
            corpus)
        filter_tokens = nltk.pos_tag(filter_tokens)
        return filter_tokens

    def chunking_and_iob_tagging(self, corpus):
        """Implement noun phrase chunking to identify named entities using a regular expression
        consisting of rules and put IOB tagging on the tokens"""
        pattern = 'NP: {<DT>?<JJ>*<NN>}'
        rp = nltk.RegexpParser(pattern)
        res = rp.parse(self.pos_tagging(corpus))
        # res.draw()
        iob_tagged = tree2conlltags(res)
        return iob_tagged

    def named_entity_rocognition(self, corpus):
        """Implement named entity recognition, and list the results"""
        entity_list = []
        for entity in nltk.sent_tokenize(corpus):
            for chunk in nltk.ne_chunk(
                    nltk.pos_tag(nltk.word_tokenize(entity))):
                if hasattr(chunk, 'label'):
                    name = chunk[0][0]
                    type = chunk.label()
                    # res = chunk.label(), ' '.join(c[0] for c in chunk)
                    entity_list.append((name, type))
        entity_list = pd.DataFrame(entity_list, columns=['Entity Name', 'Entity Type'])
        return entity_list

    def ner_relationship_extraction(self, corpus):
        """Extract relationships between Ners"""
        pass




# total_corpus = corpus_loader(ARTICLE_PATH)
# filter_tokens, snowball_stemmer_tokens, wordnet_tokens = tokenlization_and_stemming(sentences_cleaner(total_corpus))
test_text = \
    'The Sixers enter Monday on December 2021-04-02 in German riding a four-game win streak ' \
    'with Joel Embiid averaging 34.5 points, 11.0 rebounds and ' \
    '1.5 blocks during this stretch. Embiid has 12 games this season with at least 30 points and 10 rebounds – second ' \
    'most in the league behind Giannis Antetokounmpo (14 such games).Stephen Curry has scored at least 30 points in ' \
    '10 straight games and has made at least 10 3-pointers in three of his last four games. Curry has as many games' \
    ' with 10 or more 3-pointers made this season (five) as the rest of the NBA combined. Over his past five games,' \
    ' Stephen Curry has made 44 3-pointers (more than any player in a five-game stretch ever) and has shot ' \
    '54.5% from beyond the arc.'

IE = InformationExtraction(test_text)
filter_tokens, snowball_stemmer_tokens, wordnet_tokens = IE.tokenlization_and_stemming(
    test_text)
print('Tokens are: {0}'.format(filter_tokens))
print('Stemming Tokens are: {0}'. format(snowball_stemmer_tokens))
print('Lemmatization Tokens are: {0}'. format(wordnet_tokens))
print(IE.pos_tagging(test_text))
print(IE.chunking_and_iob_tagging(test_text))
print(IE.named_entity_rocognition(test_text))

"""using Standford Ner library to extract date entities, not succeed now"""
# st = StanfordNERTagger('/Users/yangyimin/Downloads/stanford-ner-2020-11-17/classifiers
# /english.all.3class.distsim.crf.ser.gz',
#                '/Users/yangyimin/Downloads/stanford-ner-2020-11-17/stanford-ner.jar')
# print(st.tag('Rami Eid is studying at Stony Brook University in NY on Sunday in March in 2020-04-04' .split()))
