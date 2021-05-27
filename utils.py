from nltk.corpus import wordnet as wn
import os
import ujson


STRUCTURED_FILE = '/content/structured'
NORMALIZED_FILE = '/content/normalized'

# reference: https://stackoverflow.com/questions/5364493/lemmatizing-pos-tagged-words-with-nltk
# WordNet POS tags are: NOUN = 'n', ADJ = 's', VERB = 'v', ADV = 'r', ADJ_SAT = 'a'
# Penn Treebank Tag Set
lemmatization_tag_map = {
    'CC': wn.NOUN,  # coordin. conjunction (and, but, or)
    'CD': wn.NOUN,  # cardinal number (one, two)
    'DT': wn.NOUN,  # determiner (a, the)
    'EX': wn.ADV,  # existential ‘there’ (there)
    'FW': wn.NOUN,  # foreign word (mea culpa)
    'IN': wn.ADV,  # preposition/sub-conj (of, in, by)
    'JJ': wn.ADJ,  # adjective (yellow)
    'JJR': wn.ADJ,  # adj., comparative (bigger)
    'JJS': wn.ADJ,  # adj., superlative (wildest)
    'LS': wn.NOUN,  # list item marker (1, 2, One)
    'MD': wn.NOUN,  # modal (can, should)
    'NN': wn.NOUN,  # noun, sing. or mass (llama)
    'NNS': wn.NOUN,  # noun, plural (llamas)
    'NNP': wn.NOUN,  # proper noun, sing. (IBM)
    'NNPS': wn.NOUN,  # proper noun, plural (Carolinas)
    'PDT': wn.ADJ,  # predeterminer (all, both)
    'POS': wn.NOUN,  # possessive ending (’s )
    'PRP': wn.NOUN,  # personal pronoun (I, you, he)
    'PRP$': wn.NOUN,  # possessive pronoun (your, one’s)
    'RB': wn.ADV,  # adverb (quickly, never)
    'RBR': wn.ADV,  # adverb, comparative (faster)
    'RBS': wn.ADV,  # adverb, superlative (fastest)
    'RP': wn.ADJ,  # particle (up, off)
    'SYM': wn.NOUN,  # symbol (+,%, &)
    'TO': wn.NOUN,  # “to” (to)
    'UH': wn.NOUN,  # interjection (ah, oops)
    'VB': wn.VERB,  # verb base form (eat)
    'VBD': wn.VERB,  # verb past tense (ate)
    'VBG': wn.VERB,  # verb gerund (eating)
    'VBN': wn.VERB,  # verb past participle (eaten)
    'VBP': wn.VERB,  # verb non-3sg pres (eat)
    'VBZ': wn.VERB,  # verb 3sg pres (eats)
    'WDT': wn.NOUN,  # wh-determiner (which, that)
    'WP': wn.NOUN,  # wh-pronoun (what, who)
    'WP$': wn.NOUN,  # possessive (wh- whose)
    'WRB': wn.NOUN,  # wh-adverb (how, where)
    '$': wn.NOUN,  # dollar sign ($)
    '#': wn.NOUN,  # pound sign (#)
    '“': wn.NOUN,  # left quote (‘ or “)
    '”': wn.NOUN,  # right quote (’ or ”)
    '(': wn.NOUN,  # left parenthesis ([, (, {, <)
    ')': wn.NOUN,  # right parenthesis (], ), }, >)
    ',': wn.NOUN,  # comma (,)
    '.': wn.NOUN,  # sentence-final punc (. ! ?)
    ':': wn.NOUN,  # mid-sentence punc (: ; ... – -)
    "''": wn.NOUN
}


def corpus_loader(content_file):
    content_path = os.getcwd() + content_file
    file_idx = 0
    corpus = []
    while os.path.exists(content_path + str(file_idx) + '.txt'):
        print('Checking file index: {}'.format(file_idx))
        with open(content_path + str(file_idx) + '.txt', 'r') as f:
            f_content = ujson.load(f)
        for t_idx in f_content:
            corpus.append(t_idx)
        file_idx += 1
    return corpus


def corpus_saver(content_file, content, batch_size):
    content_path = os.getcwd() + content_file
    length = len(content)
    file_idx = 0
    batch_idx = 0
    batch_content = []
    for i_idx, i_content in enumerate(content):
        batch_content.append(i_content)
        batch_idx += 1
        if batch_idx == batch_size or length == i_idx + 1:
            # save
            with open(content_path + str(file_idx) + '.txt', 'w') as fcw:
                fcw.write(ujson.dumps(batch_content))
                batch_content = []
                file_idx += 1
                batch_idx = 0