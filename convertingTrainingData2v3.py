import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import jsonlines

nlp = spacy.blank("en")  # load a new spacy model
db = DocBin()  # create a DocBin object
ANNOTATED_FILE_PATH = 'annotated_corpus.jsonl'

ANNOTATED_DATA = []
with open(ANNOTATED_FILE_PATH, "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        exp = (item['data'], {'entities': item['label']})
        ANNOTATED_DATA.append(exp)

TRAIN_DATA = ANNOTATED_DATA[:550]
TEST_DATA = ANNOTATED_DATA[550:]


def converting(data, filename):

    for text, annot in tqdm(data):  # data in previous format
        doc = nlp.make_doc(text)  # create doc object from text
        ents = []
        for start, end, label in annot["entities"]:  # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        doc.ents = ents  # label the text with the ents
        db.add(doc)

    db.to_disk(filename)  # save the docbin object


converting(TRAIN_DATA, "./train.spacy")
converting(TEST_DATA, "./dev.spacy")
