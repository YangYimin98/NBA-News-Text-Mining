import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example
import jsonlines
from spacy import displacy
from pathlib import Path
from NBA_Scraper import *
import os

"""
This file aims to train custom ner model based on 'en_core_web_lg' (tradition token2vec + ner)
We also trained another custom ner model based on 'en_core_web_trf' (transformer + ner) on Colab
Reference:
https://spacy.io/usage/training#config-custom
https://towardsdatascience.com/using-spacy-3-0-to-build-a-custom-ner-model-c9256bea098
https://www.machinelearningplus.com/nlp/training-custom-ner-model-in-spacy/
https://manivannan-ai.medium.com/how-to-train-ner-with-custom-training-data-using-spacy-188e0e508c6
https://pahulpreet86.github.io/name-entity-recognition-pre-trained-models-review/
https://github.com/bond005/deep_ner
https://explosion.ai/blog/spacy-transformers
"""

ANNOTATED_FILE_PATH = 'annotated_corpus.jsonl'
NER_MODEL_PATH = os.getcwd() + '/ner_model/'
EPOCHS = 200
BATCH_SIZE = 8
TRAINING = True


nlp = spacy.load('en_core_web_lg')
ner = nlp.get_pipe("ner")

if TRAINING:
    TRAIN_DATA = []
    with open(ANNOTATED_FILE_PATH, "r+", encoding="utf8") as f:
        for item in jsonlines.Reader(f):
            exp = (item['data'], {'entities': item['label']})
            TRAIN_DATA.append(exp)

    # Add labels to the pipeline
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Disable pipeline components you dont need to change
    pipe_exceptions = ['ner']
    unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


    # TRAINING THE MODEL
    with nlp.disable_pipes(*unaffected_pipes):
        print(nlp.pipe_names)
        # Training for N iterations
        for iteration in range(EPOCHS):

            # shuufling examples  before every iteration
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch

            batches = minibatch(TRAIN_DATA, size=BATCH_SIZE)  # compounding(4.0, 32.0, 1.001)
            for batch in batches:
                for texts, annotations in batch:
                    doc = nlp.make_doc(texts)
                    example = Example.from_dict(doc, annotations)
                    nlp.update(
                        [example],  # batch of examples
                        drop=0.3,  # dropout - make it harder to memorise data
                        losses=losses,
                    )
            print("Epoch: ", iteration, " Losses: ", losses)

        # Save the  model to directory
        output_dir = Path(NER_MODEL_PATH)
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

else:
    # Load the saved model and predict
    output_dir = Path(NER_MODEL_PATH)
    print("Loading from", output_dir)
    nlp = spacy.load(output_dir)


total_corpus = corpus_loader(ARTICLE_PATH)
text = total_corpus[400]['content']
doc = nlp(text)
print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
# displacy.serve(doc, style="ent")
print(nlp.pipe_names)

