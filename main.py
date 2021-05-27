import spacy
from spacy import displacy
import os

ner_path_1 = os.getcwd() + '/ner_model_trf/output/model-best/'
nlp = spacy.load(ner_path_1)
# doc = nlp("The team that represented the Eastern Conference in the 2020 Finals sits in sixth place in the All-Star break at 18-18")
# doc = nlp("Hollis-Jefferson, 26, played for the Raptors last season, providing energy and defense off the bench.")
# doc = nlp("""Chris Boucher has shot 10-for-18 (56%) on clutch 3-pointers, the second-best mark among 39 players who’ve attempted at least 15.""")
# doc = nlp("""• The Lakers have been without both Anthony Davis and LeBron James since LeBron left in the second quarter against Atlanta on March 20 with a high ankle sprain.""")
# doc = nlp("""Giannis Antetokounmpo sat out his third consecutive due to soreness in his left knee.""")
# displacy.serve(doc)



# add pipeline (declared through entry_points in setup.py)
# nlp.add_pipe("entityLinker", last=True)
#
# doc = nlp("Hollis-Jefferson, 26, played for the Raptors last season, providing energy and defense off the bench.")
#
# # returns all entities in the whole document
# all_linked_entities = doc._.linkedEntities
# # iterates over sentences and prints linked entities
# for sent in doc.sents:
#     sent._.linkedEntities.pretty_print()


