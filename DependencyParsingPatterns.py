"""Dependency parsing patterns"""


"""
matcher attr: https://spacy.io/api/matcher
dependency matcher: https://spacy.io/usage/rule-based-matching#dependencymatcher
"""

"""
The patterns include the following scenarios:

1.Which NBA player/PERSON in which city/GPE or team/ORG. Done
2.Which coach/PERSON in which team/ORG. Done
3.Which team/ORG ranked
4.Time Done
5.Which team/ORG against which team/ORG Done
6.Which NBA player/PERSON gets injured/EVENT 
7.Which NBA player/PERSON was traded/EVENT (from which team/ORG to which team/ORG)
8.Which NBA player/PERSON got penalty/EVENT (for a time/TIME period or financial penalty/MONEY)
9.Which NBA player/PERSON created a new record/SCORE Done
10.Which team/ORG city/GPE created a new record/SCORE Done
"""

"""person-team"""
# person VERB for/at/with/without/in/of team/city :pattern_person_team_rel_1
# person VERB  team/city :pattern_person_team_rel_2
# person, team/city  :pattern_person_team_rel_3

# team/city VERB for/at/with/without/in/of person :pattern_team_person_rel_1
# team/city VERB person :pattern_team_person_rel_2
# team/city, person :pattern_team_person_rel_3
# team person :pattern_team_person_rel_4
# team NOUN person :pattern_team_person_rel_5
pattern_person_team_rel_1 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "anchor_for",
        "RIGHT_ATTRS": {"ORTH": {"IN": ["for", "at", "with", "without", "within", "in", "of"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON", "DEP": "nsubj"}
    },
    {
        "LEFT_ID": "anchor_for",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}},
    }
]

pattern_person_team_rel_2 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON", "DEP": "nsubj"}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}},
    }
]

pattern_person_team_rel_3 = [
    {
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"},
    },
    {
        "LEFT_ID": "subject",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}, "DEP": "appos"},
    }
]

"""team-person"""
pattern_team_person_rel_1 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "anchor_for",
        "RIGHT_ATTRS": {"ORTH": {"IN": ["for", "at", "with", "without", "within", "in", "of"]}}
    },
    {
        "LEFT_ID": "anchor_for",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"},
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}, "DEP": "nsubj"}
    },
]

pattern_team_person_rel_2 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB", "ROOT"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"},
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}, "DEP": "nsubj"}
    }
]

pattern_team_person_rel_3 = [
    {
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"},
    },
    {
        "LEFT_ID": "subject",
        "REL_OP": "<",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}, "DEP": "appos"},
    }
]

pattern_team_person_rel_4 = [
    {
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"},
    },
    {
        "LEFT_ID": "subject",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}, "DEP": {"IN": ["nmod", "compound", "poss"]}},
    }
]

pattern_team_person_rel_5 = [
    {
        "RIGHT_ID": "middle_compound",
        "RIGHT_ATTRS": {"DEP": {"IN": ["compound", "nmod"]}},
    },
    {
        "LEFT_ID": "middle_compound",
        "REL_OP": "<",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"},
    },
    {
        "LEFT_ID": "middle_compound",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}, "DEP": {"IN": ["compound", "nmod"]}},
    },
]

"""team-team"""
pattern_team_team_rel_1 = [
    {
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}},
    },
    {
        "LEFT_ID": "object",
        "REL_OP": ".*",
        "RIGHT_ID": "object2",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}},
    },
]

"""person-score"""
pattern_person_score_rel_1 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB", "ROOT"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "anchor_for",
        "RIGHT_ATTRS": {"ORTH": {"IN": ["for", "at", "with", "without", "within", "in", "of", "on"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON", "DEP": "nsubj"}
    },
    {
        "LEFT_ID": "anchor_for",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": "PRODUCT"},
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "score",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["SCORE", "CARDINAL", "PERCENT"]}}
    },
]

pattern_person_score_rel_2 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB", "ROOT"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">>",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["PERSON", "CARDINAL"]}, "DEP": {"IN": ["nsubj", "amod"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">>",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"POS": {"IN": ["NOUN", "PROPN"]}}
    },
    {
        "LEFT_ID": "object",
        "REL_OP": ">",
        "RIGHT_ID": "score",
        "RIGHT_ATTRS": {"DEP": "nummod", "ENT_TYPE": {"NOT_IN": ["TIME", "DATE"]}}
    },
]

"""team-score"""
pattern_team_score_rel_1 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB", "ROOT"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">>",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": {"IN": ["ORG", "GPE"]}, "DEP": "nsubj"}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "object",
        "RIGHT_ATTRS": {"ENT_TYPE": "SCORE"}
    },
]

"""person-injury"""
pattern_person_injury_rel_1 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB", "ROOT"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">>",
        "RIGHT_ID": "injury",
        "RIGHT_ATTRS": {"LEMMA": {"IN": ["tendinitie", "achilles", "ankle", "knee", "shoulder", "finger"]}}
    },
]

pattern_person_injury_rel_2 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB", "ROOT"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">>",
        "RIGHT_ID": "injury",
        "RIGHT_ATTRS": {"LEMMA": {"IN": ["tendinitie", "achilles", "ankle", "knee", "shoulder", "finger"]}}
    },
    {
        "LEFT_ID": "injury",
        "REL_OP": "<",
        "RIGHT_ID": "complain",
        "RIGHT_ATTRS": {"POS": "NOUN"}
    }
]

pattern_person_injury_rel_3 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB", "ROOT"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "subject",
        "RIGHT_ATTRS": {"ENT_TYPE": "PERSON"}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">>",
        "RIGHT_ID": "injury",
        "RIGHT_ATTRS": {"LEMMA": {"IN": ["tendinitie", "achilles", "ankle", "knee", "shoulder", "finger"]}}
    },

    {
        "LEFT_ID": "injury",
        "REL_OP": "<",
        "RIGHT_ID": "adp",
        "RIGHT_ATTRS": {"POS": "ADP"}
    },
    {
        "LEFT_ID": "adp",
        "REL_OP": "<",
        "RIGHT_ID": "complain",
        "RIGHT_ATTRS": {"POS": "NOUN"}
    }
]

"""time"""
pattern_time_1 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}}
    },

    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "date",
        "RIGHT_ATTRS": {"ENT_TYPE": "DATE"}
    },
]

pattern_time_2 = [
    {
        "RIGHT_ID": "anchor",
        "RIGHT_ATTRS": {"POS": {"IN": ["AUX", "VERB"]}}
    },
    {
        "LEFT_ID": "anchor",
        "REL_OP": ">",
        "RIGHT_ID": "anchor_for",
        "RIGHT_ATTRS": {"ORTH": {"IN": ["for", "at", "with", "without", "within", "in", "of", "on"]}}
    },
    {
        "LEFT_ID": "anchor_for",
        "REL_OP": ">",
        "RIGHT_ID": "date",
        "RIGHT_ATTRS": {"ENT_TYPE": "DATE"}
    },
]

