# -*- coding: utf-8 -*-

"""
Use NLTK for preprocessing.
Feel free to switch to spacy.
"""

from nltk.tokenize import wordpunct_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import utils

LEMMATIZER = WordNetLemmatizer()
STOPWORDS = set(stopwords.words('english'))

def tokenize(text, filter_stopwords=False, lowercase=True):
    words = wordpunct_tokenize(text)
    if filter_stopwords:
        words = [w for w in words if w not in STOPWORDS]
    return words


def lemmatize(text, filter_stopwords=False, lowercase=True):
    lemmas = [LEMMATIZER.lemmatize(w)
              for w in tokenize(text, lowercase=lowercase,
                                filter_stopwords=filter_stopwords)]
    lemmas = [w for w in lemmas if len(w) > 2]
    return lemmas


def preprocess_input(input_file, output_file, func=tokenize):
    data = []
    for d in utils.read_json_list(input_file):
        d["text"] = " ".join(func(d["text"]))
        data.append(d)
    utils.write_json_list(output_file, data)

