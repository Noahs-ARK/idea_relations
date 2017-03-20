# -*- coding: utf-8 -*-

import collections
import re
import io
import gzip
import json
import functools
import logging
import numpy as np
import scipy.stats as ss
from nltk.corpus import stopwords
import utils

STOPWORDS = set(stopwords.words("english") + ["said"])

def get_ngram_list(input_words, ngrams=1, filter_stopwords=True,
                   bigram_dict=None):
    words = [w.lower() for w in input_words.split()]
    result = []
    for start in range(len(words) - ngrams + 1):
        tmp_words = words[start:start+ngrams]
        if filter_stopwords and any([w in STOPWORDS for w in tmp_words]):
            continue
        w = " ".join(tmp_words)
        result.append(w)
    return result

            
def get_mixed_tokens(input_words, ngrams=1, filter_stopwords=True,
                     bigram_dict=None):
    words = [w.lower() for w in input_words.split()]
    result, index = [], 0
    while index < len(words):
        w = words[index]
        if filter_stopwords and w in STOPWORDS:
            index += 1
            continue
        # look forward
        if index < len(words) - 1:
            bigram = w + " " + words[index + 1]
            if bigram in bigram_dict:
                result.append(bigram)
                index += 2
                continue
        result.append(w)
        index += 1
    return result


def get_word_count(input_file, filter_stopwords=True, ngrams=1,
                   bigram_dict=None, words_func=None):
    result = collections.defaultdict(int)
    for data in utils.read_json_list(input_file):
        words = words_func(data["text"], ngrams=ngrams,
                           filter_stopwords=filter_stopwords,
                           bigram_dict=bigram_dict)
        for w in words:
            result[w] += 1
    return result


def find_bigrams(filename, output_file, filter_stopwords=True, threshold=100,
                 min_count=5):
    unigram_count = get_word_count(filename,
                                   filter_stopwords=filter_stopwords, ngrams=1,
                                   words_func=get_ngram_list)
    total_words = float(sum(unigram_count.values()))
    bigram_count = get_word_count(filename,
                                  filter_stopwords=filter_stopwords, ngrams=2,
                                  words_func=get_ngram_list)
    bigram_list = []
    for w in bigram_count:
        words = w.split()
        score = (bigram_count[w] - min_count) * total_words \
                / (unigram_count[words[0]] * unigram_count[words[1]])
        if score > threshold:
            bigram_list.append((score, w))
    bigram_list.sort(reverse=True)
    with open(output_file, "w") as fout:
        for score, w in bigram_list:
            fout.write("%s\n" % json.dumps({"word": w, "score": score}))


def load_bigrams(filename):
    bigram_dict = {}
    with open(filename) as fin:
        for line in fin:
            data = json.loads(line)
            bigram_dict[data["word"]] = data["score"]
    return bigram_dict


def get_word_dict(word_count, top=10000, filter_regex=None):
    if filter_regex:
        word_count = {w: word_count[w] for w in word_count
                      if all([re.match(filter_regex, sw) for sw in w.split()])}
    words = get_most_frequent(word_count, top=top)
    return {v[1]: i for i, v in enumerate(words)}


def get_most_frequent(word_cnt, top=10000):
    words = [(word_cnt[w], w) for w in word_cnt
            if re.match("\w+", w)]
    words.sort(reverse=True)
    min_threshold = words[top - 1][0]
    return [v for v in words if v[0] >= min_threshold]


