# -*- coding: utf-8 -*-

import gzip
import json
import math
import collections
import itertools
import numpy as np
import word_count as wc
import utils

def get_uniform_alpha(first, second, count=1.0):
    word_set = set(first.keys()) | set(second.keys())
    alpha = dict([(w, count) for w in word_set])
    return alpha


def get_informative_alpha(first, second, total=1000.0):
    word_set = set(first.keys()) | set(second.keys())
    total_words = float(sum(first.values()) + sum(second.values()))
    alpha = dict(
        [(w, (first.get(w, 0) + second.get(w, 0)) / total_words * total)
         for w in word_set])
    return alpha


def log_odds_normalized_diff(first, second, alphas):
    """The algorithm described in "Fightin' Words: Lexical Feature Selection
    and Evaluation for Identifying the Content of Political Conflict"
    Monroe, Colaresi and Quinn"""
    word_set = set(first.keys()) | set(second.keys())
    total_alpha = sum([alphas[w] for w in word_set])
    word_score = {}
    first_total_count = sum(first.values())
    second_total_count = sum(second.values())
    for w in word_set:
        first_count = float(first.get(w, 0) + alphas[w])
        first_comp = float(first_total_count - first.get(w, 0)
                           + total_alpha - alphas[w])
        second_count = float(second.get(w, 0) + alphas[w])
        second_comp = float(second_total_count - second.get(w, 0)
                            + total_alpha - alphas[w])

        # print w, second_comp, second_count
        word_score[w] = math.log(first_count) - math.log(first_comp)
        word_score[w] += math.log(second_comp) - math.log(second_count)
        variance = 1. / first_count + 1. / first_comp + \
                1. / second_count + 1. / second_comp
        word_score[w] /= math.sqrt(variance)
    return word_score


def get_top_distinguishing(input_file, other_file_list,
                           output_file, vocab_size=100):
    bigram_file = "%s/bigram_phrases.txt" % data_dir
    if not os.path.exists(bigram_file):
        wc.find_bigrams(input_file, bigram_file)
    bigram_dict = wc.load_bigrams(bigram_file)
    word_cnts = wc.get_word_count(input_file, bigram_dict=bigram_dict,
                                  words_func=wc.get_mixed_tokens)
    other_cnts = collections.defaultdict(int)
    for filename in other_file_list:
        tmp_cnts = wc.get_word_count(filename, bigram_dict=bigram_dict,
                                     words_func=wc.get_mixed_tokens)
        for w in tmp_cnts:
            other_cnts[w] += tmp_cnts[w]
    alphas = get_informative_alpha(word_cnts, other_cnts)
    word_score = log_odds_normalized_diff(word_cnts, other_cnts, alphas)
    vocab_dict = wc.get_word_dict(word_score,
                                  top=vocab_size,
                                  filter_regex="\w\w+")
    utils.write_word_dict(vocab_dict, word_cnts, output_file)


def load_word_articles(input_file, vocab_file, vocab_size=100):
    articles = []
    word_map = utils.read_word_dict(vocab_file, vocab_size=vocab_size)
    word_set = utils.get_reverse_dict(word_map)
    bigram_file = "%s/bigram_phrases.txt" % data_dir
    bigram_dict = wc.load_bigrams(bigram_file)
    words_func = functools.partial(wc.get_mixed_tokens, bigram_dict=bigram_dict)
    for data in utils.read_json_list(input_file):
        words = words_func(data["text"])
        words = set([word_set[w] for w in words if w in word_set])
        articles.append(utils.IdeaArticle(fulldate=int(data["date"]),
                                          ideas=words))
    return articles, word_set, word_map

