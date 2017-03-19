# -*- coding: utf-8 -*-

import gzip
import json
import math
import collections
import itertools
import numpy as np
import word_count as wc
import utils

WordArticle = collections.namedtuple("Article", ["fulldate", "words"])

def get_uniform_alpha(first, second, count=1.0):
    word_set = set(first.keys()) | set(second.keys())
    alpha = dict([(w, count) for w in word_set])
    return alpha


def get_informative_alpha(first, second, total=1000.0):
    word_set = set(first.keys()) | set(second.keys())
    total_words = float(sum(first.values()) + sum(second.values()))
    alpha = dict(
            [(w, (first.get(w, 0) + second.get(w, 0)) / total_words*total)
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


def get_top_distinguishing(words_file, bigram_file, other_file_list, output_file, vocab_size=100):
    bigram_dict = wc.load_bigrams(bigram_file)
    word_cnts = wc.get_word_count(words_file, bigram_dict=bigram_dict,
            words_func=wc.get_mixed_tokens)
    other_cnts = collections.defaultdict(int)
    for filename in other_file_list:
        tmp_cnts = wc.get_word_count(filename, bigram_dict=bigram_dict,
                words_func=wc.get_mixed_tokens)
        for w in tmp_cnts:
            other_cnts[w] += tmp_cnts[w]
    alphas = get_informative_alpha(word_cnts, other_cnts)
    word_score = log_odds_normalized_diff(word_cnts, other_cnts, alphas)
    vocab_dict = wc.get_word_dict(word_score, top=vocab_size, filter_regex="\w\w+")
    utils.write_word_dict(vocab_dict, word_cnts, output_file)



def get_top_distinguishing_unigrams(words_file, other_file_list, output_file, vocab_size=100):
    word_cnts = wc.get_word_count(words_file, bigram_dict=None, ngrams=1,
            words_func=wc.get_ngram_list)
    other_cnts = collections.defaultdict(int)
    for filename in other_file_list:
        tmp_cnts = wc.get_word_count(filename, bigram_dict=None, ngrams=1,
                words_func=wc.get_ngram_list)
        for w in tmp_cnts:
            other_cnts[w] += tmp_cnts[w]
    alphas = get_informative_alpha(word_cnts, other_cnts)
    word_score = log_odds_normalized_diff(word_cnts, other_cnts, alphas)
    vocab_dict = wc.get_word_dict(word_score, top=vocab_size, filter_regex="\w\w+")
    utils.write_word_dict(vocab_dict, word_cnts, output_file)


def load_word_articles(words_file, vocab_file, words_func, vocab_size=100):
    articles = []
    word_map = utils.read_word_dict(vocab_file, vocab_size=vocab_size)
    word_set = utils.get_reverse_dict(word_map)
    with gzip.open(words_file) as fin:
        for line in fin:
            data = json.loads(line)
            words = words_func(data["words"])
            words = set([word_set[w] for w in words if w in word_set])
            articles.append(WordArticle(fulldate=int(data["date"]),
                words=words))
    return articles, word_set, word_map


def generate_cooccurrence_from_word_set(articles, total_frames=100):
    matrix = np.zeros((total_frames, total_frames))
    for article in articles:
        frames = article.words
        for frame in frames:
            matrix[frame, frame] += 1
        for (i, j) in itertools.combinations(frames, 2):
            matrix[i, j] += 1
            matrix[j, i] += 1
    return matrix

