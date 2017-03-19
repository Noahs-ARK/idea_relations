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


def get_word_count(filename, filter_stopwords=True, ngrams=1,
                   bigram_dict=None, words_func=None):
    result = collections.defaultdict(int)
    with gzip.open(filename) as fin:
        for l_i, line in enumerate(fin):
            if l_i % 1000 == 0:
                logging.info("in get_word_count %d" % l_i)
            data = json.loads(line)
            words = words_func(data["words"], ngrams=ngrams,
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


def convert_word_count_libsvm(word_dict, input_file, output_file):
    doc_id = 0
    with gzip.open(input_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            doc_id += 1
            data = json.loads(line)
            words = data["word_count"].keys()
            words = [(word_dict[w], data["word_count"][w])
                     for w in words if w in word_dict]
            words.sort()
            word_cnts = ["%d:%d" % (wid, cnt) for (wid, cnt) in words]
            fout.write("%s\t%s\n" % (doc_id, " ".join(word_cnts)))


def convert_word_count_mallet(word_dict, input_file, output_file, words_func=None):
    doc_id = 0
    with gzip.open(input_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            doc_id += 1
            data = json.loads(line)
            words = collections.Counter(words_func(data["words"]))
            words = [(word_dict[w], words[w])
                     for w in words if w in word_dict]
            words.sort()
            word_cnts = [" ".join([str(wid)] * cnt) for (wid, cnt) in words]
            fout.write("%s %s %s\n" % (doc_id, data["date"], " ".join(word_cnts)))


def get_time_series(result, word_list, normalize=False):
    keys = result.keys()
    keys.sort()
    ts = {w: [] for w in word_list}
    for k in keys:
        for w in word_list:
            if normalize:
                ts[w].append(result[k]["word_count"].get(w, 0) \
                        / float(result[k]["articles"]))
            else:
                ts[w].append(result[k]["word_count"].get(w, 0))
    return {w: np.array(ts[w]) for w in word_list}


def get_total_count(result, option, key):
    total = 0
    if option == "articles":
        for k in result:
            total += result[k]["articles"]
    elif option in ["word_count", "cooccur_count"]:
        for k in result:
            total += result[k][option].get(key, 0)
    else:
        raise NotImplementedError("not supported type in get_total_count")
    return float(total)


def get_total_word_count(result):
    word_cnt = collections.defaultdict(int)
    for k in result:
        for w in result[k]["word_count"]:
            word_cnt[w] += result[k]["word_count"][w]
    return word_cnt


def get_most_frequent(word_cnt, top=10000):
    words = [(word_cnt[w], w) for w in word_cnt
            if re.match("\w+", w)]
    words.sort(reverse=True)
    min_threshold = words[top - 1][0]
    return [v for v in words if v[0] >= min_threshold]


def get_pairwise_correlation(ts_dict):
    words = ts_dict.keys()
    words.sort()
    corrs = {}
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            fst, snd = words[i], words[j]
            corr, _ = ss.pearsonr(ts_dict[fst], ts_dict[snd])
            corrs[fst, snd] = corr
    return corrs


def get_pairwise_pmi(result, words, add_one=1.0):
    total = get_total_count(result, "articles", None)
    count = {w: get_total_count(result, "word_count", w)
            for w in words}
    pmi = {}
    words.sort()
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            fst, snd = words[i], words[j]
            cooccur = get_total_count(result, "cooccur_count", (fst, snd))
            pmi[fst, snd] = utils.get_log_pmi(cooccur, count[fst], count[snd], total,
                    add_one=add_one)
    return pmi


def get_mallet_input_from_words(words_file, data_dir, vocab_size=10000):
    bigram_file = "%s/bigram_phrases.txt" % data_dir
    find_bigrams(words_file, bigram_file)
    bigram_dict = load_bigrams(bigram_file)
    word_cnts = get_word_count(words_file, bigram_dict=bigram_dict,
            words_func=get_mixed_tokens)
    vocab_dict = get_word_dict(word_cnts, top=vocab_size, filter_regex="\w\w+")
    utils.write_word_dict(vocab_dict, word_cnts,
            "%s/data.word_id.dict" % data_dir)
    convert_word_count_mallet(vocab_dict, words_file,
            "%s/data.input" % data_dir,
            words_func=functools.partial(get_mixed_tokens, bigram_dict=bigram_dict))


def unigrams_to_data(words_file, data_dir, vocab_size=10000):
    word_cnts = get_word_count(words_file, ngrams=1, words_func=get_ngram_list)
    vocab_dict = get_word_dict(word_cnts, top=vocab_size, filter_regex="\w\w+")
    utils.write_word_dict(vocab_dict, word_cnts,
                          "%s/data.word_id.dict" % data_dir)
    convert_word_count_mallet(vocab_dict, words_file,
            "%s/data.input" % data_dir,
            words_func=functools.partial(get_ngram_list, ngrams=1, bigram_dict=None))


