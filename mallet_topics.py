# -*- coding: utf-8 -*-

import collections
import gzip
import itertools
import json
import numpy as np
import word_count as wc
import utils


def convert_word_count_mallet(word_dict, input_file, output_file,
                              words_func=None):
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


def get_mallet_input_from_words(input_file, data_dir, vocab_size=10000):
    bigram_file = "%s/bigram_phrases.txt" % data_dir
    if not os.path.exists(bigram_file):
        wc.find_bigrams(input_file, bigram_file)
    bigram_dict = wc.load_bigrams(bigram_file)
    word_cnts = wc.get_word_count(input_file, bigram_dict=bigram_dict,
                                  words_func=wc.get_mixed_tokens)
    vocab_dict = wc.get_word_dict(word_cnts,
                                  top=vocab_size,
                                  filter_regex="\w\w+")
    utils.write_word_dict(vocab_dict, word_cnts,
                          "%s/data.word_id.dict" % data_dir)
    convert_word_count_mallet(vocab_dict, input_file,
                              "%s/data.input" % data_dir,
                              words_func=functools.partial(
                                  wc.get_mixed_tokens,
                                  bigram_dict=bigram_dict))


def load_topic_words(vocab, input_file, top=10):
    """Get the top 10 words for each topic"""
    topic_map = {}
    with open(input_file) as fin:
        for line in fin:
            parts = line.strip().split()
            tid = int(parts[0])
            top_words = parts[2:2+top]
            topic_map[tid] = ",".join([vocab[int(w)] for w in top_words])
    return topic_map


def load_doc_topics(input_file, doc_topic_file, threshold=0.01):
    """Load topics in each document"""
    articles = []
    with open(doc_topic_file) as tfin:
        for data in utils.read_json_list(input_file):
            topic_line = tfin.readline()
            if not topic_line:
                break
            ideas = topic_line.strip().split()[2:]
            ideas = set([i for (i, v) in enumerate(ideas) 
                         if float(v) > threshold])
            articles.append(utils.IdeaArticle(fulldate=int(data["date"]),
                                         ideas=ideas))
    return articles


def load_articles(input_file, topic_dir):
    vocab_file = "%s/data.word_id.dict" % topic_dir
    doc_topic_file = "%s/doc-topics.gz" % topic_dir
    topic_word_file = "%s/topic-words.gz" % topic_dir
    vocab = util.read_word_dict(vocab_file)
    topic_map = load_topic_words(vocab, topic_word_file)
    articles = load_doc_topics(input_file, doc_topic_file)
    return articles, vocab, topic_map


def check_mallet_directory(directory):
    vocab_file = "%s/data.word_id.dict" % topic_dir
    doc_topic_file = "%s/doc-topics.gz" % topic_dir
    topic_word_file = "%s/topic-words.gz" % topic_dir
    return all([os.path.exists(filename)
               for filename in [vocab_file, doc_topic_file, topic_word_file]])


