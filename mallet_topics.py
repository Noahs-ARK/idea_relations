# -*- coding: utf-8 -*-

import collections
import gzip
import itertools
import json
import numpy as np
import utils


TopicArticle = collections.namedtuple("Article", ["fulldate", "frames"])
START_LOC = 11

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


def load_doc_topics(word_count_file, doc_topic_file, threshold=0.01):
    """load something similar to frame_count.py"""
    articles = []
    with gzip.open(word_count_file) as wfin, open(doc_topic_file) as tfin:
        while True:
            word_line = wfin.readline()
            topic_line = tfin.readline()
            if not word_line or not topic_line:
                break
            data = json.loads(word_line)
            frames = topic_line.strip().split()[2:]
            frames = set([i for (i, v) in enumerate(frames) 
                if float(v) > threshold])
            articles.append(TopicArticle(fulldate=int(data["date"]),
                frames=frames))
    return articles


def generate_cooccurrence_from_int_set(articles, total_frames=100):
    matrix = np.zeros((total_frames, total_frames))
    for article in articles:
        frames = article.frames
        for frame in frames:
            matrix[frame, frame] += 1
        for (i, j) in itertools.combinations(frames, 2):
            matrix[i, j] += 1
            matrix[j, i] += 1
    return matrix


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


