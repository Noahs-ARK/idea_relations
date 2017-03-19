# -*- coding: utf-8 -*-

import csv
import collections
import itertools
import numpy as np
import scipy.stats as ss
import util




def get_time_grouped_articles(articles, group_by="month", start_time=1980,
        end_time=2016):
    articles_group = collections.defaultdict(list)
    for article in articles:
        key = util.get_date_key(article.fulldate, group_by=group_by)
        if int(str(key)[:4]) < start_time:
            continue
        if int(str(key)[:4]) > end_time:
            continue
        articles_group[key].append(article)
    return articles_group


def get_most_frequent_cooccur(matrix, top=3, return_value=False, reverse=True):
    result = {}
    for i in range(TOTAL_FRAMES):
        order = np.argsort(matrix[i,:])
        if reverse:
            order = order[::-1]
        if return_value:
            result[DATA_FIELDS[i]] = [(DATA_FIELDS[v], matrix[i, v])
                for v in order[:top]]
        else:
            result[DATA_FIELDS[i]] = [DATA_FIELDS[v] for v in order[:top]]
    return result


def get_most_frequent(array, top=3, reverse=True):
    order = np.argsort(array)
    if reverse:
        order = order[::-1]
    return [DATA_FIELDS[v] for v in order[:top]]


def chisquared_score(N, a, b, c, d):
    diff = (a * d - b * c)
    return float(N * diff * diff) / ((a + b) * (a + c) * (b + d) * (c + d))


def get_chisquared_score(matrix, frame_count, total):
    result = matrix.copy()
    for i in range(TOTAL_FRAMES):
        for j in range(i + 1, TOTAL_FRAMES):
            a = matrix[i, j]
            b = frame_count[i] - a
            c = frame_count[j] - a
            d = total + a - frame_count[i] - frame_count[j]
            score = chisquared_score(total, a, b, c, d)
            if np.isnan(score):
                score = 0
            # if a * d < b * c:
            #     print i, j
            result[i, j] = score
            result[j, i] = score
    return result


def get_pmi(matrix, frame_count, total, total_frames=TOTAL_FRAMES,
        add_one=1.0):
    result = matrix.copy()
    for i in range(total_frames):
        for j in range(i + 1, total_frames):
            score = util.get_log_pmi(matrix[i, j],
                    frame_count[i], frame_count[j], total,
                    add_one=add_one)
            if np.isnan(score):
                score = 0
            result[i, j] = score
            result[j, i] = score
    return result


def get_group_info(grouped_articles, cooccur_func=generate_cooccurence,
        func=get_pmi):
    info_dict, chisquare_dict = {}, {}
    for key in grouped_articles:
        info_dict[key] = get_count_cooccur(grouped_articles[key], func=cooccur_func)
        chisquare_dict[key] = func(info_dict[key]["cooccur"],
                info_dict[key]["count"], float(info_dict[key]["articles"]))
    return info_dict, chisquare_dict


def get_probability_ts(info_dict, label_array):
    keys = info_dict.keys()
    keys.sort()
    time_series = []
    probability_dict = {}
    for k in info_dict:
        arr = info_dict[k]["count"]
        total = float(info_dict[k]["articles"])
        probability_dict[k] = arr / total
    for name in label_array:
        index = DATA_FIELDS.index(name)
        label_ts = []
        for k in keys:
            label_ts.append(probability_dict[k][index])
        time_series.append(np.array(label_ts))
    return time_series


def find_negative_pairs(pmi):
    negative_set = set()
    for pair in np.argwhere(pmi < 0):
        f, s = pair
        if f < s:
            negative_set.add((DATA_FIELDS[f], DATA_FIELDS[s]))
    return negative_set


def get_time_series(info_dict, normalize=False,
        total_frames=TOTAL_FRAMES):
    keys = info_dict.keys()
    keys.sort()
    ts_matrix = np.zeros((total_frames, len(keys)))
    for i, k in enumerate(keys):
        ts_matrix[:, i] = info_dict[k]["count"]
        if normalize:
            ts_matrix[:, i] = ts_matrix[:, i] / float(info_dict[k]["articles"])
    return ts_matrix


def get_ts_correlation(info_dict, normalize=False,
        total_frames=TOTAL_FRAMES):
    ts_matrix = get_time_series(info_dict, normalize=normalize,
            total_frames=total_frames)
    correlation_matrix = np.zeros((total_frames, total_frames))
    for i in range(total_frames):
        for j in range(i + 1, total_frames):
            score, _ = ss.pearsonr(ts_matrix[i, :], ts_matrix[j, :])
            correlation_matrix[i, j] = score
            correlation_matrix[j, i] = score
    return correlation_matrix


def get_top_bottom(values, top=15, bottom=15):
    selected = values[:bottom] + values[-top:]
    edges = [(t[1], t[2]) for t in selected]
    weights = [t[0] for t in selected]
    return edges, weights


def get_top_pos_neg(values, pos=15, neg=15):
    selected = [v for v in values[:neg] if v[0] < 0] \
            + [v for v in values[-pos:] if v[0] > 0]
    edges = [(t[1], t[2]) for t in selected]
    weights = [t[0] for t in selected]
    return edges, weights


def get_nodes_edges(values, chosen=None):
    edges, weights = [], []
    for v, s, t in values:
        if s in chosen or t in chosen:
            edges.append((s, t))
            weights.append(v)
    return edges, weights


def select_edges(adjacency, func):
    values = []
    for i in range(TOTAL_FRAMES):
        for j in range(i + 1, TOTAL_FRAMES):
            values.append((adjacency[i, j], i, j))
    values.sort()
    return func(values)


