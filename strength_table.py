# -*- coding: utf-8 -*-

import functools
import collections
import io
import numpy as np
import scipy.stats as ss
import util

TopicPair = collections.namedtuple("TopicPair",
        ["combined_score", "pmi", "correlation",
            "fst_topic", "snd_topic", "rank"])


def write_type_rows(fout, data, top=5):
    fout.write("combined & pmi & correlation & first & second\\\\\n")
    types = ["friends", "romance", "head-to-head", "arms-race"]
    for t in types:
        fout.write("\\midrule\n")
        fout.write("\\multicolumn{5}{c}{%s}\\\\\n" % t)
        for r in data[t][:top]:
            fout.write("%.3f & %.3f & %.3f & %s & %s \\\\\n" % (
                r.combined_score, r.pmi, r.correlation,
                ", ".join(r.fst_topic.split(",")),
                ", ".join(r.snd_topic.split(","))))

            
def load_all_pairs(table_file):
    type_list = collections.defaultdict(list)
    with open(table_file, 'r') as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if parts[0] == "None":
                continue
            # remove substrings of each other
            if parts[4].find(parts[5]) != -1 or parts[5].find(parts[4]) != -1:
                continue
            if parts[4].startswith("ion,ing") or parts[5].startswith("ion,ing"):
                continue
            type_list[parts[0]].append(TopicPair(
                combined_score=float(parts[1]),
                pmi=float(parts[2]),
                correlation=float(parts[3]),
                fst_topic=parts[4].replace("_", " "),
                snd_topic=parts[5].replace("_", " "),
                rank=len(type_list[parts[0]]) + 1))
    return type_list


def get_top_five_relationship(table_file, output_file, top=5):
    type_list = load_all_pairs(table_file)
    util.write_latex_table(output_file, "cccp{6cm}p{6cm}",
            functools.partial(write_type_rows, data=type_list, top=top))


def get_relation_strength(table_file, top=10, normalize=False, return_sem=False, return_all=False):
    type_list = load_all_pairs(table_file)
    scores = {k: [abs(v.combined_score) for v in type_list[k][:top]] for k in type_list}
    mean = {k: np.mean(scores[k]) for k in type_list}
    if return_all:
        return scores, mean, {k: ss.sem(scores[k]) for k in type_list}
    elif return_sem:
        return mean, {k: ss.sem(scores[k]) for k in type_list}
    elif normalize:
        max_v = max(mean.values())
        return {k: mean[k] / max_v for k in mean}
    else:
        return mean

