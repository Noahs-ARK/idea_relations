# -*- coding: utf-8 -*-

import os
import gzip
import io
import numpy as np
        
def get_date_key(date, group_by="month"):
    if group_by == "month":
        key = int(date / 100)
    elif group_by == "year":
        if date < 10000:
            key = int(date)
        else:
            key = int(date / 10000)
    elif group_by == "quarter":
        month = int(date / 100)
        year = int(month / 100)
        month = month % 100
        key = year * 100 + int((month - 1) / 3)
    else:
        raise NotImplementedError(
                "there is no key function for grouping %s" % group_by)
    return key


def get_log_pmi(xy, x, y, total, add_one=1.0):
    if add_one < 0:
        add_one = 0
    return np.log(xy + add_one) + np.log(total + add_one) \
            - np.log(x + add_one) - np.log(y + add_one)


def write_word_dict(vocab_dict, word_count, filename):
    with io.open(filename, mode="w", encoding="utf-8") as fout:
        ids = vocab_dict.values()
        ids.sort()
        reverse_dict = {i: w for (w, i) in vocab_dict.iteritems()}
        for wid in ids:
            fout.write("%d\t%s\t%d\n" % (wid, reverse_dict[wid],
                word_count[reverse_dict[wid]]))

            
def read_word_dict(filename, vocab_size=-1):
    vocab_map = {}
    with io.open(filename, "r", encoding="utf-8") as fin:
        count = 0
        for line in fin:
            count += 1
            if vocab_size > 0 and count > vocab_size:
                break
            try:
                wid, word, _ = line.strip().split("\t")
                vocab_map[int(wid)] = word
            except:
                print line
    return vocab_map



def unravel_indices(indices, shape, symmetry=True):
    result = set()
    for i, j in zip(*np.unravel_index(indices, shape)):
        key = [i, j]
        if symmetry:
            key.sort()
        result.add(tuple(key))
    return result


def get_extreme_pairs(matrix, count=100, symmetry=True):
    bottom, top = set(), set()
    row, col = matrix.shape
    local = matrix.copy()
    np.fill_diagonal(local, 0)
    indices = np.argsort(local, axis=None)
    bottom = unravel_indices(indices[:count], (row, col))
    top = unravel_indices(indices[-count:], (row, col))
    return bottom, top


def write_latex_table(filename, cols, output_func):
    with open(filename, 'w') as fout:
        fout.write('\\begin{tabular}{%s}\n' % cols)
        fout.write('\\toprule\n')
        output_func(fout)
        fout.write('\\bottomrule\n')
        fout.write('\\end{tabular}\n')


def get_reverse_dict(d):
    return {i: w for (w, i) in d.iteritems()}



def read_json_list(input_file):
    with gzip.open(input_file) if input_file.endswith(".gz") \
            else open(input_file) as fin:
        for line in fin:
            obj = json.loads(line)
            yield obj


def write_json_list(output_file, data):
    with gzip.open(output_file, "w") if input_file.endswith(".gz") \
            else open(input_file, "w") as fin:
        for d in data:
            fout.write("%s\n" % json.dumps(d))


