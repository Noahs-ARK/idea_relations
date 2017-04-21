# -*- coding: utf-8 -*-

import io
import os
import collections
import json
import itertools
import numpy as np
import scipy.stats as ss
from distutils.spawn import find_executable
if find_executable('latex'):
    HAS_LATEX = True
else:
    HAS_LATEX = False
try:
    from diptest import diptest
    unimodality_test = True
except:
    unimodality_test = False
import plot_functions as pf
import strength_table as st
import tex_output as to
import utils

COLOR_DICT = {
    "friends": "blue green",
    "arms-race": "dark sky blue",
    "head-to-head": "blood red",
    "tryst": "rose",
}


def generate_cooccurrence_from_int_set(articles, num_ideas=100):
    matrix = np.zeros((num_ideas, num_ideas))
    for article in articles:
        ideas = article.ideas
        for idea in ideas:
            matrix[idea, idea] += 1
        for (i, j) in itertools.combinations(ideas, 2):
            matrix[i, j] += 1
            matrix[j, i] += 1
    return matrix


def get_pmi(matrix, idea_count, total,
            num_ideas=50,
            add_one=1.0):
    result = matrix.copy()
    for i in range(num_ideas):
        for j in range(i + 1, num_ideas):
            score = utils.get_log_pmi(matrix[i, j],
                                      idea_count[i], idea_count[j], total,
                                      add_one=add_one)
            if np.isnan(score):
                score = 0
            result[i, j] = score
            result[j, i] = score
    return result


def get_count_cooccur(articles, func=generate_cooccurrence_from_int_set):
    cooccur = func(articles)
    count = np.diag(cooccur).copy()
    np.fill_diagonal(cooccur, 0)
    return {"count": count, "cooccur": cooccur,
            "articles": len(articles)}


def get_time_grouped_articles(articles, group_by="year", start_time=1980,
        end_time=2016):
    articles_group = collections.defaultdict(list)
    for article in articles:
        key = utils.get_date_key(article.fulldate, group_by=group_by)
        if int(str(key)[:4]) < start_time:
            continue
        if int(str(key)[:4]) > end_time:
            continue
        articles_group[key].append(article)
    return articles_group


def get_time_series(info_dict, num_ideas, normalize=False):
    keys = info_dict.keys()
    keys.sort()
    ts_matrix = np.zeros((num_ideas, len(keys)))
    for i, k in enumerate(keys):
        ts_matrix[:, i] = info_dict[k]["count"]
        if normalize:
            ts_matrix[:, i] = ts_matrix[:, i] / float(info_dict[k]["articles"])
    return ts_matrix


def get_ts_correlation(info_dict, num_ideas, normalize=False):
    ts_matrix = get_time_series(info_dict, num_ideas, normalize=normalize)
    correlation_matrix = np.zeros((num_ideas, num_ideas))
    for i in range(num_ideas):
        for j in range(i + 1, num_ideas):
            score, _ = ss.pearsonr(ts_matrix[i, :], ts_matrix[j, :])
            correlation_matrix[i, j] = score
            correlation_matrix[j, i] = score
    return correlation_matrix


def generate_scatter_dist_plot(articles, num_ideas, plot_dir, prefix,
                               cooccur_func=None,
                               make_plots=True, 
                               write_tests=True,
                               group_by="year",
                               samples=1000):
    result = get_count_cooccur(articles, func=cooccur_func)
    pmi = get_pmi(result["cooccur"], result["count"],
                  float(result["articles"]), num_ideas=num_ideas)
    articles_group = get_time_grouped_articles(articles, group_by=group_by)
    info_dict = {k: get_count_cooccur(articles_group[k], func=cooccur_func)
                 for k in articles_group}
    ts_correlation = get_ts_correlation(info_dict, num_ideas,
                                        normalize=True)
    xs, ys = [], []
    for i in range(num_ideas):
        for j in range(i + 1, num_ideas):
            if np.isnan(pmi[i, j]) or np.isnan(ts_correlation[i, j]):
                continue
            if np.isinf(pmi[i, j]) or np.isinf(ts_correlation[i, j]):
                continue
            xs.append(ts_correlation[i, j])
            ys.append(pmi[i, j])
    if write_tests:
        with open("%s/%s_test.jsonlist" % (plot_dir, prefix), "w") as fout:
            k, p = ss.mstats.normaltest(xs)
            fout.write("%s\n" % json.dumps(
                {"name": "correlation normality test",
                 "k2": None if np.ma.is_masked(k) else k, "p-value": p}))
            k, p = ss.mstats.normaltest(ys)
            fout.write("%s\n" % json.dumps(
                {"name": "PMI normality test",
                 "k2": None if np.ma.is_masked(k) else k, "p-value": p}))
            if unimodality_test:
                d, p = diptest.diptest(np.array(xs))
                fout.write("%s\n" % json.dumps(
                    {"name": "correlation unimodality test",
                     "d": None if np.ma.is_masked(k) else d, "p-value": p}))
                d, p = diptest.diptest(np.array(ys))
                fout.write("%s\n" % json.dumps(
                    {"name": "PMI unimodality test",
                     "d": None if np.ma.is_masked(k) else d, "p-value": p}))
            c, p = ss.pearsonr(xs, ys)
            fout.write("%s\n" % json.dumps(
                {"name": "correlation between correlation and PMI",
                 "coef": c, "p-value": p}))
    filename = "%s/%s_joint_plot.pdf" % (plot_dir, prefix)
    if make_plots:
        fig = pf.joint_plot(np.array(xs), np.array(ys),
                            xlabel="prevalence correlation",
                            ylabel="cooccurrence",
                            xlim=(-1, 1))
        pf.savefig(fig, filename)
    return pmi, ts_correlation, filename


def get_combined_extreme_pairs(pmi, corr, idea_names, output_file, count=100):
    combined = np.multiply(pmi, corr)
    combined[np.isinf(combined)] = 0
    all_pairs = []
    _, top = utils.get_extreme_pairs(np.multiply(combined,
        (pmi > 0).astype(float), (corr > 0).astype(float)), count=count)
    all_pairs.extend([(abs(combined[i, j]), (i, j)) for i, j in top])
    _, top = utils.get_extreme_pairs(np.multiply(combined,
        (pmi < 0).astype(float), (corr < 0).astype(float)), count=count)
    all_pairs.extend([(abs(combined[i, j]), (i, j)) for i, j in top])
    _, top = utils.get_extreme_pairs(np.multiply(-combined,
        (pmi > 0).astype(float), (corr < 0).astype(float)), count=count)
    all_pairs.extend([(abs(combined[i, j]), (i, j)) for i, j in top])
    _, top = utils.get_extreme_pairs(np.multiply(-combined,
        (pmi < 0).astype(float), (corr > 0).astype(float)), count=count)
    all_pairs.extend([(abs(combined[i, j]), (i, j)) for i, j in top])
    all_pairs.sort(reverse=True)
    all_pairs = [v[1] for v in all_pairs]
    with io.open(output_file, "w", encoding="utf-8") as fout:
        used = set()
        for i, j in all_pairs:
            if (i, j) in used:
                continue
            used.add((i, j))
            pair_type = "None"
            if pmi[i, j] > 0 and corr[i, j] > 0:
                pair_type = "friends"
            if pmi[i, j] < 0 and corr[i, j] > 0:
                pair_type = "arms-race"
            if pmi[i, j] < 0 and corr[i, j] < 0:
                pair_type = "head-to-head"
            if pmi[i, j] > 0 and corr[i, j] < 0:
                pair_type = "tryst"
            fout.write(u"%s\t%f\t%f\t%f\t%s\t%s\n" % (pair_type,
                combined[i, j], pmi[i, j], corr[i, j],
                idea_names[i], idea_names[j]))


def plot_top_pairs(articles, idea_names, prefix, num_ideas,
                   strength_file, output_dir,
                   top=5, cooccur_func=None, group_by="year"):
    type_list = collections.defaultdict(list)
    articles_group = get_time_grouped_articles(articles, group_by=group_by)
    info_dict = {k: get_count_cooccur(articles_group[k], func=cooccur_func)
                 for k in articles_group}
    ts_matrix = get_time_series(info_dict, num_ideas=num_ideas, normalize=True)
    with open(strength_file, 'r') as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if parts[4].startswith("ion,ing,"):
                continue
            if parts[5].startswith("ion,ing,"):
                continue
            type_list[parts[0]].append((float(parts[2]), 
                parts[4], parts[5]))
    xvalues = range(ts_matrix.shape[1])
    filename_map = {}
    for category in ["friends", "arms-race", "head-to-head", "tryst"]:
        for rank, t in enumerate(type_list[category][:top]):
            pmi, fst, snd = t
            fig, filename = plot_pair(ts_matrix, idea_names, fst, snd,
                                      category, prefix, output_dir,
                                      save_file=True, 
                                      xticklabels=articles_group.keys(),
                                      step=5,
                                      ylabel="frequency",
                                      xlabel="time periods",
                                      xlabel_rotation=30,
                                      fig_pos=[0.2, 0.2, 0.75, 0.75])
            file_key = "%s_%d" % (category.replace("-", ""), rank + 1)
            filename_map[file_key] = filename
    return filename_map


def plot_pair(ts_matrix, idea_names, fst, snd, category, prefix, output_dir,
              save_file=True, xticks=None, xticklabels=None, 
              step=5, ylabel="frequency",
              xlabel="time periods", short_idea_names=None,
              fig_pos=[0.2, 0.15, 0.75, 0.8], xlabel_rotation=None, ylim=None,
              yticks=None, shapes=None, linewidth=5,
              rc=None, fig_size=pf.FIG_SIZE,
              despine=False, ticksize=None, style="white", xlim=None):
    if type(idea_names) == dict:
        reverse_idea_names = utils.get_reverse_dict(idea_names)
    else:
        reverse_idea_names = {d: i for (i, d) in enumerate(idea_names)}
    xvalues = range(ts_matrix.shape[1])
    filename = "%s/%s_%s_%s_%s.pdf" % (output_dir, prefix, category,
                                       fst[:15].replace(" ", "_"),
                                       snd[:15].replace(" ", "_"))
    colors = pf.sns.xkcd_palette([COLOR_DICT[category]] * 2)
    if short_idea_names:
        legend = [short_idea_names[fst], short_idea_names[snd]]
    else:
        legend = [fst, snd]
    if xticks is None:
        if xticklabels is None:
            xticklabel = None
        else:
            xticklabel = (
                [xvalues[i] for i in range(0, len(xvalues), step)],
                [xticklabels[i] for i in range(0, len(xvalues), step)]) \
                    if xticklabels else None
    else:
        if xticklabels is None:
            xticklabel = (xticks, xticks)
        else:
            xticklabel = (xticks, xticklabels)
    if yticks is not None:
        yticklabel=(yticks, yticks)
    else:
        yticklabel=None
    fig = pf.plot_lines([xvalues, xvalues],
                        [ts_matrix[reverse_idea_names[fst],:],
                         ts_matrix[reverse_idea_names[snd], :]],
                        colors=colors, legend=legend,
                        linestyles=["-", "--"],
                        xticklabel=xticklabel,
                        yticklabel=yticklabel,
                        ylabel=ylabel,
                        xlabel=xlabel,
                        xlim=xlim,
                        style=style,
                        fig_pos=fig_pos,
                        xlabel_rotation=xlabel_rotation, ylim=ylim,
                        shapes=shapes, linewidth=linewidth, rc=rc,
                        fig_size=fig_size, despine=despine, ticksize=ticksize,
                        filename=filename)
    return fig, filename


def plot_average_top_strength(strength_file, prefix, output_dir, top=25):
    relations = ["friends", "tryst", "head-to-head", "arms-race"]
    colors = pf.sns.xkcd_palette([COLOR_DICT[r] for r in relations])
    value_lists = [[] for _ in relations]
    errorbar_lists = [[] for _ in relations]
    strength, sems = st.get_relation_strength(strength_file,
                                              top=top, return_sem=True)
    for (i, r) in enumerate(relations):
        value_lists[i].append(strength[r])
        errorbar_lists[i].append(sems[r])
    filename = "%s/%s_average_top_%d.pdf" % (output_dir, prefix, top)
    fig = pf.plot_bar(value_lists, errorbar_list=errorbar_lists,
                      color_list=colors,
                      fig_size=(8, 7),
                      xticklabel=([0], [""]),
                      xlim=(0, 1),
                      fig_pos=(0.15, 0.05, 0.82, 0.85),
                      legend=relations, ylabel="strength",
                      filename=filename)
    return filename


def generate_all_outputs(articles, num_ideas, idea_names, prefix,
                         output_dir, cooccur_func, table_top=5,
                         group_by="year"):
    figure_dir = "%s/figure" % output_dir
    table_dir = "%s/table" % output_dir
    if not os.path.exists(figure_dir) or not os.path.exists(table_dir):
        os.makedirs(figure_dir)
        os.makedirs(table_dir)
    info = {}
    pmi, ts_corr, joint_file = generate_scatter_dist_plot(
        articles, num_ideas,
        figure_dir, prefix,
        cooccur_func=cooccur_func,
        group_by=group_by
    )
    info["joint_file"] = joint_file
    strength_file = "%s/%s_comb_extreme_pairs.txt" % (figure_dir, prefix)
    get_combined_extreme_pairs(pmi, ts_corr, idea_names, strength_file,
                               count=100)
    
    # generate strength figure
    average_file = plot_average_top_strength(strength_file, prefix,
                                             figure_dir, top=25)
    info["average_file"] = average_file
    
    # generate figures
    filename_map = plot_top_pairs(articles, idea_names, prefix, num_ideas,
                                  strength_file, figure_dir,
                                  top=5,
                                  cooccur_func=cooccur_func,
                                  group_by=group_by)
    for k in filename_map:
        info[k] = filename_map[k]

    # generate tables
    st.get_top_relationship(strength_file,
                            "%s/%s_top_five.tex" % (table_dir, prefix),
                            top=table_top)
    info["table_file"] = "%s/%s_top_five.tex" % (table_dir, prefix)
    st.get_top_relationship(strength_file,
                            "%s/%s_top_50.tex" % (table_dir, prefix),
                            top=50)
    # generate pdf
    for k in info.keys():
        info[k] = info[k][len(output_dir) + 1:]
        if info[k].endswith(".pdf"):
            info[k] = "{{%s}%s}" % (info[k][:-4], info[k][-4:])
        else:
            info[k] = "{%s}" % (info[k])
    tex_file = "%s/%s_main.tex" % (output_dir, prefix)
    to.write_tex_file(tex_file, info)
    if HAS_LATEX:
        cwd = os.getcwd()
        os.chdir(output_dir)
        os.system("%s/mklatex.sh %s" % (cwd, tex_file))
        os.chdir(cwd)

