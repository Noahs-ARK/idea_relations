# -*- coding: utf-8 -*-

import io
import collections
import json
import numpy as np
import scipy.stats as ss
try:
    from diptest import diptest
    normality_test = True
except:
    normality_test=False
import framework as fc
import seaborn as sns
import plot_functions as pf
import util

reload(pf)

COLOR_DICT = {
        "friends": "blue green",
        "arms-race": "dark sky blue",
        "head-to-head": "blood red",
        "romance": "rose",
        }


def generate_scatter_dist_plot(articles, total, plot_dir, prefix, cooccur_func=None,
        make_plots=True, write_tests=True, group_by="year", samples=1000):
    result = fc.get_count_cooccur(articles, func=cooccur_func)
    pmi = fc.get_pmi(result["cooccur"], result["count"],
            float(result["articles"]), total_frames=total)
    articles_group = fc.get_time_grouped_articles(articles, group_by=group_by)
    info_dict = {k: fc.get_count_cooccur(articles_group[k], func=cooccur_func)
            for k in articles_group}
    ts_correlation = fc.get_ts_correlation(info_dict, total_frames=total)
    normalized_ts_correlation = fc.get_ts_correlation(info_dict,
            normalize=True, total_frames=total)
    xs, normalized_xs, ys = [], [], []
    for i in range(total):
        for j in range(i + 1, total):
            if np.isnan(pmi[i, j]) or np.isnan(ts_correlation[i, j]) \
                    or np.isnan(normalized_ts_correlation[i, j]):
                continue
            if np.isinf(pmi[i, j]) or np.isinf(ts_correlation[i, j]) \
                    or np.isinf(normalized_ts_correlation[i, j]):
                continue
            xs.append(ts_correlation[i, j])
            normalized_xs.append(normalized_ts_correlation[i, j])
            ys.append(pmi[i, j])
    if write_tests:
        with open("%s/%s_test.jsonlist" % (plot_dir, prefix), "w") as fout:
            k, p = ss.mstats.normaltest(xs)
            fout.write("%s\n" % json.dumps({"name": "correlation normality test",
                "k2": None if np.ma.is_masked(k) else k, "p-value": p}))
            k, p = ss.mstats.normaltest(normalized_xs)
            fout.write("%s\n" % json.dumps({"name": "normalized correlation normality test",
                "k2": None if np.ma.is_masked(k) else k, "p-value": p}))
            k, p = ss.mstats.normaltest(ys)
            fout.write("%s\n" % json.dumps({"name": "PMI normality test",
                "k2": None if np.ma.is_masked(k) else k, "p-value": p}))
            d, p = diptest.diptest(np.array(xs))
            fout.write("%s\n" % json.dumps({"name": "correlation unimodality test",
                "d": None if np.ma.is_masked(k) else d, "p-value": p}))
            d, p = diptest.diptest(np.array(normalized_xs))
            fout.write("%s\n" % json.dumps({"name": "normalized correlation unimodality test",
                "d": None if np.ma.is_masked(k) else d, "p-value": p}))
            d, p = diptest.diptest(np.array(ys))
            fout.write("%s\n" % json.dumps({"name": "PMI unimodality test",
                "d": None if np.ma.is_masked(k) else d, "p-value": p}))
            c, p = ss.pearsonr(xs, ys)
            fout.write("%s\n" % json.dumps({"name": "correlation between correlation and PMI",
                "coef": c, "p-value": p}))
            c, p = ss.pearsonr(normalized_xs, ys)
            fout.write("%s\n" % json.dumps({"name": "normalized correlation between correlation and PMI",
                "coef": c, "p-value": p}))
    if make_plots:
        fig = pf.distplot(xs, xlabel="correlation", ylabel="count", fig_pos=[0.18, 0.15, 0.75, 0.8],
                xtickgap=0.4)
        pf.savefig(fig, "%s/%s_correlation_dist.pdf" % (plot_dir, prefix))
        fig = pf.distplot(normalized_xs, xlabel="correlation", ylabel="count", fig_pos=[0.18, 0.15, 0.75, 0.8],
                xtickgap=0.4)
        pf.savefig(fig, "%s/%s_normalized_correlation_dist.pdf" % (plot_dir, prefix))
        fig = pf.distplot(ys, xlabel="PMI", ylabel="count", fig_pos=[0.18, 0.15, 0.75, 0.8],
                xtickgap=0.4)
        pf.savefig(fig, "%s/%s_pmi_dist.pdf" % (plot_dir, prefix))
        if len(xs) > samples:
            indexes = np.random.choice(range(len(xs)), samples, replace=False)
            plot_xs = [xs[i] for i in indexes]
            plot_normalized_xs = [normalized_xs[i] for i in indexes]
            plot_ys = [ys[i] for i in indexes]
        else:
            plot_xs, plot_normalized_xs, plot_ys = xs, normalized_xs, ys
        # lim = np.max(np.abs(plot_ys))
        lim = 1
        fig, corr_sigma = pf.scatter_plot(plot_xs, plot_ys, xlabel="correlation", ylabel="PMI",
                fig_pos=[0.18, 0.15, 0.75, 0.8], add_pca=True, full_x=xs, full_y=ys,
                xlim=(-lim, lim), ylim=(-lim, lim))
        pf.savefig(fig, "%s/%s_correlation_pmi.pdf" % (plot_dir, prefix))
        fig, normalized_corr_sigma = pf.scatter_plot(plot_normalized_xs, plot_ys, xlabel="normalized correlation", ylabel="PMI",
                fig_pos=[0.18, 0.15, 0.75, 0.8], add_pca=True, full_x=normalized_xs, full_y=ys,
                xlim=(-lim, lim), ylim=(-lim, lim))
        pf.savefig(fig, "%s/%s_normalized_correlation_pmi.pdf" % (plot_dir, prefix))
        fig = pf.joint_plot(np.array(normalized_xs), np.array(ys), xlabel="prevalence correlation", ylabel="cooccurrence",
                xlim=(-1, 1))
        pf.savefig(fig, "%s/%s_joint_plot.pdf" % (plot_dir, prefix))
        if write_tests:
            with open("%s/%s_pca.jsonlist" % (plot_dir, prefix), "w") as fout:
                fout.write("%s\n" % json.dumps({"corr_sigma": list(corr_sigma),
                    "normalized_corr_sigma": list(normalized_corr_sigma)}))
    return pmi, ts_correlation, normalized_ts_correlation


def get_most_extreme_pairs(pmi, corr, topic_map, output_file, count=100):
    pmi_bottom, pmi_top = util.get_extreme_pairs(pmi, count=count)
    corr_bottom, corr_top = util.get_extreme_pairs(corr, count=count)
    with io.open(output_file, "w", encoding="utf-8") as fout:
        fout.write(u"friends\n")
        friends = pmi_top & corr_top
        for i, j in friends:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")
        
        fout.write(u"romance\n")
        fads = pmi_top & corr_bottom
        for i, j in fads:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")

        fout.write(u"head-to-head\n")
        comps = pmi_bottom & corr_bottom
        for i, j in comps:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")

        fout.write(u"arms-race\n")
        rivals = pmi_bottom & corr_top 
        for i, j in rivals:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")

        fout.write(u"pmi top\n")
        for i, j in pmi_top:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")
        
        fout.write(u"pmi bottom\n")
        for i, j in pmi_bottom:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")
        
        fout.write(u"corr top\n")
        for i, j in corr_top:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")
        
        fout.write(u"corr bottom\n")
        for i, j in corr_bottom:
            fout.write(u"%s\t%s\n" % (topic_map[i], topic_map[j]))
        fout.write(u"\n")

def get_combined_extreme_pairs(pmi, corr, topic_map, output_file, count=100):
    combined = np.multiply(pmi, corr)
    combined[np.isinf(combined)] = 0
    all_pairs = []
    _, top = util.get_extreme_pairs(np.multiply(combined,
        (pmi > 0).astype(float), (corr > 0).astype(float)), count=count)
    all_pairs.extend([(abs(combined[i, j]), (i, j)) for i, j in top])
    _, top = util.get_extreme_pairs(np.multiply(combined,
        (pmi < 0).astype(float), (corr < 0).astype(float)), count=count)
    all_pairs.extend([(abs(combined[i, j]), (i, j)) for i, j in top])
    _, top = util.get_extreme_pairs(np.multiply(-combined,
        (pmi > 0).astype(float), (corr < 0).astype(float)), count=count)
    all_pairs.extend([(abs(combined[i, j]), (i, j)) for i, j in top])
    _, top = util.get_extreme_pairs(np.multiply(-combined,
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
                pair_type = "romance"
            fout.write(u"%s\t%f\t%f\t%f\t%s\t%s\n" % (pair_type,
                combined[i, j], pmi[i, j], corr[i, j],
                topic_map[i], topic_map[j]))


def plot_top_pairs(articles, topic_map, prefix, total, table_file, output_dir,
        top=3, cooccur_func=None, group_by="year"):
    type_list = collections.defaultdict(list)
    articles_group = fc.get_time_grouped_articles(articles, group_by=group_by)
    info_dict = {k: fc.get_count_cooccur(articles_group[k], func=cooccur_func)
            for k in articles_group}
    ts_matrix = fc.get_time_series(info_dict, total_frames=total, normalize=True)
    if type(topic_map) == dict:
        reverse_topic_map = util.get_reverse_dict(topic_map)
    else:
        reverse_topic_map = {d: i for (i, d) in enumerate(topic_map)}
    with open(table_file, 'r') as fin:
        for line in fin:
            parts = line.strip().split("\t")
            if parts[4].startswith("ion,ing,"):
                continue
            if parts[5].startswith("ion,ing,"):
                continue
            type_list[parts[0]].append((float(parts[2]), 
                parts[4], parts[5]))
    xvalues = range(ts_matrix.shape[1])
    for category in ["friends", "arms-race", "head-to-head", "romance"]:
        for pmi, fst, snd in type_list[category][:top]:
            filename = "%s/%s_%s_%s_%s.pdf" % (output_dir, prefix, category, fst[:15], snd[:15])
            colors = sns.xkcd_palette([COLOR_DICT[category]] * 2)
            fig = pf.plot_lines([xvalues, xvalues],
                    [ts_matrix[reverse_topic_map[fst],:],
                        ts_matrix[reverse_topic_map[snd], :]],
                    colors=colors, legend=[fst.replace("_", " "), snd.replace("_", " ")],
                    linestyles=["-", "--"],
                    ylabel="frequency", xlabel="time periods",
                    fig_pos=[0.2, 0.15, 0.75, 0.8])
            pf.savefig(fig, filename)

def plot_pair(ts_matrix, topic_map, fst, snd, category, prefix, output_dir,
        save_file=True, xticks=None, xticklabels=None, step=5, ylabel="frequency",
        xlabel="time periods", short_topic_names=None,
        fig_pos=[0.2, 0.15, 0.75, 0.8], xlabel_rotation=None, ylim=None,
             yticks=None, shapes=None, linewidth=5, rc=None, fig_size=pf.FIG_SIZE,
             despine=False, ticksize=None, style="whitegrid", xlim=None):
    if type(topic_map) == dict:
        reverse_topic_map = util.get_reverse_dict(topic_map)
    else:
        reverse_topic_map = {d: i for (i, d) in enumerate(topic_map)}
    xvalues = range(ts_matrix.shape[1])
    filename = "%s/%s_%s_%s_%s.pdf" % (output_dir, prefix, category, fst[:15], snd[:15])
    colors = sns.xkcd_palette([COLOR_DICT[category]] * 2)
    if short_topic_names:
        legend = [short_topic_names[fst], short_topic_names[snd]]
    else:
        legend = [fst, snd]
    if xticks is None:
        if xticklabels is None:
            xticklabel = None
        else:
            xticklabel = ([xvalues[i] for i in range(0, len(xvalues), step)],
                    [xticklabels[i] for i in range(0, len(xvalues), step)]) if xticklabels else None
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
            [ts_matrix[reverse_topic_map[fst],:],
                ts_matrix[reverse_topic_map[snd], :]],
            colors=colors, legend=legend,
            linestyles=["-", "--"],
            xticklabel=xticklabel,
            yticklabel=yticklabel,
            ylabel=ylabel, xlabel=xlabel,
            xlim=xlim,
            style=style,
            fig_pos=fig_pos, xlabel_rotation=xlabel_rotation, ylim=ylim,
                       shapes=shapes, linewidth=linewidth, rc=rc,
                        fig_size=fig_size, despine=despine, ticksize=ticksize)    
    if save_file:
        fig.savefig(filename, bbox_inches='tight')
    return fig


def get_color_label(pair):
    if pair.pmi > 0 and pair.correlation > 0:
        return sns.xkcd_rgb[COLOR_DICT["friends"]], "friends (#%d)" % pair.rank
    elif pair.pmi > 0 and pair.correlation < 0:
        return sns.xkcd_rgb[COLOR_DICT["romance"]], "romance (#%d)" % pair.rank
    elif pair.pmi < 0 and pair.correlation > 0:
        return sns.xkcd_rgb[COLOR_DICT["arms-race"]], "arms-race (#%d)" % pair.rank
    elif pair.pmi < 0 and pair.correlation < 0:
        return sns.xkcd_rgb[COLOR_DICT["head-to-head"]], "head-to-head (#%d)" % pair.rank
    else:
        return "black", "wrong"


def plot_pair_graph(pairs, output_file=None, short_topic_names=None):
    nodes, edges, node_dict, edge_dict = [], [], {}, {}
    edge_colors, edge_labels = [], []
    node_id = 0
    graph = nx.Graph()
    for p in pairs:
        if p.fst_topic not in node_dict:
            node_dict[p.fst_topic] = node_id
            nodes.append(node_id)
            graph.add_node(node_id, label=p.fst_topic if not short_topic_names \
                    else short_topic_names[p.fst_topic],
                    fontsize=70)
            node_id += 1
        if p.snd_topic not in node_dict:
            node_dict[p.snd_topic] = node_id
            nodes.append(node_id)
            graph.add_node(node_id, label=p.snd_topic if not short_topic_names \
                    else short_topic_names[p.snd_topic],
                    fontsize=70)
            node_id += 1
        edges.append((node_dict[p.fst_topic], node_dict[p.snd_topic]))
        c, l = get_color_label(p)
        edge_colors.append(c)
        edge_labels.append(l)
        graph.add_edge(node_dict[p.fst_topic], node_dict[p.snd_topic],
                color=c, label=l, fontsize=70, penwidth=10,
                labelloc="c")
    reverse_node_dict = util.get_reverse_dict(node_dict)
    if short_topic_names:
        reverse_node_dict = {n: short_topic_names[reverse_node_dict[n]]
                for n in reverse_node_dict}
    """
    graph = igraph.Graph(directed=False)
    graph.add_vertices(nodes)
    graph.add_edges(edges)
    graph.vs["name"] = [reverse_node_dict[n] for n in nodes]
    graph.es["name"] = edge_labels
    visual_style = {}
    visual_style["vertex_shape"] = "none"
    visual_style["vertex_size"] = 20
    visual_style["vertex_label_size"] = 20
    # visual_style["vertex_size"] = 0
    visual_style["vertex_color"] = "white"
    visual_style["vertex_label"] = graph.vs["name"]
    visual_style["edge_label"] = graph.es["name"]
    visual_style["edge_label_size"] = 20
    visual_style["edge_label_dist"] = 
    # visual_style["edge_width"] = [5 * abs(v) for v in graph.es["weight"]]
    visual_style["edge_width"] = [10 for _ in edge_colors]
    visual_style["edge_color"] = edge_colors
    visual_style["bbox"] = (600, 600)
    visual_style["margin"] = 100
    visual_style["layout"] = graph.layout("fr")
    if output_file:
        igraph.plot(graph, output_file, **visual_style)
    """
    
    # switch to graphviz and pydot as the graph engine
    """
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    pos = nx.spring_layout(graph)
    # nodes
    fig, ax = pf.start_plotting([8, 8], [0, 0, 1, 1], style="white")
    nx.draw_networkx_nodes(graph, pos,
            node_color="white",
            # node_shape=None,
            node_size=10000,
            ax=ax)
    # edges
    nx.draw_networkx_edges(graph, pos,
            edgelist=edges,
            width=8,
            alpha=1.0,
            edge_color=edge_colors,
            ax=ax)
    nx.draw_networkx_labels(graph, pos,
            labels=reverse_node_dict,
            font_size=25,
            font_color="#929591",
            ax=ax)
    nx.draw_networkx_edge_labels(graph, pos,
            edge_labels=dict(zip(edges, edge_labels)),
            font_size=20,
            alpha=0.6,
            rotate=False,
            ax=ax)
    ax.set_axis_off()
    """
    return graph


def write_tex_from_edges(pairs, output_file=None, short_topic_names=None):
    graph = plot_pair_graph(pairs, short_topic_names=short_topic_names)
    with open(output_file, "w") as fout:
        fout.write("\\begin{tikzpicture}\n")
        for n, d in graph.nodes_iter(data=True):
            fout.write(r'\node[text width=1.2in]')
            fout.write("(%s) at (0, 0) {%s}" % (n,
                d.get("label", n)))
            fout.write(";\n")
        fout.write("\n")
        for u, v, d in graph.edges_iter(data=True):
            color = d.get("label").split()[0].lower().replace("-", "")
            fout.write(r"\draw [%scolor, ultra thick]" % (color))
            fout.write(" (%s) -- (%s) " % (u, v))
            fout.write("node[pos=.5,sloped,above] {%s}" % d.get("label").replace("#", r"\#"))
            fout.write(";\n")
        fout.write("\\end{tikzpicture}\n")


def sort_matrix_indices(indices, matrix, reverse=False):
    values = [(matrix[i, j], (i, j)) for i, j in indices]
    values.sort(reverse=reverse)
    return [v[1] for v in values]


def get_single_most_extreme_pairs(pmi, corr, topic_map, output_file, count=100):
    pmi_bottom, pmi_top = util.get_extreme_pairs(pmi, count=count)
    corr_bottom, corr_top = util.get_extreme_pairs(corr, count=count)
    with io.open(output_file, "w", encoding="utf-8") as fout:
        fout.write(u"top pmi\n")
        for i, j in sort_matrix_indices(pmi_top, pmi, reverse=True):
            fout.write(u"%f & %s & %s\n" % (pmi[i, j], topic_map[i], topic_map[j]))
        fout.write(u"\n")
        
        fout.write(u"bottom pmi\n")
        for i, j in sort_matrix_indices(pmi_bottom, pmi, reverse=False):
            fout.write(u"%f & %s & %s\n" % (pmi[i, j], topic_map[i], topic_map[j]))
        fout.write(u"\n")
        
        fout.write(u"top correlation\n")
        for i, j in sort_matrix_indices(corr_top, corr, reverse=True):
            fout.write(u"%f & %s & %s\n" % (corr[i, j], topic_map[i], topic_map[j]))
        fout.write(u"\n")
        
        fout.write(u"bottom correlation\n")
        for i, j in sort_matrix_indices(corr_bottom, corr, reverse=False):
            fout.write(u"%f & %s & %s\n" % (corr[i, j], topic_map[i], topic_map[j]))
        fout.write(u"\n")
        

def get_ranking(value, matrix):
    num_rows = matrix.shape[0]
    array = []
    for i in range(num_rows):
        for j in range(i + 1, num_rows):
            array.append(matrix[i, j])
    if value > 0:
        array.sort(reverse=True)
    else:
        array.sort()
    array = np.array(array)
    index = np.argmin(np.abs(array - value))
    return index + 1


def plot_pmi_by_year(pmi_dict, topic_map, fst, snd, category, prefix, output_dir,
        save_file=True, xticklabels=None, step=5, ylabel="PMI",
        xlabel="time periods", short_topic_names=None,
        fig_pos=[0.2, 0.15, 0.75, 0.8], xlabel_rotation=None, ylim=None):
    if type(topic_map) == dict:
        reverse_topic_map = util.get_reverse_dict(topic_map)
    else:
        reverse_topic_map = {d: i for (i, d) in enumerate(topic_map)}
    xvalues = pmi_dict.keys()
    xvalues.sort()
    filename = "%s/%s_%s_%s_%s.pdf" % (output_dir, prefix, category, fst[:15], snd[:15])
    colors = sns.xkcd_palette([COLOR_DICT[category]] * 2)
    i, j = reverse_topic_map[fst], reverse_topic_map[snd]
    yvalues = [pmi_dict[x][i, j] for x in xvalues]
    fig = pf.plot_lines([xvalues], [yvalues],
            colors=colors,
            linestyles=["-", "--"],
            xticklabel=([xvalues[i] for i in range(0, len(xvalues), step)],
                [xticklabels[i] for i in range(0, len(xvalues), step)]) if xticklabels else None,
            ylabel=ylabel, xlabel=xlabel,
            fig_pos=fig_pos, xlabel_rotation=xlabel_rotation, ylim=ylim)
    if save_file:
        pf.savefig(fig, filename)
    return fig



def generate_all_outputs(articles, num_ideas, topic_map, prefix,
                         output_dir, cooccur_func):
    if not os.path.exists(output_dir) or not os.path.exists:
        os.makedirs(output_dir)
        os.makedirs("%s/figure" % output_dir)
        os.makedirs("%s/table" % output_dir)
    pmi, ts_corr = generate_scatter_dist_plot(articles, total,
                                              output_dir, prefix,
                                              cooccur_func=cooccur_func)
    get_combined_extreme_pairs(pmi, normalized_ts_corr, topic_map,
                               "%s/%s_comb_extreme_pairs.txt" % (curr_dir, prefix), count=100)
    # generate figures
    ef.plot_top_pairs(articles, topic_map, prefix, total,
                      "%s/%s_comb_extreme_pairs.txt" % (curr_dir, prefix),
                      curr_dir, top=3, cooccur_func=cooccur_func, group_by="year")
    # generate tables



