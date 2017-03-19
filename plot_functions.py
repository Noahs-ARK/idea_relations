"""Adapt plot functions with seaborn to get more beautiful plots."""

from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import collections
import itertools
import numpy as np
import scipy.stats as ss
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Arrow
import matplotlib.dates as mdates
from pandas import DataFrame

LINEWIDTH = 5
sns.set(font_scale=3.0, rc={
    "lines.linewidth": LINEWIDTH,
    "lines.markersize":20,
    "ps.useafm": True,
    "font.sans-serif": ["Helvetica"],
    "pdf.use14corefonts" : True,
    "text.usetex": True,
    })
FIG_SIZE = (8, 7)
logging.basicConfig(level=logging.INFO)
COLOR_NAMES = ["cerulean", "light red", "seafoam", "dark orange",
        "burgundy", "dark magenta", "midnight blue", "light violet",
        "ocean blue", "bluish purple", "pinkish", "pale orange",
        "aqua green", "pumpkin", "chocolate", "pine green"]

def start_plotting(fig_size, fig_pos, style="whitegrid", rc=None, despine=False):
    with sns.axes_style(style, rc):
        fig = plt.figure(figsize=fig_size)
        if not fig_pos:
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_axes(fig_pos)
    if despine:
        sns.despine(left=True)
    return fig, ax


def frange(x, y, step):
    while x < y:
        yield x
        x += step


def end_plotting(fig, ax, title=None, xlabel=None,
    ylabel=None, xlim=None, ylim=None, filename=None,
    xticklabel=None, xlabel_rotation=None,
    yticklabel=None, ylabel_rotation=None, label_text=None,
    xtickgap=None):
    '''A set of common operations after plotting.'''
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if xticklabel:
        ax.set_xticks(xticklabel[0])
        ax.set_xticklabels(xticklabel[1], rotation=xlabel_rotation)
    elif xtickgap:
        xticks = ax.get_xticks()
        ax.set_xticks(list(frange(min(xticks), max(xticks) + 0.001, xtickgap)))
    else:
        ax.set_xticks(ax.get_xticks())
        ax.set_xticklabels(ax.get_xticks())
    if yticklabel:
        ax.set_yticks(yticklabel[0])
        ax.set_yticklabels(yticklabel[1], rotation=ylabel_rotation)
    else:
        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(ax.get_yticks())
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if label_text:
        for x, y, t in label_text:
            ax.text(x, y, t)


def savefig(fig, filename):
    fig.savefig(filename)
    plt.close()

def ax_plot_lines(ax, xs, ys, colors, shapes, linestyles,
        errorbar=False, linewidth=LINEWIDTH):
    lines = []
    for (x, y, c, s, l) in zip(xs, ys, colors, shapes, linestyles):
        if errorbar:
            # y should be a list of lists in this case
            mean = [np.mean(yl) for yl in y]
            error = [ss.sem(yl) for yl in y]
            l = ax.errorbar(x, mean, yerr=error, color=c,
                marker=s, linestyle=l, ecolor=c)
        else:
            l, = ax.plot(x, y, color=c, marker=s, linestyle=l, linewidth=linewidth)
        lines.append(l)
    return lines

def plot_lines(xs, ys, title=None, xlabel=None,
        ylabel=None, xlim=None, ylim=None,
        colors=None, shapes=None, linestyles=None,
        errorbar=False, legend=None, loc=0,
        xticklabel=None, yticklabel=None,
        xlabel_rotation=None, ylabel_rotation=None,
        hlines=None, vlines=None, bbox_to_anchor=None,
        fig_pos=None, fig_size=FIG_SIZE, label_text=None,
        xdate=False, linewidth=LINEWIDTH, rc=None, despine=False,
              ticksize=None, style="whitegrid"):
    '''Plot lines for all pairs of xs and ys.
    Input:
        xs: a list of x-value lists
        ys: a list of y-value lists ([[1, 2, 3]], two layers),
            or a list of y-value lists grouped by x ([[[1, 2, 3], [1, 3]]],
                three layers),
            in the second case, errorbar is computed using standard error
        ys should be with the same length of xs
    '''
    fig, ax = start_plotting(fig_size, fig_pos, rc=rc, despine=despine,
            style=style)
    if not colors:
        colors = sns.color_palette(n_colors=len(xs))
    if not shapes:
        shapes = ['o', 's', '^', 'd', '+', 'v', '*', 'x'] * (len(xs) // 8 + 1)
    if not linestyles:
        linestyles = ['-'] * len(xs)
    if xdate:
        xs = [[mdates.date2num(d) for d in l] for l in xs]
    lines = ax_plot_lines(ax, xs, ys, colors, shapes, linestyles, 
        errorbar=errorbar, linewidth=linewidth)
    if xdate:
        # this works for years and month
        # customize through xlabel and xticklabels
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        yearsFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        fig.autofmt_xdate()
    if hlines:
        for y in hlines:
            ax.axhline(y=y, linestyle='--', color='black')
    if vlines:
        for x in vlines:
            ax.axvline(x=x, linestyle='--', color='black')
    if legend:
        ax.legend(lines, legend, loc=loc, bbox_to_anchor=bbox_to_anchor,
            frameon=False)
    if not xlim:
        diff = np.max(xs) - np.min(xs)
        xlim = (np.min(xs) - 0.02 * diff, np.max(xs) + 0.02 * diff)
    end_plotting(fig, ax, title=title, xlabel=xlabel,
        ylabel=ylabel, xlim=xlim, ylim=ylim,
        xticklabel=xticklabel, yticklabel=yticklabel,
        xlabel_rotation=xlabel_rotation,
        ylabel_rotation=ylabel_rotation, label_text=label_text)
    if ticksize is not None:
        ax.tick_params(axis='both', which='major', labelsize=ticksize)
    return fig



def scatter_plot(x, y, title=None, xlabel=None,
        ylabel=None, xlim=None, ylim=None,
        legend=None, loc=0, color='#0485d1',
        xticklabel=None, yticklabel=None,
        xlabel_rotation=None, ylabel_rotation=None,
        fig_pos=None, fig_size=FIG_SIZE, label_text=None,
        xdate=False, markersize=100, fit_reg=False,
        add_pca=False, full_x=None, full_y=None):
    fig, ax = start_plotting(fig_size, fig_pos)
    if xdate:
        x = [mdates.date2num(d) for d in x]
    # sns.regplot(x=np.array(x), y=np.array(y), color=color, ax=ax,
    #         ci=None, scatter_kws={"s": markersize}, fit_reg=fit_reg)
    ax.scatter(x, y, c=color, s=markersize)
    if add_pca:
        if full_x:
            X = np.array(zip(full_x, full_y))
        else:
            X = np.array(zip(x, y))
        pca = PCA(n_components=2)
        pca.fit(X)
        center = pca.mean_
        sigma = pca.explained_variance_
        components = pca.components_
        if components[0][1] < 0:
            components[0, :] *= -1
        if components[1][1] < 0:
            components[1, :] *= -1
        unit = 1.0 / sigma[0]
        # unit = 1.0
        for p in [
                Arrow(
                    center[0], center[1],
                    components[0][0] * sigma[0] * unit,
                    components[0][1] * sigma[0] * unit,
                    width=0.2, facecolor="#ec2d01"
                    ),
                Arrow(
                    center[0], center[1],
                    components[1][0] * sigma[1] * unit,
                    components[1][1] * sigma[1] * unit,
                    width=0.2, facecolor="#ec2d01"
                    ),
                ]:
            ax.add_patch(p)
        ax.annotate("first %.3f" % sigma[0],
                xy=center + components[0, :] * sigma[0] * unit,
                fontsize=25)
        ax.annotate("second %.3f" % sigma[1],
                xy=center + components[1, :] * sigma[1] * unit,
                fontsize=25)
    if xdate:
        # this works for years and month
        # customize through xlabel and xticklabels
        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        yearsFmt = mdates.DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(yearsFmt)
        ax.xaxis.set_minor_locator(months)
        fig.autofmt_xdate()
    end_plotting(fig, ax, title=title, xlabel=xlabel,
        ylabel=ylabel, xlim=xlim, ylim=ylim,
        xticklabel=xticklabel, yticklabel=yticklabel,
        xlabel_rotation=xlabel_rotation,
        ylabel_rotation=ylabel_rotation, label_text=label_text)
    if add_pca:
        return fig, sigma
    return fig


def convert_dict_list(dict_list):
    xs, ys = [], []
    for d in dict_list:
        keys = d.keys()
        keys.sort()
        xs.append(keys)
        ys.append([d[k] for k in keys])
    return xs, ys


def plot_dict_array(dict_list, title=None, xlabel=None,
        ylabel=None, xlim=None, ylim=None,
        colors=None, shapes=None, linestyles=None,
        errorbar=False, legend=None, loc=0, xticklabel=None,
        yticklabel=None, xlabel_rotation=None, ylabel_rotation=None,
        fig_pos=None, fig_size=FIG_SIZE, hlines=None, vlines=None,
        label_text=None, style="whitegrid"):
    '''Probably the most heavily used function.
    Given a list of dictionaries, plot each dictionary as a line,
    using keys as xvalues, values as yvalues.

    Example input: dict_list = [{0: [1, 2, 3], 1: [1,3]}]'''
    xs, ys = convert_dict_list(dict_list)
    return plot_lines(xs, ys, title=title, xlabel=xlabel,
        ylabel=ylabel, xlim=xlim, ylim=ylim,
        colors=colors, shapes=shapes, linestyles=linestyles,
        errorbar=errorbar, legend=legend, loc=loc,
        xticklabel=xticklabel, yticklabel=yticklabel,
        xlabel_rotation=xlabel_rotation, ylabel_rotation=ylabel_rotation,
        fig_pos=fig_pos, fig_size=fig_size, hlines=hlines, vlines=vlines,
        label_text=label_text, style=style)


def histogram(data, title=None, xlabel=None,
        ylabel=None, xlim=None, ylim=None,
        color='#0485d1', linestyles=None,
        errorbar=False, legend=None, loc=0, xticklabel=None,
        yticklabel=None, xlabel_rotation=None, ylabel_rotation=None,
        fig_pos=None, fig_size=FIG_SIZE, hlines=None, vlines=None,
        label_text=None):
    fig, ax = start_plotting(fig_size, fig_pos)
    sns.countplot(data, color=color, ax=ax)
    end_plotting(fig, ax, title=title, xlabel=xlabel,
        ylabel=ylabel, xlim=xlim, ylim=ylim,
        xticklabel=xticklabel, yticklabel=yticklabel,
        xlabel_rotation=xlabel_rotation,
        ylabel_rotation=ylabel_rotation, label_text=label_text)
    return fig


def distplot(data, title=None, xlabel=None,
        ylabel=None, xlim=None, ylim=None,
        color='#0485d1', linestyles=None,
        errorbar=False, legend=None, loc=0, xticklabel=None,
        yticklabel=None, xlabel_rotation=None, ylabel_rotation=None,
        fig_pos=None, fig_size=FIG_SIZE, hlines=None, vlines=None,
        label_text=None, rug=False, kde=False, xtickgap=None):
    fig, ax = start_plotting(fig_size, fig_pos)
    sns.distplot(data, color=color, ax=ax, rug=rug, kde=kde)
    end_plotting(fig, ax, title=title, xlabel=xlabel,
        ylabel=ylabel, xlim=xlim, ylim=ylim,
        xticklabel=xticklabel, yticklabel=yticklabel,
        xlabel_rotation=xlabel_rotation, xtickgap=xtickgap,
        ylabel_rotation=ylabel_rotation, label_text=label_text)
    return fig


def plot_power_law(values, xlabel=None,
        fig_size=FIG_SIZE, fig_pos=None, fit=True):
    fig, ax = start_plotting(fig_size, fig_pos)
    count = collections.defaultdict(int)
    for v in values:
        count[v] += 1
    x_values = count.keys()
    x_values.sort()
    y_values = []
    for (i, v) in enumerate(x_values):
        if i == 0:
            y_values.append(count[v])
        else:
            y_values.append(count[v] + y_values[-1])
    denom = float(y_values[-1])
    y_values = [0] + y_values[:-1]
    y_values = np.array(y_values)
    y_values = 1 - y_values / denom
    x_values = np.array(x_values)
    if x_values[0] == 0:
        x_values = x_values + 1
    ax.loglog(x_values, y_values, '-or')
    if fit:
        log_x_values = np.log10(x_values)
        log_y_values = np.log10(y_values)
        m,b = np.polyfit(log_x_values, log_y_values, 1)
        log_fit_y_values = m * log_x_values + b
        fit_y_values = pow(10, log_fit_y_values)
        ax.loglog(x_values, fit_y_values, '--b')
        mid = int(len(x_values) / 2)
        ax.text(x_values[mid], y_values[mid], 'alpha=%.3f' % m)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$log(P(X>x))$')
    return fig



def plot_bar(value_lists, xlabel=None, fig_size=FIG_SIZE,
        fig_pos=None, ylabel=None, xticklabel=None,
        xlabel_rotation=None, width=-1, gap=1, yticklabel=None, legend=None,
        errorbar_list=None, color_list=sns.color_palette(n_colors=10), ylim=None, ncol=1,
        handlelength=None, loc=0, bbox_to_anchor=None,
        handletextpad=None, columnspacing=None, vlines=None, hlines=None,
        hatches=None):
    fig, ax = start_plotting(fig_size, fig_pos)
    N = len(value_lists[0])
    if width < 0:
        width = 0.75 * gap / len(value_lists)
    ind = np.arange(N) * gap
    rects = []
    for i in range(len(value_lists)):
        rect = ax.bar(ind, value_lists[i], width, color=color_list[i],
                yerr=errorbar_list[i] if errorbar_list else None, error_kw={"ecolor": "black"},
                hatch=None if not hatches else hatches[i])
        ind = ind + width
        rects.append(rect)
    xlim = (-width, max(ind) + width)
    if hlines:
        for y in hlines:
            ax.axhline(y=y, linestyle='--', color='grey')
    if vlines:
        for x in vlines:
            ax.axvline(x=x, linestyle='--', color='grey')
    if legend:
        ax.legend(rects, legend, loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                ncol=ncol, handlelength=handlelength,
                handletextpad=handletextpad,
                columnspacing=columnspacing,
                frameon=False)
    end_plotting(fig, ax, xlabel=xlabel, ylabel=ylabel,
            xticklabel=(ind - width * len(value_lists) / 2, xticklabel) if xticklabel else None,
            xlabel_rotation=xlabel_rotation, xlim=xlim,
            yticklabel=yticklabel, ylim=ylim)
    return fig


def ax_plot_dict_array(ax, dict_list, color, shape, line_style, xlabel):
    xs, ys = convert_dict_list(dict_list)
    ax_plot_lines(ax, xs, ys, color, shape, line_style,
            errorbar=True)
    ax.set_xticks(range(len(xlabel)))
    ax.set_xticklabels(xlabel, rotation=45)
    ax.set_xlim((-0.15, len(xlabel) -0.85))
    return ax


def stack_time_values(time_series, fig_size=FIG_SIZE, fig_pos=None,
        xlabel=None, ylabel=None, xticklabel=None, yticklabel=None,
        xlabel_rotation=None, ylabel_rotation=None, legend=None,
        palette=None, loc=0, bbox_to_anchor=None, xlim=None, ylim=None, legend_font=None):
    if not palette:
        palette = sns.xkcd_palette(COLOR_NAMES)
    fig, ax = start_plotting(fig_size, fig_pos)
    x = range(len(time_series[0]))
    low = np.zeros(len(time_series[0]))
    recs = []
    for i in range(len(time_series)):
        high = low + time_series[i]
        color = palette.pop(0)
        ax.fill_between(x, low, high, color=color)
        recs.append(Rectangle((0, 0), 0.001, 0.001, fc=color))
        low = high.copy()
    if legend:
        ax.legend(recs, legend, loc=loc, bbox_to_anchor=bbox_to_anchor,
            frameon=False, markerscale=0.5, fontsize=legend_font)
    end_plotting(fig, ax, xlabel=xlabel, ylabel=ylabel,
            xticklabel=xticklabel, ylim=ylim,
            xlabel_rotation=xlabel_rotation, xlim=xlim)
    return fig


class SubsampleJointGrid(sns.JointGrid):
    def plot_sub_joint(self, func, subsample, **kwargs):
        """Draw a bivariate plot of `x` and `y`.
        Parameters
        ----------
        func : plotting callable
            This must take two 1d arrays of data as the first two
            positional arguments, and it must plot on the "current" axes.
        kwargs : key, value mappings
            Keyword argument are passed to the plotting function.
        Returns
        -------
        self : JointGrid instance
            Returns `self`.
        """
        if subsample > 0 and subsample < len(self.x):
            indexes = np.random.choice(range(len(self.x)), subsample, replace=False)
            plot_x = np.array([self.x[i] for i in indexes])
            plot_y = np.array([self.y[i] for i in indexes])
            plt.sca(self.ax_joint)
            func(plot_x, plot_y, **kwargs)
        else:
            plt.sca(self.ax_joint)
            func(self.x, self.y, **kwargs)
        return self



def joint_plot(x, y, xlabel=None,
        ylabel=None, xlim=None, ylim=None,
        loc="best", color='#0485d1',
        size=8, markersize=50, kind="kde",
        scatter_color="r"):
    with sns.axes_style("darkgrid"):
        if xlabel and ylabel:
            g = SubsampleJointGrid(xlabel, ylabel,
                    data=DataFrame(data={xlabel: x, ylabel: y}),
                    space=0.1, ratio=2, size=size, xlim=xlim, ylim=ylim)
        else:
            g = SubsampleJointGrid(x, y, size=size,
                    space=0.1, ratio=2, xlim=xlim, ylim=ylim)
        g.plot_joint(sns.kdeplot, shade=True, cmap="Blues")
        g.plot_sub_joint(plt.scatter, 1000, s=20, c=scatter_color, alpha=0.3)
        g.plot_marginals(sns.distplot, kde=False, rug=False)
        g.annotate(ss.pearsonr, fontsize=25, template="{stat} = {val:.2g}\np = {p:.2g}")
        g.ax_joint.set_yticklabels(g.ax_joint.get_yticks())
        g.ax_joint.set_xticklabels(g.ax_joint.get_xticks())
    return g

