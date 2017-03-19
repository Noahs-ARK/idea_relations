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

