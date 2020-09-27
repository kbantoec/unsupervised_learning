import os
import matplotlib.pyplot as plt
from typing import Tuple
from pandas.core.frame import Series
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import pearsonr


def normabspath(basedir: str, filename: str):
    return os.path.normpath(os.path.join(basedir, filename))


def scatter_axis(xs: Series,
                 ys: Series,
                 class_labels: Series = None,
                 xlim: tuple = None,
                 ylim: tuple = None,
                 align_axis_zero: bool = False,
                 save: str = None,
                 name: str = None,
                 title: str = None,
                 xy: Tuple[float, float] = None,
                 annotate_corr: bool = True,
                 model: PCA = None):
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(1, 1, 1)

    # Move left y-axis and bottom x-axis to centre, passing through (0,0)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in the left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if class_labels is not None:
        plt.scatter(xs, ys, edgecolor='k', c=class_labels)
    else:
        plt.scatter(xs, ys, edgecolor='k')
    plt.xlabel(xs.name, rotation=0)
    plt.ylabel(ys.name, rotation=0)

    if annotate_corr:
        if xy is None:
            xy = (xlim[1] - 1, ylim[1] - 1)

        r, _ = pearsonr(xs, ys)
        plt.annotate(f'Pearson r = {np.round(r * 100)}%', xy=xy)

    if title is not None:
        fig.suptitle(title, fontsize=15, fontweight='bold')

    if align_axis_zero:
        xcoord: float = 0.5
        ycoord: float = 0.5

        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]

        if xlim[0] <= 0 and xlim[1] >= 0:
            xcoord = (0 - xlim[0]) / x_range

        if ylim[0] <= 0 and ylim[1] >= 0:
            ycoord = (0 - ylim[0]) / y_range

        ax.xaxis.set_label_coords(1.07, xcoord)
        ax.yaxis.set_label_coords(ycoord, 1.04)

    if model is not None:
        mean_coords = model.mean_
        first_pcs = model.components_[0, :]
        second_pcs = model.components_[1, :]

        plt.arrow(mean_coords[0], mean_coords[1], first_pcs[0], first_pcs[1], color='red', width=0.1)
        plt.arrow(mean_coords[0], mean_coords[1], second_pcs[0], second_pcs[1], color='red', width=0.1)

    if save:
        fig.savefig(f'out/plots/{name}.png', dpi=600, transparent=True)
    plt.show()