##################################### PLOTTING ###########################################################
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

import numpy as np

def singular_violinplot(data: list, y_label: str, title: str, out_path:str=None,) -> None:
    '''AAA'''
    fig, ax = plt.subplots(figsize=(2,5))

    parts = ax.violinplot(distances, widths=0.5)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(y_label, size=13) # "\u00C5" is Unicode for Angstrom
    ax.set_xticks([])
    
    quartile1, median, quartile3 = np.percentile(distances, [25, 50, 75]) #axis=1 if multiple violinplots.
    
    for pc in parts['bodies']:
        pc.set_facecolor('cornflowerblue')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    parts["cmins"].set_edgecolor("black")
    parts["cmaxes"].set_edgecolor("black")
    print([x for x in parts["bodies"]])
        
    ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
    ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
    ax.vlines(1, np.min(distances), np.max(distances), color="k", linestyle="-", lw=2)
    
    if out_path: fig.savefig(out_path, dpi=800, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_cols_dfs(dfs, df_names, cols, titles, y_labels, dims=None, out_path=None, colormap="tab20") -> None:
    ''''''
    def set_violinstyle(axes_subplot_parts, colors="cornflowerblue") -> None:
        '''
        '''
        for color, pc in zip(colors, axes_subplot_parts["bodies"]):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        axes_subplot_parts["cmins"].set_edgecolor("black")
        axes_subplot_parts["cmaxes"].set_edgecolor("black")
        return None
    
    # get colors from colormap
    colors = [mcolors.to_hex(color) for color in plt.get_cmap(colormap).colors]
    fig, ax_list = plt.subplots(1, len(cols), figsize=(3*len(cols), 5))
    fig.subplots_adjust(wspace=1.5, hspace=0.8)
    if not dims: dims = [None for x in cols]

    for ax, col, name, label, dim in zip(ax_list, cols, titles, y_labels, dims):
        ax.set_title(name, size=15, y=1.05)
        ax.set_ylabel(label, size=15)
        ax.set_xticks([])
        data = [df[col].to_list() for df in dfs]
        parts = ax.violinplot([df[col].to_list() for df in dfs], widths=0.5)
        if dim: ax.set_ylim(dim)
        set_violinstyle(parts, colors=colors)

        for i, d in enumerate(data):
            quartile1, median, quartile3 = np.percentile(d, [25, 50, 75])
            ax.scatter(i+1, median, marker='o', color="white", s=40, zorder=3)
            ax.vlines(i+1, quartile1, quartile3, color="k", linestyle="-", lw=10)
            ax.vlines(i+1, np.min(d), np.max(d), color="k", linestyle="-", lw=2)

        handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, df_names)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.1),
                  fancybox=True, shadow=True, ncol=5, fontsize=13)
        
    if out_path: fig.savefig(out_path, dpi=800, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_cols(df, cols, titles, y_labels, dims=None, out_path=None) -> None:
    '''AAA'''
    if not dims: dims = [None for col in cols]
    def set_violinstyle(axes_subplot_parts) -> None:
        '''
        '''
        for pc in axes_subplot_parts["bodies"]:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        axes_subplot_parts["cmins"].set_edgecolor("black")
        axes_subplot_parts["cmaxes"].set_edgecolor("black")
        return None

    fig, ax_list = plt.subplots(1, len(cols), figsize=(3*len(cols), 5))
    fig.subplots_adjust(wspace=1, hspace=0.8)

    for ax, col, title, label, dim in zip(ax_list, cols, titles, y_labels, dims):
        ax.set_title(title, size=15, y=1.05)
        ax.set_ylabel(label, size=13)
        ax.set_xticks([])
        data = df[col].to_list()
        parts = ax.violinplot(df[col].to_list(), widths=0.5)
        if dim: ax.set_ylim(dim)
        set_violinstyle(parts)
        quartile1, median, quartile3 = np.percentile(data, [25, 50, 75])
        ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
        ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
        ax.vlines(1, np.min(data), np.max(data), color="k", linestyle="-", lw=2)

    if out_path: fig.savefig(out_path, dpi=800, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_lists(lists: list, titles: list[str], y_labels: list[str], dims=None, out_path=None) -> None:
    '''AAA'''
    if not dims: dims = [None for sublist in lists]
    def set_violinstyle(axes_subplot_parts) -> None:
        '''
        '''
        for pc in axes_subplot_parts["bodies"]:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        axes_subplot_parts["cmins"].set_edgecolor("black")
        axes_subplot_parts["cmaxes"].set_edgecolor("black")
        return None

    fig, ax_list = plt.subplots(1, len(lists), figsize=(3*len(lists), 5))
    fig.subplots_adjust(wspace=1, hspace=0.8)

    for ax, sublist, title, label, dim in zip(ax_list, lists, titles, y_labels, dims):
        ax.set_title(title, size=15, y=1.05)
        ax.set_ylabel(label, size=13)
        ax.set_xticks([])
        parts = ax.violinplot(sublist, widths=0.5)
        if dim: ax.set_ylim(dim)
        set_violinstyle(parts)
        quartile1, median, quartile3 = np.percentile(sublist, [25, 50, 75])
        ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
        ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
        ax.vlines(1, np.min(sublist), np.max(sublist), color="k", linestyle="-", lw=2)

    if out_path: fig.savefig(out_path, dpi=800, format="png", bbox_inches="tight")
    else: fig.show()
    return None
