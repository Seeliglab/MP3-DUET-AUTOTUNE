#Alyssa La Fleur
#graphing and misc. calculation functions 
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from scipy import stats
from itertools import combinations
warnings.simplefilter(action='ignore', category=FutureWarning)

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def make_lower_heatmap(subs_mat, pro_names, cmap, show_names=False, size_1=4.5, size_2=2, saveName=None,
                          forcelims=None, diag_off = 1, annot = False, text_color = 'white', font_size=8, show = True):  
    """
    Graphs P-LFC values in the lower diagonal of a matrix.  Graph can be saved as a svg to the figures folder
    :param subs_mat: np.array of values, expected to be square
    :param pro_names: List of protein names to be displayed. Expected to be the same length as subs_mat.shape[0] and in the same order as the matrix rows/columns
    :param cmap: String of cmap to use to make matrix
    :param show_names: Boolean if showing the names in pro_names or not on the axes
    :param size_1: Width of saved image
    :param size_2: Height of saved image
    :param saveName: Name to save matrix graph under as an svg in the figures folder
    :param forceLims: Bounds to use in making the colormap.  Expects a tuple or list of two float values of format [min,max]
    :param diag_off: Whether to display the diagonal of the square matrix/homodimers when graphing
    :param annot: Boolean if annotating squares with values or not (annotated with 3 numbers and 2 trailing decimal points by default)
    :param text_folor: String of text color to use for annotations
    :param font_size: Int of font size to use for annotations 
    :param show: Boolean if the heatmap should be displayed or not
    :return: None
    """
    # round values
    subs_mat = subs_mat.round(3)
    # making mask matrix

    # set up figure
    figure = plt.figure()
    axes = figure.add_subplot(111)

    cmap = matplotlib.cm.get_cmap(
        cmap).copy()  
    
    cmap.set_bad(color='#9a588d', alpha=0)
    if forcelims:
        cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal', vmin=forcelims[0],
                           vmax=forcelims[1])
    else:
        cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal')

    # set up ticks
    plt.yticks(list(range(0, subs_mat.shape[0])))
    plt.xticks(list(range(0, subs_mat.shape[0])))

    if show_names:
        axes.set_yticklabels(pro_names, rotation=0)

    plt.title('')
    plt.ylabel('')
    figure.set_size_inches(size_1, size_2)
    blank_x = [" "] * len(pro_names)
    if show_names:
        axes.set_xticklabels(pro_names, rotation=-45, ha='right')
    else:
        axes.set_xticklabels(blank_x)
        axes.set_yticklabels(blank_x)
    n_major_ticks = len(axes.xaxis.get_majorticklocs())  # [::-1]
    x_axis_ticks = axes.xaxis.get_majorticklocs()
    x_axis_ticks = [0] + list(x_axis_ticks)
    x_axis_ticks_rev = x_axis_ticks[::-1]
    y_axis_ticks = axes.yaxis.get_majorticklocs()

    pairs = []
    pairs2 = []
    for i in range(1, n_major_ticks):
        pairs.append((x_axis_ticks_rev[i-1], y_axis_ticks[-(i)]))
        pairs2.append((x_axis_ticks[i], y_axis_ticks[-(i)]))
    for pair in pairs:
        axes.plot([(n_major_ticks - 0.5)- pair[0], (n_major_ticks - 0.5)- pair[0]], [(n_major_ticks - 0.5), (n_major_ticks - 0.5)- pair[1] - diag_off], color='black', linestyle='-', linewidth=0.5)

    for pair in pairs2:
        axes.plot([(- 0.5), pair[0] -0.5 + 2 * diag_off], [(n_major_ticks - 0.5) - pair[1], (n_major_ticks - 0.5) - pair[1]], color='black', linestyle='-', linewidth=0.5)

    if annot:
        height, width = subs_mat.shape
        xpos, ypos = np.meshgrid(np.arange(width), np.arange(height))
        for x, y, val in zip(xpos.flat, ypos.flat, subs_mat.flat):
            if not np.isnan(val):
                annotation = '{:03.2f}'.format(val)
                text_kwargs = dict(color=text_color, ha="center", va="center")
                axes.text(x, y, annotation, **text_kwargs, fontsize=font_size)


    # fix colorbar sizing
    divider = make_axes_locatable(axes)
    cbar_ax = divider.append_axes("right", size="3%", pad=0.05)

    cbar = figure.colorbar(cax, cbar_ax)
    cbar_ax_again = cbar.ax
    cbar_ax_again.xaxis.set_tick_params(width=1.5)
    cbar_ax_again.yaxis.set_tick_params(width=1.5)
    cbar.outline.set_visible(False) 

    axes.xaxis.set_tick_params(width=0.5)
    axes.yaxis.set_tick_params(width=0.5)
    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(0.5)
        cbar_ax_again.spines[axis].set_linewidth(0.5)

    plt.tight_layout()
    if saveName:
        plt.savefig('./figures/' + saveName, dpi=300, transparent=True)
    if show:
        plt.show()
    
def make_square_heatmap(subs_mat, cutoff, pro_names, cmap, show_names = False, size_1 = 4.5, size_2 = 2, saveName = None, forcelims=None, show = True):
    """
    Graph the PPI LFC or enrichment values in a heatmap. 
    :param subs_mat: np.array of values, expected to be square
    :param cutoff: Lower cutoff value to display (can be used to show only positive P-LFC interactions)
    :param pro_names: List of protein names to be displayed. Expected to be the same length as subs_mat.shape[0] and in the same order as the matrix rows/columns
    :param cmap: String of cmap to use to make matrix
    :param show_names: Boolean if showing the names in pro_names or not on the axes
    :param size_1: Width of saved image
    :param size_2: Height of saved image
    :param saveName: Name to save matrix graph under as an svg in the figures folder
    :param forceLims: Bounds to use in making the colormap.  Expects a tuple or list of two float values of format [min,max]
    :param show: Boolean if the graph will be displayed or not
    :return: None
    """
    subs_mat = subs_mat.round(3)
    # making mask matrix
    am1 = subs_mat <= cutoff
    am1 = np.ma.masked_where(am1 == False, am1)  # Mask the data we are not colouring
    cm1 = matplotlib.colors.ListedColormap(['red', 'white'])

    # set up figure
    figure = plt.figure()
    axes = figure.add_subplot(111)

    cmap = matplotlib.cm.get_cmap(cmap).copy()
    cmap.set_bad(color='#aa0000ff', alpha=1)

    # count values
    if forcelims:
        # count values
        cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal', vmin=forcelims[0],
                           vmax=forcelims[1])
        axes.matshow(am1, cmap=cm1, interpolation='none', aspect='equal')
    else:
            
        cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal')
        axes.matshow(am1, cmap=cm1, interpolation='none', aspect='equal')

    # set up ticks
    plt.yticks(list(range(0, subs_mat.shape[0])))
    plt.xticks(list(range(0, subs_mat.shape[0])))

    
    if show_names:
        axes.set_yticklabels(pro_names, rotation=0)

    plt.title('')
    plt.ylabel('')
    figure.set_size_inches(size_1, size_2)
    blank_x = [" "] * len(pro_names)
    if show_names:
        axes.set_xticklabels(pro_names, rotation = -45, ha='right')
    else:
        axes.set_xticklabels(blank_x)
        axes.set_yticklabels(blank_x)
    # fix colorbar sizing
    divider = make_axes_locatable(axes)
    cbar_ax = divider.append_axes("right", size="3%", pad=0.05)

    cbar = figure.colorbar(cax, cbar_ax)
    cbar_ax_again = cbar.ax
    cbar_ax_again.xaxis.set_tick_params(width=1.5)
    cbar_ax_again.yaxis.set_tick_params(width=1.5)

    for xmin in axes.xaxis.get_majorticklocs():
        axes.axvline(x=xmin - .5, color='black', linestyle='-', linewidth=0.5)

    for ymin in axes.yaxis.get_majorticklocs():
        axes.axhline(y=ymin - .5, color='black', linestyle='-', linewidth=0.5)

    axes.xaxis.set_tick_params(width=0.5)
    axes.yaxis.set_tick_params(width=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(0.5)
        cbar_ax_again.spines[axis].set_linewidth(0.5)

    plt.tight_layout()
    if saveName:
        plt.savefig(saveName, dpi = 300)
    plt.show()



def make_specific_order_lower(better_order_binders, df, col, get_diag = True, limit_lower = float('-inf')):
    square_df = np.empty((len(better_order_binders), len(better_order_binders)))
    square_df[:] = np.nan
    for i in range(0, len(better_order_binders)):
        for j in range(0, i+1):# in bettter_order_binders:
            if get_diag:
                ppi_id = [better_order_binders[i], better_order_binders[j]]
                ppi_id.sort()
                if ':'.join(ppi_id) in df.PPI.to_list():
                    if limit_lower < df[df.PPI == ':'.join(ppi_id)][col].to_numpy():
                    #print (ppi_id, df[df.PPI == ':'.join(ppi_id)][col].to_numpy())
                        square_df[i,j] = df[df.PPI == ':'.join(ppi_id)][col].to_numpy()
            elif i!= j:
                ppi_id = [better_order_binders[i], better_order_binders[j]]
                ppi_id.sort()
                if ':'.join(ppi_id) in df.PPI.to_list():
                    if limit_lower < df[df.PPI == ':'.join(ppi_id)][col].to_numpy():
                    #print (ppi_id, df[df.PPI == ':'.join(ppi_id)][col].to_numpy())
                        square_df[i,j] = df[df.PPI == ':'.join(ppi_id)][col].to_numpy()
    return square_df

def make_specific_order(better_order_binders, df, col):
    square_df = np.empty((len(better_order_binders), len(better_order_binders)))
    square_df[:] = np.nan
    for i in range(0, len(better_order_binders)):
        for j in range(0, len(better_order_binders)):# in bettter_order_binders:
            b = better_order_binders[i]
            b2 = better_order_binders[j]
            ppi_id = 'DBD:' + b + ':AD:' + b2#[b, b2]
            #ppi_id.sort()
            #print (ppi_id, df[df.PPI == ppi_id])
            if df[df.PPI == ppi_id].shape[0] != 0:
                square_df[i,j] = df[df.PPI == ppi_id][col].to_numpy()
    return square_df

def make_specific_order(better_order_binders, df, col):
    """
    Reorder contents of given column in a dataframe and return a square numpy matrix of the colun values for only the proteins specificed 
    :param better_order_binders: List of strings of proteins in the assay. If it's not present in the df, it will add empty rows and coluumns in the square matrix
    :param df: Dataframe to extract values from
    :param col: String column name to get the values for
    :return: Square numpy matrix of values in col in teh order of better_order-binders
    """
    square_df = np.empty((len(better_order_binders), len(better_order_binders)))
    square_df[:] = np.nan
    for i in range(0, len(better_order_binders)):
        for j in range(0, len(better_order_binders)):
            b = better_order_binders[i]
            b2 = better_order_binders[j]
            ppi_id = 'DBD:' + b + ':AD:' + b2
            if df[df.PPI == ppi_id].shape[0] == 1:
                square_df[i,j] = df[df.PPI == ppi_id][col].to_numpy()
    return square_df


def calc_lin_enrichment(trp_col_with_na, his_col_with_na):
    """
    Calculate the library size adjusted enrichment of the interactions
    :param trp_col_with_na: Pandas series of His+ read count values, where 0 counts have been replaced with Nones
    :param his_col_with_na: Pandas series of His- read count values, where 0 counts have been replaced with Nones
    :return: 
    """
    trp_count = trp_col_with_na.fillna(0).values.sum()
    his_count = his_col_with_na.fillna(0).values.sum()
    norm_coef = trp_count / his_count
    return norm_coef * his_col_with_na.fillna(his_col_with_na.min()) / trp_col_with_na.fillna(trp_col_with_na.min())
    
def calc_log2_enrichment(lin_enrich):
    return np.log2(lin_enrich)


def make_ppi(one, two):
    ppi = [one, two]
    ppi.sort()
    return ':'.join(ppi)

def split_by_orientations(df, proteins, col_extract, keep_homodimers=True):
    """
    Takes a full interaction matrix with DBD and AD fusion proteins and collapses it to index by PPI
    :param df: Dataframe with PPIs containing DBD and AD ordering
    :param proteins: List of protein strings to get the values for
    :param col_extract: String of column name to collapse PPI measurement values for
    :param keep_homodimers: Boolean if homodimers will be retained
    :return: Dataframe of PPI of the two proteins, with two columns for the DBDAD and ADDBD fusions of the same PPI
    """
    # all combos of 2
    combos = list(combinations(proteins, 2))
    num_hetero = len(combos)
    if keep_homodimers:
        combos = combos + [(x, x) for x in proteins]
    # df of all combos of 2
    ppis = pd.DataFrame({'DBD_fast': [c[0] for c in combos], 'AD_fast': [c[1] for c in combos]})

    ppis['or_1_ppi'] = 'DBD:' + ppis['DBD_fast'] + ':AD:' + ppis['AD_fast']
    ppis['or_2_ppi'] = 'DBD:' + ppis['AD_fast'] + ':AD:' + ppis['DBD_fast']

    ppis = ppis.merge(df, left_on='or_1_ppi', right_on='PPI')
    ppis = ppis.merge(df, left_on='or_2_ppi', right_on='PPI', suffixes=['_DBDAD', '_ADDBD'])
    ppis['PPI'] = ppis.apply(lambda row: make_ppi(row['DBD_fast'], row['AD_fast']), axis=1)
    return ppis[['PPI', col_extract + '_DBDAD', col_extract + '_ADDBD']]



def get_correls(df, colx, coly, log = True):
    heatmap_tm_no_nans = df[['PPI', colx, coly]].dropna()
    n_orig = df.shape[0]
    n = heatmap_tm_no_nans.shape[0]
    if n == 2:
        return n_orig, n, None, None
    if log:
        pearson_r = stats.pearsonr(heatmap_tm_no_nans[colx],
                             np.log(heatmap_tm_no_nans[coly]))[0]**2
        spr = stats.spearmanr(heatmap_tm_no_nans[colx],
                             np.log(heatmap_tm_no_nans[coly]))[0]
        return n_orig, n, round(pearson_r,2), round(spr,2)

    else:
        
        pearson_r_nl = stats.pearsonr(heatmap_tm_no_nans[colx],
                             heatmap_tm_no_nans[coly])[0]**2
        spr_nl = stats.spearmanr(heatmap_tm_no_nans[colx],
                             heatmap_tm_no_nans[coly])[0]

        return n_orig, n, round(pearson_r_nl,2), round(spr_nl,2)



def process_fusion_dataframe(df):
    df = df.copy()
    vals_counts = ['trp1', 'his1']
    for vc in vals_counts:
        df.loc[df[vc]  == 0, vc] = None
    df['lin_enrich'] = calc_lin_enrichment(df['trp1'], df['his1'])
    df['log2_enrich'] = calc_log2_enrichment(df['lin_enrich'])
    df.loc[(df.trp1.isna()) & (df.his1.isna()), 'lin_enrich'] = None
    df.loc[(df.trp1.isna()) & (df.his1.isna()), 'log2_enrich'] = None

    return df 



def make_double_heatmap(subs_mat1, subs_mat2, pro_names, cmap1, cmap2, show_names = False, size_1 = 4.5, size_2 = 2, saveName = None, show = True, annot = True, text_color = 'white', font_size = 6):
    """
    Graph upper and lower triangle of matrix separately with different colormaps
    :param subs_mat1: np.array of values, expected to be square, lower triangle
    :param subs_mat2: np.array of values, expected to be square, upper triangle
    :param pro_names: List of protein names to be displayed. Expected to be the same length as subs_mat.shape[0] and in the same order as the matrix rows/columns
    :param cmap1: String of cmap to use to make matrix, lower triangle 
    :param cmap2: String of cmap to use to make matrix, upper triangle 
    :param show_names: Boolean if showing the names in pro_names or not on the axes
    :param size_1: Width of saved image
    :param size_2: Height of saved image
    :param saveName: Name to save matrix graph under as an svg in the figures folder
    :param show: Boolean if the graph will be displayed or not
    :param annot: Boolean if annotating squares with values or not (annotated with 3 numbers and 2 trailing decimal points by default)
    :param text_folor: String of text color to use for annotations
    :param font_size: Int of font size to use for annotations 
    :return: None
    """

    # set up figure
    figure = plt.figure()
    axes = figure.add_subplot(111)

    cmap1 = matplotlib.cm.get_cmap(cmap1).copy()
    cmap1.set_bad(color='#9a588d', alpha=0)

    cmap2 = matplotlib.cm.get_cmap(cmap2).copy()
    cmap2.set_bad(color='#9a588d', alpha=0)

    # count values
    cax = axes.matshow(subs_mat1, cmap=cmap1, interpolation='none', aspect='equal', vmin=0, vmax=1)

    cax2 = axes.matshow(subs_mat2, cmap=cmap2, interpolation='none', aspect='equal', vmin=0, vmax=1)

    # set up ticks
    plt.yticks(list(range(0, subs_mat1.shape[0])))
    plt.xticks(list(range(0, subs_mat1.shape[0])))
    
    if show_names:
        axes.set_yticklabels(pro_names, rotation=0)

    plt.title('')
    plt.ylabel('')
    figure.set_size_inches(size_1, size_2)
    blank_x = [" "] * len(pro_names)
    if show_names:
        axes.set_xticklabels(pro_names, rotation = -45, ha='right')
    else:
        axes.set_xticklabels(blank_x)
        axes.set_yticklabels(blank_x)
    
    annot_mat =  np.where(np.isnan(subs_mat1), 0, subs_mat1) + np.where(np.isnan(subs_mat2), 0, subs_mat2)
    print (annot_mat)
    if annot:
        height, width = annot_mat.shape
        xpos, ypos = np.meshgrid(np.arange(width), np.arange(height))
        for x, y, val in zip(xpos.flat, ypos.flat, annot_mat.flat):
            if not np.isnan(val):
                annotation = '{:03.2f}'.format(val)
                text_kwargs = dict(color=text_color, ha="center", va="center")
                axes.text(x, y, annotation, **text_kwargs, fontsize=font_size)

    divider = make_axes_locatable(axes)
    cbar_ax1 = divider.append_axes("left", size="3%", pad=0.05)
    cbar_ax2 = divider.append_axes("right", size="3%", pad=0.05)

    cbar1 = figure.colorbar(cax, cbar_ax1)
    cbar_ax_again = cbar1.ax
    cbar_ax_again.xaxis.set_tick_params(width=1.5)
    cbar_ax_again.yaxis.set_tick_params(width=1.5)

    cbar2 = figure.colorbar(cax2, cbar_ax2)
    cbar_ax_again = cbar2.ax
    cbar_ax_again.xaxis.set_tick_params(width=1.5)
    cbar_ax_again.yaxis.set_tick_params(width=1.5)


    for xmin in axes.xaxis.get_majorticklocs():
        axes.axvline(x=xmin - .5, color='black', linestyle='-', linewidth=0.5)

    for ymin in axes.yaxis.get_majorticklocs():
        axes.axhline(y=ymin - .5, color='black', linestyle='-', linewidth=0.5)

    axes.xaxis.set_tick_params(width=0.5)
    axes.yaxis.set_tick_params(width=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(0.5)
        cbar_ax_again.spines[axis].set_linewidth(0.5)



    plt.tight_layout()
    if saveName:
        plt.savefig('./figures/' + saveName, dpi = 300)
    if show:
        plt.show()
