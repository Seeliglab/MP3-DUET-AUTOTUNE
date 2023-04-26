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

def make_double_heatmap(subs_mat1, subs_mat2, pro_names, cmap1, cmap2, show_names = False, size_1 = 4.5, size_2 = 2, saveName = None, forcelims=None, show = True, annot = True, text_color = 'white', font_size = 6):
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

    # set up figure
    figure = plt.figure()
    axes = figure.add_subplot(111)

    cmap1 = matplotlib.cm.get_cmap(cmap1).copy()
    cmap1.set_bad(color='#9a588d', alpha=0)

    cmap2 = matplotlib.cm.get_cmap(cmap2).copy()
    cmap2.set_bad(color='#9a588d', alpha=0)

    # count values
    cax = axes.matshow(subs_mat1, cmap=cmap1, interpolation='none', aspect='equal')

    cax2 = axes.matshow(subs_mat2, cmap=cmap2, interpolation='none', aspect='equal')

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



from processing_functions import *
from comparison_datasets import *
import matplotlib as mpl
import seaborn as sns
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

jerala_pros = ['Jerala_P1',
 'Jerala_P2',
 'Jerala_P3',
 'Jerala_P4',
 'Jerala_P5',
 'Jerala_P6',
 'Jerala_P7',
 'Jerala_P8',
 'Jerala_P9',
 'Jerala_P10',
 'Jerala_P11',
 'Jerala_P12']


#Jerala libraries
jerala_target_ors, jerala_target_ppis, jerala_true_flat, jerala_true_folded = get_jerala_values()
jerala_true_flat['ashr_log2FoldChange_HIS_TRP'] = np.log(jerala_true_flat['Fold activation'])
jerala_true_folded_avg = jerala_true_folded[['PPI', 'avg_fa']].copy()
jerala_true_folded_avg['ashr_log2FoldChange_HIS_TRP'] = np.log(jerala_true_folded_avg['avg_fa'])
jerala_true_folded_max = jerala_true_folded[['PPI', 'max_fa']].copy()
jerala_true_folded_max['ashr_log2FoldChange_HIS_TRP'] = np.log(jerala_true_folded_max['max_fa'])


#flat correlations 
replicate_1_2 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_2_flat.csv')
replicate_1_2 = replicate_1_2.rename(columns = {'Unnamed: 0': 'PPI'}) 
replciate_2_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_l62_l67_flat.csv')
replciate_2_3 = replciate_2_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_1_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_l61_l67_flat.csv')
replicate_1_3 = replicate_1_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_all_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat.csv')
replicate_all_3 = replicate_all_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
    
merged = replicate_1_2.merge(replciate_2_3, on ='PPI', suffixes = ['_1_2', '_2_3'])
print (merged.shape)
merged = replicate_1_3.merge(merged, on ='PPI')
print (merged.shape)
merged = replicate_all_3.merge(merged, on ='PPI', suffixes = ['_all','_1_3'])
print (merged.shape)
merged = jerala_true_flat.merge(merged, on = 'PPI')

combos = list(combinations(['_1_2', '_1_3', '_2_3', '_all', ''], 2))

df_to_graph = pd.DataFrame({'PPI':[], 'correl':[], 'rho':[]})
for a, b in combos:
        df_to_graph.loc[df_to_graph.shape[0]] = [make_ppi(a,b), get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[2], 
                                                 get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[3]]

order_libraries = ['_1_2', '_1_3', '_2_3', '_all', '']
graph_labels = ['R1&R2', 'R1&R3', 'R2&R3', 'R1&R2&R3', 'FA']
x = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'correl',  get_diag = False)
y = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'rho',  get_diag = False)
y = y.T
make_double_heatmap(x, y, graph_labels, 'copper_r', 'bone_r')