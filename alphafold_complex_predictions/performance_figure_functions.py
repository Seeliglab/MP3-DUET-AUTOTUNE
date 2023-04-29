#support functions for making performance comparison figures 
#plots of model performance for held out proteins 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, matthews_corrcoef, roc_auc_score, f1_score, average_precision_score
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

def draw_two_pairs(pair_list, n_draw):
    unique_draws =set([])
    while len(unique_draws) < n_draw:
        chosen = np.random.choice(6,2,replace = False)
        chosen.sort()
        unique_draws.add((chosen[0], chosen[1]))
    #get actual pairs 
    pairs = []
    for combo in unique_draws:
        pairs.append([pair_list[combo[0]], pair_list[combo[1]]])
    return pairs


def plot_df_regression_padj_final(df, savename):
    fig, ax = plt.subplots(2,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(2.5,1)
    
    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class +
    s =sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_r2', ax = ax[0], palette= 'Purples')
    #s._legend.remove()
    ax[0].set_ylim(0,1)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    s = sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_spearman', ax = ax[1], palette= 'Purples')
    #s._legend.remove()
    ax[1].set_ylim(0,1)
    
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    for i in range(0,2):
        ax[i].legend([],[], frameon=False)
        ax[i].set_xlabel('')

    plt.savefig(savename, dpi = 300)
    plt.show()
    

def plot_df_classification_padj_final(df):
    fig, ax = plt.subplots(3,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(3,1.5)
    
    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class +
    s =sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_AUCROC', ax = ax[0], palette= 'Purples')
    #s._legend.remove()
    ax[0].set_ylim(0,1)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    s = sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_avgpr', ax = ax[1], palette= 'Purples')
    #s._legend.remove()
    ax[1].set_ylim(0,1)

    s = sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_mccf1', ax = ax[2], palette= 'Purples')
    #s._legend.remove()
    #ax[1].set_ylim(0,1)
    ax[2].set_ylim(0,1)
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    for i in range(0,3):
        ax[i].legend([],[], frameon=False)
        ax[i].set_xlabel('')
    plt.show()
    


def plot_df_classification_padj_final_single_row(df, savename):
    fig, ax = plt.subplots(1,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(3,0.5)
    
    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class +
    s =sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_mccf1', ax = ax, palette= 'Purples')
    #s._legend.remove()
    ax.set_ylim(0,1)
    
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
   
    plt.savefig(savename, dpi = 300)
    plt.show()


def plot_df_regression_padj_both(df2, savename):
    fig, ax = plt.subplots(1,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(1.2,0.8)
    fsize = 3
 
    s = sns.boxplot(data = df2,
                x = 'dataset',
                y = 'test_spearman', ax = ax, palette= 'Purples', fliersize = fsize)#, zorder = 1)
    ax.axhline(0,  ls = '--', color = 'gray', alpha = 0.5)
    ax.set_ylim(-0.25,1)
    #s._legend.remove()
    plt.xticks(rotation=45)
    plt.savefig(savename, dpi = 300)
    plt.show()
    

def plot_df_classification_padj_both(df2, savename):
    fig, ax = plt.subplots(1,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(1.2,0.8)
    
    fsize = 3
    s = sns.boxplot(data = df2,
                x = 'dataset', 
                y = 'test_mccf1', ax = ax, palette= 'Blues', fliersize = fsize)#, zorder = 1)
    #ax[0].axhline(0.5,  ls = '--', color = 'gray', alpha = 0.5)
    ax.set_ylim(0,1)
   
    
    plt.xticks(rotation=45)
    
    plt.savefig(savename, dpi = 300)
    plt.show()

def plot_df_regression_padj_mlab(df, savename):
    fig, ax = plt.subplots(2,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(1.5,2)

    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class 
    s =sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_r2', ax = ax[0], palette= 'Purples')
    #s._legend.remove()
    ax[0].set_ylim(0,1)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    s = sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_rho', ax = ax[1], palette= 'Purples')
    #s._legend.remove()
    ax[1].set_ylim(-0.25,1)
    
    
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    for i in range(0,2):
        ax[i].legend([],[], frameon=False)
        ax[i].set_xlabel('')
    
    plt.savefig(savename, dpi = 300)
    plt.show()


def plot_df_regression_holdout_mlab(df, savename):
    fig, ax = plt.subplots(2,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(1.5,2)

    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class 
    s =sns.boxplot(data = df,
                x = 'dataset', 
                y = 'test_r2', ax = ax[0], palette= 'Purples')
    #s._legend.remove()
    ax[0].set_ylim(0,1)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    s = sns.boxplot(data = df,
                x = 'dataset', 
                y = 'test_rho', ax = ax[1], palette= 'Purples')
    #s._legend.remove()
    ax[1].set_ylim(-0.25,1)
    
    
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    for i in range(0,2):
        ax[i].legend([],[], frameon=False)
        ax[i].set_xlabel('')
    
    plt.savefig(savename, dpi = 300)
    plt.show()
    

def plot_df_class_padj_mlab(df, savename,default_l67, default_l70 ):
    fig, ax = plt.subplots(2,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(1.5,2)

    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class 
    s =sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_aucroc', ax = ax[0], palette= 'Purples')
    #s._legend.remove()
    ax[0].set_ylim(0,1)
    ax[0].axhline(0.5, linestyle = '--')
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    s = sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_avgpr', ax = ax[1], palette= 'Purples')
    #s._legend.remove()
    ax[1].axhline(default_l67, color = 'red', linestyle = '--')
    ax[1].axhline(default_l70, color = 'blue', linestyle = '--')
    ax[1].set_ylim(0,1)    
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    for i in range(0,2):
        ax[i].legend([],[], frameon=False)
        ax[i].set_xlabel('')
    
    plt.savefig(savename, dpi = 300)
    plt.show()
    

def plot_df_class_holdout_mlab(df, savename, default_l67, default_l70):
    fig, ax = plt.subplots(2,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(1.5,2)

    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class 
    s =sns.boxplot(data = df,
                x = 'dataset', 
                y = 'test_aucroc', ax = ax[0], palette= 'Purples')
    #s._legend.remove()
    ax[0].axhline(0.5, linestyle = '--')
    ax[0].set_ylim(0,1)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    s = sns.boxplot(data = df,
                x = 'dataset', 
                y = 'test_avgpr', ax = ax[1], palette= 'Purples')
    #s._legend.remove()
    ax[1].set_ylim(0,1)
    ax[1].axhline(default_l67, color = 'red', linestyle = '--')
    ax[1].axhline(default_l70, color = 'blue', linestyle = '--')
    
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
    for i in range(0,2):
        ax[i].legend([],[], frameon=False)
        ax[i].set_xlabel('')
    
    plt.savefig(savename, dpi = 300)
    plt.show()


def calc_lin_enrichment(trp_col_with_na, his_col_with_na, trp_count = 0, his_count = 0):
    if trp_count == 0 and his_count == 0:
        trp_count = trp_col_with_na.fillna(0).values.sum()
        his_count = his_col_with_na.fillna(0).values.sum()
        norm_coef = trp_count / his_count
        #his_col_with_na = his_col_with_na.fillna(0).values.sum()
        return norm_coef * his_col_with_na.fillna(his_col_with_na.min()) / trp_col_with_na.fillna(trp_col_with_na.min())
    else:
        norm_coef = trp_count / his_count
        #his_col_with_na = his_col_with_na.fillna(0).values.sum()
        return norm_coef * his_col_with_na.fillna(his_col_with_na.min()) / trp_col_with_na.fillna(trp_col_with_na.min())

def calc_log_enrichment(lin_enrich):
    return np.log(lin_enrich)

def calc_log2_enrichment(lin_enrich):
    return np.log2(lin_enrich)


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
            if df[df.PPI == ppi_id].shape[0] == 1:
                square_df[i,j] = df[df.PPI == ppi_id][col].to_numpy()
    return square_df

   
def display_window_mutant_counts_better_colors(subs_mat, cutoff, pro_names, cmap, show_names = False, size_1 = 4.5, size_2 = 2, saveName = None, forcelims=None):
    # round values
    subs_mat = subs_mat.round(3)
    # making mask matrix
    am1 = subs_mat <= cutoff
    am1 = np.ma.masked_where(am1 == False, am1)  # Mask the data we are not colouring
    cm1 = mpl.colors.ListedColormap(['red', 'white'])

    # set up figure
    figure = plt.figure()
    axes = figure.add_subplot(111)

    white_using = "#DDDDDD"
    #matplotlib.cm.get_cmap(cmap).copy()
    cmap = mpl.cm.get_cmap(cmap).copy()#matplotlib.colors.LinearSegmentedColormap.from_list("", ["#ffffff", "#4b2e83"])   
    #cmap = cmap.reversed()
    cmap.set_bad(color='#aa0000ff', alpha=1)

    # count values
    if forcelims:
        # count values
        cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal', vmin=forcelims[0],
                           vmax=forcelims[1])
        axes.matshow(am1, cmap=cm1, interpolation='none', aspect='equal')
        #cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal',
    else:
        # count values
        #cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal')
    
        cax = axes.matshow(subs_mat, cmap=cmap, interpolation='none', aspect='equal')
        axes.matshow(am1, cmap=cm1, interpolation='none', aspect='equal')

    # set up ticks
    plt.yticks(list(range(0, subs_mat.shape[0])))
    plt.xticks(list(range(0, subs_mat.shape[0])))

    # Minor ticks
    #axes.set_xticks(np.arange(-.5, len(pro_names) -1 , 1), minor=True)
    #axes.set_yticks(np.arange(-.5, len(pro_names), 1), minor=True)
    #blank_aas = [" "] * len(aas)
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
    #cbar.outline.set_visible(False)  # = 0.5
    #cbar_ax_again.set_yticklabels(['', '', ''])  # vertically oriented colorbar

    for xmin in axes.xaxis.get_majorticklocs():
        # print (xmin)
        axes.axvline(x=xmin - .5, color='black', linestyle='-', linewidth=0.5)

    for ymin in axes.yaxis.get_majorticklocs():
        # print (ymin)
        axes.axhline(y=ymin - .5, color='black', linestyle='-', linewidth=0.5)

    axes.xaxis.set_tick_params(width=0.5)
    axes.yaxis.set_tick_params(width=0.5)

    for axis in ['top', 'bottom', 'left', 'right']:
        axes.spines[axis].set_linewidth(0.5)
        cbar_ax_again.spines[axis].set_linewidth(0.5)

    
    # for pos in wt_positions:
    # rect(pos)
    plt.tight_layout()
    if saveName:
        plt.savefig(saveName, dpi = 300)
    plt.show()

def plot_malb_perfor_single_row(df, savename):
    fig, ax = plt.subplots(1,1, sharex= True) #plot AUCROC, MCC, n_features
    # the size of A4 paper
    fig.set_size_inches(3,0.5)
    
    plt.xlabel('')
    #plotting baseline AUCROC for imbalanced class +
    s =sns.barplot(data = df,
                x = 'dataset', 
                y = 'test_spearman', ax = ax, palette= 'Purples')
    #s._legend.remove()
    ax.set_ylim(0,1)
    plt.ylabel('rho')
    plt.xticks(rotation=45)
    plt.legend([],[], frameon=False)
    plt.xlabel('')
   
    plt.savefig(savename, dpi = 300)
    plt.show()


