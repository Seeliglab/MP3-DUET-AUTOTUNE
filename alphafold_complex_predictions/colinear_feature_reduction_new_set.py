#AML
#remove highly colinear features to simplify modeling - for the new dataset (not the main model set) 
#to compare model performances 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import re 
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none',
'font.size':4
}
mpl.rcParams.update(new_rc_params)

#dictionary for naming all features consistently between images for the supplement
naming_dict = {'plddt': 0,
 'mean_plddt': 1,
 'pae': 2,
 'ptm': 3,
 'iptm': 4,
 'max_pae': 5,
 'mean_pae_interaction_AB': 6,
 'mean_pae_interaction_BA': 7,
 'mean_pae_interaction': 8,
 'mean_pae_intra_chain_A': 9,
 'mean_pae_intra_chain_B': 10,
 'mean_pae_intra_chain': 11,
 'mean_pae': 12,
 'pTMscore': 13,
 'IA_complex_normalized': 14,
 'IA_dG_cross': 15,
 'IA_dG_cross/dSASAx100': 16,
 'IA_dG_separated': 17,
 'IA_dG_separated/dSASAx100': 18,
 'IA_dSASA_hphobic': 19,
 'IA_dSASA_int': 20,
 'IA_dSASA_polar': 21,
 'IA_delta_unsatHbonds': 22,
 'IA_hbond_E_fraction': 23,
 'IA_hbonds_int': 24,
 'IA_nres_int': 25,
 'IA_packstat': 26,
 'IA_per_residue_energy_int': 27,
 'IA_sc_value': 28,
 'IA_side1_normalized': 29,
 'IA_side1_score': 30,
 'IA_side2_normalized': 31,
 'IA_side2_score': 32,
 'cart_bonded': 33,
 'cms': 34,
 'cms_apolar': 35,
 'cms_sq5': 36,
 'complex_normalized': 37,
 'dG_cross': 38,
 'dG_cross/dSASAx100': 39,
 'dG_separated': 40,
 'dG_separated/dSASAx100': 41,
 'dSASA_hphobic': 42,
 'dSASA_int': 43,
 'dSASA_polar': 44,
 'ddg': 45,
 'delta_unsatHbonds': 46,
 'exposed_hydrop': 47,
 'fa_atr': 48,
 'fa_dun_dev': 49,
 'fa_dun_rot': 50,
 'fa_dun_semi': 51,
 'fa_elec': 52,
 'fa_intra_atr_xover4': 53,
 'fa_intra_elec': 54,
 'fa_intra_rep_xover4': 55,
 'fa_intra_sol_xover4': 56,
 'fa_rep': 57,
 'fa_sol': 58,
 'hbond_E_fraction': 59,
 'hbond_bb_sc': 60,
 'hbond_sc': 61,
 'hbond_sr_bb': 62,
 'hbonds_int': 63,
 'hxl_tors': 64,
 'lk_ball': 65,
 'lk_ball_bridge': 66,
 'lk_ball_bridge_uncpl': 67,
 'lk_ball_iso': 68,
 'nres_int': 69,
 'num_core_polar': 70,
 'omega': 71,
 'p_aa_pp': 72,
 'packstat': 73,
 'per_residue_energy_int': 74,
 'rama_prepro': 75,
 'ref': 76,
 'sc2': 77,
 'sc2_int_area': 78,
 'sc2_median_dist': 79,
 'sc_value': 80,
 'score_per_res': 81,
 'side1_normalized': 82,
 'side1_score': 83,
 'side2_normalized': 84,
 'side2_score': 85,
 'total_score': 86,
 'fraction_int_all': 87,
 'fraction_all': 88,
 'sbuns': 89,
 'rank': 90,
 'chain_a_clash_num': 91,
 'chain_b_clash_num': 92,
 'IA_nres_all':93,
 'hbond_lr_bb':94,
 'nres_all':95,
 'vbuns':96}

print (len(naming_dict))

aj_important = open('./af_prediction_values_rosetta_energy_terms/most_important_features.txt', 'r')
aj_important = [x.strip() for x in aj_important.readlines()]
label_cols = ['id1', 'id2', 'model', 'type', 'msa_depth', 'new_ppi', 'ppi', 'on_target']


on_target = []
for i in range(1,12,2):
    on_target.append('P' + str(i) + '_' +  'P' + str(i+1))
    on_target.append('P' + str(i + 1)+ '_' +   'P' + str(i))


def remove_constant_cols(df, label_cols):
    remove_cols = []
    str_cols = list(df.select_dtypes(include = object).columns)
    for col in  df.columns:
        #check if there is only 1 val in column
        if col not in label_cols:
            #print (col)s
            if col not in str_cols:
                num_unique_vals = df[col].unique().shape[0]
                if num_unique_vals <= 3:
                    #print ( df[col].unique())
                    #print (df[col].value_counts())
                    remove_cols.append(col)
            else:
                remove_cols.append(col)
            

    return df.drop(columns = remove_cols)

def reorder_df(df, aj_important, label_cols):
    all_cols = df.columns
    keep_versions_v2 = []
    other_v2 = []
    for col in all_cols:
        if col not in label_cols:
            #type = get_weight_type(col)
            if col in aj_important:
                keep_versions_v2.append(col)
            else:
                other_v2.append(col)

    #print (len(keep_versions_v2))
    #print (len(other_v2) + len(keep_versions_v2))
    #print (len(all_cols))
    #reorganize the columns
    return df[keep_versions_v2 + other_v2], keep_versions_v2



def get_selection_rank(preds_cols, label_cols, aj_important):
    other_cols = preds_cols.drop(columns = label_cols)
    numb_occurences = []
    subset_af = ['plddt', 'mean_plddt', 'pae', 'ptm', 'iptm'] #give preference for theses values
    for c in other_cols.columns:
        score = 0
        kurt_col = kurtosis(other_cols[c].dropna())
        #print (kurt_col, c)
        if c in aj_important:
            score += 1
        if c in subset_af:
            score +=1 
        numb_occurences.append({'feature':c, 'kurt_of_col': kurt_col, 'score':score, 'num_unqiue': other_cols[c].value_counts().shape[0]})

    ranking_df = pd.DataFrame(numb_occurences)
    #make it so that af features are selected first 
    ranking_df['kurt_of_col'] = ranking_df['kurt_of_col']/(ranking_df['kurt_of_col'].max())
    ranking_df['num_unqiue'] = ranking_df['num_unqiue']/(ranking_df['num_unqiue'].max())
    ranking_df['score'] = ranking_df['score'] + ranking_df['kurt_of_col'] + ranking_df['num_unqiue']
    return ranking_df.sort_values(['score'], ascending= False).reset_index(drop = True)


def strip_clashing(x, chain):
    if x != 'select clashing_res, ':
        chains_list = re.findall(r'\((.*?)\)',x)
        #print (chains_list)
        for chain_curr in chains_list:
            if chain == 'A' and 'A' in chain_curr:
                return len(chain_curr.split(' ')[-1].split(","))
            elif chain == 'B' and 'B' in chain_curr:
                return len(chain_curr.split(' ')[-1].split(","))
            else:
                return -1
    else:
        return 0

def get_all_tenths_max_interclust(v2_preds, label_cols, aj_important, save_name, dendro_w, dendro_h):
    v2_order = v2_preds.drop(columns=label_cols)
    if 'Unnamed: 0' in v2_order.columns:
        v2_order = v2_order.drop(columns=['Unnamed: 0'])
    cols = list(v2_order.columns)
    fig, ax1 = plt.subplots()

    c_mat = v2_order.corr('spearman').fillna(0).to_numpy()
    #c_mat = np.nan_to_num(c_mat)
    corr = (c_mat + c_mat.T) / 2
    np.fill_diagonal(corr, 1)
    #print (corr.shape)

    # We convert the correlation matrix to a distance matrix before performings
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    print (distance_matrix.shape)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    dendro = hierarchy.dendrogram(
        dist_linkage, labels=[naming_dict[c] for c in cols], ax=ax1, leaf_rotation=90, get_leaves = True, leaf_font_size=4
    )
    #dendro_idx = np.arange(0, len(dendro["ivl"]))
    y_max = plt.ylim()[1]
    print (y_max)
    cols_lines = ['#ffffffcc', '#ff7f0eff', '#2ca02cff', '#d62728ff', '#9467bdff', '#8c564bff', '#e377c2ff', '#7f7f7fff', '#bcbd22ff']
    dfs = []
    divs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in range(0, len(divs)):
        div = divs[i]
        print ()
        cluster_ids = hierarchy.fcluster(dist_linkage, y_max*div, criterion="distance")
        plt.axhline( y_max*div, ls = '--', color = cols_lines[i])
        df_cluster = pd.DataFrame({'feature': cols, 'cluster': cluster_ids})
        ranks_2 = get_selection_rank(v2_preds, label_cols, aj_important)
        ranks_2 = ranks_2.merge(df_cluster, on = 'feature')
        subset = ranks_2.drop_duplicates('cluster', keep = 'first')
        #save to datasets 
        print (subset.shape,'T'+str(i), y_max*div )
        v2_preds[label_cols +  subset.feature.to_list()].to_csv(save_name + 'T'+str(i) +'.csv', index = False)
        dfs.append((ranks_2, subset))
        if ranks_2.drop_duplicates('cluster', keep = 'first').shape[0] == 2:
            break
    fig.set_size_inches(dendro_h,dendro_w)
    plt.tight_layout()
    plt.savefig(save_name+'dendro.svg')
    plt.close()
    print (dendro['ivl'])
    print (dendro['leaves'])
    return dfs

#v2 of new predictions
v2_preds = pd.read_csv('./af_prediction_values_rosetta_energy_terms/AF2_rosetta_merged-v2.csv')
v2_preds = v2_preds[v2_preds['rosetta-protocol'] == 'rosetta-flex-bb'].copy().reset_index(drop = True)
v2_preds.drop(columns = ['timed'], inplace = True)
v2_preds.drop(columns = 'clashing_res', inplace=True)
#remove columns which are always the same 
v2_preds['fraction_int_all'] = v2_preds.IA_nres_int/v2_preds.IA_nres_all.fillna(1)
v2_preds['fraction_all'] = v2_preds.nres_int/v2_preds.nres_all.fillna(1)
v2_preds['id1'] = v2_preds.id1.apply(lambda x: x.replace('__','_'))
v2_preds['id2'] = v2_preds.id2.apply(lambda x: x.replace('__','_'))
v2_preds['ppi'] = v2_preds.id1 + '_' + v2_preds.id2
v2_preds['on_target'] = v2_preds.ppi.isin(on_target) 
v2_preds.rename(columns = {'model_number':'model', 'plddt': 'mean_plddt'}, inplace = True)
v2_preds['type'] = 'v2'
v2_preds['msa_depth'] = 2
v2_preds['new_ppi'] = v2_preds.apply(lambda row: row.ppi + '_' + str(row.model), axis  =1 )

#drop duplicate new_ppi
v3_values = v2_preds.drop_duplicates('new_ppi', keep = 'first')
v2_preds = remove_constant_cols(v2_preds, label_cols)
dfs = get_all_tenths_max_interclust(v2_preds, label_cols, aj_important, './datasets/malb_v2_correl_reduced_r_', dendro_h=5.25, dendro_w=0.75)

#af set 
subset_af = ['mean_plddt', 'pae', 'ptm', 'iptm',]
v2_preds[label_cols + [x for x in subset_af if x in v2_preds]].to_csv('./datasets/malb_v2_correl_reduced_r_af.csv', index = False)


#v3 needs some tlc to get num clashing res out - only version for new predictions 
v3_preds = pd.read_csv('./af_prediction_values_rosetta_energy_terms/AF2_rosetta_merged-AF-v3.csv')
v3_preds = v3_preds[v3_preds['rosetta-protocol'] == 'rosetta-flex-bb'].copy().reset_index(drop = True)
v3_preds.drop(columns = ['timed'], inplace = True)
v3_preds['chain_a_clash_num'] = v3_preds.clashing_res.apply(lambda x: strip_clashing(x, 'A'))
v3_preds['chain_b_clash_num'] = v3_preds.clashing_res.apply(lambda x: strip_clashing(x, 'B'))
v3_preds.drop(columns = 'clashing_res', inplace=True)
#remove columns which are always the same 
v3_preds['fraction_int_all'] = v3_preds.IA_nres_int/v3_preds.IA_nres_all.fillna(1)
v3_preds['fraction_all'] = v3_preds.nres_int/v3_preds.nres_all.fillna(1)
v3_preds['id1'] = v3_preds.id1.apply(lambda x: x.replace('__','_'))
v3_preds['id2'] = v3_preds.id2.apply(lambda x: x.replace('__','_'))
v3_preds['ppi'] = v3_preds.id1 + '_' + v3_preds.id2
v3_preds['on_target'] = v3_preds.ppi.isin(on_target) 
v3_preds.rename(columns = {'model_number':'model', 'plddt': 'mean_plddt'}, inplace = True)
v3_preds['type'] = 'v3'
v3_preds['msa_depth'] = 2
v3_preds['new_ppi'] = v3_preds.apply(lambda row: row.ppi + '_' + str(row.model), axis  =1 )

#drop duplicate new_ppi
v3_values = v3_preds.drop_duplicates('new_ppi', keep = 'first')
v3_preds = remove_constant_cols(v3_preds, label_cols)
dfs = get_all_tenths_max_interclust(v3_preds, label_cols, aj_important, './datasets/malb_v3_correl_reduced_r_', dendro_h=5.25, dendro_w=0.75)

#af set 
subset_af = ['mean_plddt', 'pae', 'ptm', 'iptm']
v3_preds[label_cols + [x for x in subset_af if x in v3_preds]].to_csv('./datasets/malb_v3_correl_reduced_r_af.csv', index = False)
