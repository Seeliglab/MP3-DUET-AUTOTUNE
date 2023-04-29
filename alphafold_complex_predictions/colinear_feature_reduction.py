#AML
#remove highly colinear features to simplify modeling 

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

def get_all_tenths_max_interclust(v2_preds, label_cols, aj_important, save_name):
    v2_order = v2_preds.drop(columns=label_cols)
    cols = list(v2_order.columns)
    fig, ax1 = plt.subplots(figsize=(20, 10))

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
        dist_linkage, labels=cols, ax=ax1, leaf_rotation=90, get_leaves = True, leaf_font_size=4
    )
    #dendro_idx = np.arange(0, len(dendro["ivl"]))
    y_max = plt.ylim()[1]
    print (y_max)
    
    dfs = []
    divs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in range(0, len(divs)):
        div = divs[i]
        print ()
        cluster_ids = hierarchy.fcluster(dist_linkage, y_max*div, criterion="distance")
        plt.axhline( y_max*div, ls = '--')
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
    fig.set_size_inches(4,0.75)
    plt.savefig(save_name+'dendro.svg')
    plt.close()
    print (dendro['ivl'])
    print (dendro['leaves'])
    return dfs

all_old = pd.read_csv('./af_prediction_values_rosetta_energy_terms/AF2_rosetta_merged.csv')
all_old.drop(columns = ['timed', 'elapsed_time', 'tol'], inplace = True)

all_old = all_old[all_old.type != 'monomer_ptm']
v2_preds = all_old[all_old.type == 'multimer_v2'].copy()
v1_preds_msa_512 = all_old[(all_old.type == 'multimer') & (all_old.msa_depth == 512)].copy()
v1_preds_msa_1 = all_old[(all_old.type == 'multimer') & (all_old.msa_depth == 1)].copy()

v2_preds['fraction_int_all'] = v2_preds.IA_nres_int/v2_preds.IA_nres_all.fillna(1)
v1_preds_msa_512['fraction_int_all'] = v1_preds_msa_512.IA_nres_int/v1_preds_msa_512.IA_nres_all.fillna(1)
v1_preds_msa_1['fraction_int_all'] = v1_preds_msa_1.IA_nres_int/v1_preds_msa_1.IA_nres_all.fillna(1)

#add other fraction int_all
v2_preds['fraction_all'] = v2_preds.nres_int/v2_preds.nres_all.fillna(1)
v1_preds_msa_512['fraction_all'] = v1_preds_msa_512.nres_int/v1_preds_msa_512.nres_all.fillna(1)
v1_preds_msa_1['fraction_all'] = v1_preds_msa_1.nres_int/v1_preds_msa_1.nres_all.fillna(1)

v2_preds['ppi'] = v2_preds.id1 + '_' + v2_preds.id2
v1_preds_msa_512['ppi'] = v1_preds_msa_512.id1 + '_' + v1_preds_msa_512.id2
v1_preds_msa_1['ppi'] = v1_preds_msa_1.id1 + '_' + v1_preds_msa_1.id2

v2_preds['on_target'] = v2_preds.ppi.isin(on_target) 
v1_preds_msa_512['on_target'] = v1_preds_msa_512.ppi.isin(on_target) 
v1_preds_msa_1['on_target'] = v1_preds_msa_1.ppi.isin(on_target) 
v2_preds['new_ppi'] = v2_preds.apply(lambda row: row.ppi + '_' + str(row.model), axis  =1 )
v1_preds_msa_512['new_ppi'] = v1_preds_msa_512.apply(lambda row: row.ppi + '_' + str(row.model), axis  =1 )
v1_preds_msa_1['new_ppi'] = v1_preds_msa_1.apply(lambda row: row.ppi + '_' + str(row.model), axis  =1 )

#drop duplicate new_ppi
v2_preds = v2_preds.drop_duplicates('new_ppi', keep = 'first')
v1_preds_msa_512 = v1_preds_msa_512.drop_duplicates('new_ppi', keep = 'first')
v1_preds_msa_1 = v1_preds_msa_1.drop_duplicates('new_ppi', keep = 'first')

#v3 needs some tlc to get num clashing res out 
v3_preds = pd.read_csv('./af_prediction_values_rosetta_energy_terms/af3_with_resi_strs.csv')
v3_preds.drop(columns = ['timed'], inplace = True)
v3_preds['chain_a_clash_num'] = v3_preds.clashing_res.apply(lambda x: strip_clashing(x, 'A'))
v3_preds['chain_b_clash_num'] = v3_preds.clashing_res.apply(lambda x: strip_clashing(x, 'B'))
v3_preds.drop(columns = 'clashing_res', inplace=True)
#remove columns which are always the same 
v3_preds['fraction_int_all'] = v3_preds.IA_nres_int/v3_preds.IA_nres_all.fillna(1)
v3_preds['fraction_all'] = v3_preds.nres_int/v3_preds.nres_all.fillna(1)
v3_preds['ppi'] = v3_preds.id1 + '_' + v3_preds.id2
v3_preds['on_target'] = v3_preds.ppi.isin(on_target) 
v3_preds.rename(columns = {'model_number':'model', 'plddt': 'mean_plddt'}, inplace = True)
v3_preds['type'] = 'v3'
v3_preds['msa_depth'] = 2
v3_preds['new_ppi'] = v3_preds.apply(lambda row: row.ppi + '_' + str(row.model), axis  =1 )
#drop duplicate new_ppi
v3_values = v3_preds.drop_duplicates('new_ppi', keep = 'first')


v2_preds = remove_constant_cols(v2_preds, label_cols)
v1_preds_msa_512 = remove_constant_cols(v1_preds_msa_512, label_cols)
v3_preds = remove_constant_cols(v3_preds, label_cols)

dfs = get_all_tenths_max_interclust(v2_preds, label_cols, aj_important, './datasets/v2_correl_reduced_r_')
dfs = get_all_tenths_max_interclust(v1_preds_msa_512, label_cols, aj_important, './datasets/v1_512_correl_reduced_r_')
dfs = get_all_tenths_max_interclust(v3_preds, label_cols, aj_important, './datasets/v3_correl_reduced_r_')

#monomer version of AF 
all_old = pd.read_csv('./af_prediction_values_rosetta_energy_terms/AF2_rosetta_merged.csv')
all_old.drop(columns = ['timed', 'elapsed_time', 'tol'], inplace = True)
monomer_ptm = all_old[all_old.type == 'monomer_ptm'].copy()
monomer_ptm['fraction_int_all'] = monomer_ptm.IA_nres_int/monomer_ptm.IA_nres_all.fillna(1)
monomer_ptm['fraction_all'] = monomer_ptm.nres_int/monomer_ptm.nres_all.fillna(1)
monomer_ptm['ppi'] = monomer_ptm.id1 + '_' + monomer_ptm.id2

on_target = []
for i in range(1,12,2):
    on_target.append('P' + str(i) + '_' +  'P' + str(i+1))
    on_target.append('P' + str(i + 1)+ '_' +   'P' + str(i))

monomer_ptm['on_target'] = monomer_ptm.ppi.isin(on_target) 
monomer_ptm['new_ppi'] = monomer_ptm.apply(lambda row: row.ppi + '_' + str(row.model), axis  =1 )
monomer_ptm = monomer_ptm.drop_duplicates('new_ppi', keep = 'first')
mono_preds = remove_constant_cols(monomer_ptm, label_cols)
df = get_all_tenths_max_interclust(mono_preds, label_cols, aj_important,'./datasets/mono_correl_reduced_r_')

#make af output only version of features 
subset_af = ['plddt', 'mean_plddt', 'pae', 'ptm', 'iptm', 'max_pae']
v1_preds_msa_512[label_cols + [x for x in subset_af if x in v1_preds_msa_512]].to_csv('./datasets/v1_512_correl_reduced_r_af.csv', index = False)
v2_preds[label_cols + [x for x in subset_af if x in v2_preds]].to_csv('./datasets/v2_correl_reduced_r_af.csv', index = False)
v3_preds[label_cols + [x for x in subset_af if x in v3_preds]].to_csv('./datasets/v3_correl_reduced_r_af.csv', index = False)

mono_only_pae_cols = [ 'mean_pae_interaction_AB',
       'mean_pae_interaction_BA', 'mean_pae_interaction',
       'mean_pae_intra_chain_A', 'mean_pae_intra_chain_B',
       'mean_pae_intra_chain', 'mean_pae', 'pTMscore']
mono_preds[label_cols + [x for x in subset_af + mono_only_pae_cols if x in mono_preds]].to_csv('./datasets/mono_correl_reduced_r_af.csv', index = False)