#since these aren't trained on the data, we can run them on all the PPIs and the test set 

from performance_figure_functions import *
import numpy as np
from sklearn.metrics import mean_squared_error, matthews_corrcoef, roc_auc_score, f1_score, average_precision_score
from scipy.stats import spearmanr, pearsonr
import matplotlib as mpl
from mccf1_functions import *
from general_funcs import label_LFC_bins, load_dataset_single
import re
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


def load(path):
    header = np.fromfile(path, dtype=np.uint64, count=4)
    n, m = header[1], header[2]
    offset = int(header[3])
    del header
    return np.memmap(path, dtype='float32', mode='r', offset=offset, shape=(n, m))



#open the PNIC predictions from popatov that Ajasja sent 
class SymDict(dict):
    """A dictionary subclass that uses symetric tuples.
    dict[A, B] == dict[B, A].
    taken from: http://stackoverflow.com/a/23865214/952600
    """
    
    @staticmethod
    def symm(key):
        return key if key[0] < key[1] else (key[1], key[0])
    
    def __getitem__(self, key):
        return dict.__getitem__(self, self.symm(key))

    def __setitem__(self, key, value):
        dict.__setitem__(self, self.symm(key), value)

    def __delitem__(self, key):
        return dict.__delitem__(self, self.symm(key))

    def __contains__(self, key):
        return dict.__contains__(self, self.symm(key))

    def ids(self):
        """Returns the list of all ids,
        So if keys are (A, B), (A, A) and (B, B) this returns [A, B]"""
        s = set()
        for key in self.keys():
            s.update(key)
        return s     
    
def parse_line(line):
    """parses a line from the score file
    Example:
    A,B,7.3423
    returns [A, B, 7.3423]
    """
    res=line.split(',')
    return res[0], res[1], float(res[2])

parse_line("A,B,7.345")

def load_score_file(file_name):
    """Loads the score file into a symetric dictionary"""
    s = SymDict()
    
    with open(file_name, "r") as f:
        for line in f:
            line = line.strip()
            if (line=="") or (line[0]=="#"):
                continue
            a,b,score=parse_line(line)
            s[a,b]=score
    return s  

def symdict_to_array(symdict, ids=None):
    """converts a symetric dictionary to a numpy array, needed for plotting""" 
    if ids is None:
        ids = symdict.ids()
    res = np.zeros((len(ids), len(ids)))
    for ni, i in enumerate(ids):
        for nj, j in enumerate(ids):
            res[ni,nj]=symdict[i,j]
    return res        

def natural_key(string_):
    """
    Sorts by numbers that appear in a string.
    
    For example: P11 P1 P2 si correctly sorted into P1 P2 P11.
    Taken from http://stackoverflow.com/questions/2545532/python-analog-of-natsort-function-sort-a-list-using-a-natural-order-algorithm
    See also http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
    
    
def get_all_baseline_sets():
    bicpa_scores = load('PNIC_bcipa.bin')
    icipa_core_vert_scores = load('PNIC_icipa_core_vert.bin')
    icipa_nter_core_scores = load('PNIC_icipa_nter_core.bin')
    qcipa_scores = load('PNIC_qcipa.bin')

    d=load_score_file("PNIC-complete.out")
    ids = sorted(d.ids(), key=natural_key)
    mat = symdict_to_array(d, ids)

    mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_autotune_5_small.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})

    mp3_seq_values['P1'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[1])
    mp3_seq_values['P2'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[3])

    test_train_split = ['test', 'train'] * int((0.1 * mp3_seq_values.shape[0]))
    test_train_split = test_train_split + ['train'] * int(mp3_seq_values.shape[0] - len(test_train_split))
    mp3_seq_values = mp3_seq_values.sort_values('ashr_padj_HIS_TRP')
    mp3_seq_values['order_padj_sets'] = test_train_split
    mp3_seq_values['binned'] = mp3_seq_values.ashr_log2FoldChange_HIS_TRP.apply(lambda x: label_LFC_bins(x))

    #make df of popatov value
    df_rows = []
    for i in range(0,12):
        for j in range(0,12):
            df_rows.append({
                'P1': 'Jerala_P' + str(i + 1),
                'P2': 'Jerala_P' + str(j + 1),
                'popatov': mat[i][j],
                'bicpa_scores':bicpa_scores[i][j],
                'icipa_core_vert_scores':icipa_core_vert_scores[i][j],
                'icipa_nter_core_scores':icipa_nter_core_scores[i][j],
                'qcipa_scores':qcipa_scores[i][j],
            })
    pop_df = pd.DataFrame(df_rows)

    pop_merge =pop_df.merge(mp3_seq_values, how = 'left', on = ['P1', 'P2'] )
    to_correl = pop_merge[['P1', 'P2', 'popatov', 'ashr_log2FoldChange_HIS_TRP', 'bicpa_scores', 'icipa_core_vert_scores', 'icipa_nter_core_scores', 'order_padj_sets', 'binned']].dropna()
    #to_correl = to_correl[to_correl.order_padj_sets == 'test']

    #average each test point like in other evaluations 

    #using thse for plotting purposes later on to order bars at the end of the plots
    z_names = {'popatov':'z1', 'bicpa_scores':'z2', 'icipa_core_vert_scores':'z3', 'icipa_nter_core_scores':'z4' }

    rows = []
    for col in ['popatov', 'bicpa_scores', 'icipa_core_vert_scores', 'icipa_nter_core_scores' ]:
            rows.append({
                'dataset': z_names[col] + '_all', 
                'mode': 'all',
                'test_r2':  pearsonr(to_correl['ashr_log2FoldChange_HIS_TRP'], to_correl[col] * -1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(to_correl['ashr_log2FoldChange_HIS_TRP'], to_correl[col] * -1)[0]),
                ##'test_mccf1': get_mcc_f1_from_msrmts(to_correl['binned'] ==1 , to_correl[col]*-1)[0],
                'test_AUCROC':roc_auc_score(to_correl['binned'], to_correl['popatov'] *-1 ),
                'test_avgpr':average_precision_score(to_correl['binned'], to_correl[col] *-1 )})
    
    #test set only
    to_correl = to_correl[to_correl.order_padj_sets == 'test']

    for col in ['popatov', 'bicpa_scores', 'icipa_core_vert_scores', 'icipa_nter_core_scores' ]:
            rows.append({
                'dataset': z_names[col] + '_test', 
                'mode': 'test',
                'test_r2':  pearsonr(to_correl['ashr_log2FoldChange_HIS_TRP'], to_correl[col] * -1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(to_correl['ashr_log2FoldChange_HIS_TRP'], to_correl[col] * -1)[0]),
                ##'test_mccf1': get_mcc_f1_from_msrmts(to_correl['binned'] ==1 , to_correl[col]*-1)[0],
                'test_AUCROC':roc_auc_score(to_correl['binned'], to_correl['popatov'] *-1 ),
                'test_avgpr':average_precision_score(to_correl['binned'], to_correl[col] *-1 )})
            
    df_barplot = pd.DataFrame(rows)

    rows = []
    for i in range(1,13,2):
        test_Pros = ['Jerala_P' + str(i), 'Jerala_P' + str(i+1)]
        pop_df_remaining = pop_merge[(pop_df.P1.isin(test_Pros)) | (pop_df.P2.isin(test_Pros))].dropna()
        print (roc_auc_score(pop_df_remaining['binned'], pop_df_remaining['popatov'] ))
        print (roc_auc_score(pop_df_remaining['binned'], pop_df_remaining['popatov'] * -1 ))
        for col in ['popatov', 'bicpa_scores', 'icipa_core_vert_scores', 'icipa_nter_core_scores' ]:
            rows.append({
                'dataset': z_names[col], 'test_r2':  pearsonr(pop_df_remaining['ashr_log2FoldChange_HIS_TRP'], pop_df_remaining[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(pop_df_remaining['ashr_log2FoldChange_HIS_TRP'], pop_df_remaining[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(pop_df_remaining['binned'] ==1 , pop_df_remaining[col]*-1)[0],
                'test_AUCROC':roc_auc_score(pop_df_remaining['binned'], pop_df_remaining['popatov'] *-1 ),
                'test_avgpr':average_precision_score(pop_df_remaining['binned'], pop_df_remaining[col] *-1 )})

    df_heldout_pop = pd.DataFrame(rows)

    #two acc
    pairs = []
    for i in range(1,13,2):
            pairs.append(('Jerala_P' + str(i), 'Jerala_P' + str(i+1)))
    rows = []
    drawn_pairs = draw_two_pairs(pairs, 15)
    for two_pair in drawn_pairs:
        test_Pros = [x for p in two_pair for x in p]
        #test_Pros = ['Jerala_P' + str(i), 'Jerala_P' + str(i+1)]
        pop_df_remaining = pop_merge[(pop_df.P1.isin(test_Pros)) | (pop_df.P2.isin(test_Pros))].dropna()
        for col in ['popatov', 'bicpa_scores', 'icipa_core_vert_scores', 'icipa_nter_core_scores' ]:
            rows.append({
                'dataset': z_names[col], 'test_r2':  pearsonr(pop_df_remaining['ashr_log2FoldChange_HIS_TRP'], pop_df_remaining[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(pop_df_remaining['ashr_log2FoldChange_HIS_TRP'], pop_df_remaining[col])[0])})

    df_heldout_pop_two = pd.DataFrame(rows)

    return df_barplot, df_heldout_pop, df_heldout_pop_two

def get_holdout_padj_baselines():    
    #get baseline for AUCROC values 
    baselines = []
    pairs = []
    ds = 'v2'
    r2 = 'af'
    for i in range(1,13,2):
        #pairs.append('P' + str(i) + '_' + 'P' + str(i+1))
        test_pros = ['P' + str(i), 'P' + str(i+1)]
        large_dataset_padj_order, not_model_cols =load_dataset_single(ds, r2,set_name= 'ncip')
        df_test = large_dataset_padj_order[(large_dataset_padj_order.id1.isin(test_pros)) | (large_dataset_padj_order.id2.isin(test_pros))]
        test_y = df_test.binned.to_numpy()
        print (sum(test_y == 1))
        print (sum(test_y == 1) / test_y.shape[0])

        baselines.append((sum(test_y == 1) / test_y.shape[0]))
        
    mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_autotune_5_small.csv')
    mp3_seq_values['binned'] = mp3_seq_values.ashr_log2FoldChange_HIS_TRP.apply(lambda x: label_LFC_bins(x))
    test_train_split = ['test', 'train'] * int((0.1 * mp3_seq_values.shape[0]))
    test_train_split = test_train_split + ['train'] * int(mp3_seq_values.shape[0] - len(test_train_split))
    mp3_seq_values = mp3_seq_values.sort_values('ashr_padj_HIS_TRP')
    mp3_seq_values['order_padj_sets'] = test_train_split

    overall_baseline = mp3_seq_values[(mp3_seq_values.order_padj_sets == 'test' )]
    overall_baseline = overall_baseline[overall_baseline.binned == 1].shape[0]/overall_baseline.shape[0]
    all_binned_baseline = mp3_seq_values[mp3_seq_values.binned == 1].shape[0]/mp3_seq_values.shape[0]
    
    return baselines, overall_baseline, all_binned_baseline


#correlation of just individual AF metrics and ashr as last good benchmark - trying to show that measurements + AF is good, not just AF by its self 
#v2

def get_af_metrics_baselines_train_test():
    #AF preds 

    rows = []

    af_df,et  = load_dataset_single('v2', 'af' ,set_name='ncip')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
            
    af_df,et  = load_dataset_single('v2', 'af',set_name='ncip')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col])})

    #next df 
    af_df,et  = load_dataset_single('v1_512', 'af',set_name='ncip')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v1_512',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col])[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col]  )})
            
    
    af_df,et  = load_dataset_single('v1_512', 'af',set_name='ncip')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v1_512',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col])[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col]  )})
    #next df 
    af_df,et  = load_dataset_single('mono', 'af',set_name='ncip')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in [ 'mean_plddt',
        'mean_pae_interaction_AB', 'mean_pae_interaction_BA',
        'mean_pae_interaction', 'mean_pae_intra_chain_A',
        'mean_pae_intra_chain_B', 'mean_pae_intra_chain', 'mean_pae',
        'pTMscore']:
            if 'pae' in col:
                 rows.append({
                'dataset': 'mono',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * -1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'mono',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
    
    af_df,et  = load_dataset_single('mono', 'af',set_name='ncip')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in [ 'mean_plddt',
        'mean_pae_interaction_AB', 'mean_pae_interaction_BA',
        'mean_pae_interaction', 'mean_pae_intra_chain_A',
        'mean_pae_intra_chain_B', 'mean_pae_intra_chain', 'mean_pae',
        'pTMscore']:
            if 'pae' in col:
                 rows.append({
                'dataset': 'mono',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * -1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'mono',
                    'mode': 'test',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})


    af_df,et  = load_dataset_single('v3', 'af',set_name='ncip')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm']:
            if 'pae' in col:
                 rows.append({
                'dataset': 'v3',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * - 1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'v3',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
            
    af_df,et  = load_dataset_single('v3', 'af',set_name='ncip')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm']:
            if 'pae' in col:
                 rows.append({
                'dataset': 'v3',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * - 1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col] * -1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'v3',
                    'mode': 'test',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col])[0],
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})

    
    df_barplot = pd.DataFrame(rows)
    return df_barplot


def get_af_metrics_baselines_train_test_model_1():
    #AF preds 

    rows = []

    af_df,et  = load_dataset_single('v2', 'af',set_name='ncip')
    test_df =  af_df[af_df.model == 1]
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
            
    af_df,et  = load_dataset_single('v2', 'af',set_name='ncip')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col])})

    #next df 
    af_df,et  = load_dataset_single('v1_512', 'af',set_name='ncip')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v1_512',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col])[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col]  )})
            
    
    af_df,et  = load_dataset_single('v1_512', 'af',set_name='ncip')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v1_512',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col])[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col]  )})
    #next df 
    af_df,et  = load_dataset_single('mono', 'af',set_name='ncip')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in [ 'mean_plddt',
        'mean_pae_interaction_AB', 'mean_pae_interaction_BA',
        'mean_pae_interaction', 'mean_pae_intra_chain_A',
        'mean_pae_intra_chain_B', 'mean_pae_intra_chain', 'mean_pae',
        'pTMscore']:
            if 'pae' in col:
                rows.append({
                'dataset': 'mono',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * -1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'mono',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
    
    af_df,et  = load_dataset_single('mono', 'af',set_name='ncip')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in [ 'mean_plddt',
        'mean_pae_interaction_AB', 'mean_pae_interaction_BA',
        'mean_pae_interaction', 'mean_pae_intra_chain_A',
        'mean_pae_intra_chain_B', 'mean_pae_intra_chain', 'mean_pae',
        'pTMscore']:
            if 'pae' in col:
                rows.append({
                'dataset': 'mono',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * -1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'mono',
                    'mode': 'test',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})


    af_df,et  = load_dataset_single('v3', 'af',set_name='ncip')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm']:
       
            if 'pae' in col:
                rows.append({
                'dataset': 'v3',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * - 1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'v3',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
            
    af_df,et  = load_dataset_single('v3', 'af',set_name='ncip')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm']:
        
            if 'pae' in col:
                rows.append({
                'dataset': 'v3',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * - 1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col] * -1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'v3',
                    'mode': 'test',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col])[0],
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})

    
    df_barplot = pd.DataFrame(rows)

    return df_barplot

def get_all_baseline_sets_and_predictions():
    bicpa_scores = load('PNIC_bcipa.bin')
    icipa_core_vert_scores = load('PNIC_icipa_core_vert.bin')
    icipa_nter_core_scores = load('PNIC_icipa_nter_core.bin')
    qcipa_scores = load('PNIC_qcipa.bin')

    d=load_score_file("PNIC-complete.out")
    ids = sorted(d.ids(), key=natural_key)
    mat = symdict_to_array(d, ids)

    mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_autotune_5_small.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})

    mp3_seq_values['P1'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[1])
    mp3_seq_values['P2'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[3])

    test_train_split = ['test', 'train'] * int((0.1 * mp3_seq_values.shape[0]))
    test_train_split = test_train_split + ['train'] * int(mp3_seq_values.shape[0] - len(test_train_split))
    mp3_seq_values = mp3_seq_values.sort_values('ashr_padj_HIS_TRP')
    mp3_seq_values['order_padj_sets'] = test_train_split
    mp3_seq_values['binned'] = mp3_seq_values.ashr_log2FoldChange_HIS_TRP.apply(lambda x: label_LFC_bins(x))

    #make df of popatov value
    df_rows = []
    for i in range(0,12):
        for j in range(0,12):
            df_rows.append({
                'P1': 'Jerala_P' + str(i + 1),
                'P2': 'Jerala_P' + str(j + 1),
                'popatov': mat[i][j],
                'bicpa_scores':bicpa_scores[i][j],
                'icipa_core_vert_scores':icipa_core_vert_scores[i][j],
                'icipa_nter_core_scores':icipa_nter_core_scores[i][j],
                'qcipa_scores':qcipa_scores[i][j],
            })
    pop_df = pd.DataFrame(df_rows)

    pop_merge =pop_df.merge(mp3_seq_values, how = 'left', on = ['P1', 'P2'] )
    to_correl = pop_merge[['P1', 'P2', 'popatov', 'ashr_log2FoldChange_HIS_TRP', 'bicpa_scores', 'icipa_core_vert_scores', 'icipa_nter_core_scores', 'order_padj_sets', 'binned']].dropna()
    #to_correl = to_correl[to_correl.order_padj_sets == 'test']

    #average each test point like in other evaluations 

    #using thse for plotting purposes later on to order bars at the end of the plots
    z_names = {'popatov':'z1', 'bicpa_scores':'z2', 'icipa_core_vert_scores':'z3', 'icipa_nter_core_scores':'z4' }

    rows = []
    for col in ['popatov', 'bicpa_scores', 'icipa_core_vert_scores', 'icipa_nter_core_scores' ]:
            rows.append({
                'dataset': z_names[col] + '_all', 
                'mode': 'all',
                'test_r2':  pearsonr(to_correl['ashr_log2FoldChange_HIS_TRP'], to_correl[col] * -1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(to_correl['ashr_log2FoldChange_HIS_TRP'], to_correl[col]* -1)[0]),
                ##'test_mccf1': get_mcc_f1_from_msrmts(to_correl['binned'] ==1 , to_correl[col]*-1)[0],
                'test_AUCROC':roc_auc_score(to_correl['binned'], to_correl['popatov'] *-1 ),
                'test_avgpr':average_precision_score(to_correl['binned'], to_correl[col] *-1 )})
            
    df_barplot = pd.DataFrame(rows)

    return df_barplot, to_correl

#malb baselines 
#only ran V2 and V3 for these
def get_af_metrics_baselines_malb():
    #AF preds 

    rows = []

    af_df,et  = load_dataset_single('malb_v2', 'af', 'malb')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'malb_v2_z',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
            
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'malb_v2_z',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
            
    af_df,et  = load_dataset_single('malb_v3', 'af', 'malb')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm']:
       
            if 'pae' in col:
                 rows.append({
                'dataset': 'malb_v3_z',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * - 1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'malb_v3_z',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
    
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm']:
       
            if 'pae' in col:
                 rows.append({
                'dataset': 'malb_v3_z',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * - 1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'malb_v3_z',
                    'mode': 'test',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
    
    #bcipa 
    bcipa_preds = pd.read_csv('malb_bcipa_scores.csv')
    bcipa_preds=bcipa_preds.rename(columns = {'pro_1':'DBD', 'pro_2':'AD'})
    #remove double underscore

    bcipa_preds['PPI'] = 'DBD:' + bcipa_preds['DBD'] + ':AD:' + bcipa_preds['AD']
    mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_new_smaller_flat_autotune.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})
    merged = bcipa_preds.merge(mp3_seq_values, on = 'PPI', how = 'left')
    merged = merged[['PPI', 'bcipa', 'ashr_log2FoldChange_HIS_TRP']].dropna()
    col = 'bcipa'
    rows.append({
                    'dataset': 'zbcipa',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(merged['ashr_log2FoldChange_HIS_TRP'], merged[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(merged['ashr_log2FoldChange_HIS_TRP'], merged[col])[0]),
                    'test_AUCROC':0 ,
                    'test_avgpr':0 })
    df_barplot = pd.DataFrame(rows)
    return df_barplot


def get_af_metrics_baselines_malbs_model_1():
    #AF preds 
    rows = []

    af_df,et  = load_dataset_single('malb_v2', 'af')
    test_df =  af_df[af_df.model == 1]
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                #'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})

    af_df,et  = load_dataset_single('malb_v3', 'af')
    af_df =  af_df[af_df.model == 1]
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm']:
       
            if 'pae' in col:
                rows.append({
                'dataset': 'v3',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col] * - 1)[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            else:
                rows.append({
                    'dataset': 'v3',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                    'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col]  ),
                    'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] )})
            
    #bcipa 
    bcipa_preds = pd.read_csv('malb_bcipa_scores.csv') #icipa nterm version 
    bcipa_preds=bcipa_preds.rename(columns = {'pro_1':'DBD', 'pro_2':'AD'})
    bcipa_preds['PPI'] = 'DBD:' + bcipa_preds['DBD'] + ':AD:' + bcipa_preds['AD']
    mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_new_smaller_flat_autotune.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})
    merged = bcipa_preds.merge(mp3_seq_values, on = 'PPI', how = 'left')
    merged = merged[['PPI', 'bcipa', 'ashr_log2FoldChange_HIS_TRP']].dropna()
    col = 'bcipa'
    rows.append({
                    'dataset': 'zbcipa',
                    'mode': 'all',
                    'r2_cutoff': col, 
                    'test_r2':  pearsonr(merged['ashr_log2FoldChange_HIS_TRP'], merged[col])[0] ** 2,
                    'test_spearman':  np.abs(spearmanr(merged['ashr_log2FoldChange_HIS_TRP'], merged[col])[0]),
                    'test_AUCROC':0 ,
                    'test_avgpr':0 })
    df_barplot = pd.DataFrame(rows)

    return df_barplot


#getting holdout set baselines 


def get_holdout_baseline_performances(set_name, held_out_pro_lists):
    if set_name == 'ncip':
        vals_to_try = ['v2', 'v3']
    else:
        vals_to_try = ['malb_v2', 'malb_v3']
    rows = []
    for version in vals_to_try:
        df_total, _ = load_dataset_single(version, 'af', set_name )
        for subset in held_out_pro_lists:
            slice_df_test = df_total[(df_total.id1.isin(subset)) | (df_total.id2.isin(subset))]
            slice_df_test_avg =  slice_df_test.groupby(['ppi']).mean()
            # sns.scatterplot(data = slice_df_test_avg, x = 'iptm', y = 'lfc_all', style = 'binned')
            # plt.show()
            rows.append({
                        'dataset': version + '_z',
                        'mode': 'holdout',
                        'r2_cutoff':'af', 
                        'test_r2':  pearsonr(slice_df_test_avg['lfc_all'], slice_df_test_avg['iptm'])[0] ** 2,
                        'test_spearman':  np.abs(spearmanr(slice_df_test_avg['lfc_all'], slice_df_test_avg['iptm'])[0]),
                        'test_AUCROC':roc_auc_score(slice_df_test_avg['binned'], slice_df_test_avg['iptm']),
                        'test_avgpr':average_precision_score(slice_df_test_avg['binned'], slice_df_test_avg['iptm'] ) })
    df_barplot = pd.DataFrame(rows)
    return df_barplot

def get_holdout_train_baseline_performances(set_name, held_out_pro_lists):
    if set_name == 'ncip':
        vals_to_try = ['v2', 'v3']
    else:
        vals_to_try = ['malb_v2', 'malb_v3']
    rows = []
    for version in vals_to_try:
        df_total, _ = load_dataset_single(version, 'af', set_name )
        for subset in held_out_pro_lists:
            slice_df_test = df_total[~(df_total.id1.isin(subset)) & ~(df_total.id2.isin(subset))]
            slice_df_test_avg =  slice_df_test.groupby(['ppi']).mean()
            # sns.scatterplot(data = slice_df_test_avg, x = 'iptm', y = 'lfc_all', style = 'binned')
            # plt.show()
            rows.append({
                        'dataset': version +'_z',
                        'mode': 'holdout',
                        'r2_cutoff':'z', 
                        'train_r2':  pearsonr(slice_df_test_avg['lfc_all'], slice_df_test_avg['iptm'])[0] ** 2,
                        'train_r2_not_avgd':  pearsonr(slice_df_test['lfc_all'], slice_df_test['iptm'])[0] ** 2,
                        'train_spearman':  np.abs(spearmanr(slice_df_test_avg['lfc_all'], slice_df_test_avg['iptm'])[0]),
                         'train_spearman_not_avgd':  np.abs(spearmanr(slice_df_test['lfc_all'], slice_df_test['iptm'])[0]),
                        'rocauc_train':roc_auc_score(slice_df_test_avg['binned'], slice_df_test_avg['iptm']),
                        'rocauc_train_not_avgd':roc_auc_score(slice_df_test['binned'], slice_df_test['iptm']),
                        'avgpr_train':average_precision_score(slice_df_test_avg['binned'], slice_df_test_avg['iptm'] ),
                        'avgpr_train_not_avgd':average_precision_score(slice_df_test['binned'], slice_df_test['iptm'] )})
    df_barplot = pd.DataFrame(rows)
    return df_barplot