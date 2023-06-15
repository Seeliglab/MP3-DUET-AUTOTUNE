from pipeline_interaction_regression_l2 import load_dataset_single
from predictor_comparisons import *
from processing_functions import *
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
#split up v1, v2, v3
from sklearn import linear_model
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.utils import resample
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from malb_test_sets import *
import seaborn as sns

def padj_train_best_lasso_eval_on_test_get_preds(best_l1, large_dataset_padj_order, new_not_model_cols, save_name, use_oversample = False, weights = False):

    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.lfc_all.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1.1 - df_slice.ashr_padj.to_numpy()
    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], weights_oversampled))
    df_test = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'test'].copy()
    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.lfc_all.to_numpy()

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    x_test_scaled = scaler_x_full.transform(test_x)

    scaler_y_full = preprocessing.StandardScaler().fit(full_lfc.reshape(-1, 1))
    y_train_scaled = scaler_y_full.transform(full_lfc.reshape(-1, 1))
    y_test_scaled = scaler_y_full.transform(test_y.reshape(-1, 1))

    if full_train.shape[1] <= 4: 
        clf = linear_model.Ridge(alpha=best_l1, max_iter= 20000)
    else:
        clf = linear_model.Ridge(alpha=best_l1, max_iter= 20000)

    if weights:
        clf.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, y_train_scaled)
    
    train_preds = clf.predict(x_train_scaled)
    
    num_nonzero = sum(clf.coef_ != 0)
    r2_train = clf.score(x_train_scaled, y_train_scaled)
    #r2_test = clf.score(x_test_scaled, y_test_scaled)
    sr_train = spearmanr(y_train_scaled, clf.predict(x_train_scaled))[0]
    #sr_test = spearmanr(y_test_scaled, clf.predict(x_test_scaled))[0]
    pr_train = pearsonr(y_train_scaled.flatten(), clf.predict(x_train_scaled).flatten())[0]
    #pr_test = pearsonr(y_test_scaled.flatten(), clf.predict(x_test_scaled))[0]
    
    val_preds = clf.predict(x_test_scaled)
    df_test['preds'] = val_preds
    df_test['scaled_y'] = y_test_scaled
    avgs_test = df_test.groupby(['ppi']).mean()

    return avgs_test

def get_af_metrics_baselines():
    #AF preds 
    af_df,et  = load_dataset_single('v2', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    rows = []
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})

    #next df 
    af_df,et  = load_dataset_single('v1_512', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v1_512',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})

    #next df 
    af_df,et  = load_dataset_single('mono', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in [ 'mean_plddt',
        'mean_pae_interaction_AB', 'mean_pae_interaction_BA',
        'mean_pae_interaction', 'mean_pae_intra_chain_A',
        'mean_pae_intra_chain_B', 'mean_pae_intra_chain', 'mean_pae',
        'pTMscore']:
            rows.append({
                'dataset': 'mono',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})

    af_df,et  = load_dataset_single('v3', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm',
        'max_pae']:
            rows.append({
                'dataset': 'v3',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})

    df_barplot = pd.DataFrame(rows)
    df_barplot = df_barplot.sort_values(by = ['test_spearman','test_avgpr'], ascending=False).drop_duplicates( subset = 'dataset', keep = 'first')
    #now we have the ones which worked the best on train - run them on the test 
    test_perfs = []

    af_df,et  = load_dataset_single('v3', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()
    best_col = df_barplot[df_barplot.dataset == 'v3'].r2_cutoff.to_list()[0]
    test_perfs.append({
                'dataset': 'v3',
                'r2_cutoff': best_col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[best_col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[best_col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[best_col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[best_col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[best_col] *-1 )})

    af_df,et  = load_dataset_single('v2', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()
    best_col = df_barplot[df_barplot.dataset == 'v2'].r2_cutoff.to_list()[0]
    test_perfs.append({
                'dataset': 'v2',
                'r2_cutoff': best_col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[best_col])[0] ** 2,
                'test_spearman':   np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[best_col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[best_col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[best_col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[best_col] *-1 )})
    
    af_df,et  = load_dataset_single('v1_512', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()
    best_col = df_barplot[df_barplot.dataset == 'v1_512'].r2_cutoff.to_list()[0]
    test_perfs.append({
                'dataset': 'v1_512',
                'r2_cutoff': best_col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[best_col])[0] ** 2,
                'test_spearman':   np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[best_col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[best_col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[best_col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[best_col] *-1 )})
    
    af_df,et  = load_dataset_single('mono', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()
    best_col = df_barplot[df_barplot.dataset == 'mono'].r2_cutoff.to_list()[0]
    test_perfs.append({
                'dataset': 'mono',
                'r2_cutoff': best_col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[best_col])[0] ** 2,
                'test_spearman':   np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[best_col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[best_col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[best_col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[best_col] *-1 )})
    
    return  pd.DataFrame(test_perfs)


from pipeline_interaction_regression_l2 import load_dataset_single
from predictor_comparisons import *
from processing_functions import *
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
#split up v1, v2, v3
from sklearn import linear_model
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr, pearsonr
from sklearn.utils import resample
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=FutureWarning)
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from multiprocessing import Pool
from sklearn.model_selection import StratifiedKFold
from malb_test_sets import *
import seaborn as sns

def padj_train_best_lasso_eval_on_test_get_preds(best_l1, large_dataset_padj_order, new_not_model_cols, save_name, use_oversample = False, weights = False):

    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.lfc_all.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1.1 - df_slice.ashr_padj.to_numpy()
    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], weights_oversampled))
    df_test = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'test'].copy()
    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.lfc_all.to_numpy()

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    x_test_scaled = scaler_x_full.transform(test_x)

    scaler_y_full = preprocessing.StandardScaler().fit(full_lfc.reshape(-1, 1))
    y_train_scaled = scaler_y_full.transform(full_lfc.reshape(-1, 1))
    y_test_scaled = scaler_y_full.transform(test_y.reshape(-1, 1))

    if full_train.shape[1] <= 4: 
        clf = linear_model.Ridge(alpha=best_l1, max_iter= 20000)
    else:
        clf = linear_model.Ridge(alpha=best_l1, max_iter= 20000)

    if weights:
        clf.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, y_train_scaled)
    
    train_preds = clf.predict(x_train_scaled)
    
    num_nonzero = sum(clf.coef_ != 0)
    r2_train = clf.score(x_train_scaled, y_train_scaled)
    #r2_test = clf.score(x_test_scaled, y_test_scaled)
    sr_train = spearmanr(y_train_scaled, clf.predict(x_train_scaled))[0]
    #sr_test = spearmanr(y_test_scaled, clf.predict(x_test_scaled))[0]
    pr_train = pearsonr(y_train_scaled.flatten(), clf.predict(x_train_scaled).flatten())[0]
    #pr_test = pearsonr(y_test_scaled.flatten(), clf.predict(x_test_scaled))[0]
    
    val_preds = clf.predict(x_test_scaled)
    df_test['preds'] = val_preds
    df_test['scaled_y'] = y_test_scaled
    avgs_test = df_test.groupby(['ppi']).mean()

    return avgs_test

def get_af_metrics_baselines_train_test():
    #AF preds 

    rows = []

    af_df,et  = load_dataset_single('v2', 'af')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            
    af_df,et  = load_dataset_single('v2', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()   
    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v2',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman': np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})

    #next df 
    af_df,et  = load_dataset_single('v1_512', 'af')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v1_512',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            
    
    af_df,et  = load_dataset_single('v1_512', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'ptm', 'iptm']:
            rows.append({
                'dataset': 'v1_512',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
    #next df 
    af_df,et  = load_dataset_single('mono', 'af')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in [ 'mean_plddt',
        'mean_pae_interaction_AB', 'mean_pae_interaction_BA',
        'mean_pae_interaction', 'mean_pae_intra_chain_A',
        'mean_pae_intra_chain_B', 'mean_pae_intra_chain', 'mean_pae',
        'pTMscore']:
            rows.append({
                'dataset': 'mono',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
    
    af_df,et  = load_dataset_single('mono', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in [ 'mean_plddt',
        'mean_pae_interaction_AB', 'mean_pae_interaction_BA',
        'mean_pae_interaction', 'mean_pae_intra_chain_A',
        'mean_pae_intra_chain_B', 'mean_pae_intra_chain', 'mean_pae',
        'pTMscore']:
            rows.append({
                'dataset': 'mono',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})


    af_df,et  = load_dataset_single('v3', 'af')
    test_df =  af_df#[af_df.order_padj_sets == 'train']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm',
        'max_pae']:
            rows.append({
                'dataset': 'v3',
                'mode': 'all',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})
            
    af_df,et  = load_dataset_single('v3', 'af')
    test_df =  af_df[af_df.order_padj_sets == 'test']
    avgs_test = test_df.groupby(['ppi']).mean()

    for col in ['mean_plddt', 'pae', 'ptm', 'iptm',
        'max_pae']:
            rows.append({
                'dataset': 'v3',
                'mode': 'test',
                'r2_cutoff': col, 
                'test_r2':  pearsonr(avgs_test['lfc_all'], avgs_test[col])[0] ** 2,
                'test_spearman':  np.abs(spearmanr(avgs_test['lfc_all'], avgs_test[col])[0]),
                'test_mccf1': get_mcc_f1_from_msrmts(avgs_test['binned'] ==1 , avgs_test[col]*-1)[0],
                'test_AUCROC':roc_auc_score(avgs_test['binned'], avgs_test[col] *-1 ),
                'test_avgpr':average_precision_score(avgs_test['binned'], avgs_test[col] *-1 )})

    
    df_barplot = pd.DataFrame(rows)

    return df_barplot


print (get_af_metrics_baselines_train_test())