import pickle
import os
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
from sklearn.metrics import mean_squared_error, matthews_corrcoef, roc_auc_score, f1_score, average_precision_score
from mccf1_functions import *

#housekeeping 
def get_weight_type(x):
    temp = x.split('_')
    if len(temp) == 1:
        return x
    else:
        return '_'.join(x.split('_')[0:-1])
                        
def get_set(x):
    temp = x.split('_')
    if len(temp) == 1:
        return 'v3'
    else:
        return x.split('_')[-1].replace('_', '')

def merge_vals(v2_preds, endings = ['']):

    mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})

    mp3_seq_values['P1'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[1])
    mp3_seq_values['P2'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[3])
    mp3_seq_values['id1'] = mp3_seq_values.P1.apply(lambda x: x.replace('Jerala_', '')) 
    mp3_seq_values['id2'] = mp3_seq_values.P2.apply(lambda x: x.replace('Jerala_', '')) 
    mp3_seq_values['ppi'] = mp3_seq_values.id1 + '_' + mp3_seq_values.id2
    

    test_train_split = ['test', 'train'] * int((0.1 * mp3_seq_values.shape[0]))
    test_train_split = test_train_split + ['train'] * int(mp3_seq_values.shape[0] - len(test_train_split))
    mp3_seq_values = mp3_seq_values.sort_values('ashr_padj_HIS_TRP')
    mp3_seq_values['order_padj_sets'] = test_train_split

    mp3_seq_values_ppis = mp3_seq_values.ppi.to_list()
    mp3_seq_lfcs = mp3_seq_values.ashr_log2FoldChange_HIS_TRP.to_list()
    mp3_seq_lfcSEs = mp3_seq_values.ashr_lfcSE_HIS_TRP.to_list()
    ashr_padj = mp3_seq_values.ashr_padj_HIS_TRP.fillna(1).to_list()
    test_train = mp3_seq_values.order_padj_sets.to_list()
    ppis_dummy = []

    for i in range(1,6):
        ppis_dummy =ppis_dummy + [x + '_' + str(i) for x in mp3_seq_values_ppis]

    correl_df = pd.DataFrame({'new_ppi': ppis_dummy, 'lfc': mp3_seq_lfcs * 5, 'lfcsSE': mp3_seq_lfcSEs * 5, 'ashr_padj':ashr_padj * 5, 'order_padj_sets': test_train * 5})

    correl_df_sig = pd.DataFrame({'new_ppi': ppis_dummy, 'lfc': mp3_seq_lfcs * 5, 'lfcsSE': mp3_seq_lfcSEs * 5})
    print(correl_df_sig.shape)

    correl_df_merge = v2_preds.merge(correl_df, on = 'new_ppi', how = 'inner')
    correl_df_merge =correl_df_merge.merge(correl_df_sig, on = 'new_ppi', suffixes = ['_all', '_sig'], how = 'left')

    #trying some generic l1 models 
    not_model_cols = ['id1', 'id2', 'model', 'type', 'msa_depth','ppi', 'on_target']
    new_not_model_cols = []
    if len(endings) != 0:
        for ending in  endings:
            new_not_model_cols = new_not_model_cols + [x + ending for x in not_model_cols] 
    else:
        new_not_model_cols = not_model_cols
    new_not_model_cols = new_not_model_cols + ['new_ppi','lfc_all', 'lfcsSE_all', 'lfc_sig', 'lfcsSE_sig', 'ashr_padj', 'order_padj_sets'] 

    return correl_df_merge, new_not_model_cols

def label_LFC_bins(x, sort_level = 1):
    if x >= sort_level:
        return 1
    else:
        return -1

def load_dataset_single(df_set, r, endings = []):
    v2_preds = pd.read_csv('./datasets/' + df_set + '_correl_reduced_r_' + str(r) +'.csv') 
    v2_preds.dropna(axis = 0, inplace = True)
    correl_df_merge, new_not_model_cols = merge_vals(v2_preds, endings)
    large_dataset = correl_df_merge
    large_dataset['binned'] = large_dataset.lfc_all.apply(lambda x: label_LFC_bins(x))
    new_not_model_cols = new_not_model_cols + ['order_padj_sets', 'binned']
    return large_dataset, new_not_model_cols


#holdout 
#get_best_l1_holdout(results_df, asc = False, group_col= 'l1_penalty', sort_col = 'validation_mcc')
def get_best_l1_holdout(lasso_data, asc = False, group_col = 'l1_penalty', sort_col = 'validation_mcc'):
    lasso_data = lasso_data.drop(columns = 'val_pair') #drop to not throw off mean 
    results = lasso_data.groupby([group_col]).mean()
    best_l1 = results.sort_values(sort_col, ascending = asc).index[0]
    #print(results.sort_values(sort_col, ascending = asc))
    print(results.sort_values(sort_col, ascending = asc).head())
    return best_l1

@ignore_warnings(category=ConvergenceWarning)
def run_holdout_cv(large_dataset_padj_order, new_not_model_cols, test_pairs, train_pairs, use_oversample = False, weights = False):
  
    test_pros = [x for p in test_pairs for x in p]
    train_df = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pros)) & ~(large_dataset_padj_order.id2.isin(test_pros))]
    print (train_df.shape, large_dataset_padj_order.shape)

    l1_lambdas = np.logspace(-5, 5, 51, base=10)
    data = []
    l1_names = [f'coefficients [L2={l1_lambda:.0e}]' for l1_lambda in l1_lambdas]
    
    for l1, l1_col_name in zip(l1_lambdas, l1_names):# in range(7):
        for i in range(0,len(train_pairs)):
            #decide which pair to hold out as validation this time 
            val_pair = train_pairs[i]
            val_pros = [x for p in [val_pair] for x in p]
            df_slice = train_df[~(train_df.id1.isin(val_pros)) & ~(train_df.id2.isin(val_pros))]
            df_slice_valid = train_df[(train_df.id1.isin(val_pros)) | (train_df.id2.isin(val_pros))]

            train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
            train_y = df_slice.binned.to_numpy()
            weights_train = 1.1 - df_slice.ashr_padj.to_numpy()
            if use_oversample:
                binned_y = df_slice.binned.to_numpy()
                to_sample = train_x[binned_y == -1].shape[0]
                X_oversampled, y_oversampled, weights_oversampled= resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
                train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
                train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
                weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled)) 

            val_x = df_slice_valid.drop(columns = new_not_model_cols).to_numpy()
            val_y = df_slice_valid.binned.to_numpy()

            scaler_x = preprocessing.StandardScaler().fit(train_x)
            x_train_scaled = scaler_x.transform(train_x)
            x_val_scaled = scaler_x.transform(val_x)
            #x_test_scaled = scaler_x.transform(test_x)
            if train_x.shape[1] <= 4:
                model = linear_model.LogisticRegression(penalty = 'l2', C=1/l1, solver = 'liblinear', max_iter=1000)
            else:
                model = linear_model.LogisticRegression(penalty = 'l2', C=1/l1, solver = 'liblinear', max_iter=1000)
            if weights:
                model.fit(x_train_scaled, train_y, sample_weight = weights_train)
            else:
                model.fit(x_train_scaled, train_y)

            data.append({
                'val_pair': val_pair[0] + '_' + val_pair[1],
                'l1_penalty': l1,
                'validation_mcc': matthews_corrcoef(val_y == 1, model.predict(x_val_scaled)),
                'n_non_zero': sum(model.coef_ [0]!= 0)
            })
    lasso_data = pd.DataFrame(data)
    #print (lasso_data)
    return lasso_data


def train_best_lasso_eval_on_test_holdout(best_l1, large_dataset_padj_order, new_not_model_cols, save_name, test_pairs, use_oversample = False, weights = False):

    test_pros = [x for p in test_pairs for x in p]
    df_slice = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pros)) & ~(large_dataset_padj_order.id2.isin(test_pros))]
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.binned.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1.1 - df_slice.ashr_padj.to_numpy()
    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, w_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], w_oversampled))

    df_test = large_dataset_padj_order[(large_dataset_padj_order.id1.isin(test_pros)) | (large_dataset_padj_order.id2.isin(test_pros))]
    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.binned.to_numpy()

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    x_test_scaled = scaler_x_full.transform(test_x)
    
    if full_train.shape[1] <= 4:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    else:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    if weights:
        clf.fit(x_train_scaled, full_lfc, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, full_lfc)
    predict_test = clf.predict_proba(x_test_scaled)[:,1]
    #print (predict_test)
    df_test['preds'] = predict_test
    avgs_test = df_test.groupby(['ppi']).mean()
    avgs_test['pred_label'] = avgs_test['preds'] >= 0.5

    #make roc plot 
    num_nonzero = sum(clf.coef_[0] != 0)
    rocauc_train = roc_auc_score(full_lfc.flatten(), clf.predict_proba(x_train_scaled)[:,1])
    rocauc_test = roc_auc_score(avgs_test['binned'], avgs_test['preds'])
    avgpr_train = average_precision_score(full_lfc.flatten(), clf.predict_proba(x_train_scaled)[:,1])
    avgpr_test = average_precision_score(avgs_test['binned'], avgs_test['preds'])
    avgs_test['pred_label'] = avgs_test['preds'] >= 0.5

    mccf1 = get_mcc_f1_from_msrmts(avgs_test['binned'] == 1, avgs_test['preds'])[0]
    
    return   rocauc_train, rocauc_test, avgpr_train, avgpr_test, mccf1


def run_one_model_combo_holdout_classification(dataset, r2, temp_pairs, test_pair, binned, weights):
    if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
        if dataset != 'all':
            ds, not_model_cols =load_dataset_single(dataset, r2)
        else:
            ds, not_model_cols = load_dataset_single(dataset, r2, endings = ['_v1', '_v1512', '_v2', '_v3'])
            ds['ppi'] = ds.ppi_v1.apply(lambda x: x.replace('_1', ''))
            ds['id1'] = ds.id1_v1.apply(lambda x: x.replace('_1', ''))
            ds['id2'] = ds.id2_v1.apply(lambda x: x.replace('_1', ''))
            not_model_cols.append('id1')
            not_model_cols.append('id2')
            not_model_cols.append('ppi')
        print (ds.shape, dataset, r2)
        results_df = run_holdout_cv(ds, not_model_cols, [test_pair], temp_pairs, use_oversample=binned, weights= weights)
        best_l1 = get_best_l1_holdout(results_df)
        mname = 'lasso_holdout_' + dataset + '_r2_cutoff_' + str(r2) + '_oversample_' + str(binned) + '_weights_' + str(weights)
        rocauc_train, rocauc_test, avgpr_train, avgpr_test, mccf1 = train_best_lasso_eval_on_test_holdout(best_l1, ds, not_model_cols, mname + '_' + test_pair[0] + '_' + test_pair[1], [test_pair], use_oversample = binned, weights = weights)
        return {
            'dataset': dataset,
            'r2_cutoff': r2, 
            'l1_penalty': best_l1,
            'train_AUCROC': rocauc_train, 
            'train_avgpr': avgpr_train, 
            'test_AUCROC': rocauc_test, 
            'test_avgpr': avgpr_test,
            'test_mccf1': mccf1
        }
    else:
        return {
            'dataset': None,
            'r2_cutoff': None, 
            'l1_penalty': None,
            'train_AUCROC': None, 
            'train_avgpr': None, 
            'test_AUCROC': None, 
            'test_avgpr': None,
            'test_mccf1': None
        }


def holdout_cv_classification(binned = False, weights = False):
    pairs = []
    for i in range(1,13,2):
        pairs.append(('P' + str(i), 'P' + str(i+1)))
    datasets = [ 'mono', 'v1_512', 'v2', 'v3' ]#'all',
    r2_cutoffs = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'af']
    runs = []
    for p_ind in range(0, len(pairs)):
        temp_pairs = pairs.copy()
        test_pair = temp_pairs[p_ind]
        temp_pairs.remove(test_pair)
        #runs all singel cv sets 
        for dataset in datasets:
            for r2 in r2_cutoffs:
                if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
                #run_one_model_combo(dataset, r2, temp_pairs, test_pair)
                #print (dataset, r2, test_pair)
                    runs.append((dataset, r2, temp_pairs, test_pair, binned, weights))
    #now multithread 
    with Pool() as pool:
        data = pool.starmap(run_one_model_combo_holdout_classification, runs)
    return pd.DataFrame(data)

#best holdout l1 values

def holdout_classification_final_models_compare_weights(dataset, r2, binned, weights, best_l1, max_or_min):
    pairs = []
    for i in range(1,13,2):
        pairs.append(('P' + str(i), 'P' + str(i+1)))
    datasets = [dataset]
    r2_cutoffs = [r2]
    runs = []
    for p_ind in range(0, len(pairs)):
        temp_pairs = pairs.copy()
        test_pair = temp_pairs[p_ind]
        temp_pairs.remove(test_pair)
        #runs all singel cv sets 
        for dataset in datasets:
            for r2 in r2_cutoffs:
                runs.append((dataset, r2, test_pair, binned, weights, best_l1, max_or_min))
    #now multithread 
    with Pool() as pool:
        data = pool.starmap(run_defined_classification_l1_all_test_pairs, runs)
    return pd.DataFrame(data)



def run_defined_classification_l1_all_test_pairs(dataset, r2, test_pair, binned, weights, best_l1, max_or_min):
    if dataset != 'all':
        ds, not_model_cols =load_dataset_single(dataset, r2)
    else:
        ds, not_model_cols = load_dataset_single(dataset, r2, endings = ['_v1', '_v1512', '_v2', '_v3'])
        ds['id1'] = ds.id1_v1.apply(lambda x: x.replace('_1', ''))
        ds['id2'] = ds.id2_v1.apply(lambda x: x.replace('_1', ''))
        ds['ppi'] = ds.ppi_v1.apply(lambda x: x.replace('_1', ''))
        not_model_cols.append('id1')
        not_model_cols.append('id2')
        not_model_cols.append('ppi')
    mname = 'lasso_classification_holdout_' + dataset + '_r2_cutoff_' + str(r2) + '_oversample_' + str(binned) + '_weights_' + str(weights) + '_l1_' + str(best_l1) + max_or_min
    rocauc_train, rocauc_test, avgpr_train, avgpr_test, mccf1 = train_best_lasso_eval_on_test_holdout(best_l1, ds, not_model_cols, mname + '_' + test_pair[0] + '_' + test_pair[1], [test_pair], use_oversample = binned, weights= weights)
    return {
        'test_pair': test_pair[0] + '_' + test_pair[1],
        'dataset': dataset,
        'r2_cutoff': r2, 
        'l1_penalty': best_l1,
        'train_AUCROC': rocauc_train, 
        'train_avgpr': avgpr_train, 
        'test_AUCROC': rocauc_test, 
        'test_avgpr': avgpr_test,
        'test_mccf1': mccf1
    }

def final_train_best_classification(best_l1, large_dataset_padj_order, new_not_model_cols, save_name, test_pairs, use_oversample, weights):

    test_pros = [x for p in test_pairs for x in p]
    df_slice = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pros)) & ~(large_dataset_padj_order.id2.isin(test_pros))]
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.binned.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1.1 - df_slice.ashr_padj.to_numpy() 
    if use_oversample: 
        print ('use oversample')
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1],  weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], weights_oversampled)) 
        
    df_test = large_dataset_padj_order[(large_dataset_padj_order.id1.isin(test_pros)) | (large_dataset_padj_order.id2.isin(test_pros))]
    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.binned.to_numpy()

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    x_test_scaled = scaler_x_full.transform(test_x)
    
    
    if full_train.shape[1] <= 4:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    else:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)

    if weights:
        clf.fit(x_train_scaled, full_lfc, sample_weight=weights_train)
    else:
        clf.fit(x_train_scaled, full_lfc)
    predict_test = clf.predict_proba(x_test_scaled)[:,1]
    #print (predict_test)
    df_test['preds'] = predict_test
    avgs_test = df_test.groupby(['ppi']).mean()
    avgs_test['pred_label'] = avgs_test['preds'] >= 0.5

    #make roc plot 
    num_nonzero = sum(clf.coef_[0] != 0)
    rocauc_train = roc_auc_score(full_lfc.flatten(), clf.predict_proba(x_train_scaled)[:,1])
    rocauc_test = roc_auc_score(avgs_test['binned'], avgs_test['preds'])
    avgpr_train = average_precision_score(full_lfc.flatten(), clf.predict_proba(x_train_scaled)[:,1])
    avgpr_test = average_precision_score(avgs_test['binned'], avgs_test['preds'])
    
    mccf1 = get_mcc_f1_from_msrmts(avgs_test['binned'] == 1, avgs_test['preds'])[0]
   
    return  rocauc_train, rocauc_test, num_nonzero, avgpr_train, avgpr_test, mccf1


def remove_bad_model_groups(df):
    count_occ = df.dropna().value_counts('total_name').index
    #count_occ = count_occ[count_occ == 6].index
    return df[df.total_name.isin(count_occ)]

def get_l1_modes(df, max_or_min = 'max'):
    if max_or_min == 'max':
        series_df_modes = df.groupby(['total_name']).l1_penalty.apply(lambda x: max(x.mode()))
    else:
        series_df_modes = df.groupby(['total_name']).l1_penalty.apply(lambda x: min(x.mode()))
    print (series_df_modes)
    new_df = pd.DataFrame({'dataset': ['_'.join(x.split('_')[0:-1]) for x in series_df_modes.index], 
                        'r2_cutoff':[str(x.split('_')[-1]) for x in series_df_modes.index],
                        'l1_penalty': series_df_modes.values})
    return new_df

def process_holdout_results_classification(df_name, binned, weights, max_or_min):
    #looking at mcc vs aucroc for each test set 
    df = pd.read_csv(df_name)
    df = df.dropna()
    if 'aj_only' in df_name:
        df['r2_cutoff'] = 0
    df['total_name'] = df.apply(lambda row: row['dataset'] + '_' + str(row['r2_cutoff']), axis = 1)
    df = remove_bad_model_groups(df)
    l1_df = get_l1_modes(df, max_or_min)
    all_dfs = []
    for ind, row in l1_df.iterrows():
        l1_val = row['l1_penalty'] #use the largest mode found
        print (row)
        df_now = holdout_classification_final_models_compare_weights(row['dataset'], str(row['r2_cutoff']), binned, weights, l1_val, max_or_min)
        print (df_now)
        all_dfs.append(df_now)
    df_total = pd.concat(all_dfs)
    df_total.to_csv('final_results_' + max_or_min + df_name, index = False)

#padj test set 

def padj_cv_classification(binned = False, weights = False):
    datasets = [ 'mono', 'v1_512', 'v2', 'v3' ]#'all',
    r2_cutoffs = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'af']
    runs = []
   
    for dataset in datasets:
        for r2 in r2_cutoffs:
            if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' +r2 +'.csv'):
                runs.append((dataset, r2,  binned, weights))
    #now multithread 
    with Pool() as pool:
        data = pool.starmap(padj_run_one_model_combo_classification, runs)
    return pd.DataFrame(data)

def padj_run_one_model_combo_classification(dataset, r2, binned, weights):
    if dataset != 'all':
        ds, not_model_cols =load_dataset_single(dataset, r2)
    else:
        ds, not_model_cols = load_dataset_single(dataset, r2, endings = ['_v1', '_v1512', '_v2', '_v3'])
        ds['id1'] = ds.id1_v1.apply(lambda x: x.replace('_1', ''))
        ds['id2'] = ds.id2_v1.apply(lambda x: x.replace('_1', ''))
        ds['ppi'] = ds.ppi_v1.apply(lambda x: x.replace('_1', ''))
        not_model_cols.append('id1')
        not_model_cols.append('id2')
        not_model_cols.append('ppi')
    results_df = padj_run_cv_classification(ds, not_model_cols, use_oversample=binned, weights = weights)
    best_l1 = get_best_l1_holdout(results_df)
    mname = 'lasso_padj_' + dataset + '_r2_cutoff_' + str(r2) + '_oversample_' + str(binned) +'_weights_' + str(weights)
    rocauc_train, rocauc_test, avgpr_train, avgpr_test, mccf1 = padj_train_best_lasso_eval_on_test_classification(best_l1, ds, not_model_cols, mname, use_oversample = binned, weights = weights)
    return {
        'dataset': dataset,
        'r2_cutoff': r2, 
        'l1_penalty': best_l1,
        'train_AUCROC': rocauc_train, 
        'train_avgpr': avgpr_train, 
        'test_AUCROC': rocauc_test, 
        'test_avgpr': avgpr_test,
        'test_mccf1': mccf1
    }


@ignore_warnings(category=ConvergenceWarning)
def padj_run_cv_classification(large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):
  
    train_df = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
    print (train_df.shape, large_dataset_padj_order.shape)
    #do CV indicies 
    sk_split = StratifiedKFold(random_state= 1, shuffle = True)
    cv_inds = list(sk_split.split(X = np.zeros(train_df.shape[0]), y = train_df.binned.to_numpy()))
    
    l1_lambdas = np.logspace(-5, 5, 51, base=10)
    data = []
    l1_names = [f'coefficients [L2={l1_lambda:.0e}]' for l1_lambda in l1_lambdas]
    
    for l1, l1_col_name in zip(l1_lambdas, l1_names):# in range(7):
        for i in range(0,5):
            ind_pair = cv_inds[i]
            #decide which pair to hold out as validation this time 
            df_slice = train_df.iloc[ind_pair[0]]
            df_slice_valid = train_df.iloc[ind_pair[1]]

            train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
            train_y = df_slice.binned.to_numpy()
            weights_train = 1.1 - df_slice.ashr_padj.to_numpy()
            if use_oversample:
                binned_y = df_slice.binned.to_numpy()
                to_sample = train_x[binned_y == -1].shape[0]
                X_oversampled, y_oversampled, weights_oversampled = resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
                train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
                train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
                weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled))

            val_x = df_slice_valid.drop(columns = new_not_model_cols).to_numpy()
            val_y = df_slice_valid.binned.to_numpy()
            scaler_x = preprocessing.StandardScaler().fit(train_x)
            x_train_scaled = scaler_x.transform(train_x)
            x_val_scaled = scaler_x.transform(val_x)
            #x_test_scaled = scaler_x.transform(test_x)


            
            if train_x.shape[1] <= 4:
                model = linear_model.LogisticRegression(penalty = 'l2', C=1/l1, solver = 'liblinear', max_iter=1000)
            else:
                model = linear_model.LogisticRegression(penalty = 'l2', C=1/l1, solver = 'liblinear', max_iter=1000)
            if weights:
                model.fit(x_train_scaled, train_y, sample_weight = weights_train)
            else:   
                model.fit(x_train_scaled, train_y)
            data.append({
                'val_pair': i,
                'l1_penalty': l1,
                'validation_mcc': matthews_corrcoef(val_y == 1, model.predict(x_val_scaled)),
                'n_non_zero': sum(model.coef_ [0]!= 0)
            })
    
    lasso_data = pd.DataFrame(data)
    return lasso_data


def padj_train_best_lasso_eval_on_test_classification(best_l1, large_dataset_padj_order, new_not_model_cols, save_name, use_oversample = False, weights = False):

    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.binned.to_numpy()
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
    test_y = df_test.binned.to_numpy()

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    x_test_scaled = scaler_x_full.transform(test_x)

    if full_train.shape[1] <= 4:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    else:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    if weights:
        clf.fit(x_train_scaled, full_lfc, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, full_lfc)
    
    predict_test = clf.predict_proba(x_test_scaled)[:,1]
    #print (predict_test)
    df_test['preds'] = predict_test
    avgs_test = df_test.groupby(['ppi']).mean()

    rocauc_train = roc_auc_score(full_lfc.flatten(), clf.predict_proba(x_train_scaled)[:,1])
    rocauc_test = roc_auc_score(avgs_test['binned'], avgs_test['preds'])
    avgpr_train = average_precision_score(full_lfc.flatten(), clf.predict_proba(x_train_scaled)[:,1])
    avgpr_test = average_precision_score(avgs_test['binned'], avgs_test['preds'])
    mccf1 = get_mcc_f1_from_msrmts(avgs_test['binned'] == 1, avgs_test['preds'])[0]
    print (save_name, mccf1)
    return   rocauc_train, rocauc_test, avgpr_train, avgpr_test, mccf1

def open_features_dict(version):
    if version == 'v2':
        with open('v2_correl_features.pickle', 'rb') as handle:
            v2_features = pickle.load(handle)
        return v2_features
    else:
        with open('v3_correl_features.pickle', 'rb') as handle:
            v2_features = pickle.load(handle)
        return v2_features

#apply to other test sets

def padj_get_model_classification(dataset, r2, binned, weights):
    if dataset != 'all':
        ds, not_model_cols =load_dataset_single(dataset, r2)
    else:
        ds, not_model_cols = load_dataset_single(dataset, r2, endings = ['_v1', '_v1512', '_v2', '_v3'])
        ds['id1'] = ds.id1_v1.apply(lambda x: x.replace('_1', ''))
        ds['id2'] = ds.id2_v1.apply(lambda x: x.replace('_1', ''))
        not_model_cols.append('id1')
        not_model_cols.append('id2')
    if binned and not weights:
        best_l1_df = pd.read_csv('padj_classification_oversample_no_weights_ridge.csv')
        #dataset,r2_cutoff,l1_penalty
        best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == str(r2))].l1_penalty.values[0]
        print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    elif binned and weights:
        best_l1_df = pd.read_csv('padj_classification_oversample_yes_weights_ridge.csv')
        #dataset,r2_cutoff,l1_penalty
        best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)].l1_penalty.values[0] 
        print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    else:
        best_l1_df = pd.read_csv('padj_classifcation_ridge.csv')
        #dataset,r2_cutoff,l1_penalty
        best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)].l1_penalty.values[0] 
        print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    #print ('best l1: ', best_l1)
    return padj_train_model_classification(best_l1, ds, not_model_cols, use_oversample = binned, weights = weights)


def padj_train_model_classification(best_l1, large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):
    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.binned.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1.1 - df_slice.ashr_padj.to_numpy()
    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], weights_oversampled))
    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    if full_train.shape[1] <= 4:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    else:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    if weights:
        clf.fit(x_train_scaled, full_lfc, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, full_lfc)
    
    return clf, list(df_slice.drop(columns = new_not_model_cols).columns), scaler_x_full


def holdout_get_model_classification(dataset, r2, test_pair, binned, weights, max_or_min):
    if dataset != 'all':
        ds, not_model_cols =load_dataset_single(dataset, r2)
    else:
        ds, not_model_cols = load_dataset_single(dataset, r2, endings = ['_v1', '_v1512', '_v2', '_v3'])
        ds['id1'] = ds.id1_v1.apply(lambda x: x.replace('_1', ''))
        ds['id2'] = ds.id2_v1.apply(lambda x: x.replace('_1', ''))
        not_model_cols.append('id1')
        not_model_cols.append('id2')
    if binned and not weights:
        best_l1_df = pd.read_csv('final_results_' + max_or_min + 'holdout_classification_oversample_no_weights_ridge.csv')
        #dataset,r2_cutoff,l1_penalty
        best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)].l1_penalty.values[0]
        print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    elif binned and weights:
        best_l1_df = pd.read_csv('final_results_' + max_or_min + 'holdout_classification_oversample_yes_weights_ridge.csv')
        #dataset,r2_cutoff,l1_penalty
        best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)].l1_penalty.values[0] 
        print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    else:
        best_l1_df = pd.read_csv('final_results_maxholdout_classification_lasso.csv')
        #dataset,r2_cutoff,l1_penalty
        best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)].l1_penalty.values[0] 
        print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    return holdout_train_model_classification(best_l1, ds, not_model_cols, test_pair, use_oversample = binned, weights = weights)


def holdout_train_model_classification(best_l1, large_dataset_padj_order, new_not_model_cols, test_pros, use_oversample = False, weights = False):

    #test_pros = [x for p in test_pairs for x in p]
    df_slice = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pros)) & ~(large_dataset_padj_order.id2.isin(test_pros))]
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.binned.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1.1 - df_slice.ashr_padj.to_numpy()
    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], weights_oversampled))

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    if full_train.shape[1] <= 4:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    else:
        clf = linear_model.LogisticRegression(penalty = 'l2', C=1/best_l1, solver = 'liblinear', max_iter=1000)
    if weights:
        clf.fit(x_train_scaled, full_lfc, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, full_lfc)
    
    return clf, list(df_slice.drop(columns = new_not_model_cols).columns), scaler_x_full

        

