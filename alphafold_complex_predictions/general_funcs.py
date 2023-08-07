#general modeling functions 
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
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
pd.options.mode.chained_assignment = None  
import os
import itertools
from sklearn.metrics import mean_squared_error, matthews_corrcoef, roc_auc_score, f1_score, average_precision_score

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


def label_LFC_bins(x, sort_level = 5):
    if x >= sort_level:
        return 1
    else:
        return -1

def get_best_l2_holdout(lasso_data, asc = True, group_col = 'l2_penalty', sort_col = 'validation_rmse'):
    lasso_data = lasso_data.drop(columns = 'val_pair') #drop to not throw off mean 
    results = lasso_data.groupby([group_col]).mean()
    best_l1 = results.sort_values(sort_col, ascending = asc).index[0]
    return best_l1


def merge_vals(v2_preds, set = 'ncip', size_test = 'small'):
    if set == 'ncip':
        mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_autotune_5_small.csv')
    elif set == 'malb' or set == 'combo':
        mp3_seq_values = pd.read_csv('../processing_pipeline/merged_replicates/deseq_new_smaller_flat_autotune.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})
    mp3_seq_values['P1'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[1])
    mp3_seq_values['P2'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[3])
    mp3_seq_values['id1'] = mp3_seq_values.P1.apply(lambda x: x.replace('Jerala_', '')) 
    mp3_seq_values['id2'] = mp3_seq_values.P2.apply(lambda x: x.replace('Jerala_', '')) 
    mp3_seq_values['ppi'] = mp3_seq_values.id1 + '_' + mp3_seq_values.id2
    #filter to only important proteins
    if set == 'ncip':
        mp3_seq_values = mp3_seq_values[(mp3_seq_values.id1.isin(ncip_pros)) & (mp3_seq_values.id2.isin(ncip_pros))].copy()
    elif set == 'malb':
        mp3_seq_values = mp3_seq_values[(mp3_seq_values.id1.isin(malb_list)) & (mp3_seq_values.id2.isin(malb_list))].copy()
    elif set == 'combo': #ncip and malb
        mp3_seq_values = mp3_seq_values[(mp3_seq_values.id1.isin(malb_list + ncip_pros)) & (mp3_seq_values.id2.isin(malb_list + ncip_pros))].copy()
    
    if size_test == 'small':
        test_train_split = ['test', 'train'] * int((0.1 * mp3_seq_values.shape[0]))
        #print (len(test_train_split))
        test_train_split = test_train_split + ['train'] * int(mp3_seq_values.shape[0] - len(test_train_split))
    else:
        #size I want is in size_test
        test_train_split = ['test', 'train'] *size_test
        #print (len(test_train_split))
        test_train_split = test_train_split + ['train'] * int(mp3_seq_values.shape[0] - len(test_train_split))

    mp3_seq_values = mp3_seq_values.sort_values('ashr_padj_HIS_TRP')
    mp3_seq_values['order_padj_sets'] = test_train_split
    print (mp3_seq_values.order_padj_sets.value_counts())

    mp3_seq_values_ppis = mp3_seq_values.ppi.to_list()
    mp3_seq_lfcs = mp3_seq_values.ashr_log2FoldChange_HIS_TRP.to_list()
    mp3_seq_lfcSEs = mp3_seq_values.ashr_lfcSE_HIS_TRP.to_list()
    ashr_padj = mp3_seq_values.ashr_padj_HIS_TRP.fillna(1).to_list()
    test_train = mp3_seq_values.order_padj_sets.to_list()
    ppis_dummy = []

    for i in range(1,6):
        ppis_dummy =ppis_dummy + [x + '_' + str(i) for x in mp3_seq_values_ppis]

    correl_df = pd.DataFrame({'new_ppi': ppis_dummy, 'lfc_all': mp3_seq_lfcs * 5, 'lfcsSE_all': mp3_seq_lfcSEs * 5, 'ashr_padj':ashr_padj * 5, 'order_padj_sets': test_train * 5})
    correl_df_merge = v2_preds.merge(correl_df, on = 'new_ppi', how = 'inner')
  
    new_not_model_cols = ['id1', 
                          'id2', 
                          'model', 
                          'type', 
                          'msa_depth',
                          'ppi', 
                          'on_target',
                          'new_ppi',
                          'lfc_all',
                          'lfcsSE_all',
                          'ashr_padj',
                          'order_padj_sets'] 

    return correl_df_merge, new_not_model_cols


def load_dataset_single(df_set, r, set_name, size_test = 'small'):
    v2_preds = pd.read_csv('./datasets/' + df_set + '_correl_reduced_r_' + str(r) +'.csv') 
    v2_preds.dropna(axis = 0, inplace = True) #drop any failed simulation rows 
    correl_df_merge, new_not_model_cols = merge_vals(v2_preds, set_name,size_test)
    large_dataset = correl_df_merge
    large_dataset['binned'] = large_dataset.lfc_all.apply(lambda x: label_LFC_bins(x)) #label the data to aid with cross validation
    new_not_model_cols = new_not_model_cols + ['binned']
    return large_dataset, new_not_model_cols



def run_holdout_cv(large_dataset_padj_order, new_not_model_cols, test_pairs, train_pairs, use_oversample = False, weights = False):
  
    test_pros = [x for p in test_pairs for x in p]
    train_df = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pros)) & ~(large_dataset_padj_order.id2.isin(test_pros))]
    print ('test pros: ', test_pros, 'train size: ', train_df.shape, 'total size:', large_dataset_padj_order.shape)
    
    sk_split = StratifiedKFold(random_state= 1, shuffle = True)
    cv_inds = list(sk_split.split(X = np.zeros(train_df.shape[0]), y = train_df.binned.to_numpy()))
    
    l1_lambdas = np.logspace(-5, 5, 51, base=10)
    data = []
    for l1 in l1_lambdas:# in range(7):
        for i in range(0,len(cv_inds)):
            ind_pair = cv_inds[i]
            df_slice = train_df.iloc[ind_pair[0]]
            df_slice_valid = train_df.iloc[ind_pair[1]]
            train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
            train_y = df_slice.lfc_all.to_numpy()
            weights_train = 1 - df_slice.ashr_padj.to_numpy()
            if use_oversample:
                binned_y = df_slice.binned.to_numpy()
                to_sample = train_x[binned_y == -1].shape[0]
                X_oversampled, y_oversampled, weights_oversampled= resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
                train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
                train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
                weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled)) 

            val_x = df_slice_valid.drop(columns = new_not_model_cols).to_numpy()
            val_y = df_slice_valid.lfc_all.to_numpy()
            scaler_x = preprocessing.StandardScaler().fit(train_x)
            x_train_scaled = scaler_x.transform(train_x)
            x_val_scaled = scaler_x.transform(val_x)
            #x_test_scaled = scaler_x.transform(test_x)
            scaler_y = preprocessing.StandardScaler().fit(train_y.reshape(-1, 1))
            y_train_scaled = scaler_y.transform(train_y.reshape(-1, 1))
            y_val_scaled = scaler_y.transform(val_y.reshape(-1, 1))
            
            model = linear_model.Ridge(alpha=l1, max_iter= 20000)

            if weights:
                model.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
            else:
                model.fit(x_train_scaled, y_train_scaled)
            train_rmse = sqrt(mean_squared_error(y_train_scaled, model.predict(x_train_scaled)))
            validation_rmse = sqrt(mean_squared_error(y_val_scaled, model.predict(x_val_scaled)))
            data.append({
                'val_pair': i,
                'l2_penalty': l1,
                'train_rmse': train_rmse,
                'validation_rmse': validation_rmse,
                'train_r2': model.score(x_train_scaled, y_train_scaled),
                'validation_r2': model.score(x_val_scaled, y_val_scaled),
                'n_non_zero': sum(np.array(model.coef_).flatten() != 0)
            })
    
    lasso_data = pd.DataFrame(data)
    return lasso_data


def padj_cv_regression(set_name,  binned = False, weights = False, pool = True,size_test='small',):
    datasets = [ 'mono', 'v1_512', 'v2', 'v3']
    if set_name == 'malb' or set_name == 'combo':
        datasets= ['malb_v2', 'malb_v3' ]
    r2_cutoffs =['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'af']
    runs = []
    for dataset in datasets:
        for r2 in r2_cutoffs:
            if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
                runs.append((dataset, r2,set_name,size_test,  binned, weights))
    #now multithread 
    data = []
    if pool:
        with Pool() as pool:
            data = pool.starmap(padj_run_one_model_combo, runs)
    else:
        for r in runs:
            data.append(padj_run_one_model_combo(r[0], r[1], r[2], r[3], r[4],r[5]))
    return pd.DataFrame(data)




def padj_run_one_model_combo(dataset, r2, set_name, size_test, binned, weights):
    if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
       
        ds, not_model_cols =load_dataset_single(dataset, r2, set_name, size_test)
        results_df = padj_run_cv(ds, not_model_cols, use_oversample=binned, weights = weights)
        best_l1 = get_best_l2_holdout(results_df)
        sr_train_not_avgd, pr_train_not_avgd, sr_train, pr_train, sr_test_not_avgd, pr_test_not_avgd, sr_test, pr_test, num_nonzero = padj_train_best_ridge_eval_on_test(best_l1, ds, not_model_cols, use_oversample = binned, weights = weights)
        
        return {
            'dataset': dataset,
            'r2_cutoff':r2,
            'l2_penalty': best_l1,
            'train_r2_not_avgd':pr_train_not_avgd**2,
            'train_r2': pr_train**2,
            'train_spearman_not_avgd': sr_train_not_avgd,
            'train_spearman': sr_train,
            'test_r2_not_avgd':pr_test_not_avgd**2,
            'test_r2': pr_test**2,
            'test_spearman_not_avgd': sr_test_not_avgd,
            'test_spearman':sr_test,
            'n_non_zero': num_nonzero
        }


def padj_run_cv(large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):
  
    train_df = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy() 
    sk_split = StratifiedKFold(random_state= 1, shuffle = True)
    cv_inds = list(sk_split.split(X = np.zeros(train_df.shape[0]), y = train_df.binned.to_numpy()))
    
    l1_lambdas = np.logspace(-5, 5, 51, base=10)
    data = []
    for l1 in l1_lambdas:
        for i in range(0,5):
            ind_pair = cv_inds[i]
            df_slice = train_df.iloc[ind_pair[0]]
            df_slice_valid = train_df.iloc[ind_pair[1]]

            train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
            train_y = df_slice.lfc_all.to_numpy()
            weights_train = 1 - df_slice.ashr_padj.to_numpy()
            if use_oversample:
                binned_y = df_slice.binned.to_numpy()
                to_sample = train_x[binned_y == -1].shape[0]
                X_oversampled, y_oversampled, weights_oversampled = resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
                train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
                train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
                weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled))

            val_x = df_slice_valid.drop(columns = new_not_model_cols).to_numpy()
            val_y = df_slice_valid.lfc_all.to_numpy()
            scaler_x = preprocessing.StandardScaler().fit(train_x)
            x_train_scaled = scaler_x.transform(train_x)
            x_val_scaled = scaler_x.transform(val_x)
            scaler_y = preprocessing.StandardScaler().fit(train_y.reshape(-1, 1))
            y_train_scaled = scaler_y.transform(train_y.reshape(-1, 1))
            y_val_scaled = scaler_y.transform(val_y.reshape(-1, 1))
           
            model = linear_model.Ridge(alpha=l1, max_iter= 20000)
            if weights:
                model.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
            else:   
                model.fit(x_train_scaled, y_train_scaled)
            train_rmse = sqrt(mean_squared_error(y_train_scaled, model.predict(x_train_scaled)))
            validation_rmse = sqrt(mean_squared_error(y_val_scaled, model.predict(x_val_scaled)))
            data.append({
                'val_pair': i,
                'l2_penalty': l1,
                'train_rmse': train_rmse,
                'validation_rmse': validation_rmse,
            })
    
    lasso_data = pd.DataFrame(data)
    return lasso_data



def padj_train_best_ridge_eval_on_test(best_l1, large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):

    df_test = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'test'].copy()
    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.lfc_all.to_numpy()

    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.lfc_all.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1 - df_slice.ashr_padj.to_numpy()

    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], weights_oversampled))

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    x_test_scaled = scaler_x_full.transform(test_x)

    scaler_y_full = preprocessing.StandardScaler().fit(full_lfc.reshape(-1, 1))
    y_train_scaled = scaler_y_full.transform(full_lfc.reshape(-1, 1))
    y_test_scaled = scaler_y_full.transform(test_y.reshape(-1, 1))

    clf = linear_model.Ridge(alpha=best_l1, max_iter= 20000)

    if weights:
        clf.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, y_train_scaled)
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.lfc_all.to_numpy()
    x_train_scaled = scaler_x_full.transform(full_train)
    y_train_scaled = scaler_y_full.transform(full_lfc.reshape(-1, 1))
    num_nonzero = sum(np.array(clf.coef_).flatten() != 0)

    sr_train_not_avgd = spearmanr(y_train_scaled, clf.predict(x_train_scaled))[0]
    pr_train_not_avgd = pearsonr(y_train_scaled.flatten(), clf.predict(x_train_scaled).flatten())[0]
    x_preds = clf.predict(x_train_scaled)
    df_slice['preds'] = x_preds
    df_slice['scaled_y'] = y_train_scaled
    avgs_train = df_slice.groupby(['ppi']).mean()

    sr_train = spearmanr(avgs_train['scaled_y'], avgs_train['preds'])[0]
    pr_train = pearsonr(avgs_train['scaled_y'], avgs_train['preds'])[0]
    
    sr_test_not_avgd = spearmanr(y_test_scaled, clf.predict(x_test_scaled))[0]
    pr_test_not_avgd = pearsonr(y_test_scaled.flatten(), clf.predict(x_test_scaled).flatten())[0]

    val_preds = clf.predict(x_test_scaled)
    df_test['preds'] = val_preds
    df_test['scaled_y'] = y_test_scaled
    avgs_test = df_test.groupby(['ppi']).mean()

    sr_test = spearmanr(avgs_test['scaled_y'], avgs_test['preds'])[0]
    pr_test = pearsonr(avgs_test['scaled_y'], avgs_test['preds'])[0]

    return sr_train_not_avgd, pr_train_not_avgd, sr_train, pr_train, sr_test_not_avgd, pr_test_not_avgd, sr_test, pr_test, num_nonzero



def padj_get_model_regression(dataset, r2, set_name, results_frame, binned, weights,size_test='small',):
    
    ds, not_model_cols =load_dataset_single(dataset, r2, set_name,size_test)
    best_l1_df = pd.read_csv(results_frame)
    best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)].l2_penalty.values[0]
    print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    return padj_train_model_regression(best_l1, ds, not_model_cols, use_oversample = binned, weights = weights)


def padj_train_model_regression(best_l1, large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):

    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.lfc_all.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1 - df_slice.ashr_padj.to_numpy()

    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], weights_oversampled))

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)

    scaler_y_full = preprocessing.StandardScaler().fit(full_lfc.reshape(-1, 1))
    y_train_scaled = scaler_y_full.transform(full_lfc.reshape(-1, 1))

    clf = linear_model.Ridge(alpha=best_l1, max_iter= 20000)

    if weights:
        clf.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, y_train_scaled)
      
    return clf, scaler_x_full, scaler_y_full, list( df_slice.drop(columns = new_not_model_cols).columns)


def padj_get_model_classif(dataset, r2, set_name, results_frame, binned, weights,size_test='small',):
    
    ds, not_model_cols =load_dataset_single(dataset, r2, set_name,size_test)
    best_l1_df = pd.read_csv(results_frame)
    best_l1 = best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)].l2_penalty.values[0]
    print ('best l1: ', best_l1_df[(best_l1_df.dataset == dataset) & (best_l1_df.r2_cutoff == r2)], best_l1)
    return padj_train_model_classif(best_l1, ds, not_model_cols, use_oversample = binned, weights = weights)


def padj_train_model_classif(l2, large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):

    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
    df_test = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'test'].copy()

    train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
    train_y = df_slice.binned.to_numpy()
    weights_train = 1 - df_slice.ashr_padj.to_numpy()
    if use_oversample:
        binned_y = df_slice.binned.to_numpy()
        to_sample = train_x[binned_y == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
        train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
        train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
        weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled))

    scaler_x = preprocessing.StandardScaler().fit(train_x)
    x_train_scaled = scaler_x.transform(train_x)

    model = linear_model.LogisticRegression(penalty = 'l2', C=1/l2, solver = 'liblinear', max_iter=2000)

    if weights:
        model.fit(x_train_scaled, train_y, sample_weight = weights_train)
    else:   
        model.fit(x_train_scaled, train_y)

    return   model, scaler_x, list( df_slice.drop(columns = new_not_model_cols).columns)



malb_list = [ 'mALb8_A',  'mALb8_cutT1_A', 'mALb8_cutT2_A', '1002_mALb8x1_rprtc_A',
    '1001_mALb8x1_fdrtc_A',
    '1004_mALb8x2_rprtc_A',
 '1003_mALb8x2_fdrtc_A',
 '1006_mALb8x12_rprtc_A',
 '1005_mALb8x12_fdrtc_A',
 '1008_mALb8x12j_rprtc_A',
  '1007_mALb8x12j_fdrtc_A',
 'mALb8_B', 
 '1002_mALb8x1_rprtc_B',
 '1001_mALb8x1_fdrtc_B',
 '1004_mALb8x2_rprtc_B',
 '1003_mALb8x2_fdrtc_B',
 '1006_mALb8x12_rprtc_B',
 '1005_mALb8x12_fdrtc_B',
  '1008_mALb8x12j_rprtc_B',
  '1007_mALb8x12j_fdrtc_B',
   'mALb8_cutT1_B','mALb8_cutT2_B',]
#holdout protein training 
ncip_pros = ['P' + str(i) for i in list(range(1,13))]

#filter malb test sets - some don't have any positive class if selected as only protien 

def get_sample_pairs(hold_size, num_samples, pro_list):
    if hold_size == 1:
        if num_samples > len(pro_list):
            num_samples = len(pro_list)
            print ('forcing unique proteins to sample to length of possible proteins', len(pro_list))
        if pro_list == malb_list:
            bad = ['1007_mALb8x12j_fdrtc_A','mALb8_cutT2_B']
            malb_list_clean = list(set(malb_list) - set(bad))
            pro_list = malb_list_clean
            if num_samples > len(pro_list):
                num_samples = len(pro_list)
        return [[pro_list[x]] for x in np.random.choice(len(pro_list), num_samples, replace = False)]
    else:
        possible_combinations = list(itertools.combinations(pro_list, hold_size))
        return [possible_combinations[x] for x in np.random.choice(len(possible_combinations), num_samples, replace = False)]

def holdout_cv_regression(set_name, hold_size, num_samples = 0, binned = False, weights = False, pool = True):
    #set up heldout protein sets 
    if set_name == 'ncip':
        datasets = [ 'mono', 'v1_512', 'v2', 'v3' ]
        if hold_size == 1:
            pairs = get_sample_pairs(hold_size, 12, ncip_pros)
        elif hold_size == 2: #only find the designed 
            pairs = []
            for i in range(1,13,2):
                pairs.append(('P' + str(i), 'P' + str(i+1)))
    elif set_name == 'malb':
        datasets= ['malb_v2', 'malb_v3' ]
        #sample proteins and num to sample 
        pairs = get_sample_pairs(hold_size, num_samples, malb_list)
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
                    runs.append((dataset, r2, test_pair, set_name, 'small', binned, weights))
    #now multithread
    if pool: 
        with Pool() as pool:
            data = pool.starmap(run_one_model_combo_holdout, runs)
    else:
        data = []
        for r in runs:
            data.append(run_one_model_combo_holdout(r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    return pd.DataFrame(data).dropna()


def run_one_model_combo_holdout(dataset, r2, test_pair, set_name, size_test, binned, weights):
    if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
        ds, not_model_cols =load_dataset_single(dataset, r2, set_name, size_test)
        results_df = run_holdout_cv(ds, not_model_cols, test_pair, use_oversample=binned, weights= weights)
        best_l2 = get_best_l2_holdout(results_df)
        sr_train_not_avgd, pr_train_not_avgd, sr_train, pr_train, sr_test_not_avgd, pr_test_not_avgd, sr_test, pr_test, num_nonzero = train_best_ridge_eval_on_test_holdout(best_l2, ds, not_model_cols, test_pair, use_oversample = binned, weights= weights)
        return {
            'dataset': dataset,
            'test_pair': '|'.join(test_pair),
            'r2_cutoff':r2,
            'l2_penalty': best_l2,
            'train_r2_not_avgd':pr_train_not_avgd**2,
            'train_r2': pr_train**2,
            'train_spearman_not_avgd': sr_train_not_avgd,
            'train_spearman': sr_train,
            'test_r2_not_avgd':pr_test_not_avgd**2,
            'test_r2': pr_test**2,
            'test_spearman_not_avgd': sr_test_not_avgd,
            'test_spearman':sr_test,
            'n_non_zero': num_nonzero
        }
   

def run_holdout_cv(large_dataset_padj_order, new_not_model_cols, test_pairs, use_oversample = False, weights = False):
  
    #test_pros = [x for p in test_pairs for x in p]
    train_df = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pairs)) & ~(large_dataset_padj_order.id2.isin(test_pairs))]
    print ('test pros: ', test_pairs, 'train size: ', train_df.shape, 'total size:', large_dataset_padj_order.shape)
    
    sk_split = StratifiedKFold(random_state= 1, shuffle = True)
    cv_inds = list(sk_split.split(X = np.zeros(train_df.shape[0]), y = train_df.binned.to_numpy()))
    
    l1_lambdas = np.logspace(-5, 5, 51, base=10)
    data = []

    for l1 in l1_lambdas:# in range(7):
        for i in range(0,len(cv_inds)):
            ind_pair = cv_inds[i]
            df_slice = train_df.iloc[ind_pair[0]]
            df_slice_valid = train_df.iloc[ind_pair[1]]
            train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
            train_y = df_slice.lfc_all.to_numpy()
            weights_train = 1 - df_slice.ashr_padj.to_numpy()
            if use_oversample:
                binned_y = df_slice.binned.to_numpy()
                to_sample = train_x[binned_y == -1].shape[0]
                X_oversampled, y_oversampled, weights_oversampled= resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
                train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
                train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
                weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled)) 

            val_x = df_slice_valid.drop(columns = new_not_model_cols).to_numpy()
            val_y = df_slice_valid.lfc_all.to_numpy()

            scaler_x = preprocessing.StandardScaler().fit(train_x)
            x_train_scaled = scaler_x.transform(train_x)
            x_val_scaled = scaler_x.transform(val_x)

            scaler_y = preprocessing.StandardScaler().fit(train_y.reshape(-1, 1))
            y_train_scaled = scaler_y.transform(train_y.reshape(-1, 1))
            y_val_scaled = scaler_y.transform(val_y.reshape(-1, 1))
           
            model = linear_model.Ridge(alpha=l1, max_iter= 20000)

            if weights:
                model.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
            else:
                model.fit(x_train_scaled, y_train_scaled)
            train_rmse = sqrt(mean_squared_error(y_train_scaled, model.predict(x_train_scaled)))
            validation_rmse = sqrt(mean_squared_error(y_val_scaled, model.predict(x_val_scaled)))
            data.append({
                'val_pair': i,
                'l2_penalty': l1,
                'train_rmse': train_rmse,
                'validation_rmse': validation_rmse,
                'n_non_zero': sum(np.array(model.coef_).flatten() != 0)
            })
    
    lasso_data = pd.DataFrame(data)
    return lasso_data



def train_best_ridge_eval_on_test_holdout(best_l1, large_dataset_padj_order, new_not_model_cols, test_pros, use_oversample = False, weights = False):

    #test_pros = [x for p in test_pairs for x in p]
    df_slice = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pros)) & ~(large_dataset_padj_order.id2.isin(test_pros))]
  
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.lfc_all.to_numpy()
    full_bins = df_slice.binned.to_numpy()
    weights_train = 1 - df_slice.ashr_padj.to_numpy()

    if use_oversample: 
        to_sample = full_train[full_bins == -1].shape[0]
        X_oversampled, y_oversampled, w_oversampled = resample(full_train[full_bins == 1], full_lfc[full_bins == 1], weights_train[full_bins == 1], replace=True, n_samples=to_sample,random_state = 1)           
        full_train = np.vstack((full_train[full_bins == -1], X_oversampled))
        full_lfc = np.hstack((full_lfc[full_bins == -1], y_oversampled))
        weights_train = np.hstack((weights_train[full_bins == -1], w_oversampled))

    df_test = large_dataset_padj_order[(large_dataset_padj_order.id1.isin(test_pros)) | (large_dataset_padj_order.id2.isin(test_pros))]
    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.lfc_all.to_numpy()

    scaler_x_full = preprocessing.StandardScaler().fit(full_train)
    x_train_scaled = scaler_x_full.transform(full_train)
    x_test_scaled = scaler_x_full.transform(test_x)

    scaler_y_full = preprocessing.StandardScaler().fit(full_lfc.reshape(-1, 1))
    y_train_scaled = scaler_y_full.transform(full_lfc.reshape(-1, 1))
    y_test_scaled = scaler_y_full.transform(test_y.reshape(-1, 1))
    
    clf = linear_model.Ridge(alpha=best_l1, max_iter= 20000)
    
    if weights:
        clf.fit(x_train_scaled, y_train_scaled, sample_weight = weights_train)
    else:
        clf.fit(x_train_scaled, y_train_scaled)
    
    full_train = df_slice.drop(columns = new_not_model_cols).to_numpy()
    full_lfc = df_slice.lfc_all.to_numpy()
    x_train_scaled = scaler_x_full.transform(full_train)
    y_train_scaled = scaler_y_full.transform(full_lfc.reshape(-1, 1))
    num_nonzero = sum(np.array(clf.coef_).flatten() != 0)

    sr_train_not_avgd = spearmanr(y_train_scaled, clf.predict(x_train_scaled))[0]
    pr_train_not_avgd = pearsonr(y_train_scaled.flatten(), clf.predict(x_train_scaled).flatten())[0]
    x_preds = clf.predict(x_train_scaled)
    df_slice['preds'] = x_preds
    df_slice['scaled_y'] = y_train_scaled
    avgs_train = df_slice.groupby(['ppi']).mean()

    sr_train = spearmanr(avgs_train['scaled_y'], avgs_train['preds'])[0]
    pr_train = pearsonr(avgs_train['scaled_y'], avgs_train['preds'])[0]
    
    sr_test_not_avgd = spearmanr(y_test_scaled, clf.predict(x_test_scaled))[0]
    pr_test_not_avgd = pearsonr(y_test_scaled.flatten(), clf.predict(x_test_scaled).flatten())[0]

    val_preds = clf.predict(x_test_scaled)
    df_test['preds'] = val_preds
    df_test['scaled_y'] = y_test_scaled
    avgs_test = df_test.groupby(['ppi']).mean()

    sr_test = spearmanr(avgs_test['scaled_y'], avgs_test['preds'])[0]
    pr_test = pearsonr(avgs_test['scaled_y'], avgs_test['preds'])[0]

    return sr_train_not_avgd, pr_train_not_avgd, sr_train, pr_train, sr_test_not_avgd, pr_test_not_avgd, sr_test, pr_test, num_nonzero




def padj_cv_classification(set_name,  binned = False, weights = False, pool = True,size_test='small',):
    datasets = [ 'mono', 'v1_512', 'v2', 'v3']
    if set_name == 'malb' or set_name == 'combo':
        datasets= ['malb_v2', 'malb_v3' ]
    r2_cutoffs = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'af']
    runs = []
    for dataset in datasets:
        for r2 in r2_cutoffs:
            if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
                runs.append((dataset, r2,set_name,size_test,  binned, weights))
    data = []
    if pool:
        with Pool() as pool:
            data = pool.starmap(padj_run_one_model_combo_classification, runs)
    else:
        for r in runs:
            data.append(padj_run_one_model_combo_classification(r[0], r[1], r[2], r[3], r[4],r[5]))
    return pd.DataFrame(data)

def padj_run_one_model_combo_classification(dataset, r2, set_name, size_test, binned, weights):
    if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
       
        ds, not_model_cols =load_dataset_single(dataset, r2, set_name, size_test)
        results_df = padj_run_cv_classification(ds, not_model_cols, use_oversample=binned, weights = weights)
        best_l2 = get_best_l2_holdout(results_df, asc = False, group_col= 'l2_penalty', sort_col = 'validation_mcc')
        rocauc_train_not_avgd, avgpr_train_not_avgd, rocauc_train, avgpr_train, rocauc_test_not_avgd, avgpr_test_not_avgd, rocauc_test, avgpr_test, num_nonzero = padj_train_best_classif_eval_on_test(best_l2, ds, not_model_cols, use_oversample = binned, weights = weights)
        
        return {
            'dataset': dataset,
            'r2_cutoff':r2,
            'l2_penalty': best_l2,
            'rocauc_train_not_avgd':rocauc_train_not_avgd,
            'rocauc_train': rocauc_train,
            'avgpr_train_not_avgd': avgpr_train_not_avgd,
            'avgpr_train': avgpr_train,
            'rocauc_test_not_avgd':rocauc_test_not_avgd,
            'rocauc_test': rocauc_test,
            'avgpr_test_not_avgd': avgpr_test_not_avgd,
            'avgpr_test':avgpr_test,
            'n_non_zero': num_nonzero
        }


def padj_run_cv_classification(large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):
  
    train_df = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
    print (train_df.shape, large_dataset_padj_order.shape)
    
    sk_split = StratifiedKFold(random_state= 1, shuffle = True)
    cv_inds = list(sk_split.split(X = np.zeros(train_df.shape[0]), y = train_df.binned.to_numpy()))
    
    l1_lambdas = np.logspace(-5, 5, 51, base=10)
    data = []

    for l2 in l1_lambdas:# in range(7):
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
            
            model = linear_model.LogisticRegression(penalty = 'l2', C=1/l2, solver = 'liblinear', max_iter=2000)

            if weights:
                model.fit(x_train_scaled, train_y, sample_weight = weights_train)
            else:   
                model.fit(x_train_scaled, train_y)
            data.append({
                'val_pair': i,
                'l2_penalty': l2,
                'train_mcc':matthews_corrcoef(train_y == 1, model.predict(x_train_scaled)),
                'validation_mcc': matthews_corrcoef(val_y == 1, model.predict(x_val_scaled)),
                'n_non_zero': sum(np.array(model.coef_).flatten() != 0)
            })
    
    lasso_data = pd.DataFrame(data)
    return lasso_data


def padj_train_best_classif_eval_on_test(l2, large_dataset_padj_order, new_not_model_cols, use_oversample = False, weights = False):

    df_slice = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'train'].copy()
    df_test = large_dataset_padj_order[large_dataset_padj_order.order_padj_sets == 'test'].copy()

    train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
    train_y = df_slice.binned.to_numpy()
    weights_train = 1 - df_slice.ashr_padj.to_numpy()
    if use_oversample:
        binned_y = df_slice.binned.to_numpy()
        to_sample = train_x[binned_y == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
        train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
        train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
        weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled))

    scaler_x = preprocessing.StandardScaler().fit(train_x)
    x_train_scaled = scaler_x.transform(train_x)

    model = linear_model.LogisticRegression(penalty = 'l2', C=1/l2, solver = 'liblinear', max_iter=2000)

    if weights:
        model.fit(x_train_scaled, train_y, sample_weight = weights_train)
    else:   
        model.fit(x_train_scaled, train_y)

    num_nonzero = sum(np.array(model.coef_).flatten() != 0)
    train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
    train_y = df_slice.binned.to_numpy()
    x_train_scaled = scaler_x.transform(train_x)
    rocauc_train_not_avgd = roc_auc_score(train_y.flatten(), model.predict_proba(x_train_scaled)[:,1])
    avgpr_train_not_avgd = average_precision_score(train_y.flatten(), model.predict_proba(x_train_scaled)[:,1])
    df_slice['pred_probs'] = model.predict_proba(x_train_scaled)[:,1]

    avgs_train = df_slice.groupby(['ppi']).mean()
    rocauc_train = roc_auc_score(avgs_train['binned'], avgs_train['pred_probs'])
    avgpr_train = average_precision_score(avgs_train['binned'], avgs_train['pred_probs'])

    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.binned.to_numpy()
    x_test_scaled = scaler_x.transform(test_x)
    predict_test = model.predict_proba(x_test_scaled)[:,1]
    rocauc_test_not_avgd = roc_auc_score(test_y.flatten(), predict_test)
    avgpr_test_not_avgd = average_precision_score(test_y.flatten(), predict_test)
    df_test['pred_probs'] = predict_test
    avgs_test = df_test.groupby(['ppi']).mean()
    rocauc_test = roc_auc_score(avgs_test['binned'], avgs_test['pred_probs'])
    avgpr_test = average_precision_score(avgs_test['binned'], avgs_test['pred_probs'])

    return   rocauc_train_not_avgd, avgpr_train_not_avgd, rocauc_train, avgpr_train, rocauc_test_not_avgd, avgpr_test_not_avgd, rocauc_test, avgpr_test, num_nonzero

#classification holdout

def holdout_cv_classification(set_name, hold_size, num_samples = 0, binned = False, weights = False, pool = True):
    #set up heldout protein sets 
    if set_name == 'ncip':
        datasets = [ 'mono', 'v1_512', 'v2', 'v3' ]
        if hold_size == 1:
            pairs = get_sample_pairs(hold_size, 12, ncip_pros)
        elif hold_size == 2: #only find the designed 
            pairs = []
            for i in range(1,13,2):
                pairs.append(('P' + str(i), 'P' + str(i+1)))
    elif set_name == 'malb':
        datasets= ['malb_v2', 'malb_v3' ]
        #sample proteins and num to sample 
        pairs = get_sample_pairs(hold_size, num_samples, malb_list)
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
                    runs.append((dataset, r2, test_pair, set_name, 'small', binned, weights))
    #now multithread
    if pool: 
        with Pool() as pool:
            data = pool.starmap(run_one_model_combo_holdout_classif, runs)
    else:
        data = []
        for r in runs:
            data.append(run_one_model_combo_holdout_classif(r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    return pd.DataFrame(data).dropna()


def run_one_model_combo_holdout_classif(dataset, r2, test_pair, set_name, size_test, binned, weights):
    if os.path.exists('./datasets/' + dataset + '_correl_reduced_r_' + str(r2) +'.csv'):
        ds, not_model_cols =load_dataset_single(dataset, r2, set_name, size_test)
        results_df = run_holdout_cv_classif(ds, not_model_cols, test_pair, use_oversample=binned, weights= weights)
        best_l2 = get_best_l2_holdout(results_df, asc = False, group_col= 'l2_penalty', sort_col = 'validation_mcc')
        rocauc_train_not_avgd, avgpr_train_not_avgd, rocauc_train, avgpr_train, rocauc_test_not_avgd, avgpr_test_not_avgd, rocauc_test, avgpr_test, num_nonzero = train_best_classif_eval_on_test_holdout(best_l2, ds, not_model_cols, test_pair, use_oversample = binned, weights = weights)
        
        return {
            'dataset': dataset,
            'test_pair': '|'.join(test_pair),
            'r2_cutoff':r2,
            'l2_penalty': best_l2,
            'rocauc_train_not_avgd':rocauc_train_not_avgd,
            'rocauc_train': rocauc_train,
            'avgpr_train_not_avgd': avgpr_train_not_avgd,
            'avgpr_train': avgpr_train,
            'rocauc_test_not_avgd':rocauc_test_not_avgd,
            'rocauc_test': rocauc_test,
            'avgpr_test_not_avgd': avgpr_test_not_avgd,
            'avgpr_test':avgpr_test,
            'n_non_zero': num_nonzero
        }
   

def run_holdout_cv_classif(large_dataset_padj_order, new_not_model_cols, test_pairs, use_oversample = False, weights = False):
  
    #test_pros = [x for p in test_pairs for x in p]
    train_df = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pairs)) & ~(large_dataset_padj_order.id2.isin(test_pairs))]
    print (train_df.shape, large_dataset_padj_order.shape)
    
    sk_split = StratifiedKFold(random_state= 1, shuffle = True)
    cv_inds = list(sk_split.split(X = np.zeros(train_df.shape[0]), y = train_df.binned.to_numpy()))
    
    l1_lambdas = np.logspace(-5, 5, 51, base=10)
    data = []

    for l2 in l1_lambdas:# in range(7):
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
            
            model = linear_model.LogisticRegression(penalty = 'l2', C=1/l2, solver = 'liblinear', max_iter=2000)

            if weights:
                model.fit(x_train_scaled, train_y, sample_weight = weights_train)
            else:   
                model.fit(x_train_scaled, train_y)
            data.append({
                'val_pair': i,
                'l2_penalty': l2,
                'train_mcc':matthews_corrcoef(train_y == 1, model.predict(x_train_scaled)),
                'validation_mcc': matthews_corrcoef(val_y == 1, model.predict(x_val_scaled)),
                'n_non_zero': sum(np.array(model.coef_).flatten() != 0)
            })
    
    lasso_data = pd.DataFrame(data)
    return lasso_data



def train_best_classif_eval_on_test_holdout(l2, large_dataset_padj_order, new_not_model_cols, test_pros, use_oversample = False, weights = False):

    #test_pros = [x for p in test_pairs for x in p]
    df_slice = large_dataset_padj_order[~(large_dataset_padj_order.id1.isin(test_pros)) & ~(large_dataset_padj_order.id2.isin(test_pros))]
    df_test = large_dataset_padj_order[(large_dataset_padj_order.id1.isin(test_pros)) | (large_dataset_padj_order.id2.isin(test_pros))]

    train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
    train_y = df_slice.binned.to_numpy()
    weights_train = 1 - df_slice.ashr_padj.to_numpy()
    if use_oversample:
        binned_y = df_slice.binned.to_numpy()
        to_sample = train_x[binned_y == -1].shape[0]
        X_oversampled, y_oversampled, weights_oversampled = resample(train_x[binned_y == 1], train_y[binned_y == 1], weights_train[binned_y == 1], replace=True, n_samples=to_sample,random_state = 1)           
        train_x = np.vstack((train_x[binned_y == -1], X_oversampled))
        train_y = np.hstack((train_y[binned_y == -1], y_oversampled))
        weights_train = np.hstack((weights_train[binned_y == -1], weights_oversampled))

    scaler_x = preprocessing.StandardScaler().fit(train_x)
    x_train_scaled = scaler_x.transform(train_x)

    model = linear_model.LogisticRegression(penalty = 'l2', C=1/l2, solver = 'liblinear', max_iter=2000)

    if weights:
        model.fit(x_train_scaled, train_y, sample_weight = weights_train)
    else:   
        model.fit(x_train_scaled, train_y)

    num_nonzero = sum(np.array(model.coef_).flatten() != 0)
    train_x = df_slice.drop(columns = new_not_model_cols).to_numpy()
    train_y = df_slice.binned.to_numpy()
    x_train_scaled = scaler_x.transform(train_x)
    rocauc_train_not_avgd = roc_auc_score(train_y.flatten(), model.predict_proba(x_train_scaled)[:,1])
    avgpr_train_not_avgd = average_precision_score(train_y.flatten(), model.predict_proba(x_train_scaled)[:,1])
    df_slice['pred_probs'] = model.predict_proba(x_train_scaled)[:,1]

    avgs_train = df_slice.groupby(['ppi']).mean()
    rocauc_train = roc_auc_score(avgs_train['binned'], avgs_train['pred_probs'])
    avgpr_train = average_precision_score(avgs_train['binned'], avgs_train['pred_probs'])

    test_x = df_test.drop(columns = new_not_model_cols).to_numpy()
    test_y = df_test.binned.to_numpy()
    x_test_scaled = scaler_x.transform(test_x)
    predict_test = model.predict_proba(x_test_scaled)[:,1]
    rocauc_test_not_avgd = roc_auc_score(test_y.flatten(), predict_test)
    avgpr_test_not_avgd = average_precision_score(test_y.flatten(), predict_test)
    df_test['pred_probs'] = predict_test
    avgs_test = df_test.groupby(['ppi']).mean()
    rocauc_test = roc_auc_score(avgs_test['binned'], avgs_test['pred_probs'])
    avgpr_test = average_precision_score(avgs_test['binned'], avgs_test['pred_probs'])

    return   rocauc_train_not_avgd, avgpr_train_not_avgd, rocauc_train, avgpr_train, rocauc_test_not_avgd, avgpr_test_not_avgd, rocauc_test, avgpr_test, num_nonzero