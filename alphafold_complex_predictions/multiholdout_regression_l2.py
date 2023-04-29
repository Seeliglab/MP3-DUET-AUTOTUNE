#seeing if af and all features will get better with more held out test pairs 

from pipeline_interaction_regression_l2 import *

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



def holdout_cv_classification_multi_select(n_draw_pairs, binned = False, weights = False):
    pairs = []
    for i in range(1,13,2):
        pairs.append(('P' + str(i), 'P' + str(i+1)))
    datasets = [ 'v2', 'v3' ]#'all',
    r2_cutoffs = ['T1','T2','T3','T4','T5', 'af']
    runs = []
    drawn_pairs = draw_two_pairs(pairs, n_draw_pairs)
    #print (drawn_pairs)
    for two_pair in drawn_pairs:
        temp_pairs = pairs.copy()
        for pair in two_pair:
            temp_pairs.remove(pair)
        #runs all singel cv sets 
        for dataset in datasets:
            for r2 in r2_cutoffs:
                #run_one_model_combo(dataset, r2, temp_pairs, test_pair)
                #print (dataset, r2, test_pair)
                runs.append((dataset, r2, temp_pairs, two_pair, binned, weights))
    #print (runs)
    #now multithread 
    with Pool(processes= 8) as pool:
        data = pool.starmap(run_one_model_combo_holdout_classification_multi_test, runs)
    return pd.DataFrame(data)



def run_one_model_combo_holdout_classification_multi_test(dataset, r2, temp_pairs, test_pair, binned, weights):
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
    results_df = run_holdout_cv(ds, not_model_cols, test_pair, temp_pairs, use_oversample=binned, weights= weights)
    best_l1 = get_best_l1_holdout(results_df)

    mname = 'lasso_holdout_' + dataset + '_r2_cutoff_' + str(r2) + '_oversample_' + str(binned) + '_weights_' + str(weights)
    rocauc_train, rocauc_test, avgpr_train, avgpr_test, mccf1 = train_best_lasso_eval_on_test_holdout(best_l1, ds, not_model_cols, mname + '_' + test_pair[0][0] + '_' + test_pair[0][1] + '_' + test_pair[1][0] + '_' + test_pair[1][1], test_pair, use_oversample = binned, weights = weights)
    return {
        'test_pair': test_pair[0][0] + '_' + test_pair[0][1] + '_' + test_pair[1][0] + '_' + test_pair[1][1],
        'dataset': dataset,
        'r2_cutoff': r2, 
        'l1_penalty': best_l1,
        'train_AUCROC': rocauc_train, 
        'train_avgpr': avgpr_train, 
        'test_AUCROC': rocauc_test, 
        'test_avgpr': avgpr_test,
        'test_mccf1': mccf1
    }





def process_holdout_results_classification_double_select(df_name, binned, weights, max_or_min):
    #looking at mcc vs aucroc for each test set 
    df = pd.read_csv(df_name)
    if 'aj_only' in df_name:
        df['r2_cutoff'] = 0
    df['total_name'] = df.apply(lambda row: row['dataset'] + '_' + str(row['r2_cutoff']), axis = 1)
    df = remove_bad_model_groups(df)
    l1_df = get_l1_modes(df, max_or_min)
    all_dfs = []
    for ind, row in l1_df.iterrows():
        l1_val = row['l1_penalty'] #use the largest mode found
        print (row)
        df_now = holdout_classification_final_models_compare_weights_double_select(row['dataset'], row['r2_cutoff'], binned, weights, l1_val, max_or_min)
        print (df_now)
        all_dfs.append(df_now)
    df_total = pd.concat(all_dfs)
    df_total.to_csv('final_results_' + max_or_min + df_name, index = False)

def holdout_classification_final_models_compare_weights_double_select(dataset, r2, binned, weights, best_l1, max_or_min):
    pairs = []
    for i in range(1,13,2):
        pairs.append(('P' + str(i), 'P' + str(i+1)))
    runs = []
    drawn_pairs = draw_two_pairs(pairs, 15)
    datasets = [dataset]
    r2_cutoffs = [r2]
    runs = []
    for two_pair in drawn_pairs:
        temp_pairs = pairs.copy()
        for pair in two_pair:
            temp_pairs.remove(pair)
        #runs all singel cv sets 
        for dataset in datasets:
            for r2 in r2_cutoffs:
                #run_one_model_combo(dataset, r2, temp_pairs, test_pair)
                #print (dataset, r2, test_pair)
                runs.append((dataset, r2, two_pair, binned, weights, best_l1, max_or_min))
    #now multithread 
    with Pool(processes= 8) as pool:
        data = pool.starmap(run_defined_classif_l1_all_double_select_pairs, runs)
    return pd.DataFrame(data)



def run_defined_classif_l1_all_double_select_pairs(dataset, r2, test_pair, binned, weights, best_l1, max_or_min):
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
    mname = 'lasso_regression_holdout_' + dataset + '_r2_cutoff_' + str(r2) + '_oversample_' + str(binned) + '_weights_' + str(weights) + '_l1_' + str(best_l1) + max_or_min
    rocauc_train, rocauc_test, num_nonzero, avgpr_train, avgpr_test, mccf1 = final_train_best_classification(best_l1, ds, not_model_cols,  mname + '_' + test_pair[0][0] + '_' + test_pair[0][1] + '_' + test_pair[1][0] + '_' + test_pair[1][1], test_pair, use_oversample = binned, weights = weights)
    return {
        'test_pair': test_pair[0][0] + '_' + test_pair[0][1] + '_' + test_pair[1][0] + '_' + test_pair[1][1],
        'dataset': dataset,
        'r2_cutoff': r2, 
        'l1_penalty': best_l1,
        'train_AUCROC': rocauc_train, 
        'train_avgpr': avgpr_train, 
        'test_AUCROC': rocauc_test, 
        'test_avgpr': avgpr_test,
        'test_mccf1': mccf1,
        'num_non_zero': num_nonzero
    }

demo = holdout_cv_classification_multi_select(15, True, True)
print (demo.sort_values('test_mccf1', ascending = False).head(20))
demo.to_csv('hold2_ridge_results_classif.csv', index = False)
process_holdout_results_classification_double_select('hold2_ridge_results_classif.csv', True, True, 'max')