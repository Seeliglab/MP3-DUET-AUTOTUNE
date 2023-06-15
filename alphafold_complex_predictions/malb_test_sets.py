import re 
import pandas as pd

def label_LFC_bins(x, sort_level = 1):
    if x >= sort_level:
        return 1
    else:
        return -1

def strip_clashing(x, chain):
    if x != 'select clashing_res, ':
        chains_list = re.findall(r'\((.*?)\)',x)
        print (chains_list)
        for chain_curr in chains_list:
            if chain == 'A' and 'A' in chain_curr:
                return len(chain_curr.split(' ')[-1].split(","))
            elif chain == 'B' and 'B' in chain_curr:
                return len(chain_curr.split(' ')[-1].split(","))
            else:
                return -1
    else:
        return 0

def open_malb_predictions(dataset):
    if dataset == 'v2':
        v2_preds_malbs = pd.read_csv('af_prediction_values_rosetta_energy_terms\AF2_rosetta_merged-v2.csv')

        #remove columns which are always the same 
        v2_preds_malbs['fraction_int_all'] = v2_preds_malbs.IA_nres_int/v2_preds_malbs.IA_nres_all.fillna(1)

        #add other fraction int_all
        v2_preds_malbs['fraction_all'] = v2_preds_malbs.nres_int/v2_preds_malbs.nres_all.fillna(1)

        v2_preds_malbs['ppi'] = v2_preds_malbs.id1 + '|' + v2_preds_malbs.id2

        v2_preds_malbs['new_ppi'] = v2_preds_malbs.apply(lambda row: row.ppi + '|' + str(row.model_number), axis  =1 )

        min_sc_v2 = v2_preds_malbs[v2_preds_malbs['rosetta-protocol'] == 'rosetta-min-sc'].copy()
        flex_bb_v2 = v2_preds_malbs[v2_preds_malbs['rosetta-protocol'] != 'rosetta-min-sc'].copy()

        return min_sc_v2, flex_bb_v2
    else:
        #v3 needs some tlc to get num clashing res out 
        v3_preds = pd.read_csv('af_prediction_values_rosetta_energy_terms\AF2_rosetta_merged-AF-v3.csv')
        v3_preds.drop(columns = ['timed'], inplace = True)

        v3_preds['chain_a_clash_num'] = v3_preds.clashing_res.apply(lambda x: strip_clashing(x, 'A'))
        v3_preds['chain_b_clash_num'] = v3_preds.clashing_res.apply(lambda x: strip_clashing(x, 'B'))
        v3_preds.drop(columns = 'clashing_res', inplace=True)
        #remove columns which are always the same 
        v3_preds['fraction_int_all'] = v3_preds.IA_nres_int/v3_preds.IA_nres_all.fillna(1)
        v3_preds['fraction_all'] = v3_preds.nres_int/v3_preds.nres_all.fillna(1)
        v3_preds['ppi'] = v3_preds.id1 + '|' + v3_preds.id2
        v3_preds.rename(columns = {'model_number':'model', 'plddt': 'mean_plddt'}, inplace = True)
        v3_preds['type'] = 'v3'
        v3_preds['msa_depth'] = 2
        v3_preds['new_ppi'] = v3_preds.apply(lambda row: row.ppi + '|' + str(row.model), axis  =1 )
        #drop duplicate new_ppi

        min_sc_v2 = v3_preds[v3_preds['rosetta-protocol'] == 'rosetta-min-sc'].copy()
        flex_bb_v2 = v3_preds[v3_preds['rosetta-protocol'] != 'rosetta-min-sc'].copy()
        return min_sc_v2, flex_bb_v2

def attach_mp3_values(af_df, use_orientation_1):
    mp3_seq_values = pd.read_csv('deseq_l70_psuedoreplicate_autotune.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})

    mp3_seq_values['P1'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[0])
    mp3_seq_values['P2'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[1])
    mp3_seq_values['ppi'] = mp3_seq_values.P1 + '_' + mp3_seq_values.P2
    mp3_seq_values['ppi2'] = mp3_seq_values.P2 + '_' + mp3_seq_values.P1

    if use_orientation_1: 
        mp3_seq_values_ppis = mp3_seq_values.ppi.to_list() #+ mp3_seq_values.ppi2.to_list()
    else:
        mp3_seq_values_ppis = mp3_seq_values.ppi2.to_list()
    mp3_seq_lfcs = mp3_seq_values.ashr_log2FoldChange_HIS_TRP.to_list()# * 2
    ppis_dummy = []

    for i in range(1,6):
        ppis_dummy =ppis_dummy + [x + '_' + str(i) for x in mp3_seq_values_ppis]

    correl_df = pd.DataFrame({'new_ppi': ppis_dummy, 'lfc': mp3_seq_lfcs * 5})
    correl_df_merge_l70 = af_df.merge(correl_df, on = 'new_ppi', how = 'inner')
    correl_df_merge_l70['binned'] = correl_df_merge_l70.lfc.apply(lambda x: label_LFC_bins(x) )

    mp3_seq_values = pd.read_csv('deseq_l67_psuedoreplicate_autotune.csv')
    mp3_seq_values = mp3_seq_values.rename(columns = {'Unnamed: 0': 'PPI'})


    mp3_seq_values['P1'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[0])
    mp3_seq_values['P2'] = mp3_seq_values.PPI.apply(lambda x: x.split(':')[1])
    mp3_seq_values['ppi'] = mp3_seq_values.P1 + '_' + mp3_seq_values.P2
    mp3_seq_values['ppi2'] = mp3_seq_values.P2 + '_' + mp3_seq_values.P1

    if use_orientation_1: 
        mp3_seq_values_ppis = mp3_seq_values.ppi.to_list() #+ mp3_seq_values.ppi2.to_list()
    else:
        mp3_seq_values_ppis = mp3_seq_values.ppi2.to_list()
    mp3_seq_lfcs = mp3_seq_values.ashr_log2FoldChange_HIS_TRP.to_list() 
    ppis_dummy = []

    for i in range(1,6):
        ppis_dummy =ppis_dummy + [x + '_' + str(i) for x in mp3_seq_values_ppis]

    correl_df = pd.DataFrame({'new_ppi': ppis_dummy, 'lfc': mp3_seq_lfcs * 5})
    correl_df_merge_l67 = af_df.merge(correl_df, on = 'new_ppi', how = 'inner')
    correl_df_merge_l67['binned'] = correl_df_merge_l67.lfc.apply(lambda x: label_LFC_bins(x) )
    return correl_df_merge_l67, correl_df_merge_l70

def open_malb_sets_with_labels(dataset, use_orientation_1):
    set_sc, set_bb = open_malb_predictions(dataset)
    sc_1, sc_2 = attach_mp3_values(set_sc, use_orientation_1)
    bb_1, bb_2 = attach_mp3_values(set_bb, use_orientation_1)
    return sc_1, sc_2, bb_1, bb_2

