# Alyssa La Fleur
# Raw read count processing functions to prepare inputs for DESeq2
import os
import warnings
from functools import reduce
from itertools import combinations
from math import ceil, isnan
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)


def check_destination_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)


def flatten(df_list_full, binders):
    single_col_trp = make_single_col_from_df(df_list_full[0], binders)
    single_col_trp = single_col_trp.rename(columns={'count': 'trp1'})
    single_col_his = make_single_col_from_df(df_list_full[1], binders)
    single_col_his = single_col_his.rename(columns={'count': 'his1'})
    df_list = [single_col_trp, single_col_his]
    concat_df = reduce(lambda left, right: pd.merge(left, right, on=['PPI'], how='outer'), df_list)
    print('Total: ', concat_df.shape[0], ' failed rows: ',
          concat_df[(concat_df.trp1 == 0) & (concat_df.his1 == 0)].shape[0])
    return concat_df


def autofill_bad_dbds(flat_df, binders, list_bad_dbds):
    flat_df['DBD'] = flat_df.PPI.apply(lambda x: x.split(':')[1])
    flat_df['AD'] = flat_df.PPI.apply(lambda x: x.split(':')[3])

    vals_counts = ['trp1', 'his1']
    for vc in vals_counts:
        flat_df.loc[flat_df[vc] == 0, vc] = None
    flat_df['lin_enrich'] = calculate_enrichment(flat_df['trp1'], flat_df['his1'])
    keep_cols = ['trp1', 'his1', 'AD', 'DBD', 'lin_enrich']

    split_for_graphing = [split_by_orientation(flat_df, binders, x, True) for x in keep_cols]
    # make one df
    split_df = reduce(lambda left, right: pd.merge(left, right, on=['PPI'], how='outer'), split_for_graphing)
    # DBDAD is lower, ADDBD is upper
    split_df['trp_fraction_ad_old'] = split_df['trp1_ADDBD'].fillna(0) / split_df['trp1_ADDBD'].sum()
    split_df['trp_fraction_dbd_old'] = split_df['trp1_DBDAD'].fillna(0) / split_df['trp1_DBDAD'].sum()
    split_df['his_fraction_ad_old'] = split_df['his1_ADDBD'].fillna(0) / split_df['his1_ADDBD'].sum()
    split_df['his_fraction_dbd_old'] = split_df['his1_DBDAD'].fillna(0) / split_df['his1_DBDAD'].sum()

    sum_trp_addbd = split_df.trp1_ADDBD.sum()
    sum_his_addbd = split_df.his1_ADDBD.sum()
    sum_trp_dbdad = split_df.trp1_DBDAD.sum()
    sum_his_dbdad = split_df.his1_DBDAD.sum()

    min_trp_addbd = split_df.trp1_ADDBD.min()
    min_his_addbd = split_df.his1_ADDBD.min()
    min_trp_dbdad = split_df.trp1_DBDAD.min()
    min_his_dbdad = split_df.his1_DBDAD.min()

    for bad_dbd in list_bad_dbds:
        subset = split_df[(split_df.DBD_DBDAD == bad_dbd) | (split_df.DBD_ADDBD == bad_dbd)]
        subset = subset[subset.DBD_DBDAD != subset.AD_DBDAD]
        # print (subset)
        for ind, row in subset.iterrows():
            # print (row)
            if row['DBD_DBDAD'] == bad_dbd:
                # using ADDBD to replace DBDAD
                if isnan(row['trp1_ADDBD']):  # if the trp count in the replacement is none, copy the none
                    trp_new = min_trp_dbdad
                    his_new = ceil(row['his_fraction_ad_old'] * sum_his_dbdad)
                elif isnan(row['his1_ADDBD']):
                    trp_new = ceil(row['trp_fraction_ad_old'] * sum_trp_dbdad)
                    his_new = min_his_dbdad
                else:
                    trp_new, his_new = back_calculate_pairs(sum_trp_dbdad, sum_his_dbdad, sum_trp_addbd, sum_his_addbd,
                                                            row['lin_enrich_ADDBD'], row['trp_fraction_ad_old'])
                split_df.loc[ind, 'trp1_DBDAD'] = trp_new
                split_df.loc[ind, 'his1_DBDAD'] = his_new
            elif row['DBD_ADDBD'] == bad_dbd:
                # using DBDAD to replace ADDBD
                if isnan(row['trp1_ADDBD']):
                    trp_new = min_trp_addbd
                    his_new = ceil(row['his_fraction_dbd_old'] * sum_his_addbd)
                elif isnan(row['his1_ADDBD']):
                    trp_new = ceil(row['trp_fraction_dbd_old'] * sum_trp_addbd)
                    his_new = min_his_addbd
                else:
                    trp_new, his_new = back_calculate_pairs(sum_trp_addbd, sum_his_addbd, sum_trp_dbdad, sum_his_dbdad,
                                                            row['lin_enrich_DBDAD'], row['trp_fraction_dbd_old'])
                split_df.loc[ind, 'trp1_ADDBD'] = trp_new
                split_df.loc[ind, 'his1_ADDBD'] = his_new

    # set all homodimers of the bad protein to 0 manually for both values (these cannot be recovered)
    bad_ppis = [x + ':' + x for x in list_bad_dbds]
    for bad_int in bad_ppis:
        split_df.loc[split_df.PPI == bad_int, 'trp1_DBDAD'] = None
        split_df.loc[split_df.PPI == bad_int, 'his1_DBDAD'] = None
        split_df.loc[split_df.PPI == bad_int, 'trp1_ADDBD'] = None
        split_df.loc[split_df.PPI == bad_int, 'his1_ADDBD'] = None

    # dealing with homodimers
    for ind, row in split_df[split_df.DBD_DBDAD == split_df.AD_DBDAD].iterrows():
        if not isnan(row['trp1_DBDAD']) and not isnan(row['his1_DBDAD']):
            trp_new_DBDAD, his_new_DBDAD = back_calculate_homodimers(row['trp1_DBDAD'],
                                                                     sum_trp_dbdad,
                                                                     sum_his_dbdad,
                                                                     row['lin_enrich_DBDAD'])
            trp_new_ADDBD, his_new_ADDBD = back_calculate_homodimers(row['trp1_DBDAD'],
                                                                     sum_trp_addbd,
                                                                     sum_his_addbd,
                                                                     row['lin_enrich_DBDAD'])

            split_df.loc[ind, 'his1_DBDAD'] = his_new_DBDAD
            split_df.loc[ind, 'his1_ADDBD'] = his_new_ADDBD

    # flatten this df - save a flattened version + DESeq2 ready versions
    split_df['PPI_DBDAD'] = 'DBD:' + split_df['DBD_DBDAD'] + ':AD:' + split_df['AD_DBDAD']
    split_df['PPI_ADDBD'] = 'DBD:' + split_df['DBD_ADDBD'] + ':AD:' + split_df['AD_ADDBD']

    half_df = split_df[['PPI_DBDAD', 'trp1_DBDAD', 'his1_DBDAD']].copy()
    half_df.rename(columns={'PPI_DBDAD': 'PPI', 'trp1_DBDAD': 'trp1', 'his1_DBDAD': 'his1'}, inplace=True)

    half_df2 = split_df[split_df.DBD_DBDAD != split_df.AD_DBDAD][['PPI_ADDBD', 'trp1_ADDBD', 'his1_ADDBD']].copy()
    half_df2.rename(columns={'PPI_ADDBD': 'PPI', 'trp1_ADDBD': 'trp1', 'his1_ADDBD': 'his1'}, inplace=True)

    flat_again = pd.concat([half_df, half_df2])

    flat_again = flat_again.fillna(0)
    flat_again['temp_sum'] = flat_again.sum(axis=1)

    # remove bad rows
    flat_again = flat_again[flat_again.temp_sum != 0].copy()
    # drop temp col
    flat_again.drop('temp_sum', inplace=True, axis=1)

    # set up with homodimer retention
    prep_deseq2 = split_df[['PPI', 'trp1_DBDAD', 'trp1_ADDBD', 'his1_DBDAD', 'his1_ADDBD']]
    prep_deseq2 = prep_deseq2.fillna(0)
    # rename the columns
    prep_deseq2 = prep_deseq2.rename(columns={'trp1_DBDAD': 'count_DBDAD_trp1',
                                              'trp1_ADDBD': 'count_ADDBD_trp1',
                                              'his1_DBDAD': 'count_DBDAD_his1',
                                              'his1_ADDBD': 'count_ADDBD_his1'})

    prep_deseq2['is_homodimer'] = prep_deseq2.PPI.apply(lambda x: x.split(':')[0] == x.split(':')[1])
    without_homo = prep_deseq2[~prep_deseq2.is_homodimer].copy()
    prep_deseq2.drop('is_homodimer', inplace=True, axis=1)
    without_homo.drop('is_homodimer', inplace=True, axis=1)

    return flat_again, prep_deseq2, without_homo


def autofill_bad_ads(flat_df, binders, list_bad_ads):
    flat_df['DBD'] = flat_df.PPI.apply(lambda x: x.split(':')[1])
    flat_df['AD'] = flat_df.PPI.apply(lambda x: x.split(':')[3])

    vals_counts = ['trp1', 'his1']
    for vc in vals_counts:
        flat_df.loc[flat_df[vc] == 0, vc] = None
    flat_df['lin_enrich'] = calculate_enrichment(flat_df['trp1'], flat_df['his1'])
    keep_cols = ['trp1', 'his1', 'AD', 'DBD', 'lin_enrich']

    split_for_graphing = [split_by_orientation(flat_df, binders, x, True) for x in keep_cols]
    # make one df
    split_df = reduce(lambda left, right: pd.merge(left, right, on=['PPI'], how='outer'), split_for_graphing)
    # DBDAD is lower, ADDBD is upper

    split_df['trp_fraction_ad_old'] = split_df['trp1_ADDBD'] / split_df['trp1_ADDBD'].sum()
    split_df['trp_fraction_dbd_old'] = split_df['trp1_DBDAD'] / split_df['trp1_DBDAD'].sum()
    split_df['his_fraction_ad_old'] = split_df['his1_ADDBD'] / split_df['his1_ADDBD'].sum()
    split_df['his_fraction_dbd_old'] = split_df['his1_DBDAD'] / split_df['his1_DBDAD'].sum()

    sum_trp_addbd = split_df.trp1_ADDBD.sum()
    sum_his_addbd = split_df.his1_ADDBD.sum()
    sum_trp_dbdad = split_df.trp1_DBDAD.sum()
    sum_his_dbdad = split_df.his1_DBDAD.sum()

    min_trp_addbd = split_df.trp1_ADDBD.min()
    min_his_addbd = split_df.his1_ADDBD.min()
    min_trp_dbdad = split_df.trp1_DBDAD.min()
    min_his_dbdad = split_df.his1_DBDAD.min()

    for bad_ad in list_bad_ads:
        subset = split_df[(split_df.AD_DBDAD == bad_ad) | (split_df.AD_ADDBD == bad_ad)]
        subset = subset[subset.DBD_DBDAD != subset.AD_DBDAD]  #
        for ind, row in subset.iterrows():
            # print (row)
            if row['AD_DBDAD'] == bad_ad:
                # using ADDBD to replace DBDAD
                if isnan(row['trp1_DBDAD']):  # if the trp count in the replacement is none, copy the none
                    trp_new = min_trp_dbdad
                    his_new = ceil(row['his_fraction_dbd_old'] * sum_his_dbdad)
                elif isnan(row['his1_ADDBD']):
                    trp_new = ceil(row['trp_fraction_dbd_old'] * sum_trp_dbdad)
                    his_new = min_his_dbdad
                else:
                    trp_new, his_new = back_calculate_pairs(sum_trp_dbdad, sum_his_dbdad, sum_trp_addbd, sum_his_addbd,
                                                            row['lin_enrich_ADDBD'], row['trp_fraction_ad_old'])
                split_df.loc[ind, 'trp1_DBDAD'] = trp_new
                split_df.loc[ind, 'his1_DBDAD'] = his_new
            elif row['AD_ADDBD'] == bad_ad:
                # using DBDAD to replace ADDBD
                if isnan(row['trp1_ADDBD']):
                    trp_new = min_trp_addbd
                    his_new = ceil(row['his_fraction_dbd_old'] * sum_trp_addbd)
                elif isnan(row['his1_ADDBD']):
                    trp_new = ceil(row['trp_fraction_dbd_old'] * sum_his_addbd)
                    his_new = min_his_addbd
                else:
                    trp_new, his_new = back_calculate_pairs(sum_trp_addbd, sum_his_addbd, sum_trp_dbdad, sum_his_dbdad,
                                                            row['lin_enrich_DBDAD'], row['trp_fraction_dbd_old'])
                split_df.loc[ind, 'trp1_ADDBD'] = trp_new
                split_df.loc[ind, 'his1_ADDBD'] = his_new

    # set all homodimers of the bad protein to 0 manually for both values (these cannot be recovered)
    bad_ppis = [x + ':' + x for x in list_bad_ads]
    for bad_int in bad_ppis:
        split_df.loc[split_df.PPI == bad_int, 'trp1_DBDAD'] = None
        split_df.loc[split_df.PPI == bad_int, 'his1_DBDAD'] = None
        split_df.loc[split_df.PPI == bad_int, 'trp1_ADDBD'] = None
        split_df.loc[split_df.PPI == bad_int, 'his1_ADDBD'] = None

    for ind, row in split_df[split_df.DBD_DBDAD == split_df.AD_DBDAD].iterrows():
        if not isnan(row['trp1_DBDAD']) and not isnan(row['his1_DBDAD']):
            trp_new_DBDAD, his_new_DBDAD = back_calculate_homodimers(row['trp1_DBDAD'],
                                                                     sum_trp_dbdad,
                                                                     sum_his_dbdad,
                                                                     row['lin_enrich_DBDAD'])
            trp_new_ADDBD, his_new_ADDBD = back_calculate_homodimers(row['trp1_DBDAD'],
                                                                     sum_trp_addbd,
                                                                     sum_his_addbd,
                                                                     row['lin_enrich_DBDAD'])

            split_df.loc[ind, 'his1_DBDAD'] = his_new_DBDAD
            split_df.loc[ind, 'his1_ADDBD'] = his_new_ADDBD
    # flatten this df - save a flattened version + deseq ready versions
    split_df['PPI_DBDAD'] = 'DBD:' + split_df['DBD_DBDAD'] + ':AD:' + split_df['AD_DBDAD']
    split_df['PPI_ADDBD'] = 'DBD:' + split_df['DBD_ADDBD'] + ':AD:' + split_df['AD_ADDBD']

    half_df = split_df[['PPI_DBDAD', 'trp1_DBDAD', 'his1_DBDAD']].copy()
    half_df.rename(columns={'PPI_DBDAD': 'PPI', 'trp1_DBDAD': 'trp1', 'his1_DBDAD': 'his1'}, inplace=True)

    half_df2 = split_df[split_df.DBD_DBDAD != split_df.AD_DBDAD][['PPI_ADDBD', 'trp1_ADDBD', 'his1_ADDBD']].copy()
    half_df2.rename(columns={'PPI_ADDBD': 'PPI', 'trp1_ADDBD': 'trp1', 'his1_ADDBD': 'his1'}, inplace=True)

    flat_again = pd.concat([half_df, half_df2])

    flat_again = flat_again.fillna(0)
    flat_again['temp_sum'] = flat_again.sum(axis=1)
    # remove bad rows
    flat_again = flat_again[flat_again.temp_sum != 0].copy()
    # drop temp col
    flat_again.drop('temp_sum', inplace=True, axis=1)

    # set up with homodimer retention
    prep_deseq2 = split_df[['PPI', 'trp1_DBDAD', 'trp1_ADDBD', 'his1_DBDAD', 'his1_ADDBD']]
    prep_deseq2 = prep_deseq2.fillna(0)
    # rename the columns
    prep_deseq2 = prep_deseq2.rename(columns={'trp1_DBDAD': 'count_DBDAD_trp1',
                                              'trp1_ADDBD': 'count_ADDBD_trp1',
                                              'his1_DBDAD': 'count_DBDAD_his1',
                                              'his1_ADDBD': 'count_ADDBD_his1'})

    prep_deseq2['is_homodimer'] = prep_deseq2.PPI.apply(lambda x: x.split(':')[0] == x.split(':')[1])
    without_homo = prep_deseq2[~prep_deseq2.is_homodimer].copy()

    prep_deseq2.drop('is_homodimer', inplace=True, axis=1)
    without_homo.drop('is_homodimer', inplace=True, axis=1)

    return flat_again, prep_deseq2, without_homo


# naming functions
def make_ppi(one, two):
    """
    Makes a PPI name by sorting protein names and joining
    :param one: One input protein name (str)
    :param two: Other input protein name (str)
    :return: Final PPI name (str), formatted as P1:P2 where P1 is before P2 when sorted
    """
    ppi = [one, two]
    ppi.sort()
    return ':'.join(ppi)


def fix_ppi(ppi):
    """
    Corrects PPI name (for errors introduced during name standardizations across datasets)
    :param ppi: Input PPI to fix (str), format of P1:P2
    :return: New PPI name (str)
    """
    dbd = ppi.split(':')[0]
    ad = ppi.split(':')[1]
    ppi_name = [dbd, ad]
    ppi_name.sort()
    ppi_name = ":".join(ppi_name)
    return ppi_name


# dataframe manipulation functions
def open_frames(drive_location, trp_counts, his_counts):
    """
    Opens trp and his read count files for processing
    :param drive_location: location of data files directory
    :param trp_counts: name of the Trp csv file (str)
    :param his_counts: name of the His csv file (str)
    :return: list of trp and his count Pandas dataframes, and a list of all proteins measured in the screen
    """
    trp_counts = pd.read_csv(drive_location + trp_counts)
    trp_counts = trp_counts.rename(columns={'Binder1': 'Binder'})
    his_counts = pd.read_csv(drive_location + his_counts)
    his_counts = his_counts.rename(columns={'Binder1': 'Binder'})
    trp_counts = trp_counts.fillna(0)
    his_counts = his_counts.fillna(0)
    binders = trp_counts.Binder.to_list()
    df_list_full = [trp_counts, his_counts, ]
    return df_list_full, binders


def make_single_col_from_df(df, binders_list):
    final_counts = []
    final_binder_pairs = []
    for b in binders_list:
        for b2 in binders_list:
            if b2 in df:
                val_row = df[df.Binder == b][b2].to_list()
                if len(val_row) > 0:
                    final_counts.append(val_row[0])
                    final_binder_pairs.append("DBD:" + b + ':AD:' + b2)
    return pd.DataFrame({'PPI': final_binder_pairs, 'count': final_counts})


def process_df_for_deseq2(concat_df, save_name):
    # remove the first header col when saving for deseq2 input
    cols = list(concat_df.columns)
    concat_df.to_csv(save_name, index=False, header=[''] + cols[1:])


def back_calculate_homodimers(trp_homo, trp_count, his_count, lin_enrich_infill):
    inv_norm_coefficient = his_count / trp_count
    trp_now = trp_homo
    his_now = ceil(trp_now * lin_enrich_infill * inv_norm_coefficient)
    return trp_now, his_now


def back_calculate_pairs(trp_old, his_old, trp_count_other, his_count_other, lin_enrich_infill, trp_fraction_other_or=0,
                         his_fraction_other_or=0):
    # return trp values =
    trp_now = None
    his_now = None
    inv_norm_coefficient = his_count_other / trp_count_other
    if trp_fraction_other_or != 0:
        trp_now = ceil(trp_fraction_other_or * trp_old)
        his_now = ceil(trp_now * lin_enrich_infill * inv_norm_coefficient)
        # return trp_now, his_now
    elif his_fraction_other_or != 0:
        his_now = ceil(his_fraction_other_or * his_old)
        trp_now = ceil(his_now * (1 / (lin_enrich_infill * inv_norm_coefficient)))
        # return trp_now, his_now
    return trp_now, his_now


def split_by_orientation(df, proteins, col_extract, keep_homodimers=True):
    # for a col, reorder to 2 rows for each PPI

    # all combos of 2
    combos = list(combinations(proteins, 2))
    if keep_homodimers:
        combos = combos + [(x, x) for x in proteins]
    # df of all combos of 2
    ppis = pd.DataFrame({'DBD_fast': [c[0] for c in combos], 'AD_fast': [c[1] for c in combos]})

    ppis['or_1_ppi'] = 'DBD:' + ppis['DBD_fast'] + ':AD:' + ppis['AD_fast']
    ppis['or_2_ppi'] = 'DBD:' + ppis['AD_fast'] + ':AD:' + ppis['DBD_fast']

    ppis = ppis.merge(df, left_on='or_1_ppi', right_on='PPI')
    ppis = ppis.merge(df, left_on='or_2_ppi', right_on='PPI', suffixes=('_DBDAD', '_ADDBD'))
    ppis['PPI'] = ppis.apply(lambda row: make_ppi(row['DBD_fast'], row['AD_fast']), axis=1)
    return ppis[['PPI', col_extract + '_DBDAD', col_extract + '_ADDBD']]


def calculate_enrichment(trp_col_with_na, his_col_with_na, trp_count=0, his_count=0):
    if trp_count == 0 and his_count == 0:
        trp_count = trp_col_with_na.fillna(0).values.sum()
        his_count = his_col_with_na.fillna(0).values.sum()
        norm_coefficient = trp_count / his_count
        return norm_coefficient * his_col_with_na.fillna(his_col_with_na.min()) / trp_col_with_na.fillna(trp_col_with_na.min())
    else:
        norm_coefficient = trp_count / his_count
        return norm_coefficient * his_col_with_na.fillna(his_col_with_na.min()) / trp_col_with_na.fillna(trp_col_with_na.min())
