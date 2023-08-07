# Alyssa La Fleur
# identify any autoactivators in the replicates
import numpy as np
import seaborn as sns
from processing_functions import *
from scipy import stats
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_binder_list_remove_swap(trp_table_name, drive_loc):
    trp_counts = pd.read_csv(drive_loc + trp_table_name)
    cols = list(trp_counts.columns)
    cols_new = []
    for c in cols:
        if 'swap' in c:
            parts = c.split('_')
            if parts[-1] == 'swap':
                cols_new.append('_'.join(parts[:-1]))
        else:
            cols_new.append(c)
    return cols_new[1:]


def flag_outliers_high(tms, limit=3):
    """
    Flag any trimmed means which are above the extreme IQR outlier upper bound
    :param tms: trimmed means (Pandas series)
    :param limit: definition of IQR outliers bounds (1.5 is outliers, 3 is extreme outliers)
    :return: Boolean series of if the trimmed mean is a high outlier or not (Pandas series)
    """
    # flag outlier tms
    q3, q1 = np.percentile(tms.to_list(), [75, 25])
    IQR = q3 - q1
    upper_bound = q3 + (limit * IQR)
    # lower_bound = q1 - 1.5 * IQR
    # print(q3, q1, IQR, upper_bound)
    return tms > upper_bound


# noinspection PyRedundantParentheses
def find_autoactivators(flat_df, save_name, trim_val=0.25):
    """
    Scans a dataframe of trp and his read counts for the replicate for autoactivators
    :param flat_df: input dataframe of the pre- and post-selection read counts (Pandas dataframe)
    :param save_name: Name to save calculated trimmed means and graphs of trimmed mean outliers under in the figures folder
    :param trim_val: Value to use in trimmed mean calculations
    :return: None
    """
    flat_df = flat_df.rename(columns={'Unnamed: 0': 'PPI'})
    vals_counts = ['trp1', 'his1']
    for vc in vals_counts:
        flat_df.loc[flat_df[vc] == 0, vc] = None

    flat_df['lin_enrich'] = calculate_enrichment(flat_df['trp1'], flat_df['his1'])
    flat_df['log2_enrich'] = flat_df['lin_enrich'].apply(lambda x: np.log2(x))  # , deseq2_results['his1'])

    flat_df['bad_row'] = flat_df.trp1.isna() & flat_df.his1.isna()

    flat_df.loc[flat_df['bad_row'], 'trp1'] = None
    flat_df.loc[flat_df['bad_row'], 'his1'] = None
    flat_df.loc[flat_df['bad_row'], 'lin_enrich'] = None
    flat_df.loc[flat_df['bad_row'], 'log2_enrich'] = None

    flat_df = flat_df[~flat_df.bad_row]
    flat_df['DBD'] = flat_df.PPI.apply(lambda x: x.split(':')[1])
    flat_df['AD'] = flat_df.PPI.apply(lambda x: x.split(':')[3])

    pros = []
    trimmed_means_dbd = []

    for period, group in flat_df.groupby('DBD'):
        trimmed_means_dbd.append(stats.trim_mean(group.log2_enrich.to_list(), trim_val))
        pros.append(period)
    dbd_df = pd.DataFrame({'Pro': pros, 'tms_dbd': trimmed_means_dbd})

    pros = []
    trimmed_means_ad = []

    for period, group in flat_df.groupby('AD'):
        trimmed_means_ad.append(stats.trim_mean(group.log2_enrich.to_list(), trim_val))
        pros.append(period)
    ad_df = pd.DataFrame({'Pro': pros, 'tms_ad': trimmed_means_ad})

    tm_df_total = dbd_df.merge(ad_df, on='Pro')
    tm_df_total['dbd_outliers'] = flag_outliers_high(tm_df_total['tms_dbd'], 3)
    tm_df_total['ad_outliers'] = flag_outliers_high(tm_df_total['tms_ad'], 3)
    if tm_df_total[(tm_df_total.ad_outliers) | (tm_df_total.dbd_outliers)].shape[0] != 0:
        print(tm_df_total[(tm_df_total.ad_outliers) | (tm_df_total.dbd_outliers)])
        print(tm_df_total[(tm_df_total.ad_outliers) | (tm_df_total.dbd_outliers)].Pro.to_list())
        print('num outliers: ', len(tm_df_total[(tm_df_total.ad_outliers) | (tm_df_total.dbd_outliers)].Pro.to_list()))
        sns.scatterplot(data=tm_df_total, x='tms_dbd', y='tms_ad', hue='ad_outliers', style='dbd_outliers')
        plt.savefig('./figures/autoactivators_' + save_name + '.svg')
    tm_df_total.to_csv('autoactivator_trimmed_means_' + save_name + '.csv')


def flatten_library_search_for_autoactivators(drive_location, stub_name, trp_table, his_table):
    """
    Run autoactivator search on a replicate
    :param drive_location: location of folder containing read counts
    :param stub_name: name to save trimmed means and outlier csvs to
    :param trp_table: Name of the pre-selection read count csv table
    :param his_table: Name of the post-selection read count csv table
    :return: None
    """
    check_destination_folder('./figures/')  # make sure the figures folder exists
    df_list_full, binders = open_frames(drive_location, trp_table, his_table)
    flat_df = flatten(df_list_full, binders)
    find_autoactivators(flat_df, stub_name)


# final method screens
# Jerala P1-P12 series coils

print('l61')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l61/", 'l61_2mM', 'L61_TRP_44h_table.csv',
                                          'L61_2mM_3AT_HIS_table.csv')
print('-' * 50)
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l61/", 'l61_10mM', 'L61_TRP_44h_table.csv',
                                          'L61_10mM_3AT_HIS_table.csv')
print('-' * 50)
print('l62')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l62/", 'l62_2mM', 'L62_TRP_table.csv',
                                          'L62_2mM_3AT_OD05_HIS_table.csv')
print('-' * 50)
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l62/", 'l62_10mM', 'L62_TRP_table.csv',
                                          'L62_10mM_3AT_HIS_table.csv')
print('-' * 50)
# Jerala P1-P12, mALb8 truncations
print('l67')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l67/", 'l67', 'L67_TRP_table.csv',
                                          'L67_HIS_table.csv')
print('-' * 50)
# Plaper N and PA series, Thomas AN and BN coils
print('l68')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l68/", 'l68', 'L68_TRP_table.csv',
                                          'L68_HIS_table.csv')
print('-' * 50)
# Bcl-2 designed binders
print('l66')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l66/", 'l66', 'L66_TRP_table.csv',
                                          'L66_1mM_3AT_36h_HIS_table.csv')
print('-' * 50)
# mALb8 deleted hydrogen bond networks
print('l70')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/l70/", 'l70', 'L70_TRP_table.csv',
                                          'L70_HIS_table.csv')
print('-' * 50)
# older method screens
# DHD1 + DHD0
print('l39')
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l39_1/", 'l39_1', 'L39_TRP_table.csv',
                                          'L39_HIS_table.csv')
print('-' * 50)
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l39_2/", 'l39_2', 'L39_TRP_table.csv',
                                          'L39_HIS_0mM_3AT_table.csv')
print('-' * 50)
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l39_3/", 'l39_3', 'L39_TRP_table.csv',
                                          'L39_HIS_table.csv')
print('-' * 50)
# DHD0 + DHD2 + mALb
print('l43')
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l43/", 'l43', 'L43_TRP1_table.csv',
                                          'L43_HIS1_1mM_3AT_table.csv')
print('-' * 50)
# DHD0-3 + mALb
print('l44')
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l44/", 'l44', 'L44_TRP_table.csv',
                                          'L44_HIS_1mM_3AT_table.csv')
print('-' * 50)
print('l49')
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l49/", 'l49', 'L49_TRP_table.csv',
                                          'L49_HIS_main.csv')
print('-' * 50)
# Bcl-2 designed 4
print('l45')
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l45/", 'l45', 'L45_TRP_V1_table.csv',
                                          'L45_HIS_V2_125_Repl1_table.csv')
print('-' * 50)
print('l48')
flatten_library_search_for_autoactivators("../data/older_mp3seq_method/l48/", 'l48', 'L48_TRP_table.csv',
                                          'L48_HIS_table.csv')
print('-' * 50)

#data\final_mp3seq_method\large_replicate_2023
#removed junk rows (Bcls were faulty, and some BW proteins were missing their expected partners)
print('new 1')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/large_rep_nochanges/", 'new_1_smaller', 'TRP_table_Trp1_dropJunk.csv',
                                          'HIS_table_His1_dropJunk.csv')
print('new 2')
flatten_library_search_for_autoactivators("../data/final_mp3seq_method/large_rep_nochanges/", 'new_2_smaller', 'TRP_table_Trp2_dropJunk.csv',
                                          'HIS_table_His2_dropJunk.csv')


