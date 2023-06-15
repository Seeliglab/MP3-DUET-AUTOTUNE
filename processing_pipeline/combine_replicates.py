# Alyssa La Fleur
# combining replicates and subsets of replicates with proteins of interest for input to DESeq2
from processing_functions import *
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def merge_frames(list_replicate_names, save_name, mode='flat', join='inner', start_folder="./processed_replicates/"):
    frames = []
    sizes = []
    for replicate in list_replicate_names:
        # checking what kind of file we are merging now
        df_current = pd.DataFrame({})
        if mode == 'flat':
            af_name = start_folder + replicate + '_flat.csv'
            df_current = pd.read_csv(af_name)
        elif mode == 'flat_af':
            af_name = start_folder + replicate + '_flat_autotune.csv'
            df_current = pd.read_csv(af_name)
        elif mode == 'psuedoreplicate':
            af_name = start_folder + replicate + '_psuedoreplicate_autotune.csv'
            df_current = pd.read_csv(af_name)
            df_current = df_current[(df_current.count_DBDAD_trp1 != 0) | (df_current.count_DBDAD_his1 != 0) | (
                    df_current.count_ADDBD_trp1 != 0) | (df_current.count_ADDBD_his1 != 0)]

        df_current.rename(columns={'Unnamed: 0': 'PPI'}, inplace=True)

        if replicate in ['l39_1', 'l39_2', 'l39_3', 'l42', 'l43', 'l44'] and mode == 'psuedoreplicate':
            df_current['PPI'] = df_current.PPI.apply(lambda x: x.replace('_swap', ''))
            df_current['PPI'] = df_current.PPI.apply(lambda x: make_ppi(x.split(':')[0], x.split(':')[1]))
        else:
            df_current['PPI'] = df_current.PPI.apply(lambda x: x.replace('_swap', ''))
        # rename columns (not PPI, which should always be the first one)
        non_PPI_cols = df_current.columns.to_list()[1:]
        new_col_names = dict(zip(non_PPI_cols, [x + '_' + replicate for x in non_PPI_cols]))
        df_current.rename(columns=new_col_names, inplace=True)
        # now add to the list to merge together
        frames.append(df_current)
        sizes.append(df_current.shape[0])
    # condense the frames
    split_df = reduce(lambda left, right: pd.merge(left, right, on=['PPI'], how=join), frames)
    # save using the deseq prep function
    split_df = split_df.fillna(0)
    process_df_for_deseq2(split_df, save_name)


# check merged_replicates folder exists
check_destination_folder('./merged_replicates/')

#new replicates (jerala)
merge_frames(['l61_2mM', 'l62_2mM', 'l67', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/jerala_flat_autotune_5_small.csv', 'flat_af')
merge_frames(['l61_2mM', 'l62_2mM', 'l67', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/jerala_psuedoreplicate_autotune_5_small.csv',
             'psuedoreplicate')

merge_frames(['l61_2mM', 'l62_2mM', 'new_2_smaller'], './merged_replicates/jerala_flat_autotune_3_small.csv', 'flat_af')
merge_frames(['l61_2mM', 'l62_2mM',  'new_2_smaller'], './merged_replicates/jerala_psuedoreplicate_autotune_3_small.csv',
             'psuedoreplicate')

merge_frames(['l61_2mM', 'l62_2mM', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/jerala_flat_autotune_4_small.csv', 'flat_af')
merge_frames(['l61_2mM', 'l62_2mM',  'new_1_smaller', 'new_2_smaller'], './merged_replicates/jerala_psuedoreplicate_autotune_4_small.csv',
             'psuedoreplicate')


# merging mALb8 cuts
merge_frames(['l44', 'l67','new_1_smaller', 'new_2_smaller'], './merged_replicates/malb_truncations_4_smaller_flat_autotune.csv', 'flat_af')
merge_frames(['l44', 'l67','new_1_smaller', 'new_2_smaller'], './merged_replicates/malb_truncations_4_smaller_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames(['l44', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/malb_truncations_3_smaller_flat_autotune.csv', 'flat_af')
merge_frames(['l44', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/malb_truncations_3_smaller_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames(['l44', 'l67','new_1c_fixed', 'new_2c_fixed'], './merged_replicates/malb_truncations_4_flat_autotune.csv', 'flat_af')
merge_frames(['l44', 'l67','new_1c_fixed', 'new_2c_fixed'], './merged_replicates/malb_truncations_4_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames(['l44', 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/malb_truncations_3_flat_autotune.csv', 'flat_af')
merge_frames(['l44', 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/malb_truncations_3_psuedoreplicate_autotune.csv', 'psuedoreplicate')

#l70 h bond removals
merge_frames(['l70', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/hbond_malbs_3_smaller_flat_autotune.csv', 'flat_af')
merge_frames(['l70', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/hbond_malbs_3_smaller_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames(['l70', 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/hbond_malbs_3_flat_autotune.csv', 'flat_af')
merge_frames(['l70', 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/hbond_malbs_3_psuedoreplicate_autotune.csv', 'psuedoreplicate')

#just new 
merge_frames(['new_1_smaller', 'new_2_smaller'], './merged_replicates/new_smaller_flat_autotune.csv', 'flat_af')
merge_frames([ 'new_1_smaller', 'new_2_smaller'], './merged_replicates/new_smaller_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames([ 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/new_flat_autotune.csv', 'flat_af')
merge_frames([ 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/new_psuedoreplicate_autotune.csv', 'psuedoreplicate')

#l68 
merge_frames(['l68', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/plaper_3_smaller_flat_autotune.csv', 'flat_af')
merge_frames(['l68', 'new_1_smaller', 'new_2_smaller'], './merged_replicates/plaper_3_smaller_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames(['l68', 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/plaper_3_flat_autotune.csv', 'flat_af')
merge_frames(['l68', 'new_1c_fixed', 'new_2c_fixed'], './merged_replicates/plaper_3_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames(['l68', 'new_1_smaller',], './merged_replicates/plaper_2a_smaller_flat_autotune.csv', 'flat_af')
merge_frames(['l68', 'new_1_smaller', ], './merged_replicates/plaper_2a_smaller_psuedoreplicate_autotune.csv', 'psuedoreplicate')

merge_frames(['l68',  'new_2_smaller'], './merged_replicates/plaper_2b_smaller_flat_autotune.csv', 'flat_af')
merge_frames(['l68',  'new_2_smaller'], './merged_replicates/plaper_2b_smaller_psuedoreplicate_autotune.csv', 'psuedoreplicate')

"""
# merging Jerala replicates

merge_frames(['l61_2mM', 'l62_2mM', 'l67'], './merged_replicates/jerala_flat.csv', 'flat')
merge_frames(['l61_2mM', 'l62_2mM', 'l67'], './merged_replicates/jerala_flat_autotune.csv', 'flat_af')
merge_frames(['l61_2mM', 'l62_2mM', 'l67'], './merged_replicates/jerala_psuedoreplicate_autotune.csv',
             'psuedoreplicate')

merge_frames(['l61_2mM', 'l62_2mM'], './merged_replicates/jerala_2_flat.csv', 'flat')
merge_frames(['l61_2mM', 'l62_2mM'], './merged_replicates/jerala_2_flat_autotune.csv', 'flat_af')
merge_frames(['l61_2mM', 'l62_2mM'], './merged_replicates/jerala_2_psuedoreplicate_autotune.csv',
             'psuedoreplicate')

merge_frames(['l61_2mM', 'l67'], './merged_replicates/l61_l67_flat.csv', 'flat')
merge_frames(['l61_2mM', 'l67'], './merged_replicates/l61_l67_flat_autotune.csv', 'flat_af')
merge_frames(['l61_2mM', 'l67'], './merged_replicates/jerala_l61_l67_psuedoreplicate_autotune.csv',
             'psuedoreplicate')

merge_frames(['l62_2mM', 'l67'], './merged_replicates/l62_l67_flat.csv', 'flat')
merge_frames(['l62_2mM', 'l67'], './merged_replicates/l62_l67_flat_autotune.csv', 'flat_af')
merge_frames(['l62_2mM', 'l67'], './merged_replicates/jerala_l62_l67_psuedoreplicate_autotune.csv',
             'psuedoreplicate')

# merging Bcl-2 homolog binders
merge_frames(['l45', 'l66'], './merged_replicates/bcl_flat.csv', 'flat')
merge_frames(['l45', 'l66'], './merged_replicates/bcl_psuedoreplicate_autotune.csv', 'psuedoreplicate')

# merging DHD1 & DHD0
merge_frames(['l39_1', 'l39_2', 'l39_3', 'l44', 'l49'], './merged_replicates/dhd1_dhd0_flat.csv', 'flat')
merge_frames(['l39_1', 'l39_2', 'l39_3', 'l44', 'l49'], './merged_replicates/dhd1_dhd0_psuedoreplicate_autotune.csv',
             'psuedoreplicate')

# merging DHD2 & DHD0 & mALb
merge_frames(['l43', 'l44', 'l49'], './merged_replicates/dhd2_dhd0_malb_flat.csv', 'flat')
merge_frames(['l43', 'l44', 'l49'], './merged_replicates/dhd2_dhd0_malb_psuedoreplicate_autotune.csv',
             'psuedoreplicate')

# merging all designs
merge_frames(['l44', 'l49'], './merged_replicates/all_designed_coils_flat.csv', 'flat')
merge_frames(['l44', 'l49'], './merged_replicates/all_designed_coils_psuedoreplicate_autotune.csv', 'psuedoreplicate')

# merging mALb8 cuts
merge_frames(['l44', 'l67'], './merged_replicates/malb_truncations_flat.csv', 'flat')
merge_frames(['l44', 'l67'], './merged_replicates/malb_truncations_psuedoreplicate_autotune.csv', 'psuedoreplicate')



"""