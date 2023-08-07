# Alyssa La Fleur
# processing of each replicate of different screens (assuming autoactivators are known)
# uses starmap to speed up processing
from processing_functions import *
from multiprocessing import Pool
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def run_one_library(drive_location, destination_folder, stub_name, trp_table, his_table, bad_dbds, bad_ads = []):
    """
    Process one replicate with autotune and prepare psuedoreplicate and non-psuedoreplicate read count csv files for DESeq2 input
    :param drive_location: Location folder of input trp and his read count csv files
    :param destination_folder: Folder to save DESeq2 ready input files
    :param stub_name: shared beginning for DESeq2 ready input files being generated (str):
        stub_name_flat.csv - pre and post His read counts per DBD and AD fusion, autoactivators removed
        stub_name__flat_autotune.csv - pre and post His read counts per DBD and AD fusion, autotuned
        stub_name_psuedoreplicate_autotune.csv - pre and post HIs read counts per PPI, psuedoreplicated and autotuned
    :param trp_table: pre his selection read count csv name in drive_location (str)
    :param his_table: post his selection read count csv name in drive_location (str)
    :param bad_dbds: list of autoactivating dbd proteins (list of str)
    :return: None
    """
    df_list_full, binders = open_frames(drive_location, trp_table, his_table)
    # save all unique binders in the screening
    binders_dummy_df = pd.DataFrame({'Binder': binders})
    binders_dummy_df.to_csv(destination_folder +stub_name + 'binders.csv', index=False)

    # dataframe of pre and post His selection read counts combined into one dataframe
    flat_df = flatten(df_list_full, binders)

    # fill autoactivators identified using autotune
    flat_again, deseq_homo, _ = autofill_both(flat_df.copy(), binders, bad_dbds, bad_ads)

    # removing autoactivating interactions from the dataframe
    flat_df = flat_df[(flat_df.trp1 != 0) | (flat_df.his1 != 0)]
    flat_df.to_csv(destination_folder + stub_name + '_flat.csv', index=False)

    flat_df['DBD'] = flat_df.PPI.apply(lambda x: x.split(":")[1])
    flat_df['AD'] = flat_df.PPI.apply(lambda x: x.split(":")[3])
    flat_df = flat_df[~flat_df.DBD.isin(bad_dbds)]  # drop autoactiavtors DBD
    flat_df = flat_df.fillna(0)
    flat_df.drop('DBD', inplace=True, axis=1)
    flat_df.drop('AD', inplace=True, axis=1)
    flat_df = flat_df[(flat_df.trp1 != 0) | (flat_df.his1 != 0)]
    flat_df.to_csv(destination_folder + stub_name + '_flat_no_autoactivators.csv', index=False)

    # save autotuned dataframe
    flat_again.to_csv(destination_folder + stub_name + '_flat_autotune.csv', index=False)

    # remove undetected interactions from the psuedoreplicate (homodimers included) and save
    deseq_homo = deseq_homo[
        (deseq_homo.count_DBDAD_trp1 != 0) | (deseq_homo.count_DBDAD_his1 != 0) | (deseq_homo.count_ADDBD_trp1 != 0) | (
                deseq_homo.count_ADDBD_his1 != 0)]
    process_df_for_deseq2(deseq_homo, destination_folder + stub_name + '_psuedoreplicate_autotune.csv')
    print("FINISHED: ", stub_name)


check_destination_folder("./processed_replicates/")  # check the destination folder exists
library_inputs_list = []

library_inputs_list.append(("../data/final_mp3seq_method/large_rep_nochanges/",
                            "./processed_replicates/",
                            'new_1c_fixed',
                            'TRP_table_Trp1.csv',
                            'HIS_table_His1.csv',
                            ['1005_mALb8x12_fdrtc_B', '1007_mALb8x12j_fdrtc_B', '1008_mALb8x12j_rprtc_B', 'AN4', 'Bcl-2', 'Mcl1[151-321]', 'N1', 'XCDP07', 'alphaBCL2', 'alphaBCLB', 'alphaMCL1'] ))

library_inputs_list.append(("../data/final_mp3seq_method/large_rep_nochanges/",
                            "./processed_replicates/",
                            'new_1_smaller',
                            'TRP_table_Trp1_dropJunk.csv',
                            'HIS_table_His1_dropJunk.csv',
                            ['1004_mALb8x2_rprtc_B', '1005_mALb8x12_fdrtc_B', '1007_mALb8x12j_fdrtc_B', '1008_mALb8x12j_rprtc_B', 'AN4', 'IL14_A', 'MCL1', 'N1'] ))



library_inputs_list.append(("../data/final_mp3seq_method/large_rep_nochanges/",
                            "./processed_replicates/",
                            'new_2c_fixed',
                            'TRP_table_Trp2.csv',
                            'HIS_table_His2.csv',['1005_mALb8x12_fdrtc_B', '1007_mALb8x12j_fdrtc_B', 'Mcl1[151-321]', 'N1', 'alphaBCLB']
                           ))

library_inputs_list.append(("../data/final_mp3seq_method/large_rep_nochanges/",
                            "./processed_replicates/",
                            'new_2_smaller',
                            'TRP_table_Trp2_dropJunk.csv',
                            'HIS_table_His2_dropJunk.csv',
                            ['1004_mALb8x2_rprtc_B', '1005_mALb8x12_fdrtc_B', '1007_mALb8x12j_fdrtc_B', '1008_mALb8x12j_rprtc_B', 'AN4', 'N1'] ))

# Jerala P1-P12 replicates
library_inputs_list.append(("../data/final_mp3seq_method/l61/",
                            "./processed_replicates/",
                            'l61_2mM',
                            'L61_TRP_44h_table.csv',
                            'L61_2mM_3AT_HIS_table.csv',
                            []))

library_inputs_list.append(("../data/final_mp3seq_method/l61/",
                            "./processed_replicates/",
                            'l61_10mM',
                            'L61_TRP_44h_table.csv',
                            'L61_10mM_3AT_HIS_table.csv',
                            []))
library_inputs_list.append(("../data/final_mp3seq_method/l62/",
                            "./processed_replicates/",
                            'l62_2mM',
                            'L62_TRP_table.csv',
                            'L62_2mM_3AT_OD05_HIS_table.csv',
                            []))
library_inputs_list.append(("../data/final_mp3seq_method/l62/",
                            "./processed_replicates/",
                            'l62_10mM',
                            'L62_TRP_table.csv',
                            'L62_10mM_3AT_HIS_table.csv',
                            []))
# Jerala P1-P12, mALb8 truncations
library_inputs_list.append(("../data/final_mp3seq_method/l67/",
                            "./processed_replicates/",
                            'l67',
                            'L67_TRP_table.csv',
                            'L67_HIS_table.csv',
                            ['IL14_A', 'Jerala_P11']))

# Plaper N and PA series, Thomas AN and BN coils
library_inputs_list.append(("../data/final_mp3seq_method/l68/",
                            "./processed_replicates/",
                            'l68',
                            'L68_TRP_table.csv',
                            'L68_HIS_table.csv',
                            ['AN4', 'Jerala_P11', 'N1']))

# Bcl-2 designed binders
library_inputs_list.append(("../data/final_mp3seq_method/l66/",
                            "./processed_replicates/",
                            'l66',
                            'L66_TRP_table.csv',
                            'L66_1mM_3AT_36h_HIS_table.csv',
                            []))
# mALb8 deleted hydrogen bond networks
library_inputs_list.append(("../data/final_mp3seq_method/l70/",
                            "./processed_replicates/",
                            'l70',
                            'L70_TRP_table.csv',
                            'L70_HIS_table.csv',
                            ['1003__mALb8x2_fdrtc_B', '1004__mALb8x2_rprtc_B', '1005__mALb8x12_fdrtc_B',
                             '1007__mALb8x12j_fdrtc_B', '1008__mALb8x12j_rprtc_B', 'Jerala_P11']))
# older method screens
#BCL values
library_inputs_list.append(("../data/older_mp3seq_method/l33/",
                            "./processed_replicates/",
                            'l33',
                            'L33_TRP_table.csv',
                            'L33_HIS_2mM_3AT_table.csv',
                            ['XCDP07']))
# DHD1 + DHD0
library_inputs_list.append(("../data/older_mp3seq_method/l39_1/",
                            "./processed_replicates/",
                            'l39_1',
                            'L39_TRP_table.csv',
                            'L39_HIS_table.csv',
                            ['A_HT_DHD_100_swap', 'A_HT_DHD_33_swap', 'A_HT_DHD_96_swap', 'A_HT_DHD_97_swap',
                             'B_HT_DHD_94_swap']))
library_inputs_list.append(("../data/older_mp3seq_method/l39_2/",
                            "./processed_replicates/",
                            'l39_2',
                            'L39_TRP_table.csv',
                            'L39_HIS_0mM_3AT_table.csv',
                            ['A_HT_DHD_97_swap', 'A_HT_DHD_33_swap']))
library_inputs_list.append(("../data/older_mp3seq_method/l39_3/",
                            "./processed_replicates/",
                            'l39_3',
                            'L39_TRP_table.csv',
                            'L39_HIS_table.csv',
                            ['A_HT_DHD_100_swap', 'A_HT_DHD_33_swap', 'A_HT_DHD_71_swap', 'A_HT_DHD_96_swap',
                             'A_HT_DHD_97_swap', 'B_HT_DHD_45_swap', 'B_HT_DHD_94_swap']))
# DHD0 + DHD2 + mALb
library_inputs_list.append(("../data/older_mp3seq_method/l43/",
                            "./processed_replicates/",
                            'l43',
                            'L43_TRP1_table.csv',
                            'L43_HIS1_1mM_3AT_table.csv',
                            ['ZCON_155_cutT5_B']))
# DHD0-3 + mALb
library_inputs_list.append(("../data/older_mp3seq_method/l44/",
                            "./processed_replicates/",
                            'l44',
                            'L44_TRP_table.csv',
                            'L44_HIS_1mM_3AT_table.csv',
                            ['A_HT_DHD_33_swap', 'A_HT_DHD_97_swap', 'B_HT_DHD_29_swap', 'Bcl-B', 'mALb8_cutT1_B']))
library_inputs_list.append(("../data/older_mp3seq_method/l49/",
                            "./processed_replicates/",
                            'l49',
                            'L49_TRP_table.csv',
                            'L49_HIS_main.csv',
                            []))
# Bcl-2 designed 4
library_inputs_list.append(("../data/older_mp3seq_method/l45/",
                            "./processed_replicates/",
                            'l45',
                            'L45_TRP_V1_table.csv',
                            'L45_HIS_V2_125_Repl1_table.csv',
                            []))
library_inputs_list.append(("../data/older_mp3seq_method/l48/",
                            "./processed_replicates/",
                            'l48',
                            'L48_TRP_table.csv',
                            'L48_HIS_table.csv',
                            []))
def main():
    with Pool() as pool:
        pool.starmap(run_one_library, library_inputs_list)



if __name__ == "__main__":
    main()