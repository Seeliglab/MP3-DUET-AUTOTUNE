#load true values
from processing_functions import * 
import numpy as np
import pandas as pd 
from functools import reduce

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def get_jerala_values():
    target_orientations = []
    target_ppis = []
    binders = []
    for i in range(1,13,2):
        #print (i, i + 1)
        target_orientations.append('DBD:Jerala_P' + str(i) + ':AD:Jerala_P' + str(i+1))
        target_orientations.append('DBD:Jerala_P' + str(i+1) + ':AD:Jerala_P' + str(i))
        ppi_label = ['Jerala_P' + str(i) , 'Jerala_P' + str(i+1) ]
        ppi_label.sort()
        target_ppis.append(':'.join(ppi_label))
        binders.append('Jerala_P' + str(i))
        binders.append('Jerala_P' + str(i+1))
    
    binders = list(set(binders))


    ## compare with real values now
    true_vals = pd.read_csv('./benchmark_values_misc/NICP_invivo-data.csv')
    #DBD:Jerala_P1:AD:Jerala_P1
    #true_vals['on_target'] = true_vals.apply(lambda row: abs(row['Peptide1'] - row['Peptide2'])  == 1, axis =1 ) 
    true_vals['str_1'] = true_vals['Peptide1'].apply(lambda x: 'Jerala_P' + str(x))
    true_vals['str_2'] = true_vals['Peptide2'].apply(lambda x: 'Jerala_P' + str(x))
    true_vals['PPI'] = 'DBD:' + true_vals['str_2'] + ':AD:' + true_vals['str_1'] #[df.PPI == 'DBD:' + dbd + ':AD:' + ad]
    true_vals['on_target'] = true_vals.PPI.apply(lambda x: x in target_orientations)
    #print (true_vals.on_target.value_counts())
    
    #make folded version with avg fold activation 
    split_for_graphing = split_by_orientations(true_vals, binders, 'Fold activation', True)
    split_for_graphing['avg_fa'] = split_for_graphing.apply( lambda row: sum([row['Fold activation_DBDAD'], row['Fold activation_ADDBD']]) * 0.5, axis = 1)
    split_for_graphing['max_fa'] = split_for_graphing.apply( lambda row: max([row['Fold activation_DBDAD'], row['Fold activation_ADDBD']]), axis = 1)

    return target_orientations, target_ppis, true_vals, split_for_graphing


def get_puma_values():
    #pumas 

    KD_MCL1_PUMA_M144A = 0.181 * 10**-9#nM
    KD_MCL1_PUMA_M144A_error = 0.017 * 10**-9#nM
    KD_MCL1_PUMA_M144A_L148A = 102* 10**-9#nM
    KD_MCL1_PUMA_M144A_L148A_error = 8 * 10**-9#nM
    KD_MCL1_PUMA_M144A_L141A = 432 * 10**-9#nM
    KD_MCL1_PUMA_M144A_L141A_error = 32 * 10**-9#nM
    KD_mALB8_A_mALB8_B = 1.0* 10**-9#nM
    KD_mALB8_A_mALB8_B_error = 1.0* 10**-9#nM

    colors = ['red','red','red','red']
    ppis =  [('MCL1', 'PUMA_M144A'), ('MCL1','PUMA_M144A_L141A'), ('MCL1', 'PUMA_M144A_L148A'), 
             ('mALb8_A', 'mALb8_B')]
    actual_ids = []
    for ppi in ppis:
        id_ppi = [ppi[0], ppi[1]]
        id_ppi.sort()
        actual_ids.append(":".join(id_ppi))
    print (actual_ids)
    kds_df = pd.DataFrame({'PPI':actual_ids, 'kd':[KD_MCL1_PUMA_M144A,
                                                  KD_MCL1_PUMA_M144A_L148A,
                                                  KD_MCL1_PUMA_M144A_L141A,
                                                 KD_mALB8_A_mALB8_B],
                          'kd_error': [KD_MCL1_PUMA_M144A_error,
                                       KD_MCL1_PUMA_M144A_L148A_error,
                                       KD_MCL1_PUMA_M144A_L141A_error,
                                      KD_mALB8_A_mALB8_B_error]})
    return kds_df

def fix_ppi(ppi):
    dbd = ppi.split(':')[0]
    ad = ppi.split(':')[1]
    ppi_name = [dbd, ad]
    ppi_name.sort()
    ppi_name = ":".join(ppi_name)
    return ppi_name

def make_ppi(one, two):
    ppi = [one, two]
    ppi.sort()
    return ':'.join(ppi)

def bcl_binders():

    davids_with_errors = pd.read_csv('./benchmark_values_misc/flat_format_david.csv')
    davids_with_errors['PPI'] = davids_with_errors['PPI'].apply(lambda x: fix_ppi(x))
    steph_with_errors = pd.read_csv('./benchmark_values_misc/flat_format_steph.csv')
    steph_with_errors['PPI'] = steph_with_errors['PPI'].apply(lambda x: fix_ppi(x))

    bcls = ['Bcl-2', 'Bcl-xL','Bcl-w','Bfl-1', 'Bcl-B','Mcl1[151-321]']

    on_target_interactions = [
                          ['Bcl-2','alphaBCL2'],
                          ['Mcl1[151-321]','alphaMCL1'],
                          ['Bfl-1','alphaBFL1'],
                          ['Bcl-B','alphaBCLB']
                         ]
    on_target_ppis = []
    on_traget_ors = []
    #F-ECM04 and B-ECM01 ['Bcl-B','BECM01']
    for ot in on_target_interactions:
        ot.sort()
        #print (ot)
        on_target_ppis.append(':'.join(ot))
        on_traget_ors.append('DBD:' + ot[0] + ':AD:' + ot[1])
        on_traget_ors.append('DBD:' + ot[1] + ':AD:' + ot[0])
        
    #FECM04
    promisc = [['FECM04', x] for x in bcls] + [['BECM01', x] for x in bcls]

    promisc_ppis = []
    #F-ECM04 and B-ECM01 ['Bcl-B','BECM01']
    for ot in promisc:
        ot.sort()
        promisc_ppis.append(':'.join(ot))
    
    def mark_type(x):
        if x in on_target_ppis:
            return 'ON'
        if x in promisc_ppis:
            return 'PRO'
        return 'OFF'
    steph_with_errors['capped_out'] = steph_with_errors.count_error_bars.isna()

    davids_with_errors = davids_with_errors[['PPI', 'pairwise', 'batched']].copy()
    davids_with_errors['pairwise_val'] = davids_with_errors.pairwise.apply(lambda x: float(x.split('+')[0]))
    davids_with_errors['pairwise_error'] = davids_with_errors.pairwise.apply(lambda x: float(x.split('+')[1]))
    davids_with_errors['batched_val'] = davids_with_errors.batched.apply(lambda x: float(x.split('+')[0]))
    davids_with_errors['batched_error'] = davids_with_errors.batched.apply(lambda x: float(x.split('+')[1]))
    davids_with_errors['coef_error_d'] = np.abs(davids_with_errors['pairwise_error']/davids_with_errors['pairwise_val'])
    steph_with_errors['coef_error_s'] = np.abs(steph_with_errors['count_error_bars']/steph_with_errors['count'])
    merger = davids_with_errors.merge(steph_with_errors, on = 'PPI')

    merger['type'] = merger.PPI.apply(lambda x: mark_type(x)) #on_target_ppis)
    return merger, on_target_ppis, on_traget_ors


def woolfson_plaper_ppi_kds():

    KD_N5_N6 = 1.0 * 10**-9#nM
    KD_N5_N6_error = 0.5 * 10**-9#nM
    KD_N7_N8 = 10.0 * 10**-9#nM
    KD_N7_N8_error = 2 * 10**-9#nM
    KD_P5A_P6A = 4.0 * 10**-9#nM
    KD_P5A_P6A_error = 0.5 * 10**-9#nM
    KD_P7A_P8A = 19.0  * 10**-9#nM
    KD_P7A_P8A_error = 2 * 10**-9#nM
    KD_N1_N2 = 14.0  * 10**-9#nM
    KD_N1_N2_error = 1 * 10**-9#nM

    KD_BN3_AN3 = 3.10 * 10**-6 #nM
    KD_BN3_AN3_error = 0.58 * 10 **-6
    KD_BN3_AN35 = 2.79 * 10 **-7 #nM
    KD_BN3_AN35_error = 1.42 * 10**-7

    KD_BN3_AN4 = 7.47 * 10**-8#nM
    KD_BN3_AN4_error =3.30 * 10**-8 

    KD_BN35_AN3 = 6.40 * 10**-7 #nM
    KD_BN35_AN3_error = 2.77 * 10**-7
    KD_BN35_AN35 = 5.15 *10**-9#nM
    KD_BN35_AN35_error = 2.05*10**-9
    KD_BN35_AN4 =  2.08*10**-10#nM
    KD_BN35_AN4_error = 0.73*10**-10

    KD_BN4_AN3 = 3.87*10**-7 #nM
    KD_BN4_AN3_error = 1.45*10**-7
    KD_BN4_AN35 = 4.74*10**-10 #nM
    KD_BN4_AN35_error = 4.49*10**-10
    KD_BN4_AN4 = 1*10**-10 #NOTE THIS IS A UPPERBOUND, less than this...
    KD_BN4_AN4_error = 1*10**-10

    ppis = [('N5', 'N6'), ('N7','N8'), ('P5A', 'P6A'), ('P7A', 'P8A'),  ('N1', 'N2'), ('BN3', 'AN3'), ('BN3', 'AN35'), ('BN3', 'AN4'), 
           ('BN35', 'AN3'), ('BN35', 'AN35'), ('BN35', 'AN4'), ('BN4', 'AN3'), ('BN4', 'AN35'), ('BN4', 'AN4')] #('N1', 'N2'),
    actual_ids = []
    for ppi in ppis:
        id_ppi = [ppi[0], ppi[1]]
        id_ppi.sort()
        actual_ids.append(":".join(id_ppi))
    kds_df = pd.DataFrame({'PPI':actual_ids, 'kd':[KD_N5_N6,KD_N7_N8,KD_P5A_P6A, KD_P7A_P8A,KD_N1_N2,
                                                      KD_BN3_AN3, KD_BN3_AN35,KD_BN3_AN4,
                                                      KD_BN35_AN3,KD_BN35_AN35,KD_BN35_AN4,
                                                      KD_BN4_AN3,KD_BN4_AN35,KD_BN4_AN4
                                                     ], 'kd_error': [KD_N5_N6_error,KD_N7_N8_error,KD_P5A_P6A_error, 
                                                                     KD_P7A_P8A_error,KD_N1_N2_error,
                                                KD_BN3_AN3_error, KD_BN3_AN35_error,KD_BN3_AN4_error,
                                                      KD_BN35_AN3_error,KD_BN35_AN35_error,KD_BN35_AN4_error,
                                                      KD_BN4_AN3_error,KD_BN4_AN35_error,KD_BN4_AN4_error
                                                     ]})

    kds_df['-ln_kd'] = kds_df['kd'].apply(lambda x: -np.log10(x))
    kds_df['set'] = ['plaper(covid)'] * 5 + ['woolfson'] * 9 
    return kds_df

def plaper_other_values():
    load_tms = pd.read_csv('./benchmark_values_misc/plapers_true.csv')
    #load_tms['PPI'] = load_tms.apply(lambda row: make_ppi(row['nLuc'], row['cLuc']), axis = 1)
    return load_tms