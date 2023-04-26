#AML
#permutation tests to demonstrate that the homolog TRP read counts are in general lower than heterodimer read counts 

import os
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns 
from scipy.stats import permutation_test
from scipy.stats import ttest_ind
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

library_inputs_list = []



library_inputs_list.append(("../data/final_mp3seq_method/l61/",            
                            'L61_TRP_44h_table.csv',
                            ))
library_inputs_list.append(("../data/final_mp3seq_method/l62/",
                            
                            'L62_TRP_table.csv',
                            ))

library_inputs_list.append(("../data/final_mp3seq_method/l67/",
                            
                            'L67_TRP_table.csv',
                            ))

# Plaper N and PA series, Thomas AN and BN coils
library_inputs_list.append(("../data/final_mp3seq_method/l68/",
                           
                            'L68_TRP_table.csv',
                           ))

# Bcl-2 designed binders
library_inputs_list.append(("../data/final_mp3seq_method/l66/",
                            
                            'L66_TRP_table.csv',
                            ))
# mALb8 deleted hydrogen bond networks
library_inputs_list.append(("../data/final_mp3seq_method/l70/",
                            
                            'L70_TRP_table.csv',
                            ))
# older method screens

# DHD1 + DHD0
library_inputs_list.append(("../data/older_mp3seq_method/l39_1/",
                            
                            'L39_TRP_table.csv',
                           ))
library_inputs_list.append(("../data/older_mp3seq_method/l39_2/",
                            
                            'L39_TRP_table.csv',
                          ))
library_inputs_list.append(("../data/older_mp3seq_method/l39_3/",
                           
                            'L39_TRP_table.csv',
                            ))
# DHD0 + DHD2 + mALb
library_inputs_list.append(("../data/older_mp3seq_method/l43/",
                            
                            'L43_TRP1_table.csv',
                            ))
# DHD0-3 + mALb
library_inputs_list.append(("../data/older_mp3seq_method/l44/",
                          
                            'L44_TRP_table.csv',
                            ))
library_inputs_list.append(("../data/older_mp3seq_method/l49/",
                           
                            'L49_TRP_table.csv',
                           ))
# Bcl-2 designed 4
library_inputs_list.append(("../data/older_mp3seq_method/l45/",
                            
                            'L45_TRP_V1_table.csv',
                            ))
library_inputs_list.append(("../data/older_mp3seq_method/l48/",
                           
                            'L48_TRP_table.csv',
                            ))



def make_homo_and_het_reshuffle_matrix(library_list):
    total_trp_count_dfs = []
    for l in library_list:
        trp_counts = pd.read_csv(l[0] + l[1])
        trp_counts = trp_counts.rename(columns={'Binder1': 'Binder'})
        trp_counts = trp_counts.fillna(0)
        binder_list = trp_counts.Binder.to_list()
        final_counts = []
        final_binder_pairs = []
        for b in binder_list:
            for b2 in binder_list:
                if b2 in trp_counts:
                    val_row = trp_counts[trp_counts.Binder == b][b2].to_list()
                    if len(val_row) > 0:
                        final_counts.append(val_row[0])
                        final_binder_pairs.append("DBD:" + b + ':AD:' + b2)
        trp_counts_acutal = pd.DataFrame({'PPI': final_binder_pairs, 'count': final_counts})
        # save all unique binders in the screening
        trp_counts_acutal['dbd'] = trp_counts_acutal['PPI'].apply(lambda x: x.split(':')[1])
        trp_counts_acutal['ad'] = trp_counts_acutal['PPI'].apply(lambda x: x.split(':')[3])
        trp_counts_acutal['homodimer'] = trp_counts_acutal['dbd'] == trp_counts_acutal['ad']
        print (trp_counts_acutal.homodimer.value_counts())
        #correct trp counts to fractional ones 
        trp_counts_acutal['count'] = trp_counts_acutal['count']/trp_counts_acutal['count'].sum()
        #print (trp_counts_acutal)
        trp_counts_acutal = trp_counts_acutal.fillna(0)
        min_max_scaler =  MinMaxScaler()
        transformed = min_max_scaler.fit_transform(trp_counts_acutal['count'].to_numpy().reshape(-1,1))
        #transformed = min_max_scaler.transform(trp_counts_acutal['count'].to_numpy().reshape(1,-1))
        #print (transformed)
        trp_counts_acutal['count'] = transformed
        total_trp_count_dfs.append(trp_counts_acutal)
    #make one df and return it 
    return pd.concat(total_trp_count_dfs)

def save_all_trp_values(library_inputs_list):
    if not os.path.exists('all_trp_counts.csv'):
        all_trp_values = make_homo_and_het_reshuffle_matrix(library_inputs_list)
        print (all_trp_values)
        all_trp_values.to_csv('all_trp_counts.csv', index = False)

#t-test 
#reject null, homodimers mean trp is less than over proteins (very high significance)
all_trp_values = pd.read_csv('all_trp_counts.csv')
print (ttest_ind(all_trp_values[all_trp_values.homodimer]['count'].to_numpy(), all_trp_values[~all_trp_values.homodimer]['count'].to_numpy(), alternative = 'less'))

#run per-assay permutation test and overall one 


def permutation_values(df_counts, n_to_sample):
    homo_val = np.average(df_counts[df_counts.homodimer]['count'].to_numpy())
    n_homos = df_counts[df_counts.homodimer].shape[0]
    samples = []
    for s in range(0, n_to_sample):
        #make a random sample
        sample_counts = np.random.choice(df_counts['count'].to_numpy(), n_homos, replace = False)
        samples.append(sample_counts.mean())
    return homo_val, samples
    


def make_homo_and_het_reshuffle_plot(library_list, n = 1000):
    fig, square_axs = plt.subplots(4, 4, tight_layout=True, sharey='row')
    locs = np.arange(16).reshape(4,4)        
    print (square_axs[0])
    for i in range(0, len(library_list)):
        l = library_list[i]
        l_name = l[0].split('/')[-2]
        print (l_name)
        trp_counts = pd.read_csv(l[0] + l[1])
        trp_counts = trp_counts.rename(columns={'Binder1': 'Binder'})
        trp_counts = trp_counts.fillna(0)
        binder_list = trp_counts.Binder.to_list()
        final_counts = []
        final_binder_pairs = []
        for b in binder_list:
            for b2 in binder_list:
                if b2 in trp_counts:
                    val_row = trp_counts[trp_counts.Binder == b][b2].to_list()
                    if len(val_row) > 0:
                        final_counts.append(val_row[0])
                        final_binder_pairs.append("DBD:" + b + ':AD:' + b2)
        trp_counts_acutal = pd.DataFrame({'PPI': final_binder_pairs, 'count': final_counts})
        # save all unique binders in the screening
        trp_counts_acutal['dbd'] = trp_counts_acutal['PPI'].apply(lambda x: x.split(':')[1])
        trp_counts_acutal['ad'] = trp_counts_acutal['PPI'].apply(lambda x: x.split(':')[3])
        trp_counts_acutal['homodimer'] = trp_counts_acutal['dbd'] == trp_counts_acutal['ad']
        print (trp_counts_acutal.homodimer.value_counts())
        #correct trp counts to fractional ones 
        trp_counts_acutal['count'] = trp_counts_acutal['count']/trp_counts_acutal['count'].sum()
        #print (trp_counts_acutal)
        trp_counts_acutal = trp_counts_acutal.fillna(0)
        min_max_scaler =  MinMaxScaler()
        transformed = min_max_scaler.fit_transform(trp_counts_acutal['count'].to_numpy().reshape(-1,1))
        trp_counts_acutal['count'] = transformed

        homo_val, samples = permutation_values(trp_counts_acutal, n)
        sample_p_val = sum([homo_val <= s for s in samples])
        
        # We can also normalize our inputs by the total number of counts
        #locs = np.where(locs == i)
        print (l_name, str(1 - round(sample_p_val/n , 2)))
        if i >= 6:
            square_axs[np.where(locs == i)[0][0],np.where(locs == i)[1][0]].hist(samples, density=False, color = '#0000bcff')
        else:
            square_axs[np.where(locs == i)[0][0],np.where(locs == i)[1][0]].hist(samples, density=False, color = '#00aad4ff')
        square_axs[np.where(locs == i)[0][0],np.where(locs == i)[1][0]].axvline(homo_val, color = '#c8ab37ff')
        
    plt.savefig('permutation_test_homodimer_means_no_share_x.svg')
    plt.close()


#first 6 are final method libraries
make_homo_and_het_reshuffle_plot(library_inputs_list)
print (len(library_inputs_list))