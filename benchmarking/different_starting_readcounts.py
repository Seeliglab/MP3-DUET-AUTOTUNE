#Alyssa La Fleur

#Demonstrating the effect of different starting minimum readcount values on DESeq2 analysis and correlations 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
from processing_functions import *
from comparison_datasets import *

import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
woolfsons = ['AN4', 'AN35', 'AN3', 'BN4', 'BN35', 'BN3']

just_kds = woolfson_plaper_ppi_kds()
just_kds['kd_nm'] = just_kds['kd']/(10**-9)
just_kds['kd_error_nm'] = just_kds['kd_error']/(10**-9)
just_kds['log2_enrich'] = np.log(just_kds['kd_nm'])

#load values for comparison 
deseq_homo_af_batch = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l68_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_1 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l68_psuedoreplicate_autotune_f_1.csv')
deseq_homo_af_batch_f_1 = deseq_homo_af_batch_f_1.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_3 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l68_psuedoreplicate_autotune_f_3.csv')
deseq_homo_af_batch_f_3 = deseq_homo_af_batch_f_3.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_5 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l68_psuedoreplicate_autotune_f_5.csv')
deseq_homo_af_batch_f_5 = deseq_homo_af_batch_f_5.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_10 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l68_psuedoreplicate_autotune_f_10.csv')
deseq_homo_af_batch_f_10 = deseq_homo_af_batch_f_10.rename(columns = {'Unnamed: 0': 'PPI'})

merged = deseq_homo_af_batch.merge(deseq_homo_af_batch_f_1, on = 'PPI', suffixes= ['_f_0', '_f_1'], how = 'left')
merged = deseq_homo_af_batch_f_3.merge(merged, on = 'PPI', how = 'right')
merged = deseq_homo_af_batch_f_5.merge(merged, on  = 'PPI', suffixes = [ '_f_5', '_f_3'])
merged = deseq_homo_af_batch_f_10.merge(merged, on = 'PPI', how = 'right')
merged = just_kds.merge(merged, on  = 'PPI', suffixes = [ '_kd', ''], how = 'right')

plot_r2_woolfson = []
plot_rho_woolfson = []
plot_n_woolfson = []

pairs = ['_f_0', '_f_1', '_f_3', '_f_5', '']
for p in pairs:
    vals = get_correls(merged[merged.set == 'woolfson'], 'ashr_log2FoldChange_HIS_TRP' + p,  'kd_nm', True)
    plot_r2_woolfson.append(vals[2])
    plot_rho_woolfson.append(vals[3])
    plot_n_woolfson.append(vals[1])
    print (get_correls(merged[merged.set == 'woolfson'], 'ashr_log2FoldChange_HIS_TRP' + p,  'kd_nm', True))

#Jerala 
jerala_pros = ['Jerala_P1', 'Jerala_P2', 'Jerala_P3', 'Jerala_P4', 'Jerala_P5', 'Jerala_P6', 'Jerala_P7', 'Jerala_P8', 'Jerala_P9', 'Jerala_P10', 'Jerala_P11', 'Jerala_P12']

#Jerala libraries
jerala_target_ors, jerala_target_ppis, jerala_true_flat, jerala_true_folded = get_jerala_values()

deseq_homo_af_batch = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_1 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_f_1.csv')
deseq_homo_af_batch_f_1 = deseq_homo_af_batch_f_1.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_f_3.csv')
deseq_homo_af_batch_f_3 = deseq_homo_af_batch_f_3.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_5 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_f_5.csv')
deseq_homo_af_batch_f_5 = deseq_homo_af_batch_f_5.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch_f_10 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_f_10.csv')
deseq_homo_af_batch_f_10 = deseq_homo_af_batch_f_10.rename(columns = {'Unnamed: 0': 'PPI'})

merged = deseq_homo_af_batch.merge(deseq_homo_af_batch_f_1, on = 'PPI', suffixes= ['_f_0', '_f_1'], how = 'left')
merged = deseq_homo_af_batch_f_3.merge(merged, on = 'PPI', how = 'right')
merged = deseq_homo_af_batch_f_5.merge(merged, on  = 'PPI', suffixes = [ '_f_5', '_f_3'], how = 'right')
merged = deseq_homo_af_batch_f_10.merge(merged, on = 'PPI', how = 'right')
merged = jerala_true_flat.merge(merged, on  = 'PPI', suffixes = [ '_kd', ''], how = 'right')

plot_r2_jerala = []
plot_rho_jerala = []
plot_n_jerala = []
pairs = ['_f_0', '_f_1', '_f_3', '_f_5', '']
for p in pairs:
    vals = get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + p,  'Fold activation', True)
    plot_r2_jerala.append(vals[2])
    plot_rho_jerala.append(vals[3])
    plot_n_jerala.append(vals[1])
    print (get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + p,  'Fold activation', True))

#make line plot 
# data1&data2

cutoff = [0,10,20,30,40]
# plot graphs 
off_targets = "#782167ff"
f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows =4, ncols = 1, sharex = True)
ax1.plot(cutoff, plot_r2_jerala, color = 'gray', alpha = 0.5)
ax2.plot(cutoff, plot_rho_jerala, color = 'gray', alpha = 0.5)
ax3.plot(cutoff, plot_r2_woolfson, color = 'gray', alpha = 0.5)
ax4.plot(cutoff, plot_rho_woolfson, color = 'gray', alpha = 0.5)
sns.scatterplot(ax = ax1, x = cutoff, linestyle = '-', y = plot_r2_jerala, size = plot_n_jerala, color = off_targets)
sns.scatterplot(ax = ax2, x = cutoff, linestyle = '-', y = plot_rho_jerala, size = plot_n_jerala, color = off_targets)
sns.scatterplot(ax = ax3, x = cutoff, linestyle = '-', y = plot_r2_woolfson, size = plot_n_woolfson, color = off_targets)
sns.scatterplot(ax = ax4, x = cutoff, linestyle = '-', y = plot_rho_woolfson, size = plot_n_woolfson, color = off_targets)
#plt.tight_layout()
f.set_size_inches(5,2)
plt.savefig('./figures/jerala_r2_changes_size.svg', dpi = 300)
plt.show()