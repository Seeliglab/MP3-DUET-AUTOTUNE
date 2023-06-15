#ALyssa La Fleur
#Correlation with kd values for the truncated coil pair
#6/14/2023: updated to use all three replicates 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

woolfsons = ['AN4', 'BN4', 'AN35', 'BN35', 'AN3', 'BN3']

#sets of interactions
woolfson_woolfson_ppis = []

for w in woolfsons:
    for w2 in woolfsons:
        ppi = [w, w2]
        ppi.sort()
        woolfson_woolfson_ppis.append(':'.join(ppi))

on_targ = ['N5:N6', 'N7:N8', 'P5A:P6A', 'P7A:P8A']
just_kds = woolfson_plaper_ppi_kds()
just_kds['on_target'] = just_kds.PPI.isin(on_targ)
just_kds['kd_nm'] = just_kds['kd']/(10**-9)
just_kds['kd_error_nm'] = just_kds['kd_error']/(10**-9)

#load values for comparison 
deseq_homo_af_batch = pd.read_csv('../processing_pipeline/merged_replicates/deseq_plaper_3_smaller_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})

merge_kds_d = deseq_homo_af_batch.merge(just_kds, on = 'PPI')
merge_kds_d['true_p'] = 1 - merge_kds_d['ashr_padj_HIS_TRP'].fillna(1)

merge_kds_d = merge_kds_d[merge_kds_d.set == 'woolfson']


off_targets = "#782167ff"

f, ax = plt.subplots()
plt.errorbar(y = merge_kds_d['kd_nm'],
             x = merge_kds_d['ashr_log2FoldChange_HIS_TRP'], 
             xerr= merge_kds_d['ashr_lfcSE_HIS_TRP'], 
             yerr= merge_kds_d['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.errorbar(y = merge_kds_d.iloc[-1]['kd_nm'],
             x = merge_kds_d.iloc[-1]['ashr_log2FoldChange_HIS_TRP'], 
             xerr= merge_kds_d.iloc[-1]['ashr_lfcSE_HIS_TRP'], 
             yerr= merge_kds_d.iloc[-1]['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = 'black', capsize = 3, markersize = 5, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(2,2)
plt.savefig('./figures/woolfson_kd_correlations.svg', dpi = 300)
plt.show()

kds_no_nans = merge_kds_d[['PPI', 'ashr_log2FoldChange_HIS_TRP','kd_nm']].dropna()
print (kds_no_nans.shape)

print(stats.pearsonr(kds_no_nans['ashr_log2FoldChange_HIS_TRP'],
                     np.log(kds_no_nans['kd_nm']))[0]**2)

print(stats.spearmanr(kds_no_nans['ashr_log2FoldChange_HIS_TRP'],
                     np.log(kds_no_nans['kd_nm']))[0])


#making heatmap for Figure 1
flat_df = pd.read_csv('../processing_pipeline/merged_replicates/deseq_plaper_3_smaller_flat_autotune.csv')
flat_df = flat_df.rename(columns = {'Unnamed: 0': 'PPI'})

#fill in missing AN4:AN4 homodimer in reps 1 and 2 with l68 value 
flat_df = pd.read_csv('../processing_pipeline/merged_replicates/deseq_plaper_3_smaller_flat_autotune.csv')
flat_df = flat_df.rename(columns = {'Unnamed: 0': 'PPI'})

woolfsons = ['AN4', 'AN35',  'AN3', 'BN4','BN35', 'BN3' ]

#save heatmap 
make_square_heatmap(make_specific_order(woolfsons, flat_df, 'ashr_log2FoldChange_HIS_TRP'),
                                           float('-inf'), woolfsons, 'bone_r', False, 2.5, 2, 'woolfson_heatmap.svg', )
