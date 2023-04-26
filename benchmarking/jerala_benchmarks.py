#Alyssa La Fleur
#Figure 1 Jerala content

from processing_functions import *
from comparison_datasets import *
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)


jerala_pros = ['Jerala_P1',
 'Jerala_P2',
 'Jerala_P3',
 'Jerala_P4',
 'Jerala_P5',
 'Jerala_P6',
 'Jerala_P7',
 'Jerala_P8',
 'Jerala_P9',
 'Jerala_P10',
 'Jerala_P11',
 'Jerala_P12']


#Make heatmap of measurements
#all 3
flat_df_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat.csv')
print (flat_df_3)
flat_df_3.rename(columns = {'Unnamed: 0':'PPI'}, inplace = True)
all_3s = make_specific_order(jerala_pros, flat_df_3, 'ashr_log2FoldChange_HIS_TRP')
print (all_3s)

#all 2
flat_df = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_2_flat.csv')
flat_df.rename(columns = {'Unnamed: 0':'PPI'}, inplace = True)
all_2s = make_specific_order(jerala_pros, flat_df, 'ashr_log2FoldChange_HIS_TRP')

#for those with nans, fill with 2 values 
all_2s[np.isnan(all_3s)]
all_3s[np.isnan(all_3s)] = all_2s[np.isnan(all_3s)]
print (all_3s)

make_dict = {}
for i in range(0, len(jerala_pros)):
    make_dict[jerala_pros[i]] = all_2s[:,i]
pro_series = pd.Series(jerala_pros)
as_pd_df = pd.DataFrame(make_dict)
as_pd_df.set_index(pro_series, inplace= True)
as_pd_df.to_csv('jerala_ashr_lfc_vals.csv')
#save heatmap 
make_square_heatmap(all_3s, float('-inf'), jerala_pros, 'bone_r', False,2.5,2, 'jerala_heatmap.svg')


jerala_target_ors, jerala_target_ppis, jerala_true_flat, jerala_true_folded = get_jerala_values()

#all 3
on_targets = "#ffcc00ff"
off_targets = "#782167ff"
f, ax = plt.subplots()
subset = jerala_true_flat.merge(flat_df_3, on = 'PPI')
print (subset.on_target.value_counts())
plt.errorbar(y = subset[subset.on_target == False]['Fold activation'],
             x = subset[subset.on_target == False]['ashr_log2FoldChange_HIS_TRP'], 
             xerr= subset[subset.on_target == False]['ashr_lfcSE_HIS_TRP'], 
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.errorbar(y = subset[subset.on_target == True]['Fold activation'],
             x = subset[subset.on_target == True]['ashr_log2FoldChange_HIS_TRP'], 
             xerr= subset[subset.on_target == True]['ashr_lfcSE_HIS_TRP'], 
             fmt="o", elinewidth = 2, markersize = 5, alpha = 0.9, color = on_targets,capsize = 3, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(1.7,2.3)
plt.savefig('./figures/' + 'jerala_all_vals_3_reps_only.svg', dpi = 300)
plt.show()

print(get_correls(subset, 'ashr_log2FoldChange_HIS_TRP', 'Fold activation', log=True))
print(get_correls(subset, 'ashr_log2FoldChange_HIS_TRP', 'Fold activation', log=False))


