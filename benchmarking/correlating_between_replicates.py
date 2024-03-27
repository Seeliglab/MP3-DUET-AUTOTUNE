from processing_functions import *
from comparison_datasets import *
import matplotlib as mpl
import seaborn as sns
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
"""
jerala_pros = ['Jerala_P1', 'Jerala_P2', 'Jerala_P3', 'Jerala_P4', 'Jerala_P5', 'Jerala_P6', 'Jerala_P7', 'Jerala_P8', 'Jerala_P9', 'Jerala_P10', 'Jerala_P11', 'Jerala_P12']

#Jerala libraries
jerala_target_ors, jerala_target_ppis, jerala_true_flat, jerala_true_folded = get_jerala_values()
jerala_true_flat['ashr_log2FoldChange_HIS_TRP'] = np.log(jerala_true_flat['Fold activation'])
jerala_true_folded_avg = jerala_true_folded[['PPI', 'avg_fa']].copy()
jerala_true_folded_avg['ashr_log2FoldChange_HIS_TRP'] = np.log(jerala_true_folded_avg['avg_fa'])
jerala_true_folded_max = jerala_true_folded[['PPI', 'max_fa']].copy()
jerala_true_folded_max['ashr_log2FoldChange_HIS_TRP'] = np.log(jerala_true_folded_max['max_fa'])

#enrichment correlations vs LFC and P-LFC
replicate_1_flat_no_autoact = pd.read_csv('../processing_pipeline/processed_replicates/l61_2mM_flat_no_autoactivators.csv')
replicate_1_flat_no_autoact =  process_fusion_dataframe(replicate_1_flat_no_autoact)
replicate_2_flat_no_autoact = pd.read_csv('../processing_pipeline/processed_replicates/l62_2mM_flat_no_autoactivators.csv')
replicate_2_flat_no_autoact =  process_fusion_dataframe(replicate_2_flat_no_autoact)
replicate_3_flat_no_autoact = pd.read_csv('../processing_pipeline/processed_replicates/l67_flat_no_autoactivators.csv')
replicate_3_flat_no_autoact =  process_fusion_dataframe(replicate_3_flat_no_autoact)
replicate_4_flat_no_autoact = pd.read_csv('../processing_pipeline/processed_replicates/new_1_smaller_flat_no_autoactivators.csv')
replicate_4_flat_no_autoact =  process_fusion_dataframe(replicate_4_flat_no_autoact)
replicate_5_flat_no_autoact = pd.read_csv('../processing_pipeline/processed_replicates/new_2_smaller_flat_no_autoactivators.csv')
replicate_5_flat_no_autoact =  process_fusion_dataframe(replicate_5_flat_no_autoact)


#merge together
merged = replicate_1_flat_no_autoact.merge(replicate_2_flat_no_autoact, on = 'PPI', suffixes= ['_1', '_2'])
merged = replicate_3_flat_no_autoact.merge(merged, on = 'PPI', how = 'right')
merged = merged.merge(replicate_4_flat_no_autoact, on = 'PPI', suffixes=['_3', '_4'])
merged = merged.merge(replicate_5_flat_no_autoact, on = 'PPI')
merged['log2_enrich_avg'] = merged.apply(lambda row: np.log2(np.nanmean([row['lin_enrich'], row['lin_enrich_1'], row['lin_enrich_2'], row['lin_enrich_3'], row['lin_enrich_4']])), axis = 1)
merged['log2_enrich_max'] = merged.apply(lambda row: np.log2(np.max([row['lin_enrich'], row['lin_enrich_1'], row['lin_enrich_2'],row['lin_enrich_3'], row['lin_enrich_4']])), axis = 1)

#merge in flat df wtih and without autoactivators for all 3 replicates and flat AF values 
replicate_all_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat.csv')
replicate_all_3 = replicate_all_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_all_3['log2_enrich'] = replicate_all_3['ashr_log2FoldChange_HIS_TRP']
replicate_all_3_at = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_flat_autotune_5_small.csv')
replicate_all_3_at = replicate_all_3_at.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_all_3_at['log2_enrich'] = replicate_all_3_at['ashr_log2FoldChange_HIS_TRP']
jerala_true_flat['log2_enrich'] = np.log(jerala_true_flat['Fold activation'])

merged = replicate_all_3.merge(merged, on = 'PPI', suffixes= ['_lfc_noauto', '_5'], how = 'right')
merged = replicate_all_3_at.merge(merged, on = 'PPI', how = 'right')
merged = jerala_true_flat.merge(merged, on  = 'PPI', suffixes = [ '_FA', '_lfc_autotune'])
print (merged.shape)

combos = list(combinations(['_1', '_2', '_3', '_4', '_5', '_avg', '_max', '_lfc_autotune', '_FA'], 2))

df_to_graph = pd.DataFrame({'PPI':[], 'correl':[], 'rho':[]})
for a, b in combos:
        df_to_graph.loc[df_to_graph.shape[0]] = [make_ppi(a,b), get_correls(merged, 'log2_enrich' + a, 'log2_enrich' + b, False)[2], 
                                                 get_correls(merged, 'log2_enrich' + a, 'log2_enrich' + b, False)[3]]

order_libraries = ['_1', '_2', '_3', '_4', '_5', '_avg', '_max', '_lfc_autotune', '_FA']
graph_labels = ['R1', 'R2', 'R3', 'R3', 'R4','Avg E', '_Max E', 'LFC', 'FA']
x = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'correl',  get_diag = False)
y = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'rho',  get_diag = False)
y = y.T
make_double_heatmap(x, y, graph_labels, 'copper_r', 'bone_r', size_1 = 3.5, size_2 = 3.5,show_names = False, annot = True, font_size = 8, saveName= 'correls_flat_enrich_v_lfc.svg', show=False)

merged_flat = merged.copy()

#P-LFC correlations 
#Correlating individual replicates (for different processing versions)
replicate_1 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l61_2mM_psuedoreplicate_autotune.csv')
replicate_1 = replicate_1.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_2  = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l62_2mM_psuedoreplicate_autotune.csv')
replicate_2 = replicate_2.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_3  = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l67_psuedoreplicate_autotune.csv')
replicate_3 = replicate_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_1_2 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_2_psuedoreplicate_autotune.csv')
replicate_1_2 = replicate_1_2.rename(columns = {'Unnamed: 0': 'PPI'}) 
replciate_2_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_l62_l67_psuedoreplicate_autotune.csv')
replciate_2_3 = replciate_2_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_1_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_l61_l67_psuedoreplicate_autotune.csv')
replicate_1_3 = replicate_1_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_all_3 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_psuedoreplicate_autotune.csv')
replicate_all_3 = replicate_all_3.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_4 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_new_1_smaller_psuedoreplicate_autotune.csv')
replicate_4 = replicate_4.rename(columns = {'Unnamed: 0': 'PPI'}) 
replicate_5 = pd.read_csv('../processing_pipeline/merged_replicates/deseq_new_2_smaller_psuedoreplicate_autotune.csv')
replicate_5 = replicate_5.rename(columns = {'Unnamed: 0': 'PPI'}) 



merged = replicate_1.merge(replicate_2, on = 'PPI', suffixes = ['_1', '_2'])
merged = replicate_3.merge(merged, on ='PPI')
merged = replicate_1_2.merge(merged, on ='PPI', suffixes = ['_1_2', '_3'])
merged = replciate_2_3.merge(merged, on ='PPI')
merged = replicate_1_3.merge(merged, on ='PPI', suffixes = ['_1_3', '_2_3'])
merged = replicate_all_3.merge(merged, on ='PPI')
merged = jerala_true_folded_avg.merge(merged, on = 'PPI', suffixes= ['_avg_fa', '_all'])
merged = jerala_true_folded_max.merge(merged, on = 'PPI')
merged = merged.merge(replicate_4, on = 'PPI', suffixes = ['_max_fa', '_4'])
merged = merged.merge(replicate_5, on = 'PPI')

#fold and correlate with P-LFC values per replciate
log_2_enrich_avg = split_by_orientations(merged_flat, jerala_pros, 'log2_enrich_avg', True)
log_2_enrich_avg['ashr_log2FoldChange_HIS_TRP'] = log_2_enrich_avg.apply(lambda row: np.nanmean([row['log2_enrich_avg_ADDBD'], row['log2_enrich_avg_DBDAD']]), axis = 1)
merged = log_2_enrich_avg.merge(merged, on = 'PPI', suffixes = ['_avg_enrich', '_5'])
log_2_enrich_avg = split_by_orientations(merged_flat, jerala_pros, 'log2_enrich_max', True)
log_2_enrich_avg['ashr_log2FoldChange_HIS_TRP'] = log_2_enrich_avg.apply(lambda row: np.nanmean([row['log2_enrich_max_ADDBD'], row['log2_enrich_max_DBDAD']]), axis = 1)
merged = log_2_enrich_avg.merge(merged, on = 'PPI')

combos = list(combinations(['_1', '_2', '_3', '_4', '_5', '_all', '_avg_fa', '_max_fa', '_avg_enrich', ''], 2))

df_to_graph = pd.DataFrame({'PPI':[], 'correl':[], 'rho':[]})
for a, b in combos:
        df_to_graph.loc[df_to_graph.shape[0]] = [make_ppi(a,b), get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[2], 
                                                 get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[3]]
order_libraries = ['_1', '_2', '_3', '_4', '_5', '_all', '_avg_enrich', '', '_avg_fa', '_max_fa' ]
graph_labels = ['R1', 'R2', 'R3', 'R4', 'R5',  'R1-5', 'Avg En', 'Max En', 'Avg FA', 'Max FA']
x = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'correl',  get_diag = False)
y = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'rho',  get_diag = False)
y = y.T
make_double_heatmap(x, y, graph_labels, 'copper_r', 'bone_r', size_1 = 4.5, size_2 = 4.5,show_names = False, annot = True, font_size = 8, saveName= 'correls_enrich_v_plfc.svg', show=False)
"""
#woolfson kd P-LFC and enrichemnt comparisons 

woolfsons = ['AN4', 'AN35', 'AN3', 'BN4', 'BN35', 'BN3']

just_kds = woolfson_plaper_ppi_kds()
just_kds['kd_nm'] = just_kds['kd']/(10**-9)
just_kds['kd_error_nm'] = just_kds['kd_error']/(10**-9)
just_kds['log2_enrich'] = np.log(just_kds['kd_nm'])

#load values for comparison 
deseq_homo_af_batch = pd.read_csv('../processing_pipeline/processed_replicates/deseq_plaper_3_smaller_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})
deseq_homo_af_batch['log2_enrich'] = deseq_homo_af_batch['ashr_log2FoldChange_HIS_TRP']

#making heatmap for Figure 1
flat_df = pd.read_csv('../processing_pipeline/processed_replicates/l68_flat.csv')
flat_df['DBD'] = flat_df.PPI.apply(lambda x: x.split(':')[1])
flat_df['AD'] = flat_df.PPI.apply(lambda x: x.split(':')[3])

vals_counts = ['trp1', 'his1']
for vc in vals_counts:
    flat_df.loc[flat_df[vc]  == 0, vc] = None

flat_df['lin_enrich'] = calc_lin_enrichment(flat_df['trp1'], flat_df['his1'])
flat_df['log2_enrich'] = calc_log2_enrichment(flat_df['lin_enrich'])
flat_df.loc[(flat_df.trp1.isna()) & (flat_df.his1.isna()), 'lin_enrich'] = None
flat_df.loc[(flat_df.trp1.isna()) & (flat_df.his1.isna()), 'log2_enrich'] = None

log_2_enrich_avg = split_by_orientations(flat_df, woolfsons, 'log2_enrich', True)
log_2_enrich_avg['log2_enrich'] = log_2_enrich_avg.apply(lambda row: np.nanmean([row['log2_enrich_ADDBD'], row['log2_enrich_DBDAD']]), axis = 1)
log_2_enrich_max = split_by_orientations(flat_df, woolfsons, 'log2_enrich', True)
log_2_enrich_max['log2_enrich'] = log_2_enrich_max.apply(lambda row: np.nanmean([row['log2_enrich_ADDBD'], row['log2_enrich_DBDAD']]), axis = 1)

merged = log_2_enrich_avg.merge(log_2_enrich_max, on = 'PPI', suffixes= ['_avg', '_max'], how = 'right')
merged = deseq_homo_af_batch.merge(merged, on = 'PPI', how = 'right')
merged = just_kds.merge(merged, on  = 'PPI', suffixes = [ '_PLFC', '_kd'])

combos = list(combinations(['_avg', '_max', '_PLFC', '_kd'], 2))

df_to_graph = pd.DataFrame({'PPI':[], 'correl':[], 'rho':[]})
for a, b in combos:
        df_to_graph.loc[df_to_graph.shape[0]] = [make_ppi(a,b), get_correls(merged, 'log2_enrich' + a, 'log2_enrich' + b, False)[2], 
                                                 np.abs(get_correls(merged, 'log2_enrich' + a, 'log2_enrich' + b, False)[3])]

order_libraries = ['_avg', '_max', '_PLFC', '_kd']
x = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'correl',  get_diag = False)
y = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'rho',  get_diag = False)
y = y.T
make_double_heatmap(x, y, order_libraries, 'copper_r', 'bone_r', size_1 = 2.5, size_2 = 2.5, show_names = True, annot = True, font_size = 8, saveName= 'correls_kd_woolfson.svg', show=False)

off_targets = "#782167ff"

f, ax = plt.subplots()
plt.errorbar(y = merged['kd_nm'],
             x = merged['log2_enrich_avg'], 
             yerr= merged['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.errorbar(y = merged.iloc[-1]['kd_nm'],
             x = merged.iloc[-1]['log2_enrich_avg'], 
             yerr= merged.iloc[-1]['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = 'black', capsize = 3, markersize = 5, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(1.5,1.5)
plt.savefig('./figures/woolfson_kd_correlations_avg.svg', dpi = 300)
#plt.show()

f, ax = plt.subplots()

f, ax = plt.subplots()
plt.errorbar(y = merged['kd_nm'],
             x = merged['ashr_log2FoldChange_HIS_TRP'], 
             xerr= merged['ashr_lfcSE_HIS_TRP'], 
             yerr= merged['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.errorbar(y = merged.iloc[-1]['kd_nm'],
             x = merged.iloc[-1]['ashr_log2FoldChange_HIS_TRP'], 
             xerr= merged.iloc[-1]['ashr_lfcSE_HIS_TRP'], 
             yerr= merged.iloc[-1]['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = 'black', capsize = 3, markersize = 5, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(1.5,1.5)
plt.savefig('./figures/woolfson_kd_correlations_plfc.svg', dpi = 300)
#plt.show()

f, ax = plt.subplots()
plt.errorbar(y = merged['kd_nm'],
             x = merged['log2_enrich_max'], 
             yerr= merged['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.errorbar(y = merged.iloc[-1]['kd_nm'],
             x = merged.iloc[-1]['log2_enrich_max'], 
             yerr= merged.iloc[-1]['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = 'black', capsize = 3, markersize = 5, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(1.5,1.5)
plt.savefig('./figures/woolfson_kd_correlation_max.svg', dpi = 300)
#plt.show()

#look at lower points only
sorted_values = merged.sort_values('kd_nm', ascending = False).reindex()

f, ax = plt.subplots()
plt.errorbar(y = sorted_values.iloc[0:5]['kd_nm'],
             x = sorted_values.iloc[0:5]['log2_enrich_avg'], 
             yerr= sorted_values.iloc[0:5]['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(1.5,1)
plt.savefig('./figures/woolfson_kd_correlations_avg_small.svg', dpi = 300)

print (get_correls(sorted_values.iloc[0:5],  'log2_enrich_avg','kd_nm',  True))

f, ax = plt.subplots()
plt.errorbar(y = sorted_values.iloc[0:5]['kd_nm'],
             x = sorted_values.iloc[0:5]['log2_enrich_max'], 
             yerr= sorted_values.iloc[0:5]['kd_error_nm'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(1.5,1)
plt.savefig('./figures/woolfson_kd_correlation_max_small.svg', dpi = 300)

print (get_correls(sorted_values.iloc[0:5],  'log2_enrich_max', 'kd_nm', True))

f, ax = plt.subplots()
plt.errorbar(y = sorted_values.iloc[0:5]['kd_nm'],
             x = sorted_values.iloc[0:5]['ashr_log2FoldChange_HIS_TRP'], 
             yerr= sorted_values.iloc[0:5]['kd_error_nm'],
             xerr=sorted_values.iloc[0:5]['ashr_lfcSE_HIS_TRP'], 
             fmt="o", elinewidth = 2, alpha = 0.75, color = off_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.yscale('log')
plt.tight_layout()
f.set_size_inches(1.5,1)
plt.savefig('./figures/woolfson_kd_correlation_plfc_small.svg', dpi = 300)


print (get_correls(sorted_values.iloc[0:5], 'ashr_log2FoldChange_HIS_TRP',  'kd_nm', True))

#Correlating old and final versions of MP3-seq replicates 


homo_l39c = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l39_1_psuedoreplicate_autotune.csv')
homo_l39c = homo_l39c.rename(columns = {'Unnamed: 0': 'PPI'})
homo_l39c['PPI'] = homo_l39c.PPI.apply(lambda x: x.replace('_swap', ''))
homo_l39c['PPI'] = homo_l39c.PPI.apply(lambda x:make_ppi(x.split(':')[0], x.split(':')[1]))

homo_l39d = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l39_2_psuedoreplicate_autotune.csv')
homo_l39d = homo_l39d.rename(columns = {'Unnamed: 0': 'PPI'})
homo_l39d['PPI'] = homo_l39d.PPI.apply(lambda x: x.replace('_swap', ''))
homo_l39d['PPI'] = homo_l39d.PPI.apply(lambda x:make_ppi(x.split(':')[0], x.split(':')[1]))

homo_l39e = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l39_3_psuedoreplicate_autotune.csv')
homo_l39e = homo_l39e.rename(columns = {'Unnamed: 0': 'PPI'})
homo_l39e['PPI'] = homo_l39e.PPI.apply(lambda x: x.replace('_swap', ''))
homo_l39e['PPI'] = homo_l39e.PPI.apply(lambda x:make_ppi(x.split(':')[0], x.split(':')[1]))

homo_l44 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l44_psuedoreplicate_autotune.csv')
homo_l44 = homo_l44.rename(columns = {'Unnamed: 0': 'PPI'})
homo_l44['PPI'] = homo_l44.PPI.apply(lambda x: x.replace('_swap', ''))
homo_l44['PPI'] = homo_l44.PPI.apply(lambda x:make_ppi(x.split(':')[0], x.split(':')[1]))

homo_l49 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l49_psuedoreplicate_autotune.csv')
homo_l49 = homo_l49.rename(columns = {'Unnamed: 0': 'PPI'})
 
dhd0_1 = pd.read_csv("../processing_pipeline/merged_replicates/deseq_dhd1_dhd0_psuedoreplicate_autotune.csv")
dhd0_1 = dhd0_1.rename(columns = {'Unnamed: 0': 'PPI'})

final = pd.read_csv('../processing_pipeline/merged_replicates/deseq_all_designed_coils_psuedoreplicate_autotune.csv')
final = final.rename(columns = {'Unnamed: 0': 'PPI'})
print (final.shape)
 
    
merged = homo_l39c.merge(homo_l39d, on = 'PPI', suffixes = ['_l39c', '_l39d'])
print (merged.shape)
merged = homo_l39e.merge(merged, on ='PPI')
print (merged.shape)
merged = homo_l44.merge(merged, on ='PPI', suffixes = ['_l44', '_l39e'])
print (merged.shape)
merged = homo_l49.merge(merged, on ='PPI')
print (merged.shape)
merged = dhd0_1.merge(merged, on ='PPI', suffixes = ['_dhd0_1','_l49'])
print (merged.shape)
merged = final.merge(merged, on = 'PPI')
print (merged.shape)

combos = list(combinations(['_l39c', '_l39d', '_l39e', '_l44', '_l49', '_dhd0_1', ''], 2))

df_to_graph = pd.DataFrame({'PPI':[], 'correl':[], 'rho':[]})
for a, b in combos:
    
        df_to_graph.loc[df_to_graph.shape[0]] = [make_ppi(a,b), get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[2], 
                                                 get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[3]]
        
order_libraries = ['_l39c', '_l39d', '_l39e', '_l44', '_l49', '_dhd0_1', '']
x = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'correl',  get_diag = False)
y = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'rho',  get_diag = False)
y = y.T
make_double_heatmap(x, y, order_libraries, 'copper_r', 'bone_r', size_1 = 2.5, size_2 = 2.5, show_names = False, annot = True, font_size = 8, saveName= 'correls_large_scale_l39.svg', show=False)


homo_l43 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l43_psuedoreplicate_autotune.csv')
homo_l43 = homo_l43.rename(columns = {'Unnamed: 0': 'PPI'})
homo_l43['PPI'] = homo_l43.PPI.apply(lambda x: x.replace('_swap', ''))
homo_l43['PPI'] = homo_l43.PPI.apply(lambda x:make_ppi(x.split(':')[0], x.split(':')[1]))

dhd1_2_malbs = pd.read_csv("../processing_pipeline/merged_replicates/deseq_dhd2_dhd0_malb_psuedoreplicate_autotune.csv")
dhd1_2_malbs = dhd1_2_malbs.rename(columns = {'Unnamed: 0': 'PPI'})

final = pd.read_csv('../processing_pipeline/merged_replicates/deseq_all_designed_coils_psuedoreplicate_autotune.csv')
final = final.rename(columns = {'Unnamed: 0': 'PPI'})
 
merged = homo_l43.merge(homo_l44, on = 'PPI', suffixes = ['_l43', '_l44'])
merged = homo_l49.merge(merged, on ='PPI')
merged = dhd1_2_malbs.merge(merged, on ='PPI', suffixes = ['_dhd1_2_malbs', '_l49'])
print (merged.shape)
merged = final.merge(merged, on = 'PPI', how = 'right')
print (merged.shape)

combos = list(combinations(['_l43', '_l44', '_l49', '_dhd1_2_malbs', ''], 2))

df_to_graph = pd.DataFrame({'PPI':[], 'correl':[], 'rho':[]})
for a, b in combos:
        
        df_to_graph.loc[df_to_graph.shape[0]] = [make_ppi(a,b), get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[2], 
                                                 get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[3]]


order_libraries = ['_l43', '_l44', '_l49', '_dhd1_2_malbs', '']
x = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'correl',  get_diag = False)
y = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'rho',  get_diag = False)
y = y.T
make_double_heatmap(x, y, order_libraries, 'copper_r', 'bone_r', size_1 = 2.5, size_2 = 2.5, show_names = False, annot = True, font_size = 8, saveName= 'correls_large_scale_l43.svg', show=False)

#bcl replicate correlations 
homo_l66 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l66_psuedoreplicate_autotune.csv')
homo_l66 = homo_l66.rename(columns = {'Unnamed: 0': 'PPI'})

homo_l45 = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l45_psuedoreplicate_autotune.csv')
homo_l45 = homo_l45.rename(columns = {'Unnamed: 0': 'PPI'})

homo_bcl= pd.read_csv('../processing_pipeline/merged_replicates/deseq_bcl_psuedoreplicate_autotune.csv')
homo_bcl = homo_bcl.rename(columns = {'Unnamed: 0': 'PPI'})

merged = homo_l45.merge(homo_l66, on = 'PPI', suffixes = ['_l45', '_l66'])
print (merged.shape)
merged = homo_bcl.merge(merged, on ='PPI')
print (merged.shape)

combos = list(combinations(['_l45', '_l66', ''], 2))

df_to_graph = pd.DataFrame({'PPI':[], 'correl':[], 'rho':[]})
for a, b in combos:
        
        df_to_graph.loc[df_to_graph.shape[0]] = [make_ppi(a,b), get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[2], 
                                                 get_correls(merged, 'ashr_log2FoldChange_HIS_TRP' + a, 'ashr_log2FoldChange_HIS_TRP' + b, False)[3]]


order_libraries = ['_l45', '_l66', '']
x = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'correl',  get_diag = False)
y = make_specific_order_lower_triangle(order_libraries, df_to_graph, 'rho',  get_diag = False)
y = y.T
make_double_heatmap(x, y, order_libraries, 'copper_r', 'bone_r', size_1 = 2, size_2 = 2, show_names = False, annot = True, font_size = 8, saveName= 'correls_bcl_replicates.svg', show=False)
