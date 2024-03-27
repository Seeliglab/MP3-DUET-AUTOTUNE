

from processing_functions import *
from comparison_datasets import *
import matplotlib as mpl
import seaborn as sns
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

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


