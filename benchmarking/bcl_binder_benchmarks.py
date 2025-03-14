#Alyssa La Fleur
#Correlating Bcl-2 homolog and binder measurements from Beger & Alpha-Seq
from processing_functions import *
from comparison_datasets import *
import pandas as pd
import seaborn as sns
import os
from scipy.stats import ttest_ind
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

bcl_values, on_target_ppis, on_target_ors = bcl_binders()
bcl_values['type'] = bcl_values.PPI.apply(lambda x: x in on_target_ppis) #on_target_ppis)
print (bcl_values.type.value_counts())

homo_bcl= pd.read_csv('../processing_pipeline/merged_replicates/deseq_bcl_psuedoreplicate_autotune.csv')
homo_bcl = homo_bcl.rename(columns = {'Unnamed: 0': 'PPI'})

both = homo_bcl.merge(bcl_values, on = 'PPI')
kd_exists =  both[both.capped_out == False] #removed points which were outside instrument limits


on_targets = "#ffcc00ff"
off_targets = "#782167ff"
f, ax = plt.subplots()
subset = kd_exists
subset['count'] = subset['count']/(10**-9)
subset['count_error'] = subset['count_error_bars']/(10**-9)
print (subset.type.value_counts())
plt.errorbar(y = subset[subset.type]['count'],
             x = subset[subset.type]['ashr_log2FoldChange_HIS_TRP'], 
             xerr= subset[subset.type]['ashr_lfcSE_HIS_TRP'], 
             yerr = subset[subset.type]['count_error'],
             fmt="o", elinewidth = 2, alpha = 0.75, color = on_targets, capsize = 3, markersize = 5, ecolor = 'gray')
plt.errorbar(y = subset[~subset.type]['count'],
             x = subset[~subset.type]['ashr_log2FoldChange_HIS_TRP'], 
             xerr=  subset[~subset.type]['ashr_lfcSE_HIS_TRP'], 
             yerr =  subset[~subset.type]['count_error'],
             fmt="o", elinewidth = 2, markersize = 5, alpha = 0.9, color = off_targets,capsize = 3, ecolor = 'gray')
ax.set_yscale('log')
locmaj = matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, ), numticks=100)
ax.yaxis.set_major_locator(locmaj)

locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(-1, 6) * .1,
                                      numticks=100)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.tight_layout()
f.set_size_inches(1.5,2)
plt.savefig('./figures/' + 'bcl_new.svg', dpi = 300)
plt.show()

print ('plaper orig values:')
print(get_correls(subset, 'ashr_log2FoldChange_HIS_TRP', 'count', log=True))
print(get_correls(subset, 'ashr_log2FoldChange_HIS_TRP', 'count', log=False))






f, ax = plt.subplots(figsize=(3,3))

plt.errorbar(x = kd_exists[kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[kd_exists.type]['count']/(10**-9),
                yerr= kd_exists[kd_exists.type]['count_error_bars']/(10**-9),
                xerr = kd_exists[kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = on_targets, capsize = 1, markersize = 4, ecolor = 'gray')
plt.errorbar(x = kd_exists[~kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[~kd_exists.type]['count']/(10**-9),
                yerr= kd_exists[~kd_exists.type]['count_error_bars']/(10**-9),
                xerr = kd_exists[~kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = off_targets, capsize = 1, markersize = 4, ecolor = 'gray')

plt.yscale('log')
f.set_size_inches(1.5,2)
plt.savefig('./figures/' + 'bcl_correl_kds.svg', dpi = 300)
plt.show()
both['count_nm'] = both['count']/(10**-9)
print(get_correls(both[both.capped_out == False], 'ashr_log2FoldChange_HIS_TRP', 'count_nm', False))
print(get_correls(both[both.capped_out == False], 'ashr_log2FoldChange_HIS_TRP' , 'count_nm', True))


print ('alpha seq')
#alpha seq correls
f, ax = plt.subplots(figsize=(3,3))

plt.errorbar(x = kd_exists[kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[kd_exists.type]['batched_val'],
                yerr= kd_exists[kd_exists.type]['batched_error'],
                xerr = kd_exists[kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = on_targets, capsize = 1, markersize = 4, ecolor = 'gray')
plt.errorbar(x = kd_exists[~kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[~kd_exists.type]['batched_val'],
                yerr= kd_exists[~kd_exists.type]['batched_error'],
                xerr = kd_exists[~kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = off_targets, capsize = 1, markersize = 4, ecolor = 'gray')

plt.yscale('log')
f.set_size_inches(1.5,2)
plt.savefig('./figures/' + 'bcl_correl_batched_kds_exist.svg', dpi = 300)
plt.show()

print(get_correls(both[both.capped_out == False], 'ashr_log2FoldChange_HIS_TRP', 'batched_val', False))
print(get_correls(both[both.capped_out == False], 'ashr_log2FoldChange_HIS_TRP' , 'batched_val', True))

f, ax = plt.subplots(figsize=(3,3))

plt.errorbar(x = kd_exists[kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[kd_exists.type]['batched_val'],
                yerr= kd_exists[kd_exists.type]['batched_error'],
                xerr = kd_exists[kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = on_targets, capsize = 1, markersize = 4, ecolor = 'gray')
plt.errorbar(x = kd_exists[~kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[~kd_exists.type]['batched_val'],
                yerr= kd_exists[~kd_exists.type]['batched_error'],
                xerr = kd_exists[~kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = off_targets, capsize = 1, markersize = 4, ecolor = 'gray')

plt.yscale('log')
f.set_size_inches(1.5,2)
plt.savefig('./figures/' + 'bcl_correl_batched_all.svg', dpi = 300)
plt.show()
both['count_nm'] = both['count']/(10**-9)
print(get_correls(both, 'ashr_log2FoldChange_HIS_TRP', 'batched_val', False))
print(get_correls(both, 'ashr_log2FoldChange_HIS_TRP' , 'batched_val', True))

f, ax = plt.subplots(figsize=(3,3))

plt.errorbar(x = kd_exists[kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[kd_exists.type]['pairwise_val'],
                yerr= kd_exists[kd_exists.type]['pairwise_error'],
                xerr = kd_exists[kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = on_targets, capsize = 1, markersize = 4, ecolor = 'gray')
plt.errorbar(x = kd_exists[~kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[~kd_exists.type]['pairwise_val'],
                yerr= kd_exists[~kd_exists.type]['pairwise_error'],
                xerr = kd_exists[~kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = off_targets, capsize = 1, markersize = 4, ecolor = 'gray')

plt.yscale('log')
f.set_size_inches(1.5,2)
plt.savefig('./figures/' + 'bcl_correl_batched_kds_exist.svg', dpi = 300)
plt.show()

print(get_correls(both[both.capped_out == False], 'ashr_log2FoldChange_HIS_TRP', 'pairwise_val', False))
print(get_correls(both[both.capped_out == False], 'ashr_log2FoldChange_HIS_TRP' , 'pairwise_val', True))

f, ax = plt.subplots(figsize=(3,3))

plt.errorbar(x = kd_exists[kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[kd_exists.type]['pairwise_val'],
                yerr= kd_exists[kd_exists.type]['pairwise_error'],
                xerr = kd_exists[kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = on_targets, capsize = 1, markersize = 4, ecolor = 'gray')
plt.errorbar(x = kd_exists[~kd_exists.type]['ashr_log2FoldChange_HIS_TRP'],
                y = kd_exists[~kd_exists.type]['pairwise_val'],
                yerr= kd_exists[~kd_exists.type]['pairwise_error'],
                xerr = kd_exists[~kd_exists.type]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = off_targets, capsize = 1, markersize = 4, ecolor = 'gray')

plt.yscale('log')
f.set_size_inches(1.5,2)
plt.savefig('./figures/' + 'bcl_correl_batched_all.svg', dpi = 300)
plt.show()
both['count_nm'] = both['count']/(10**-9)
print(get_correls(both, 'ashr_log2FoldChange_HIS_TRP', 'pairwise_val', False))
print(get_correls(both, 'ashr_log2FoldChange_HIS_TRP' , 'pairwise_val', True))


#heatmap 
order_binders = ['Bcl-2', 'Mcl1[151-321]','Bcl-B', 'Bfl-1', 'Bcl-xL','Bcl-w',
                   'alphaBCL2', 'alphaMCL1', 'alphaBCLB','alphaBFL1', '2CDP06', 'BCDP01', 'XCDP07',  
                  'FECM04', 'BECM01']
x = make_specific_order_lower_triangle(order_binders, both, 'ashr_log2FoldChange_HIS_TRP',  get_diag = True)
make_lower_heatmap(x, order_binders, 'bone_r',size_1 = 4, size_2 = 4, saveName = 'kd_versus_bcl_heatmap.svg')

#boxplots & p-value analysis 
h = 1.25
w = 1.1

sns.stripplot(data = both, y = 'pairwise_val', x = 'capped_out', size = 3, color = 'black')
sns.boxplot(data = both, y = 'pairwise_val', x = 'capped_out', palette = {True: '#d2eedeff', False: '#d7eef4ff'})#, alpha = 0.75)#, kind = 'box')
plt.ylabel('')
plt.xlabel('')
plt.xticks([])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(h,w*0.95)
#plt.ylabel('')
plt.savefig("./figures/pair_log_cap.svg" )
plt.show()
            
            

sns.stripplot(data = both, y = 'batched_val', x = 'capped_out', size = 3, color = 'black')
sns.boxplot(data = both, y = 'batched_val', x = 'capped_out', palette = {True: '#d2eedeff', False: '#d7eef4ff'})#, alpha = 0.75)#, kind = 'box')
plt.ylabel('')
plt.xlabel('')
plt.xticks([])
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(h,w*0.95)
#plt.ylabel('')
plt.savefig("./figures/pair_batch_cap.svg" )
plt.show()


sns.stripplot(data = both, y = 'ashr_log2FoldChange_HIS_TRP', x = 'capped_out', size = 3, color = 'black')
sns.boxplot(data = both, y = 'ashr_log2FoldChange_HIS_TRP', x = 'capped_out', palette = {True: '#f4e3d7ff', False: '#b17047ff'})#, alpha = 0.75)#, kind = 'box')
plt.ylabel('')
plt.xlabel('')
plt.xticks([])
plt.ylim(-1.5,5)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(h,w*2)
plt.savefig("./figures/" + 'ashr_log2FoldChange_HIS_TRP'  + '_cap.svg' )
plt.show()

#t-tests 
print ('alpha-seq batched')
res = ttest_ind(both[both.capped_out == False].batched_val, both[both.capped_out == True].batched_val, equal_var=True, alternative = 'greater')
print (res)#, res.confidence_interval())
print ('alphaseq pairwise')
res = ttest_ind(both[both.capped_out == False].pairwise_val, both[both.capped_out == True].pairwise_val, equal_var=True, alternative = 'greater')
print (res)#, res.confidence_interval())
print ('mp3seq')
res = ttest_ind(both[both.capped_out == False].ashr_log2FoldChange_HIS_TRP, both[both.capped_out == True].ashr_log2FoldChange_HIS_TRP, equal_var=True, alternative = 'greater')
print (res)#, res.confidence_interval())
#low values 
f, ax = plt.subplots(figsize=(3,3))

plt.errorbar(x = both[both.capped_out]['ashr_log2FoldChange_HIS_TRP'],
                y = both[both.capped_out]['pairwise_val'],
                yerr= both[both.capped_out]['pairwise_error'],
                xerr = both[both.capped_out]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = off_targets, capsize = 1, markersize = 4, ecolor = 'gray')

plt.yscale('log')
f.set_size_inches(1,1.5)
plt.savefig('./figures/' + 'bcl_plfc_pairwise_no_kds.svg', dpi = 300)
plt.show()
print(get_correls( both[both.capped_out], 'ashr_log2FoldChange_HIS_TRP', 'pairwise_val', False))
print(get_correls( both[both.capped_out], 'ashr_log2FoldChange_HIS_TRP' , 'pairwise_val', True))

#low values 
f, ax = plt.subplots()

plt.errorbar(x = both[both.capped_out]['ashr_log2FoldChange_HIS_TRP'],
                y = both[both.capped_out]['batched_val'],
                yerr= both[both.capped_out]['batched_error'],
                xerr = both[both.capped_out]['ashr_lfcSE_HIS_TRP'],
                fmt="o", elinewidth = 0.75, alpha = 0.75, color = off_targets, capsize = 1, markersize = 4, ecolor = 'gray')

plt.yscale('log')
f.set_size_inches(1,1.5)
plt.savefig('./figures/' + 'bcl_plfc_batched_no_kds.svg', dpi = 300)
plt.show()
print(get_correls( both[both.capped_out], 'ashr_log2FoldChange_HIS_TRP', 'batched_val', False))
print(get_correls( both[both.capped_out], 'ashr_log2FoldChange_HIS_TRP' , 'batched_val', True))

