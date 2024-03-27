#Alyssa La Fleur
#Correlating Bcl-2 homolog and binder measurements from Beger & Alpha-Seq
from processing_functions import *
from comparison_datasets import *
import pandas as pd
import seaborn as sns
import os
from scipy.stats import ttest_ind
from scipy.stats import t
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


def calc_t_conf_interval(sample1, sample2, alpha = 0.9):
    df = len(sample1) + len(sample2) - 2  # Degrees of freedom
    sample_mean = sample1.mean() - sample2.mean()
    sample_std = (sample1.std()**2 / len(sample1) + sample2.std()**2 / len(sample2))**0.5
    n = len(sample1) + len(sample2)

    confidence_interval = t.interval(alpha, df, loc=sample_mean, scale=sample_std / n**0.5)
    return confidence_interval

bcl_values, on_target_ppis, on_target_ors = bcl_binders()
bcl_values['type'] = bcl_values.PPI.apply(lambda x: x in on_target_ppis) #on_target_ppis)
print (bcl_values.type.value_counts())

homo_bcl= pd.read_csv('../processing_pipeline/merged_replicates/deseq_bcl_psuedoreplicate_autotune.csv')
homo_bcl = homo_bcl.rename(columns = {'Unnamed: 0': 'PPI'})

both = homo_bcl.merge(bcl_values, on = 'PPI')

#t-tests 
print ('alpha-seq batched')
res = ttest_ind(both[both.capped_out == False].batched_val, both[both.capped_out == True].batched_val, equal_var=True, alternative = 'greater')
print (res)
print (calc_t_conf_interval(both[both.capped_out == False].batched_val, both[both.capped_out == True].batched_val))
print ('alphaseq pairwise')
res = ttest_ind(both[both.capped_out == False].pairwise_val, both[both.capped_out == True].pairwise_val, equal_var=True, alternative = 'greater')
print (res)
print ('mp3seq')
res = ttest_ind(both[both.capped_out == False].ashr_log2FoldChange_HIS_TRP, both[both.capped_out == True].ashr_log2FoldChange_HIS_TRP, equal_var=True, alternative = 'greater')
print (res)
#low values 
