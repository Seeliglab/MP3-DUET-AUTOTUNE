#AML
from performance_figure_functions import *
from predictor_comparisons import *

comp_datasets, comp_holdout, comp_holdout_two = get_all_baseline_sets()
avgpr_baselines_holdout, overall_baseline_avgpr = get_holdout_padj_baselines()

#Making main padj figures (all datasets, cutoffs)
padj_classification_results_nw = pd.read_csv('padj_classification_oversample_no_weights_ridge.csv')
padj_classification_results_nw['total_dataset'] = padj_classification_results_nw['dataset'] + '_' + padj_classification_results_nw['r2_cutoff']
padj_classification_results_yw = pd.read_csv('padj_classification_oversample_yes_weights_ridge.csv')
both_results = padj_classification_results_nw.merge(padj_classification_results_yw, on = ['dataset', 'r2_cutoff'], suffixes = ['_nw', '_yw'])

fig, axes = plt.subplots(3, sharex = True)
sns.scatterplot(data = both_results, x = 'test_AUCROC_nw', y = 'test_AUCROC_yw', hue = 'r2_cutoff', ax = axes[0] )
axes[0].plot([0,1], [0,1])
sns.scatterplot(data = both_results, x = 'test_avgpr_nw', y = 'test_avgpr_yw', hue = 'r2_cutoff', ax = axes[1])
axes[1].plot([0,1], [0,1])
sns.scatterplot(data = both_results, x = 'test_mccf1_nw', y = 'test_mccf1_yw', hue = 'r2_cutoff', ax = axes[2])
axes[2].plot([0,1], [0,1])
fig.set_size_inches(1,3)
plt.close()

#plotting barplots of all cutoff performances 



padj_regression_results_nw = pd.read_csv('padj_regression_oversample_ridge.csv')
padj_regression_results_nw['total_dataset'] = padj_regression_results_nw['dataset'] + '_' + padj_regression_results_nw['r2_cutoff']
padj_regression_results_yw = pd.read_csv('padj_regression_oversample_yes_weights_ridge.csv')
both_result_reg = padj_regression_results_nw.merge(padj_regression_results_yw, on = ['dataset', 'r2_cutoff'], suffixes = ['_nw', '_yw'])

fig, axes = plt.subplots(2, sharex = True)
sns.scatterplot(data = both_result_reg, x = 'test_r2_nw', y = 'test_r2_yw', hue = 'r2_cutoff', ax = axes[0])
axes[0].plot([0,1], [0,1])
sns.scatterplot(data = both_result_reg, x = 'test_spearman_nw', y = 'test_spearman_yw', hue = 'r2_cutoff', ax = axes[1])
axes[1].plot([0,1], [0,1])
fig.set_size_inches(1,2)
plt.show()

