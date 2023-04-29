from pipeline_interaction_regression_l2 import *

#1) Run padj 
padj_with_weights = padj_cv_regression(binned = True, weights = False)
padj_with_weights.to_csv('padj_regression_oversample_ridge.csv', index = False)
padj_with_weights = padj_cv_regression(binned = True, weights = True)
padj_with_weights.to_csv('padj_regression_oversample_yes_weights_ridge.csv', index = False)

#2) Run holdout and holdout final (DONE)
all_data = holdout_cv_regression(binned = True, weights = False)
all_data.to_csv('holdout_regression_oversample_no_weights_ridge.csv', index = False)
process_holdout_results_regression('holdout_regression_oversample_no_weights_ridge.csv', True, False, 'min')
process_holdout_results_regression('holdout_regression_oversample_no_weights_ridge.csv', True, False, 'max')
all_data.to_csv('holdout_regression_oversample_yes_weights_ridge.csv', index = False)
process_holdout_results_regression('holdout_regression_oversample_yes_weights_ridge.csv', True, False, 'min')
process_holdout_results_regression('holdout_regression_oversample_yes_weights_ridge.csv', True, False, 'max')

