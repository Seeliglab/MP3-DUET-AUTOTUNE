from pipeline_interaction_classification_l2 import *

#1) Run padj (DONE)
padj_no_weights = padj_cv_classification(binned = True, weights = False)
padj_no_weights.to_csv('padj_classification_oversample_no_weights_ridge.csv', index = False)
padj_with_weights = padj_cv_classification(binned = True, weights = True)
padj_with_weights.to_csv('padj_classification_oversample_yes_weights_ridge.csv', index = False)

#2) Run holdout and holdout final (DONE)
all_data = holdout_cv_classification(binned = True, weights = False)
all_data.to_csv('holdout_classification_oversample_no_weights_ridge.csv', index = False)
process_holdout_results_classification('holdout_classification_oversample_no_weights_ridge.csv', True, False, 'min')
process_holdout_results_classification('holdout_classification_oversample_no_weights_ridge.csv', True, False, 'max')
all_data = holdout_cv_classification(binned = True, weights = True)
all_data.to_csv('holdout_classification_oversample_yes_weights_ridge.csv', index = False)
process_holdout_results_classification('holdout_classification_oversample_yes_weights_ridge.csv', True, True, 'min')
process_holdout_results_classification('holdout_classification_oversample_yes_weights_ridge.csv', True, True, 'max')

