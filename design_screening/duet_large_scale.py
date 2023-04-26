from duet_functions import *


#DHD0 and DHD1 DUET 
deseq_homo_af_batch = pd.read_csv('../processing_pipeline/merged_replicates/deseq_dhd1_dhd0_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})
subset = deseq_homo_af_batch[(deseq_homo_af_batch.ashr_log2FoldChange_HIS_TRP > 0 ) &  (deseq_homo_af_batch.ashr_padj_HIS_TRP <= 0.01)].copy()
subset.rename(columns = {'Unnamed: 0': 'PPI'}, inplace=True)
subset['pro1'] = subset.PPI.apply(lambda x: x.split(':')[0])
subset['pro2'] = subset.PPI.apply(lambda x: x.split(':')[1])
#remove non coiled designs from the set 
to_remove =['2CDP06','BCDP01','BECM01','Bcl-2','Bcl-B','Bcl-w','Bcl-xL','Bfl-1','FECM04','Mcl1[151-321]','XCDP07','alphaBCL2','alphaBCLB','alphaBFL1','alphaMCL1']
subset = subset[~(subset.pro1.isin(to_remove)) & ~(subset.pro2.isin(to_remove))].copy()
keep= list(set(subset.pro1.to_list() + subset.pro2.to_list()))


good_verts, peak_degree_1, tracking_frame = scan_score_highest_removal(subset)
tracking_frame.to_csv('dhd0_dhd1_tracking_high_removal.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('high_removal_dhd0_dhd1_results.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset)
tracking_frame.to_csv('duet_dhd0_dhd1_tracking.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_dhd0_dhd1_results.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset, 50)
tracking_frame.to_csv('duet_dhd0_dhd1_tracking_50_iterations.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_dhd0_dhd1_results_50_iterations.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset, 75)
tracking_frame.to_csv('duet_dhd0_dhd1_tracking_75_iterations.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_dhd0_dhd1_results_75_iterations.csv', index = False)


#DHD0 and DHD1 DUET 
deseq_homo_af_batch = pd.read_csv('../processing_pipeline/merged_replicates/deseq_dhd2_dhd0_malb_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})
subset = deseq_homo_af_batch[(deseq_homo_af_batch.ashr_log2FoldChange_HIS_TRP > 0 ) &  (deseq_homo_af_batch.ashr_padj_HIS_TRP <= 0.01)].copy()
subset.rename(columns = {'Unnamed: 0': 'PPI'}, inplace=True)
subset['pro1'] = subset.PPI.apply(lambda x: x.split(':')[0])
subset['pro2'] = subset.PPI.apply(lambda x: x.split(':')[1])
#remove non coiled designs from the set 
to_remove =['2CDP06','BCDP01','BECM01','Bcl-2','Bcl-B','Bcl-w','Bcl-xL','Bfl-1','FECM04','Mcl1[151-321]','XCDP07','alphaBCL2','alphaBCLB','alphaBFL1','alphaMCL1']
subset = subset[~(subset.pro1.isin(to_remove)) & ~(subset.pro2.isin(to_remove))].copy()
keep= list(set(subset.pro1.to_list() + subset.pro2.to_list()))


good_verts, peak_degree_1, tracking_frame = scan_score_highest_removal(subset)
tracking_frame.to_csv('dhd2_dhd0_malb_tracking_high_removal.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('high_removal_dhd2_dhd0_malb_results.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset)
tracking_frame.to_csv('duet_dhd2_dhd0_malb_tracking.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_dhd2_dhd0_malb_results.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset,10)
tracking_frame.to_csv('duet_dhd2_dhd0_malb_tracking_50_iterations.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_dhd2_dhd0_malb_results_10_iterations.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset, 15)
tracking_frame.to_csv('duet_dhd2_dhd0_malb_tracking_15_iterations.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_dhd2_dhd0_malb_results_75_iterations.csv', index = False)

#DHD0 and DHD1 DUET 
deseq_homo_af_batch = pd.read_csv('../processing_pipeline/merged_replicates/deseq_all_designed_coils_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})
subset = deseq_homo_af_batch[(deseq_homo_af_batch.ashr_log2FoldChange_HIS_TRP > 0 ) &  (deseq_homo_af_batch.ashr_padj_HIS_TRP <= 0.01)].copy()
subset.rename(columns = {'Unnamed: 0': 'PPI'}, inplace=True)
subset['pro1'] = subset.PPI.apply(lambda x: x.split(':')[0])
subset['pro2'] = subset.PPI.apply(lambda x: x.split(':')[1])
#remove non coiled designs from the set 
to_remove =['2CDP06','BCDP01','BECM01','Bcl-2','Bcl-B','Bcl-w','Bcl-xL','Bfl-1','FECM04','Mcl1[151-321]','XCDP07','alphaBCL2','alphaBCLB','alphaBFL1','alphaMCL1']
subset = subset[~(subset.pro1.isin(to_remove)) & ~(subset.pro2.isin(to_remove))].copy()
keep= list(set(subset.pro1.to_list() + subset.pro2.to_list()))


good_verts, peak_degree_1, tracking_frame = scan_score_highest_removal(subset)
tracking_frame.to_csv('all_designed_coils_tracking_high_removal.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('high_removal_all_designed_coils_malb_results.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset)
tracking_frame.to_csv('duet_all_designed_coils_malb_tracking.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_all_designed_coils_malb_results.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset, 50)
tracking_frame.to_csv('duet_all_designed_coils_malb_tracking_50_iterations.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_all_designed_coils_malb_results_50_iterations.csv', index = False)

good_verts, peak_degree_1, tracking_frame = scan_score(subset, 75)
tracking_frame.to_csv('duet_all_designed_coils_tracking_75_iterations.csv')
subset2 = subset[(subset.pro1.isin(good_verts)) & (subset.pro2.isin(good_verts))].copy()
subset2.to_csv('duet_all_designed_coils_results_75_iterations.csv', index = False)
