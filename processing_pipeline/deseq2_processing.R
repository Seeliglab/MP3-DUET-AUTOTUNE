#Alyssa La Fleur
#DESeq2 processing script for all replicates and combined replicates
#install DESeq2 (v 1.38.3) & R version 4.2.2
library("DESeq2")

psuedoreplicate_deseq2_with_batch_effects <- function(in_name, out_name, num_reps=1, filter=0) {
  
  #load input tables 
  counts_table <- read.csv(in_name, row.names=1)
  print (counts_table)
  counts_table <- counts_table[rowSums(counts_table >= filter) == ncol(counts_table), ]
  print (counts_table)
  counts_table = as.matrix(counts_table)
  
  
  # Generate list of conditions
  vec_trp = c(rep.int(c("TRP", "TRP","HIS","HIS"), num_reps ) )
  cond <- factor(vec_trp)
  vec_os = c(rep.int(c("o1", "o2"), num_reps ) , rep.int(c("o1", "o2"), num_reps))
  batch <- factor(vec_os)
  df1<-data.frame(cond,batch)
  
  # Dataset object construction
  dds <- DESeqDataSetFromMatrix(counts_table,df1, ~ cond + batch)
  
  # Perform negative binomial GLM fitting
  dds <- DESeq(dds)
  
  # Generate log2(TRP/HIS)
  res <- results(dds, contrast=c("cond", "HIS", "TRP"))
  res <- lfcShrink(
    dds,
    res=res,
    type="ashr"
  )
  
  new_colnames <- colnames(res)
  new_colnames[new_colnames == "baseMean"] <- paste0("ashr_baseMean_", "HIS", "_TRP")
  new_colnames[new_colnames == "log2FoldChange"] <- paste0("ashr_log2FoldChange_", "HIS", "_TRP")
  new_colnames[new_colnames == "lfcSE"] <- paste0("ashr_lfcSE_", "HIS", "_TRP")
  new_colnames[new_colnames == "pvalue"] <- paste0("ashr_pvalue_", "HIS", "_TRP")
  new_colnames[new_colnames == "padj"] <- paste0("ashr_padj_", "HIS", "_TRP")
  colnames(res) <- new_colnames
  
  counts_table <- cbind(counts_table, res)

  #save counts_table
  write.csv(counts_table, out_name)
  print("Done!")
  
}


fusion_PPI_replicates <- function(in_name, out_name, num_reps=1, filter=0) {
  
  homo_name <- paste(in_name, sep="")
  homo_out_name = paste(out_name, sep = "")
  
  #load input tables 
  counts_table <- read.csv(homo_name, row.names=1)
  print (counts_table)
  counts_table <- counts_table[rowSums(counts_table >= filter) == ncol(counts_table), ]
  print (counts_table)
  counts_df = as.matrix(counts_table)

  # Generate list of conditions
  vec_trp = rep.int(c("TRP", "HIS"), num_reps )
  cond <- factor(vec_trp)
  df1<-data.frame(cond)
  
  # Dataset object construction
  dds <- DESeqDataSetFromMatrix(counts_df,df1, ~ cond)
  
  # Perform negative binomial GLM fitting
  dds <- DESeq(dds)
  
  # Generate log2(TRP/HIS)
  res <- results(dds, contrast=c("cond", "HIS", "TRP"))
  res <- lfcShrink(
    dds,
    res=res,
    type="ashr"
  )
  
  new_colnames <- colnames(res)
  new_colnames[new_colnames == "baseMean"] <- paste0("ashr_baseMean_", "HIS", "_TRP")
  new_colnames[new_colnames == "log2FoldChange"] <- paste0("ashr_log2FoldChange_", "HIS", "_TRP")
  new_colnames[new_colnames == "lfcSE"] <- paste0("ashr_lfcSE_", "HIS", "_TRP")
  new_colnames[new_colnames == "pvalue"] <- paste0("ashr_pvalue_", "HIS", "_TRP")
  new_colnames[new_colnames == "padj"] <- paste0("ashr_padj_", "HIS", "_TRP")
  colnames(res) <- new_colnames
  
  counts_table <- cbind(counts_table, res)
  
  #save counts_table
  write.csv(counts_table, homo_out_name)
  print("Done!")
  
}

#individual replicates
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/new_1_smaller_psuedoreplicate_autotune.csv', './processed_replicates/deseq_new_1_smaller_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/new_2_smaller_psuedoreplicate_autotune.csv', './processed_replicates/deseq_new_2_smaller_psuedoreplicate_autotune.csv')

#flat combinations 
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_5_small.csv', './merged_replicates/deseq_jerala_flat_autotune_5_small.csv', 5)
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_3_small.csv', './merged_replicates/deseq_jerala_flat_autotune_3_small.csv', 3)
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_4_small.csv', './merged_replicates/deseq_jerala_flat_autotune_4_small.csv', 4)
fusion_PPI_replicates('./merged_replicates/malb_truncations_2_flat_autotune.csv', './merged_replicates/deseq_malb_truncations_2_smaller_flat_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/malb_truncations_4_smaller_flat_autotune.csv', './merged_replicates/deseq_malb_truncations_4_smaller_flat_autotune.csv', 4)
fusion_PPI_replicates('./merged_replicates/malb_truncations_3_smaller_flat_autotune.csv', './merged_replicates/deseq_malb_truncations_3_smaller_flat_autotune.csv', 3)
fusion_PPI_replicates('./merged_replicates/malb_truncations_3b_smaller_flat_autotune.csv', './merged_replicates/deseq_malb_truncations_3b_smaller_flat_autotune.csv', 3)

fusion_PPI_replicates('./merged_replicates/hbond_malbs_3_smaller_flat_autotune.csv', './merged_replicates/deseq_hbond_malbs_3_smaller_flat_autotune.csv', 3)
fusion_PPI_replicates('./merged_replicates/new_smaller_flat_autotune.csv', './merged_replicates/deseq_new_smaller_flat_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/new_flat_autotune.csv', './merged_replicates/deseq_new_flat_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/plaper_3_smaller_flat_autotune.csv', './merged_replicates/deseq_plaper_3_smaller_flat_autotune.csv', 3)
fusion_PPI_replicates('./merged_replicates/plaper_3_flat_autotune.csv', './merged_replicates/deseq_plaper_3_flat_autotune.csv', 3)

#psuedo combinations
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/jerala_psuedoreplicate_autotune_5_small.csv', './merged_replicates/deseq_jerala_psuedoreplicate_autotune_5_small.csv', 5)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/malb_truncations_4_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_malb_truncations_4_smaller_psuedoreplicate_autotune.csv', 4)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/malb_truncations_3_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_malb_truncations_3_smaller_psuedoreplicate_autotune.csv', 3)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/malb_truncations_4_psuedoreplicate_autotune.csv', './merged_replicates/deseq_malb_truncations_4_psuedoreplicate_autotune.csv', 4)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/malb_truncations_3_psuedoreplicate_autotune.csv', './merged_replicates/deseq_malb_truncations_3_psuedoreplicate_autotune.csv', 3)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/hbond_malbs_3_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_hbond_malbs_3_smaller_psuedoreplicate_autotune.csv', 3)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/hbond_malbs_3_psuedoreplicate_autotune.csv', './merged_replicates/deseq_hbond_malbs_3_psuedoreplicate_autotune.csv', 3)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/new_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_new_smaller_psuedoreplicate_autotune.csv', 2)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/new_psuedoreplicate_autotune.csv', './merged_replicates/deseq_new_psuedoreplicate_autotune.csv', 2)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_3_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_plaper_3_smaller_psuedoreplicate_autotune.csv', 3)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_3_psuedoreplicate_autotune.csv', './merged_replicates/deseq_plaper_3_psuedoreplicate_autotune.csv', 3)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_2a_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_plaper_2a_smaller_psuedoreplicate_autotune.csv', 2)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_2b_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_plaper_2b_smaller_psuedoreplicate_autotune.csv', 2)



# #filter on different cutoffs 
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_5_small.csv', './merged_replicates/deseq_jerala_flat_f_1.csv', 5, filter = 1)
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_5_small.csv', './merged_replicates/deseq_jerala_flat_f_10.csv', 5, filter = 10)
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_5_small.csv', './merged_replicates/deseq_jerala_flat_f_20.csv', 5,  filter = 20)
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_5_small.csv', './merged_replicates/deseq_jerala_flat_f_30.csv', 5,  filter = 30)
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune_5_small.csv', './merged_replicates/deseq_jerala_flat_f_40.csv', 5,  filter = 40)

psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_3_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_l68_psuedoreplicate_autotune_f_10.csv', 3,filter = 10)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_3_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_l68_psuedoreplicate_autotune_f_20.csv',3, filter = 20)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_3_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_l68_psuedoreplicate_autotune_f_30.csv',3, filter = 30)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/plaper_3_smaller_psuedoreplicate_autotune.csv', './merged_replicates/deseq_l68_psuedoreplicate_autotune_f_40.csv', 3,filter = 40)



#individual replicates
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l33_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l33_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l43_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l43_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l44_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l44_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l45_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l45_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l48_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l48_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l49_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l49_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l66_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l66_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l67_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l67_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l68_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l68_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l70_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l70_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l39_1_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l39_1_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l39_2_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l39_2_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l39_3_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l39_3_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l61_2mM_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l61_2mM_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l62_2mM_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l62_2mM_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l61_10mM_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l61_10mM_psuedoreplicate_autotune.csv')
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l62_10mM_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l62_10mM_psuedoreplicate_autotune.csv')

#combined replicates

psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/jerala_l61_l67_psuedoreplicate_autotune.csv', './merged_replicates/deseq_jerala_l61_l67_psuedoreplicate_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/l61_l67_flat.csv', './merged_replicates/deseq_l61_l67_flat.csv', 2)
fusion_PPI_replicates('./merged_replicates/l61_l67_flat_autotune.csv', './merged_replicates/deseq_l61_l67_flat_autotune.csv', 2)

psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/jerala_l62_l67_psuedoreplicate_autotune.csv', './merged_replicates/deseq_jerala_l62_l67_psuedoreplicate_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/l62_l67_flat.csv', './merged_replicates/deseq_l62_l67_flat.csv', 2)
fusion_PPI_replicates('./merged_replicates/l62_l67_flat_autotune.csv', './merged_replicates/deseq_l62_l67_flat_autotune.csv', 2)

psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/jerala_psuedoreplicate_autotune.csv', './merged_replicates/deseq_jerala_psuedoreplicate_autotune.csv', 3)
fusion_PPI_replicates('./merged_replicates/jerala_flat.csv', './merged_replicates/deseq_jerala_flat.csv', 3)
fusion_PPI_replicates('./merged_replicates/jerala_flat_autotune.csv', './merged_replicates/deseq_jerala_flat_autotune.csv', 3)

psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/jerala_2_psuedoreplicate_autotune.csv', './merged_replicates/deseq_jerala_2_psuedoreplicate_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/jerala_2_flat_autotune.csv', './merged_replicates/deseq_jerala_2_flat_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/jerala_2_flat.csv', './merged_replicates/deseq_jerala_2_flat.csv', 2)

psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/bcl_psuedoreplicate_autotune.csv', './merged_replicates/deseq_bcl_psuedoreplicate_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/bcl_flat.csv', './merged_replicates/deseq_bcl_flat.csv', 2)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/dhd1_dhd0_psuedoreplicate_autotune.csv', './merged_replicates/deseq_dhd1_dhd0_psuedoreplicate_autotune.csv', 5)
fusion_PPI_replicates('./merged_replicates/dhd1_dhd0_flat.csv', './merged_replicates/deseq_dhd1_dhd0_flat.csv', 5)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/dhd2_dhd0_malb_psuedoreplicate_autotune.csv', './merged_replicates/deseq_dhd2_dhd0_malb_psuedoreplicate_autotune.csv', 3)
fusion_PPI_replicates('./merged_replicates/dhd2_dhd0_malb_flat.csv', './merged_replicates/deseq_dhd2_dhd0_malb_flat.csv', 3)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/all_designed_coils_psuedoreplicate_autotune.csv', './merged_replicates/deseq_all_designed_coils_psuedoreplicate_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/all_designed_coils_flat.csv', './merged_replicates/deseq_all_designed_coils_flat.csv', 2)
psuedoreplicate_deseq2_with_batch_effects('./merged_replicates/malb_truncations_psuedoreplicate_autotune.csv', './merged_replicates/deseq_malb_truncations_psuedoreplicate_autotune.csv', 2)
fusion_PPI_replicates('./merged_replicates/malb_truncations_flat.csv', './merged_replicates/deseq_malb_truncations_flat.csv', 2)

#filter on different cutoffs 
fusion_PPI_replicates('./merged_replicates/jerala_flat.csv', './merged_replicates/deseq_jerala_flat_f_1.csv', 3, filter = 1)
fusion_PPI_replicates('./merged_replicates/jerala_flat.csv', './merged_replicates/deseq_jerala_flat_f_3.csv', 3, filter = 10)
fusion_PPI_replicates('./merged_replicates/jerala_flat.csv', './merged_replicates/deseq_jerala_flat_f_5.csv', 3,  filter = 20)
fusion_PPI_replicates('./merged_replicates/jerala_flat.csv', './merged_replicates/deseq_jerala_flat_f_10.csv', 3,  filter = 30)
fusion_PPI_replicates('./merged_replicates/jerala_flat.csv', './merged_replicates/deseq_jerala_flat_f_10.csv', 3,  filter = 40)


psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l68_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l68_psuedoreplicate_autotune_f_1.csv', filter = 10)
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l68_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l68_psuedoreplicate_autotune_f_3.csv', filter = 20)
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l68_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l68_psuedoreplicate_autotune_f_5.csv', filter = 30)
psuedoreplicate_deseq2_with_batch_effects('./processed_replicates/l68_psuedoreplicate_autotune.csv', './processed_replicates/deseq_l68_psuedoreplicate_autotune_f_10.csv', filter = 40)

