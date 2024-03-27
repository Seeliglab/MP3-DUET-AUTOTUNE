MP3-Seq data processing pipeline.  The main steps are: 

1. Autoactivator screening (*identify_autoactivators.py*): Screen all measured interactions for potential autoactivators using enrichment values.  Output csv files of any autoactivators
2. Individual library prep (*process_all_libraries.py*): Using identified autoactivators, use AUTOTUNE to replace the readcount values as needed.  Also, create psuedoreplicate files
3. Combining biological replicates (*combine_replicates.py*): If applicable, combine replicates into a single csv file
4. Calcualte LFC and P-LFC values (*deseq2_processing.R*): R functions to calculate LFCs for individual replicates, and P-LFCs for combined replicates 

The entire pipeline should only take a few minutes per new dataset it is used on - with DESeq2 requiring the most time depending on the settings selected. 
