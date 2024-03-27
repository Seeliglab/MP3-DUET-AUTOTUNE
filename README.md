# MP3-DUET-AUTOTUNE
Data and analysis code for [*Massively parallel protein-protein interaction measurement by sequencing (MP3-seq) enables rapid screening of protein heterodimers*](https://www.biorxiv.org/content/10.1101/2023.02.08.527770v1)

## Dependencies 

### Python

Using Python 3.11.3

- pandas 1.5.3
- numpy 1.24.3
- seaborn 0.12.2
- matplotlib  3.7.1
- scipy 1.10.1
- scikit-learn 1.2.2
- networkx 3.2.1 
- graph-tool 2.59 (note that graph-tool has limited OS support, and is only necessary for making the graphs in the DUET figure)

### R 

Using R version 4.2.2
- DESeq2 1.38.3 

This software has been tested on Windows 11 Home and Ubuntu 18.04. 

## Contents

- *benchmarking*: Code for comparison of MP3-Seq to benchmark datasets from the paper (Figures 2-3)
- *data*: Barcode counts for libraries used in the paper (both older and final MP3-Seq method versions)
- *design_screening*: DUET analysis code for large designed binder screens, and orthogonality gap scripts
- *processing_pipeline*: MP3-Seq processing pipeline with autoactivator screening, autoactivator correction, psuedoreplication, and R scripts to run DESeq2.  The pipeline is shown for the datasets from the paper as an example, along with their expected outputs.
- *supplementary_analysis*: Misc. scripts to replicate supplementary figures in the paper.
- *Alphafold_and_Rosetta_metrics*: Scripts to calculate AlphaFold and Rosetta metrics
