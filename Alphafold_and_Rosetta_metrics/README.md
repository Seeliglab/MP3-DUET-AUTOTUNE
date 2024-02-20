# Running Alphafold and Rosetta on an all-by-all set of sequences.  

This folder includes scripts to run Alphafold and Rosetta on an all-by-all set of sequences.

## Installation

Alpafold2 and must be installed on the system. We used the excellent [ColabFold](https://github.com/sokrypton/ColabFold) with local installation instructions taken [LocalColabFold](https://github.com/YoshitakaMo/localcolabfold). 

A recent [Rosetta](https://www.rosettacommons.org/software) version (with the Rosetta Scripts application) must also be installed locally. 

The system is set up for a cluster running SLURM.

## Usage

If you would like to run your own sequences: Make a copy of the `2023-03-21__mALB8_all_by_all_v2` folder. Put your fasta file in the `data` subdirectory.

Execute the notebooks in the numbered order. Make sure to adjust the paths to Alphafold 2 and Rosetta.

### Output

The output is a gzipped CSV file located in the data directory `data/AF2_rosetta_merged.csv.gz`. The `out` directory contains all Alphafold and Rosetta intermediate files, useful for debugging. These files can be deleted. 