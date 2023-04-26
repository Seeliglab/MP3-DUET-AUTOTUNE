Code to run the Deleting Undirected Edges Thoughtfully (DUET) algorithm, and to replicate results from Fig 5 and the supplementary analysis figure for DUET.

* *duet_functions.py*: The DUET scoring function and greedy graph reduction algorithm

* *duet_graph_visualization.py*: Creates DUET visualizations for Fig 5 (the csv files are to allow specific node placement for DUET graphs)

* *duet_large_scale.py*: Running DUET on the large-scale screens 

* *on_target_screening.py*: Filtering large-scale screens and DUET results to determine the number of 'on-target' designs in teh libraries

* *orthogonality_gap_reduction.py*: Filtering DUET result graphs to be truly orthogonal with different graph cutoffs, when considering all interactions instead of just low padj interactions
