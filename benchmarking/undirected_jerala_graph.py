#Alyssa La Fleur
#creating weighted undirected graphical representation of the Jerala interactions

import pandas as pd
from graph_tool.all import *


#load DESeq2 processed Jerala P-series interactions (only those with all 3 replicates)
deseq_homo_af_batch = pd.read_csv('../processing_pipeline/merged_replicates/deseq_jerala_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})


#adding 'dummy' interactions to standardize P-LFC scale across graphs in Fig 1 
#these can be easily deletd later to reproduce the exact paper figures by deleting them in a SVG editor 

#graph setup 
keep = ['Jerala_P1',
 'Jerala_P2',
 'Jerala_P3',
 'Jerala_P4',
 'Jerala_P5',
 'Jerala_P6',
 'Jerala_P7',
 'Jerala_P8',
 'Jerala_P9',
 'Jerala_P10',
 'Jerala_P11',
 'Jerala_P12', 'd3', 'd4', 'd5', 'd6']

keep_color =  ['#008080ff'] * 12 + ['red'] * 4

position_dict = {
'Jerala_P1':(2,3),
'Jerala_P2':(3,3),
'Jerala_P3': (3.5, 3),
'Jerala_P4':(4.5,3),

'Jerala_P5':(2,2.5),
'Jerala_P6':(3,2.5),
'Jerala_P7':(3.5,2.5),
'Jerala_P8':(4.5,2.5),

'Jerala_P9':(2,2),
'Jerala_P10':(3,2),
'Jerala_P12':(4.5,2),
'Jerala_P11':(3.5,2),
'd3': (1.5,3),
'd4': (1.5,2),
'd5':(5,3),
'd6':(5,2)}


#Filtering to just the Jerala P-series proteins 
deseq_homo_af_batch['pro1'] = deseq_homo_af_batch.PPI.apply(lambda x: x.split(":")[0])
deseq_homo_af_batch['pro2'] = deseq_homo_af_batch.PPI.apply(lambda x: x.split(":")[1])
subset = deseq_homo_af_batch[(deseq_homo_af_batch.ashr_padj_HIS_TRP <= 0.05) & (deseq_homo_af_batch.pro1.isin(keep)) & (deseq_homo_af_batch.pro2.isin(keep))& (deseq_homo_af_batch.ashr_log2FoldChange_HIS_TRP>0)].copy()


#add some fake frows to scan sol
subset = subset[['pro1', 'pro2', 'ashr_log2FoldChange_HIS_TRP']].copy()
subset.loc[subset.shape[0]] = ['d3', 'd4', 9]
subset.loc[subset.shape[0]] = ['d5', 'd6', 1]
subset.reset_index(drop = True)

#creating graph tool graph
g = Graph(directed = False)
graph_inds_dict = {}
for b,c in zip(keep, keep_color):
    graph_inds_dict[b] = [keep.index(b), c]

subset['node1'] = subset.pro1.apply(lambda x: graph_inds_dict[x][0])
subset['node2'] = subset.pro2.apply(lambda x: graph_inds_dict[x][0])

#adding edges and edge weights
eprop = g.new_edge_property('double')
for ind, row in subset.iterrows():
    curr_edge = g.add_edge(row['node1'], row['node2'])
    eprop[curr_edge]= float(row['ashr_log2FoldChange_HIS_TRP'])

#adding vertex properties for graph labels and vertex locations
vprop = g.new_vertex_property("string")
color_prop = g.new_vertex_property("string")
position_v = g.new_vertex_property("vector<double>")
#vprop.
for b in graph_inds_dict:
    if '_' not in b:
        vprop[graph_inds_dict[b][0]] = b
        color_prop[graph_inds_dict[b][0]] = graph_inds_dict[b][1]
        position_v[graph_inds_dict[b][0]] = position_dict[b]
    else:
        vprop[graph_inds_dict[b][0]] = b.split('_')[1]
        color_prop[graph_inds_dict[b][0]] = graph_inds_dict[b][1]
        position_v[graph_inds_dict[b][0]] = position_dict[b]
g.vertex_properties["name"]=vprop
g.vertex_properties["color"]=color_prop
g.vertex_properties["position"]=position_v
g.edge_properties["ashr"]=eprop

#draw the grpah tool graph and save as a svg
graph_draw(
    g,
    vertex_text=g.vertex_properties["name"],
    pos=g.vertex_properties["position"],
    vertex_font_size=20,
    vertex_shape="circle",
    vertex_fill_color=g.vertex_properties["color"],
    vertex_pen_width=3,
    edge_pen_width=g.edge_properties["ashr"],
output = 'jerala_all_12.svg'
)
