#Alyssa La Fleur
#Making the undirected graphs of the Plaper N-series, P-Series, and PA-series interactions 

import pandas as pd
from graph_tool.all import *
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

deseq_homo_af_batch = pd.read_csv('../processing_pipeline/processed_replicates/deseq_l68_psuedoreplicate_autotune.csv')
deseq_homo_af_batch = deseq_homo_af_batch.rename(columns = {'Unnamed: 0': 'PPI'})

#making main figure graph
keep = ['N5', 'N6', 'N7', 'N8', 'P5A', 'P6A', 'P7A', 'P8A','Jerala_P5',
 'Jerala_P6',
 'Jerala_P7',
 'Jerala_P8', 'd3', 'd4', 'd5', 'd6']

#adding 'dummy' interactions to standardize P-LFC scale across graphs in Fig 1 
#these can be easily deletd later to reproduce the exact paper figures by deleting them in a SVG editor 

position_dict = {'N5':(1,3.75),
                 'N6':(3,4),
                 'N7':(6,3.75),
                 'N8':(4,4),
                 'P5A':(1,3),
                 'P6A':(3,3),
                 'P7A':(6,3),
                 'P8A':(4,3),
                 'Jerala_P5':(1,2),
 'Jerala_P6':(3,2.25),
 'Jerala_P7':(6,2),
 'Jerala_P8':(4,2.25),
'd3': (1.5,3),
'd4': (1.5,2),
'd5':(5,3),
'd6':(5,2)}


keep_color = ['#decd87ff'] * 4 + ['#aca793ff'] * 4 + ['#008080ff'] * 4 + ['red'] * 4
deseq_homo_af_batch['pro1'] = deseq_homo_af_batch.PPI.apply(lambda x: x.split(":")[0])
deseq_homo_af_batch['pro2'] = deseq_homo_af_batch.PPI.apply(lambda x: x.split(":")[1])


#only look at those in woolfson or the plaper subset
subset = deseq_homo_af_batch[(deseq_homo_af_batch.ashr_padj_HIS_TRP <= 0.01) & (deseq_homo_af_batch.pro1.isin(keep)) & (deseq_homo_af_batch.pro2.isin(keep))& (deseq_homo_af_batch.ashr_log2FoldChange_HIS_TRP>0)].copy()
subset.reset_index(drop = True)

subset = subset[['pro1', 'pro2', 'ashr_log2FoldChange_HIS_TRP']].copy()
subset.loc[subset.shape[0]] = ['d3', 'd4', 9]
subset.loc[subset.shape[0]] = ['d5', 'd6', 1]

g = Graph(directed = False)
graph_inds_dict = {}
for b,c in zip(keep, keep_color):
    graph_inds_dict[b] = [keep.index(b), c]

subset['node1'] = subset.pro1.apply(lambda x: graph_inds_dict[x][0])
subset['node2'] = subset.pro2.apply(lambda x: graph_inds_dict[x][0])

eprop = g.new_edge_property('double')
for ind, row in subset.iterrows():
    curr_edge = g.add_edge(row['node1'], row['node2'])
    eprop[curr_edge]= float(row['ashr_log2FoldChange_HIS_TRP'])

#add the property to vertex object
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

graph_draw(
    g,
    vertex_text=g.vertex_properties["name"],
    pos=g.vertex_properties["position"],
    vertex_font_size=20,
    vertex_shape="circle",
    vertex_fill_color=g.vertex_properties["color"],
    vertex_pen_width=3,
    edge_pen_width=g.edge_properties["ashr"],
output = 'jerala_plaper.svg'
)

#all proteins in all three sets
keep = ['N5', 'N6', 'N7', 'N8', 'P5A', 'P6A', 'P7A', 'P8A',
 'Jerala_P1',
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


keep_color = ['#decd87ff'] * 4 + ['#aca793ff'] * 4 + ['#008080ff'] * 12 + ['red'] * 4


position_dict = {
'Jerala_P1':(2,3),
'Jerala_P2':(3,3),
'P5A': (3.5, 3),
'P6A':(4.5,3),

'Jerala_P5':(2,2.5),
'Jerala_P6':(3,2.5),
'N7':(3.5,2.5),
'N8':(4.5,2.5),

'Jerala_P9':(2,2),
'Jerala_P10':(3,2),
'N6':(4.5,2),
'N5':(3.5,2),

'P7A':(3.5, 3.5),
'P8A':(4.5, 3.5),

'Jerala_P12':(6,2),
'Jerala_P11':(5,2),
'Jerala_P7':(5,2.5),
'Jerala_P8':(6,2.5),
'Jerala_P3': (5, 3),
'Jerala_P4':(6,3),

'd3': (0.5,3),
'd4': (0.5,2),
'd5':(7,3),
'd6':(7,2)}


subset = deseq_homo_af_batch[(deseq_homo_af_batch.ashr_padj_HIS_TRP <= 0.01) & (deseq_homo_af_batch.pro1.isin(keep)) & (deseq_homo_af_batch.pro2.isin(keep))& (deseq_homo_af_batch.ashr_log2FoldChange_HIS_TRP>0)].copy()
subset.reset_index(drop = True)

subset = subset[['pro1', 'pro2', 'ashr_log2FoldChange_HIS_TRP']].copy()
subset.loc[subset.shape[0]] = ['d3', 'd4', 9]
subset.loc[subset.shape[0]] = ['d5', 'd6', 1]

g = Graph(directed = False)
graph_inds_dict = {}
for b,c in zip(keep, keep_color):
    graph_inds_dict[b] = [keep.index(b), c]

subset['node1'] = subset.pro1.apply(lambda x: graph_inds_dict[x][0])
subset['node2'] = subset.pro2.apply(lambda x: graph_inds_dict[x][0])

eprop = g.new_edge_property('double')
for ind, row in subset.iterrows():
    curr_edge = g.add_edge(row['node1'], row['node2'])
    eprop[curr_edge]= float(row['ashr_log2FoldChange_HIS_TRP'])

#add the property to vertex object
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

graph_draw(
    g,
    vertex_text=g.vertex_properties["name"],
    vertex_font_size=8,
    vertex_shape="circle",
    vertex_fill_color=g.vertex_properties["color"],
    vertex_pen_width=3,
    edge_pen_width=g.edge_properties["ashr"],
output = 'jerala_plaper_supplement_graph.svg'
)

