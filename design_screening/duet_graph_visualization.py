#AML
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from multiprocessing import Pool
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from numpy import double
from scipy import stats
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import combinations
from graph_tool.all import *
import networkx as nx
from collections import Counter
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)

color_dict = {'HT_DHD': '#8787def7', 'ZCON': '#d35f5ff7', 'DHDr2':'#ff9955f7', 'MALB': '#ffe680f7', 'OTHER':'red'}


def mark_type(name_pro):
    if 'HT_DHD' in name_pro:# and 'HT_DHD' in name2 and name1 != name2:
        return 'HT_DHD'
    elif 'ZCON' in name_pro:# and 'ZCON' in name2 and name1 != name2:
         return 'ZCON'
    elif 'DHDr2' in name_pro:# and 'ZCON' in name2 and name1 != name2:
         return 'DHDr2'
    return 'MALB'


def mark_targets(name1, name2):
    if 'HT_DHD' in name1 and 'HT_DHD' in name2 and name1 != name2:
        if  name1[2:] == name2[2:]:
            return True
    elif 'ZCON' in name1 and 'ZCON' in name2 and name1 != name2:
        if  name1[2:] == name2[2:]:
            return True
    elif 'DHDr2' in name1 and 'DHDr2' in name2 and name1 != name2:
        if name1[:-2] == name2[:-2]:
         #   print("        YES")
            return True
    return False

def make_score_plots(highest_removal, final_duet, version_str):

    fig = plt.figure()
    plt.plot(highest_removal.index, highest_removal['deg1'], color = 'darkgray')
    plt.plot(final_duet.index, final_duet['deg1'], color = 'darkblue')
    fig.set_size_inches(2,1)
    plt.savefig(version_str + '_deg1.svg')
    plt.close()

    fig = plt.figure()
    plt.plot(highest_removal.index, highest_removal['score'], color = 'darkgray')
    plt.plot(final_duet.index, final_duet['score'], color = 'darkblue')
    fig.set_size_inches(2,1)
    plt.savefig(version_str + '_score.svg')
    plt.close()


    fig = plt.figure()
    plt.plot(highest_removal.index, highest_removal['score'], color = 'darkgray')
    plt.plot(final_duet.index, final_duet['score'], color = 'darkblue')
    plt.xlim(highest_removal.shape[0] - 25,highest_removal.shape[0])
    y_low = min([highest_removal['score'].to_numpy().flatten()[highest_removal.shape[0] - 25], final_duet['score'].to_numpy().flatten()[final_duet.shape[0] - 25]]) - 10
    y_high = max([highest_removal['score'][highest_removal.shape[0]-1], final_duet['score'][final_duet.shape[0] - 1]]) + 10
    plt.ylim(y_low,y_high)
    fig.set_size_inches(1,1)
    plt.savefig(version_str + '_score_box.svg')
    plt.close()


def make_graphviz_graph(scanned_sol, output_name):
    print (scanned_sol)
    scanned_sol_pros = list(set(scanned_sol.pro1.to_list() + scanned_sol.pro2.to_list()))
    g = Graph(directed=False)
    graph_inds_dict = {}

    print (len(scanned_sol_pros), len(set(scanned_sol_pros)))
    for b in scanned_sol_pros:
        graph_inds_dict[b] = [scanned_sol_pros.index(b)]

    scanned_sol['node1'] = scanned_sol.pro1.apply(lambda x: graph_inds_dict[x][0])
    scanned_sol['node2'] = scanned_sol.pro2.apply(lambda x: graph_inds_dict[x][0])

    eprop = g.new_edge_property('double')
    for ind, row in scanned_sol.iterrows():
        curr_edge = g.add_edge(row['node1'], row['node2'])
        eprop[curr_edge] = float(row['ashr_log2FoldChange_HIS_TRP'])

    #see how many are on target vs not
    color_prop = g.new_vertex_property("string")
    
    for b in graph_inds_dict:
        color_prop[graph_inds_dict[b][0]] = color_dict[mark_type(b)]
    g.vertex_properties["color"]=color_prop
    g.edge_properties["ashr"] = eprop
    

    verts_degrees = []
    index_verts = []
    for v in g.vertices():
        verts_degrees.append(v.out_degree())
    print(pd.DataFrame({'dummy': verts_degrees}).dummy.value_counts())
    better_layout =  random_layout(g)
    graph_draw(
        g,
        vertex_pen_width=0.5,
        pos=better_layout,
        vertex_shape="circle",
        vertex_fill_color=g.vertex_properties["color"],
        edge_pen_width=g.edge_properties["ashr"],
        output = output_name
    )

#plot results of DUET scores 
highest_removal = pd.read_csv('dhd0_dhd1_tracking_high_removal.csv')
final_duet = pd.read_csv('duet_dhd0_dhd1_tracking.csv')
make_score_plots(highest_removal, final_duet, 'dhd0_dhd1')

#plot results of DUET scores 
highest_removal = pd.read_csv('dhd2_dhd0_malb_tracking_high_removal.csv')
final_duet = pd.read_csv('duet_dhd2_dhd0_malb_tracking.csv')
make_score_plots(highest_removal, final_duet, 'dhd0_dhd1')

#plot results of DUET scores 
highest_removal = pd.read_csv('all_designed_coils_tracking_high_removal.csv')
final_duet = pd.read_csv('duet_all_designed_coils_malb_tracking.csv')
make_score_plots(highest_removal, final_duet, 'all_designed_coils_malb')

#making small and large DUET graphs 
make_graphviz_graph(pd.read_csv('duet_dhd2_dhd0_malb_results.csv'), 'dhd0_2_malb_graph.svg' )
make_graphviz_graph(pd.read_csv('duet_dhd2_dhd0_malb_results_10_iterations.csv'), 'dhd0_2_malb_graph_10.svg'  )
make_graphviz_graph(pd.read_csv('duet_dhd2_dhd0_malb_results_15_iterations.csv'), 'dhd0_2_malb_graph_15.svg' )

make_graphviz_graph(pd.read_csv('high_removal_dhd0_dhd1_results.csv'), 'dhd0_dhd1_graph.svg' )
make_graphviz_graph(pd.read_csv('duet_dhd0_dhd1_results_50_iterations.csv'), 'dhd0_dhd1_malb_graph_50.svg'  )
make_graphviz_graph(pd.read_csv('duet_dhd0_dhd1_results_75_iterations.csv'), 'dhd0_dhd1_malb_graph_75.svg' )

make_graphviz_graph(pd.read_csv('duet_all_designed_coils_malb_results.csv'), 'dhd0_2_malb_graph.svg' )
make_graphviz_graph(pd.read_csv('duet_all_designed_coils_malb_results_50_iterations.csv'), 'dhd0_2_malb_graph_50.svg'  )
make_graphviz_graph(pd.read_csv('duet_all_designed_coils_results_75_iterations.csv'), 'dhd0_2_malb_graph_75.svg' )
