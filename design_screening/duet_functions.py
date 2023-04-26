#ALyssa La Fleur
#functions for performing DUET on a all by all screen

#actually a really interesting algorithmic problem - similar to greedy graph traversal and a star

#import pandas as pd
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from numpy import double
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from graph_tool.all import *
import networkx as nx
from collections import Counter

#make svg text editable 
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
from functools import partial


def make_graph_tool_graph(subset, keep):
    g = Graph(directed=False)
    graph_inds_dict = {}
    for b in keep:
        graph_inds_dict[b] = [keep.index(b)]

    subset['node1'] = subset.pro1.apply(lambda x: graph_inds_dict[x][0])
    subset['node2'] = subset.pro2.apply(lambda x: graph_inds_dict[x][0])

    eprop = g.new_edge_property('double')
    for ind, row in subset.iterrows():
        curr_edge = g.add_edge(row['node1'], row['node2'])
        eprop[curr_edge] = float(row['ashr_log2FoldChange_HIS_TRP'])

    # add the property to vertex object
    vprop = g.new_vertex_property("string")

    for b in graph_inds_dict:
        # if '_' not in b:
        vprop[graph_inds_dict[b][0]] = b

    g.vertex_properties["name"] = vprop
    g.edge_properties["ashr"] = eprop

    # look for number nodes with 1 edge only
    verts_degrees = []
    index_verts = []
    for v in g.vertices():
        verts_degrees.append(v.out_degree())
    print(pd.DataFrame({'dummy': verts_degrees}).dummy.value_counts())

    better_layout =  random_layout(g)
    # setup edge weights by ashr logfoldchange
    graph_draw(
        g,
        vertex_text=g.vertex_properties["name"],
        pos=better_layout,
        vertex_font_size=20,
        vertex_shape="circle",
        vertex_pen_width=3,
        edge_pen_width=g.edge_properties["ashr"],
    )

def networkx_draw_graph(subset):
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    # made the graph
    print(g.number_of_nodes())
    print(g.number_of_edges())
    nx.draw(g, width = [g[u][v]['weight'] for u,v in g.edges()])
    plt.show()
    return g


def reduction_by_maximal_degree_deletion(subset):
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    # made the graph
    print(g.number_of_nodes())
    print(g.number_of_edges())

    tracking_frame = pd.DataFrame(
        {'deg1': [], 'deg2': [], 'deg3': [], 'deg4': [], 'deg5': [], 'max_degree': [], 'totalNodes': [],
         'totalEdges': []})
    peak_degree_1 = None
    #remove 0 degree nodes if present
    highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
    for h in highest:
        if h[1] == 0:
            g.remove_node(h[0])
    degrees = [val for (node, val) in g.degree()]
    degreeCounter = Counter(degrees)
    tracking_frame.loc[0] = [degreeCounter[1], degreeCounter[2], degreeCounter[3], degreeCounter[4], degreeCounter[5],
                             highest[0][1], g.number_of_nodes(), g.number_of_edges()]
    for round in range(0, 300):
        print('round: ', round)
        highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
        degrees = [val for (node, val) in g.degree()]
        degreeCounter = Counter(degrees)
        for h in highest:
            if h[1] == 0:
                g.remove_node(h[0])
        tracking_frame.loc[tracking_frame.shape[0]] = [degreeCounter[1], degreeCounter[2], degreeCounter[3],
                                                       degreeCounter[4], degreeCounter[5], highest[0][1],
                                                       g.number_of_nodes(), g.number_of_edges()]
        if degreeCounter[1] >= tracking_frame.deg1.max():
            peak_degree_1 = (round, [node for (node, val) in g.degree()])

        if highest[0][1] == 1:
            break
        g.remove_node(highest[0][0])

    good_verts = [node for (node, val) in g.degree()]
    return good_verts, peak_degree_1, tracking_frame


def num_degree_one_vertices_post_deletion(graph, nodes):
    dummyG = graph.copy()
    for node in nodes:
        dummyG.remove_node(node)
    highest = sorted(dummyG.degree, key=lambda x: x[1], reverse=True)
    # remove the degree 0 nodes
    for h in highest:
        if h[1] == 0:
            dummyG.remove_node(h[0])
    #get mode 1 count
    degrees = [val for (node, val) in dummyG.degree()]
    degreeCounter = Counter(degrees)
    return degreeCounter[1]

def get_mode_and_avg_degree(graph, node):
    dummyG = graph.copy()
    dummyG.remove_node(node)
    highest = sorted(dummyG.degree, key=lambda x: x[1], reverse=True)
    # remove the degree 0 nodes
    #get mode 1 count
    degrees = [val for (node, val) in dummyG.degree()]
    return stats.mode(degrees)[0], np.mean(degrees)


def scan_all_possibilities_reduction(subset):
    #maximize number of degree one nodes by deletion of vertices
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    # made the graph

    tracking_frame = pd.DataFrame(
        {'deg1': [], 'deg2': [], 'deg3': [], 'deg4': [], 'max_degree': [], 'totalNodes': [],
         'totalEdges': []})

    highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
    #remove the degree 0 nodes
    print(g.number_of_nodes())
    for h in highest:
        if h[1] == 0:
            g.remove_node(h[0])
    print (g.number_of_nodes())
    #tracking
    degrees = [val for (node, val) in g.degree()]
    degreeCounter = Counter(degrees)
    tracking_frame.loc[0] = [degreeCounter[1], degreeCounter[2], degreeCounter[3], degreeCounter[4],
                             highest[0][1], g.number_of_nodes(), g.number_of_edges()]
    peak_degree_1 = None
    for round in range(0, 300):
        print('round: ', round)
        highest = sorted(g.degree, key=lambda x: x[1], reverse=True)

        degrees = [val for (node, val) in g.degree()]
        degreeCounter = Counter(degrees)
        tracking_frame.loc[tracking_frame.shape[0]] = [degreeCounter[1], degreeCounter[2], degreeCounter[3],
                                                       degreeCounter[4], highest[0][1],
                                                       g.number_of_nodes(), g.number_of_edges()]
        if degreeCounter[1] >= tracking_frame.deg1.max():
            peak_degree_1 = (round, [node for (node, val) in g.degree()])

        if highest[0][1] == 1:
            break

        #scan through ALL nodes to find the best reduction candidate
        to_remove = [float('-inf'), None]
        for h in highest:
            if h[1] > 1:
                effect = num_degree_one_vertices_post_deletion(g, h[0])
                if effect >= to_remove[0]:
                    to_remove[0] = effect
                    to_remove[1] = h[0]
            elif h[1] == 0:
                #remove the node
                print ('removing nodes...')
                print (g.number_of_nodes())
                g.remove_node(h[0])
                print(g.number_of_nodes())

        g.remove_node(to_remove[1])
        # degrees = [val for (node, val) in G.degree()]
    good_verts = [node for (node, val) in g.degree()]
    return good_verts, peak_degree_1, tracking_frame

def scan_mode_and_avg(subset):
    #maximize number of degree one nodes by deletion of vertices
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    # made the graph

    tracking_frame = pd.DataFrame(
        {'deg1': [], 'deg2': [], 'deg3': [], 'deg4': [], 'max_degree': [], 'totalNodes': [],
         'totalEdges': [], 'score': []})

    highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
    #remove the degree 0 nodes
    print(g.number_of_nodes())
    for h in highest:
        if h[1] == 0:
            g.remove_node(h[0])
    print (g.number_of_nodes())
    #tracking
    degrees = [val for (node, val) in g.degree()]
    degreeCounter = Counter(degrees)
    tracking_frame.loc[0] = [degreeCounter[1], degreeCounter[2], degreeCounter[3], degreeCounter[4],
                             highest[0][1], g.number_of_nodes(), g.number_of_edges(), score_values(g)]
    peak_degree_1 = None
    for round in range(0, 300):
        if round %50 == 0:
            print (round)
        #print('round: ', round)
        highest = sorted(g.degree, key=lambda x: x[1], reverse=True)

        degrees = [val for (node, val) in g.degree()]
        degreeCounter = Counter(degrees)
        tracking_frame.loc[tracking_frame.shape[0]] = [degreeCounter[1], degreeCounter[2], degreeCounter[3],
                                                       degreeCounter[4], highest[0][1],
                                                       g.number_of_nodes(), g.number_of_edges(), score_values(g)]
        if degreeCounter[1] >= tracking_frame.deg1.max():
            peak_degree_1 = (round, [node for (node, val) in g.degree()])

        if highest[0][1] == 1:
            break

        #scan through ALL nodes to find the best reduction candidate
        to_remove = [float('inf'), None]
        for h in highest:
            if h[1] > 1:
                effect, effect2 = get_mode_and_avg_degree(g, h[0])
                net_effect = effect + (effect2 - (g.number_of_nodes() /2))
                #print (net_effect)
                if net_effect <= to_remove[0]:
                    to_remove[0] = net_effect
                    to_remove[1] = h[0]
                    #print ('udpate: ', to_remove[0])
            elif h[1] == 0:
                #remove the node
                #print ('removing nodes...')
                #print (g.number_of_nodes())
                g.remove_node(h[0])
                #print(g.number_of_nodes())

        g.remove_node(to_remove[1])
        # degrees = [val for (node, val) in G.degree()]
    good_verts = [node for (node, val) in g.degree()]
    return good_verts, peak_degree_1, tracking_frame

def score_values(g):
    score = 0
    homodimers_count = 0
    for e in g.edges():
        if e[0] == e[1]:
            homodimers_count += 1
        elif g.degree[e[0]] ==1 and g.degree[e[1]] == 1:
            score += g[e[0]][e[1]]["weight"]
        elif  g.degree[e[0]] ==1:
            current = g[e[0]][e[1]]["weight"]
            weights_other = [g[e2[0]][e2[1]]["weight"] for e2 in g.edges(e[1])]
            weights_other.sort()
            if weights_other[1]/current <= 0.25:
                score += current
            else:
                score -= current
        elif  g.degree[e[1]] ==1:
            current = g[e[0]][e[1]]["weight"]
            weights_other = [g[e2[0]][e2[1]]["weight"] for e2 in g.edges(e[0])]
            weights_other.sort()
            if weights_other[1] / current <= 0.25:
                score += current
            else:
                score -= current
        else:
            current = g[e[0]][e[1]]["weight"]
            weights_other_0 = [g[e2[0]][e2[1]]["weight"] for e2 in g.edges(e[0])]
            weights_other_0.sort()
            weights_other_1 = [g[e2[0]][e2[1]]["weight"] for e2 in g.edges(e[1])]
            weights_other_1.sort()
            if weights_other_0[1]/current <= 0.25 and weights_other_1[1]/current <= 0.25:
                score += current
            else:
                score -= current
    return score

def score_after_removal(node, g):
    dummyG = g.copy()
    dummyG.remove_node(node)
    return score_values(dummyG)


def scan_score(subset, max_its = 300):
    #maximize number of degree one nodes by deletion of vertices
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    # made the graph

    tracking_frame = pd.DataFrame(
        {'deg1': [], 'deg2': [], 'deg3': [], 'deg4': [], 'max_degree': [], 'totalNodes': [],
         'totalEdges': [], 'score':[]})

    highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
    #remove the degree 0 nodes
    print(g.number_of_nodes())
    for h in highest:
        if h[1] == 0:
            g.remove_node(h[0])
    print (g.number_of_nodes())
    #tracking
    degrees = [val for (node, val) in g.degree()]
    degreeCounter = Counter(degrees)
    tracking_frame.loc[0] = [degreeCounter[1], degreeCounter[2], degreeCounter[3], degreeCounter[4],
                             highest[0][1], g.number_of_nodes(), g.number_of_edges(), score_values(g)]
    peak_degree_1 = None
    for round in range(0, max_its):
        
        if round %50 == 0:
            print (round)
        
        highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
        for h in highest:
            if h[1] ==0 :
                g.remove_node(h[0])
        
        highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
        degrees = [val for (node, val) in g.degree()]
        degreeCounter = Counter(degrees)
        try:
            tracking_frame.loc[tracking_frame.shape[0]] = [degreeCounter[1], degreeCounter[2], degreeCounter[3],
                                                           degreeCounter[4], highest[0][1],
                                                           g.number_of_nodes(), g.number_of_edges(), score_values(g)]
            if degreeCounter[1] >= tracking_frame.deg1.max():
                peak_degree_1 = (round, [node for (node, val) in g.degree()])

            if highest[0][1] == 1:
                break
            
            with Pool() as pool:
                results = pool.map(partial(score_after_removal, g=g), [h[0] for  h in highest])
           
            max_node = highest[results.index(max(results))][0]
            g.remove_node(max_node)

        except:
            print ('broke')
    good_verts = [node for (node, val) in g.degree()]
    return good_verts, peak_degree_1, tracking_frame

#TODO
def scan_score_highest_removal(subset):
    #maximize number of degree one nodes by deletion of vertices
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    # made the graph

    tracking_frame = pd.DataFrame(
        {'deg1': [], 'deg2': [], 'deg3': [], 'deg4': [], 'max_degree': [], 'totalNodes': [],
         'totalEdges': [], 'score':[]})

    highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
    #remove the degree 0 nodes
    print(g.number_of_nodes())
    for h in highest:
        if h[1] == 0:
            g.remove_node(h[0])
    print (g.number_of_nodes())
    #tracking
    degrees = [val for (node, val) in g.degree()]
    degreeCounter = Counter(degrees)
    tracking_frame.loc[0] = [degreeCounter[1], degreeCounter[2], degreeCounter[3], degreeCounter[4],
                             highest[0][1], g.number_of_nodes(), g.number_of_edges(), score_values(g)]
    peak_degree_1 = None
    for round in range(0, 300):
        if round %50 == 0:
            print (round)
        
        highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
        for h in highest:
            if h[1] ==0 :
                g.remove_node(h[0])

        highest = sorted(g.degree, key=lambda x: x[1], reverse=True)
        degrees = [val for (node, val) in g.degree()]
        degreeCounter = Counter(degrees)
        tracking_frame.loc[tracking_frame.shape[0]] = [degreeCounter[1], degreeCounter[2], degreeCounter[3],
                                                       degreeCounter[4], highest[0][1],
                                                       g.number_of_nodes(), g.number_of_edges(), score_values(g)]
        if degreeCounter[1] >= tracking_frame.deg1.max():
            peak_degree_1 = (round, [node for (node, val) in g.degree()])

        if highest[0][1] == 1:
            break

        max_node = highest[0][0]
        g.remove_node(max_node)
        
    good_verts = [node for (node, val) in g.degree()]
    return good_verts, peak_degree_1, tracking_frame


