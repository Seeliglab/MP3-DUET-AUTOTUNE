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
import networkx as nx
from collections import Counter

#make svg text editable 
import matplotlib as mpl
new_rc_params = {'text.usetex': False,
"svg.fonttype": 'none'
}
mpl.rcParams.update(new_rc_params)
from functools import partial





def reduction_by_maximal_degree_deletion(subset):
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    # made the graph
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
        
        if round %25 == 0:
            print (round, g.number_of_nodes())
        
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
        
        with Pool() as pool:
            results = pool.map(partial(score_after_removal, g=g), [h[0] for  h in highest])
        
        max_node = highest[results.index(max(results))][0]
        g.remove_node(max_node)

    good_verts = [node for (node, val) in g.degree()]
    return good_verts, peak_degree_1, tracking_frame

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


def make_graph_and_score_only(subset):
    g = nx.Graph()
    for ind, row in subset.iterrows():
        g.add_edge(row['pro1'], row['pro2'], weight = row['ashr_log2FoldChange_HIS_TRP'])
    return score_values(g)