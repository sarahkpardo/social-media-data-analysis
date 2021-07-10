from collections import Counter
import functools
import itertools
from operator import itemgetter
import re
import timeit

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

from utils import *
from topic_utils import *


def add_node_attributes(G):
    """Compute various graph metrics and add to node attributes.
    """
    # degree
    nx.set_node_attributes(G,
                           dict(G.out_degree(G.nodes())), 
                           'out_degree')
    nx.set_node_attributes(G,
                           dict(G.in_degree(G.nodes())), 
                           'in_degree')
    
    if (type(G) != nx.classes.multidigraph.MultiDiGraph):
        # eigenvector centrality
        nx.set_node_attributes(G, 
                               nx.eigenvector_centrality(G),
                               'eigenvector')

        # betweenness centrality
        nx.set_node_attributes(G,
                               nx.betweenness_centrality(G), 
                               'betweenness')

        # degree centrality
        nx.set_node_attributes(G,
                               nx.degree_centrality(re),
                               'degree_centrality')
        
def top_nodes(attribute_dict, limit=20, show=False):
    """Return and optionally print the top n nodes in an attribute
    dictionary.
    """
    nodes = sorted(attribute_dict.items(), 
                   key=lambda item: item[1], 
                   reverse=True)[:limit]
    if show:
        print("Top {} nodes:".format(limit))
        for d in nodes:
            print(d)
        
    return nodes
        
def highest_value(attribute_dict):     
    """Find the node with largest value in an attribute dictionary.
    
    Return: 
        tuple(node, value)
        
    """     
    # Ordered tuple
    attr_items = [(b,a) for (a,b) in attribute_dict.iteritems()]    
    # Sort in descending order     
    attr_items.sort()  
    attr_items.reverse()
    return tuple(reversed(attr_items[0]))


def to_df(G):
    """Export graph to Pandas dataframe with attributes as columns."""
    return pd.DataFrame(dict(G.nodes(data=True))).T
    
def to_file(G, file):
    """Export graph to .gexf file."""
    nx.write_gexf(G, file)
    
def to_txt(G, file):
    """Export node attributes to .txt file."""
    results = [(k, bet_cen[k], clo_cen[k], eig_cen[k]) for k in range(len(nodes))]
    
    f = open(file,'w')
    for item in results:     
        f.write(','.join(map(str,item)))     
        f.write('\n')
    f.close()

def get_matrix(G):
    return nx.to_numpy_matrix(G)


def visualize(G, format_dict=None):
    if format_dict == None:
        format_dict = {'font_size':16,
                       'width':3,
                       'edge_color':'grey',
                       'node_color':'purple',
                       'with_labels':False,
                      }

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(G, k=2)

    nx.draw_networkx(G, 
                     pos,
                     ax=ax,
                     kwargs=format_dict,
                    )
    # Offset labels
    for key, value in pos.items():
        x, y = value[0]+.135, value[1]+.045
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='red', alpha=0.25),
                horizontalalignment='center', fontsize=13)
        
def graph_properties(graph):
    properties = []
    out = "Nodes: {}".format(graph.number_of_nodes())
    properties.append(out)
    print(out)
    
    out = "Edges: {}".format(graph.number_of_edges())
    print(out)
    properties.append(out)
    
    degrees = [val for (node, val) in graph.degree()]
    out = "Maximum degree: {}".format(np.max(degrees))
    properties.append(out)
    print(out)
    
    out = "Minimum degree: {}".format(np.min(degrees))
    properties.append(out)
    print(out)
    
    out = "Average degree: {:.1f}".format(np.mean(degrees))
    properties.append(out)
    print(out)
    
    out = "Mode: {}".format(stats.mode(degrees)[0][0])
    properties.append(out)
    print(out)
    
    return properties
