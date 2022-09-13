"""Utility functions for building and analyzing user graphs."""

import re

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import stats

from sm_analysis.utils import *
from sm_analysis.twitter.utils import get_unique_ids

def make_user_graph(campaign, campaign_users):

    unique_ids = dict(get_unique_ids(campaign, campaign_users))
    id_df = pd.DataFrame(data=unique_ids.values(),
                         index=unique_ids.keys(),
                         columns=['tweet_count'])
    id_df['in_campaign'] = (id_df.index).isin(campaign_users.index)

    user_graph = nx.Graph()
    user_graph.add_nodes_from(
        id_df.index,
        count=id_df.tweet_count,
        in_campaign=id_df.in_campaign,
    )
    return user_graph


def make_connection_graph(tweetid, user1, user2, user_graph):

    graph = nx.MultiDiGraph(user_graph)

    tweetids = [dict(tweetid=tid) for tid in tweetid]
    graph.add_edges_from(list(zip(user1, user2, tweetids)))
    add_node_attributes(graph)

    return graph


def add_node_attributes(graph):
    """Compute various graph metrics and add to node attributes.
    """
    # degree
    nx.set_node_attributes(graph, dict(graph.out_degree(graph.nodes())),
                           'out_degree')
    nx.set_node_attributes(graph, dict(graph.in_degree(graph.nodes())),
                           'in_degree')

    if (type(graph) != nx.classes.multidigraph.MultiDiGraph):
        # eigenvector centrality
        nx.set_node_attributes(graph, nx.eigenvector_centrality(graph),
                               'eigenvector')

        # betweenness centrality
        nx.set_node_attributes(graph, nx.betweenness_centrality(graph),
                               'betweenness')

        # degree centrality
        nx.set_node_attributes(graph, nx.degree_centrality(re),
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
    attr_items = [(b, a) for (a, b) in attribute_dict.iteritems()]
    # Sort in descending order
    attr_items.sort()
    attr_items.reverse()
    return tuple(reversed(attr_items[0]))


def to_df(graph):
    """Export graph to Pandas dataframe with attributes as columns."""
    return pd.DataFrame(dict(graph.nodes(data=True))).T


def to_file(graph, file):
    """Export graph to .gexf file."""
    nx.write_gexf(graph, file)


def get_matrix(graph):
    return nx.to_numpy_matrix(graph)


def visualize(graph, format_dict=None):
    if format_dict is None:
        format_dict = {
            'font_size': 16,
            'width': 3,
            'edge_color': 'grey',
            'node_color': 'purple',
            'with_labels': False,
        }

    fig, ax = plt.subplots(figsize=(10, 8))

    pos = nx.spring_layout(graph, k=2)

    nx.draw_networkx(
        graph,
        pos,
        ax=ax,
        kwargs=format_dict,
    )
    # Offset labels
    for key, value in pos.items():
        x, y = value[0] + .135, value[1] + .045
        ax.text(x,
                y,
                s=key,
                bbox=dict(facecolor='red', alpha=0.25),
                horizontalalignment='center',
                fontsize=13)

    fig.show()

    return fig


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
