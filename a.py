from loader import Loader
from parcellation import AAL
from measures.correlation import CorrelationBasedBrainNetwork
from measures.covariance import CovarianceBasedBrainNetwork
from measures.causality import GrangerCausalityBasedBrainNetwork
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output
import numpy as np
from graph import AssociationGraph
import igraph
import numpy as np
import igraph as ig
import dionysus as d
import multiprocessing
import networkx as nx
import dill
import pandas as pd
from math import ceil
import igraph as ig


group = 'SCZ'
modality = 'BOLD'
task = 'task001_run001'
kind = 'bold_mcf_brain'

agent = Loader()
data = agent.filter_data(group=group, modality=modality, task=task, kind=kind)
agent = AAL(data)
parcellation_results = agent.parcellate()

subject, coords = parcellation_results['sub001']

class TemporalLabelling:

    def __init__(
            self, 
            time_series, 
            coordinates,
            resolution, 
            overlap, 
            kind='positive', 
            threshold=80, 
            method='correlation'
            ):
        
        self.timeseries = time_series
        self.coordinates = coordinates
        self.resolution = resolution
        self.overlap = overlap
        self.kind = kind
        self.threshold = threshold
        self.method = self.__switch(method=method)
    
    @staticmethod
    def __switch(method):
        if method == 'correlation':
            return CorrelationBasedBrainNetwork
        if method == 'covariance':
            return CovarianceBasedBrainNetwork
        if method == 'causal':
            return GrangerCausalityBasedBrainNetwork
    
    @staticmethod
    def __partition_dataframe(df, window_size, overlap_size):
        step = window_size - overlap_size
        return [df.iloc[i:i + window_size] for i in range(0, len(df), step)]
    
    @staticmethod
    def __combine_networkx_graphs(graphs):
        combined_graph = nx.MultiGraph()

        for g in graphs:
            combined_graph.add_nodes_from(g.nodes(data=True))

        for g in graphs:
            for u, v, data in g.edges(data=True):
                combined_graph.add_edge(u, v, **data)

        return combined_graph
    
    def __preprocess(self, adjacency_matrix):

        node_names, adjacency_matrix = adjacency_matrix.columns, adjacency_matrix.values
        
        if self.kind == 'positive':
            adjacency_matrix = np.where(adjacency_matrix > 0, adjacency_matrix, 0)
        else:
            adjacency_matrix = np.where(adjacency_matrix < 0, adjacency_matrix, 0)
        
        edge_weights = adjacency_matrix[np.triu_indices(adjacency_matrix.shape[0])]
        threshold = np.percentile(edge_weights, self.threshold)

        adjacency_matrix = np.where(adjacency_matrix >= threshold, adjacency_matrix, 0)
        
        adjacency_matrix = pd.DataFrame(
            adjacency_matrix, 
            columns=node_names, 
            index=node_names
            )
        
        return adjacency_matrix

    def partition_data(self):

        partitions = self.__partition_dataframe(
            self.timeseries,
            self.resolution,
            self.overlap
        )
        partitions = [
            self.method(partition, sparsify=True).generate_connectivity_matrix() 
            for partition in partitions
            ]
        partitions = [
            AssociationGraph(timestep, self.__preprocess(partition), self.coordinates).create_graph()
            for timestep, partition in enumerate(partitions)
        ]

        brain_network = self.__combine_networkx_graphs(partitions)
        
        return brain_network


a = TemporalLabelling(subject, coords, 30, 15)

g = a.partition_data()


class SlidingWindows:

    def __init__(self, graph):
        self.graph = self.__convert_to_igraph(graph)

    @staticmethod
    def __convert_to_igraph(graph):
        graph = ig.Graph.from_networkx(graph)
        graph.vs["name"] = graph.vs["_nx_name"]
        del(graph.vs["_nx_name"])
        return graph
    
    def generate(self, partition_count, overlap):
        times = np.array(self.graph.es["time"])
        resolution = 1 / partition_count
        duration = round((resolution) * (times.max() - times.min()), 1)
        windows = []
        for i in range(int(1 / resolution)):
            edges = self.graph.es.select(time_gt=times.min() + duration*i - overlap,
                                time_lt=times.min() + duration*(i+1))
            windows.append(self.graph.subgraph_edges(edges))
        return windows
    

windows = SlidingWindows(g).generate(5, 1)


def max_simplicial_complex(g):
    """Return the maximal simplicial complex of a network g."""
    return d.Filtration(g.maximal_cliques())


def find_transitions(a):
    """Find the transition times in an array of presence times."""
    res = []
    prev = False
    for i, cur in enumerate(a):
        if cur != prev:
            res.append(i)
        prev = cur
    return res


def presence_times(g):
    """Compute the data required to compute zigzag persistence:
    simplicial complex and transition times.
    :param g: igraph Graph
    :return: a tuple with the maximum simplicial complex and the
    transition times of each simplex."""
    max_simplicial_complex = d.Filtration(g.cliques())
    filts = []
    for t in np.sort(np.unique(g.es["time"])):
        edges = g.es.select(time_eq=t)
        cliques = g.subgraph_edges(edges).cliques()
        filts.append(d.Filtration(cliques))
        print('hi', t)
    presences = [[s in filt for filt in filts] for s in max_simplicial_complex]
    presences = [find_transitions(p) for p in presences]
    return (max_simplicial_complex, presences)


def zigzag_network(g):
    """Compute zigzag persistence on a temporal network.
    :param g: igraph Graph
    :return: a list of persistence diagrams.
    """
    (f, t) = presence_times(g)
    _, dgms, _ = d.zigzag_homology_persistence(f, t)
    return dgms


pool = multiprocessing.Pool(processes=3)


print("Zigzag persistence...", end="", flush=True)
zz_dgms = pool.map(zigzag_network, windows)
# zz_dgms = zigzag_network(windows)
dill.dump(zz_dgms, open("generative/zz_dgms.dill", "wb"))
print("done, saved.")