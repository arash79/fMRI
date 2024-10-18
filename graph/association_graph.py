import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from nichord.convert import convert_matrix
from nichord.chord import plot_chord
from nichord.glassbrain import plot_glassbrain


class AssociationGraph:
    
    def __init__(self, timestep, association_matrix, coordinates):
        self.timestep = timestep
        self.association_matrix = association_matrix
        self.coordinates = coordinates
        self.graph = nx.Graph()

    def calculate_weights_batch(self, batch):
        for i, j in batch:
            index = self.association_matrix.index[i]
            column = self.association_matrix.columns[j]
            weight = self.association_matrix.loc[index][column]
            if weight != 0:
                self.graph.add_edge(index, column, time=self.timestep, weight=int(weight * 100))

    def create_graph(self):
        num_rows = len(self.association_matrix)
        indexes = [(i, j) for i in range(num_rows) for j in range(i + 1, num_rows)]
        batch_size = 1000 

        with ThreadPoolExecutor() as executor:
            for i in range(0, len(indexes), batch_size):
                batch = indexes[i:i + batch_size]
                executor.submit(self.calculate_weights_batch, batch)
        
        return self.graph

    def draw_chord(self, fp_chord='chord_graph.png'):

        edges, edge_weights = convert_matrix(self.association_matrix.values)
        idx_to_label = {i: self.association_matrix.columns[i] for i in range(len(self.association_matrix.columns))}

        plot_chord(
            idx_to_label, edges, 
            edge_weights=edge_weights, fp_chord=fp_chord, 
            linewidths=2, alphas=0.1, do_ROI_circles=False, label_fontsize=20, 
            do_ROI_circles_specific=False, ROI_circle_radius=0.02, vmin=1, vmax=1
            )

    def draw_glass_brain(self, fp_glass='glassbrain.png'):
        
        edges, edge_weights = convert_matrix(self.association_matrix.values)
        idx_to_label = {i: self.association_matrix.columns[i] for i in range(len(self.association_matrix.columns))}

        plot_glassbrain(
            idx_to_label, edges, 
            edge_weights, fp_glass,
            self.coordinates, linewidths=5, 
            node_size=10, vmin=1, vmax=1
            )
        