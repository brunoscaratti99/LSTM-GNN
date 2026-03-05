import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def adjacency_matrix(N_local, edge_index, edge_weight=None):
    A = torch.eye(N_local, device=edge_index.device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)
    for i in range(0, len(edge_index[0]), 2):
        u, v = edge_index[0][i], edge_index[0][i + 1]
        A[u, v] = edge_weight[i]
        A[v, u] = edge_weight[i]
    return A




def knn_topology(pos, stations, k=4):
    i = 0
    N = len(stations)
    coords = np.zeros((N,2))
    for name in stations.keys():
        pos[i]    = float(stations[name][1]), float(stations[name][0])
        coords[i] = [float(stations[name][1]), float(stations[name][0])]
        i += 1
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
    nbrs.fit(coords)
    
    distance, indices = nbrs.kneighbors(coords)
    
    E_1, E_2 = [], []
    
    for i in range(N):
        for j in indices[i][1:]:
            E_1.append(i)
            E_2.append(j)
            E_1.append(j)
            E_2.append(i)
    return torch.tensor([E_1,E_2], dtype=torch.int), pos
    

def plot_graph(N, edge_index, pos):
    graph_data = Data(x=torch.zeros(N), edge_index=edge_index)

    G = to_networkx(graph_data, to_undirected=True)
    
    plt.figure(figsize=(30,30))
    nx.draw(G, pos,
            with_labels=True,
            node_color='lightblue',
            node_size=700,
            width=3,
            font_weight='bold')
    nx.draw_networkx_labels(G, pos, font_color='black')
    plt.show()
    


"""
graph_data = Data(x=torch.zeros(62), edge_index=edge_index_knn)


G = to_networkx(graph_data, to_undirected=True)

plt.figure(figsize=(30,30))
nx.draw(G,pos,
        with_labels=True,
        node_color='lightblue',
        node_size=700,
        width=3,
        font_weight='bold')



nx.draw_networkx_labels(G,pos,font_color='black')
plt.show()
print("Construido grafo com ",N,"vértices")
"""
