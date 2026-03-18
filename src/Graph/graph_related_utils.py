import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected, coalesce
import sys
sys.path.append("../src")

from Data.feature_extraction import haversine_km


def adjacency_matrix(N, edge_index, edge_weight=None, symmetric=True):
    A = torch.zeros(N, N, device=edge_index.device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)

    src, dst = edge_index  # edge_index[0] e edge_index[1]
    A[src, dst] = edge_weight

    if symmetric:
        A[dst, src] = edge_weight

    A.fill_diagonal_(1.0)

    return A

def graph_matrix_index(A, threshold=1e-6, directed=False, include_self=False):
    # A: [N, N]
    N = A.shape[0]
    mask = A.abs() > threshold

    if not include_self:
        mask = mask & ~torch.eye(N, dtype=torch.bool, device=A.device)

    idx = mask.nonzero(as_tuple=False)  # [E, 2] com pares (i,j)

    if not directed:
        # mantém só triângulo superior e espelha -> evita duplicatas
        idx = idx[idx[:, 0] < idx[:, 1]]
        idx = torch.cat([idx, idx[:, [1, 0]]], dim=0)

    edge_index = idx.t().contiguous().long()  # [2, E]
    return edge_index


def knn_topology(stations, k=4):
    i = 0
    N = len(stations)
    pos = {}
    
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
            if i == j:
                continue
            E_1.append(i)
            E_2.append(j)

    edge_index = torch.tensor([E_1, E_2], dtype=torch.long)
    # garante bidirecional e remove duplicatas (evita grau inflado)
    edge_index = to_undirected(edge_index, num_nodes=N)
    edge_index, _ = coalesce(edge_index, None, num_nodes=N)
    return edge_index, pos
    

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

def distance_graph(stations, criterion=120):
    N = len(stations)
    E_1, E_2 = [], []
    edge_weight = []
    pos = []
    for i, name in enumerate(stations.keys()):
        pos.append([float(stations[name][1]), float(stations[name][0])])
        lat_i, lon_i = float(stations[name][1]), float(stations[name][0])
        for j, name_2 in enumerate(stations.keys()):
            if i >= j:
                continue
            lat_j, lon_j = float(stations[name_2][1]), float(stations[name_2][0])
            dist = haversine_km(lat_i, lon_i, lat_j, lon_j)
            if dist<=criterion:
                E_1.append(i)
                E_2.append(j)
                E_2.append(i)
                E_1.append(j)
                edge_weight.append(dist)
                edge_weight.append(dist)
    return torch.tensor([E_1, E_2], dtype=torch.long), pos, edge_weight

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
