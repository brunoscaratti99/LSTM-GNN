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


def experimental_graph(stations, total_neighbors:int, random:int = 0):
    
    """
    Makes a graph with a total number of neighbors in which "random" number of them are choosen randomly instead
    of by knn
    """

    # Get the default knn topology
    edge_index, pos = knn_topology(stations = stations, k = total_neighbors - random)

    plot_graph(len(pos), edge_index, pos)

    # Guarantee that it is ordered
    perm = (edge_index[0] * edge_index.size(1) + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]

    # Make and aux list of unique edges
    index_list = torch.unique(edge_index[0])

    # Make an aux list for new edges to be added
    new_edges = []

    i = 0

    # Loop while i is still in range of all indexes
    while i < edge_index.shape[1]:
        
        source = edge_index[0, i].item()
        
        # Collect neighbors of this source
        blacklist = set()
        blacklist.add(source)
        
        j = i
        # While the next node exists and is the same as the current
        while j < edge_index.shape[1] and edge_index[0, j] == source:
            # Add its neighbor to the black list
            blacklist.add(edge_index[1, j].item())
            # Go to next neighbor
            j += 1
            # NOTE: when edge_index[0, j+1] is a new node, the code will still add
            # +1 to j but it will get out of the while loop, therefore j will have
            # the first index of the next source node

        # Add random neighbors
        added = 0

        # While we still have random nodes to add
        while added < random:
            # Sample one from all possible indexes
            value = index_list[torch.randint(len(index_list), () )].item()
            
            # If it has not been blacklisted
            if value not in blacklist:
                # Add the edge to new edges
                new_edges.append([source, value])
                # Blacklist it so it doesnt get added twice
                blacklist.add(value)
                added += 1

        # Since j has the first index of the next source node we change i into it
        i = j

    # Add all new edges
    if new_edges:
        new_edges = torch.tensor(new_edges).T
        edge_index = torch.cat([edge_index, new_edges], dim = 1)

    return(edge_index, pos)


def distance_matrix_from_lat_lon(latitudes, longitudes, dtype=torch.float32, device=None):
    """
    Constrói uma matriz A [N, N] em que A[i, j] é a distância, em km,
    entre as estações i e j a partir de suas latitudes e longitudes.

    Parameters
    ----------
    latitudes : array-like
        Vetor 1D com as latitudes das estações em graus.
    longitudes : array-like
        Vetor 1D com as longitudes das estações em graus.
    dtype : torch.dtype, optional
        Tipo do tensor de saída.
    device : str ou torch.device, optional
        Dispositivo do tensor de saída.
    """
    latitudes = torch.as_tensor(latitudes, dtype=torch.float64)
    longitudes = torch.as_tensor(longitudes, dtype=torch.float64)

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("latitudes e longitudes devem ser vetores 1D.")
    if latitudes.numel() != longitudes.numel():
        raise ValueError("latitudes e longitudes devem ter o mesmo tamanho.")

    lat_rad = torch.deg2rad(latitudes)
    lon_rad = torch.deg2rad(longitudes)

    dlat = lat_rad[:, None] - lat_rad[None, :]
    dlon = lon_rad[:, None] - lon_rad[None, :]

    a = (
        torch.sin(dlat / 2) ** 2
        + torch.cos(lat_rad[:, None]) * torch.cos(lat_rad[None, :]) * torch.sin(dlon / 2) ** 2
    )
    a = torch.clamp(a, 0.0, 1.0)

    earth_radius_km = 6371.0
    distances = 2.0 * earth_radius_km * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))
    distances.fill_diagonal_(0.0)

    return distances.to(dtype=dtype, device=device)


def exp_distance_adjacency_matrix(
    N,
    edge_index,
    edge_distance,
    sigma,
    threshold=0.0,
    symmetric=True,
    fill_diagonal=1.0,
):
    """
    Constrói uma matriz de adjacência A [N, N] a partir de um edge_index [2, E],
    usando pesos A[i, j] = exp(-d_ij / sigma^2) para as arestas cujo valor
    seja maior ou igual a `threshold`.

    Parameters
    ----------
    N : int
        Número de nós.
    edge_index : torch.Tensor
        Tensor [2, E] com pares (origem, destino).
    edge_distance : array-like ou torch.Tensor
        Distância associada a cada aresta, com tamanho E.
    sigma : float
        Parâmetro da exponencial.
    threshold : float, optional
        Valor mínimo para manter a aresta na matriz.
    symmetric : bool, optional
        Se True, espelha A[i, j] em A[j, i].
    fill_diagonal : float ou None, optional
        Valor da diagonal. Use None para não alterar a diagonal.
    """
    if sigma <= 0:
        raise ValueError("sigma deve ser maior que zero.")

    edge_distance = torch.as_tensor(edge_distance, dtype=torch.float32, device=edge_index.device)
    if edge_distance.numel() != edge_index.shape[1]:
        raise ValueError("edge_distance deve ter o mesmo número de elementos que o número de arestas em edge_index.")

    edge_weight = torch.exp(-edge_distance / (sigma ** 2))
    keep = edge_weight >= threshold

    A = torch.zeros((N, N), dtype=edge_weight.dtype, device=edge_index.device)
    src, dst = edge_index[:, keep]
    A[src, dst] = edge_weight[keep]

    if symmetric:
        A[dst, src] = edge_weight[keep]

    if fill_diagonal is not None:
        A.fill_diagonal_(fill_diagonal)

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
