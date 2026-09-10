import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected, coalesce
import sys
sys.path.append("../src")

from Data.feature_extraction import haversine_km
from Evaluation.plot_style import apply_seaborn_theme, save_figure


apply_seaborn_theme()

def max_graph_aggregate(H_prev, A, include_self=True):
    # H_prev: [B, H, N]
    # A: [N, N]

    if not include_self:
        A = A.clone()
        A.fill_diagonal_(0)

    mask = (A > 0)  # [N, N]
    H_nodes = H_prev.transpose(1, 2)  # [B, N, H]

    # [B, src, dst, H]
    msgs = H_nodes.unsqueeze(2).expand(-1, -1, A.size(1), -1)

    msgs = msgs.masked_fill(
        ~mask.unsqueeze(0).unsqueeze(-1),
        float("-inf")
    )

    H_graph = msgs.amax(dim=1)  # max sobre os vizinhos src -> [B, dst, H]
    H_graph = torch.where(torch.isfinite(H_graph), H_graph, torch.zeros_like(H_graph))

    return H_graph.transpose(1, 2)  # [B, H, N]



def adjacency_matrix(N, edge_index, edge_weight=None, symmetric=True):
    A = torch.zeros(N, N, device=edge_index.device)
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.shape[1], device=edge_index.device)

    if edge_index.max() == 0:
        return torch.eye(N, device=edge_index.device)

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


STATION_SIMILARITY_CHOICES = frozenset(
    {"gaussian", "ones", "inverse_distance", "climatology_correlation"}
)


def normalize_station_similarity(similarity: str) -> str:
    """Validate and normalize a station-similarity method name."""
    if not isinstance(similarity, str):
        raise TypeError("station_similarity must be a string.")
    normalized = similarity.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in STATION_SIMILARITY_CHOICES:
        available = ", ".join(sorted(STATION_SIMILARITY_CHOICES))
        raise ValueError(
            f"Unsupported station_similarity={similarity!r}. Available values: {available}."
        )
    return normalized


def _climatology_correlation_similarity(climatology, n_stations: int) -> torch.Tensor:
    """Return non-negative Pearson station similarities from training precipitation.

    Negative correlations do not represent an attractive edge in the current
    non-negative adjacency architecture, so they receive zero similarity.
    Invalid/constant station series are treated the same way.  The caller
    subsequently floors selected edge weights to keep the chosen topology.
    """
    values = np.asarray(
        climatology.detach().cpu().numpy() if torch.is_tensor(climatology) else climatology,
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != n_stations:
        raise ValueError(
            "climatology must have shape [time, station] with one column per station."
        )

    correlation = np.zeros((n_stations, n_stations), dtype=np.float64)
    np.fill_diagonal(correlation, 1.0)
    for source in range(n_stations):
        source_values = values[:, source]
        for destination in range(source + 1, n_stations):
            destination_values = values[:, destination]
            valid = np.isfinite(source_values) & np.isfinite(destination_values)
            if valid.sum() >= 2:
                source_valid = source_values[valid]
                destination_valid = destination_values[valid]
                if (
                    np.std(source_valid) > 0.0
                    and np.std(destination_valid) > 0.0
                ):
                    value = float(np.corrcoef(source_valid, destination_valid)[0, 1])
                    if np.isfinite(value):
                        correlation[source, destination] = value
                        correlation[destination, source] = value

    return torch.from_numpy(np.clip(correlation, 0.0, 1.0).astype(np.float32))


def station_similarity_edge_weights(
    stations,
    edge_index,
    *,
    station_similarity: str = "gaussian",
    gaussian_sigma_km: float = 100.0,
    climatology=None,
    minimum_weight: float = 1e-4,
) -> torch.Tensor:
    """Build non-negative initial weights, aligned to a station ``edge_index``.

    The station order is exactly ``stations.keys()``.  Returned values are for
    off-diagonal edges only; :func:`adjacency_matrix` always supplies the
    self-loop of one.  Values are floored at ``minimum_weight`` so a zero
    similarity never silently removes an edge from a locked KNN topology.

    ``gaussian`` uses ``exp(-d_ij**2 / (2 * sigma**2))`` with haversine
    distance in kilometres.  ``inverse_distance`` uses ``1 / d_ij``.  The
    model's positive residual parameterization accepts that physical scale
    directly, rather than forcing it into a sigmoid probability.
    """
    station_similarity = normalize_station_similarity(station_similarity)
    if not torch.is_tensor(edge_index) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must be a torch tensor with shape [2, E].")
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=edge_index.device)
    if minimum_weight <= 0.0 or not np.isfinite(minimum_weight):
        raise ValueError("minimum_weight must be finite and greater than zero.")

    station_names = list(stations.keys())
    n_stations = len(station_names)
    if n_stations < 1:
        raise ValueError("stations must contain at least one station.")
    if int(edge_index.min()) < 0 or int(edge_index.max()) >= n_stations:
        raise ValueError("edge_index contains an index outside the station mapping.")

    if station_similarity == "ones":
        similarity_matrix = torch.ones(
            (n_stations, n_stations), dtype=torch.float32, device=edge_index.device
        )
    elif station_similarity == "climatology_correlation":
        if climatology is None:
            raise ValueError(
                "station_similarity='climatology_correlation' requires training climatology."
            )
        similarity_matrix = _climatology_correlation_similarity(
            climatology, n_stations
        ).to(device=edge_index.device)
    else:
        latitudes = [float(stations[name][0]) for name in station_names]
        longitudes = [float(stations[name][1]) for name in station_names]
        distances = distance_matrix_from_lat_lon(
            latitudes, longitudes, dtype=torch.float32, device=edge_index.device
        )
        if station_similarity == "gaussian":
            if gaussian_sigma_km <= 0.0 or not np.isfinite(gaussian_sigma_km):
                raise ValueError("gaussian_sigma_km must be finite and greater than zero.")
            similarity_matrix = torch.exp(
                -(distances.square()) / (2.0 * float(gaussian_sigma_km) ** 2)
            )
        else:  # inverse_distance
            similarity_matrix = torch.zeros_like(distances)
            positive_distance = distances > 0.0
            similarity_matrix[positive_distance] = 1.0 / distances[positive_distance]
            similarity_matrix.fill_diagonal_(1.0)

    source, destination = edge_index
    weights = similarity_matrix[source, destination]
    if not torch.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError("station similarity produced invalid edge weights.")
    return weights.clamp_min(float(minimum_weight)).to(dtype=torch.float32)


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
    

def plot_graph(N, edge_index, pos, output_path=None, show=True):
    graph_data = Data(x=torch.zeros(N), edge_index=edge_index)

    G = to_networkx(graph_data, to_undirected=True)
    palette = sns.color_palette("crest", 5)

    fig, ax = plt.subplots(figsize=(18, 18))
    nx.draw(G, pos,
            with_labels=True,
            node_color=[palette[3]],
            edge_color=palette[1],
            node_size=620,
            width=1.8,
            alpha=0.9,
            font_weight='bold',
            font_size=8,
            ax=ax)
    nx.draw_networkx_labels(G, pos, font_color="#1f2933", font_size=8, ax=ax)
    ax.set_title("Initial station graph topology", fontsize=18, pad=18)
    ax.set_axis_off()
    try:
        if output_path is not None:
            save_figure(fig, output_path, dpi=190)
        if show:
            plt.show()
    finally:
        if not show:
            plt.close(fig)
    return output_path

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
