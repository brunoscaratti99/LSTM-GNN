# Graph Package Documentation

`src/Graph` builds station graphs, adjacency matrices, distance matrices, KNN topology, and graph visualizations. These utilities feed the model-level adjacency shared by all maintained GLSTM layers and the graph-temporal Transformer components.

## Package-Level Conventions

- Station dictionaries are generally `{station_name: [latitude, longitude]}`.
- `edge_index` follows PyTorch Geometric convention with shape `[2, E]`.
- Dense adjacency matrices are `[N, N]`, where rows/columns correspond to station order.
- Distances are measured in kilometers unless stated otherwise.
- KNN topology uses station coordinates and returns both `edge_index` and node positions.

## Libraries Used

- `torch` for adjacency tensors and graph aggregation.
- `torch_geometric.data` and `torch_geometric.utils` for graph data and NetworkX conversion.
- `sklearn.neighbors.NearestNeighbors` for KNN graph construction.
- `numpy` for distance calculations.
- `networkx`, `matplotlib`, and `seaborn` for graph plotting.
- `Data.feature_extraction.haversine_km` for geographic distance.
- `Evaluation.plot_style` for consistent plot saving.

## `graph_related_utils.py`

Functions:

- `max_graph_aggregate(H_prev, A, include_self=True)`: aggregates hidden states over graph neighbors with a max operation. Inputs: `H_prev` shaped `[B, H, N]` and dense adjacency `[N, N]`. Output: `[B, H, N]`.
- `adjacency_matrix(N, edge_index, edge_weight=None, symmetric=True)`: converts `edge_index` and optional edge weights into a dense adjacency matrix. If `symmetric=True`, reciprocal entries are filled.
- `experimental_graph(stations, total_neighbors, random=0)`: builds an experimental graph with a mixture of nearest and random neighbors.
- `distance_matrix_from_lat_lon(latitudes, longitudes, dtype=torch.float32, device=None)`: computes an `[N, N]` matrix of haversine distances from coordinate arrays.
- `normalize_station_similarity(similarity)`: validates one of `"gaussian"`, `"ones"`, `"inverse_distance"`, or `"climatology_correlation"`.
- `station_similarity_edge_weights(stations, edge_index, station_similarity=..., gaussian_sigma_km=..., climatology=...)`: creates initial weights aligned to `edge_index`. Gaussian uses `exp(-d²/(2σ²))` in kilometres; inverse distance uses `1/d`; climatology correlation is the non-negative Pearson correlation from training precipitation only; and `ones` assigns one to every edge. Zero similarities are floored to `1e-4` so they do not delete a selected KNN edge.
- `exp_distance_adjacency_matrix(N, edge_index, edge_distance, sigma, threshold=0.0, symmetric=True, fill_diagonal=1.0)`: converts edge distances into exponential-decay weights, optionally thresholding and symmetrizing.
- `graph_matrix_index(A, threshold=1e-6, directed=False, include_self=False)`: converts a dense adjacency matrix back to `edge_index` form using a threshold.
- `knn_topology(stations, k=4)`: builds a K-nearest-neighbor station graph. Input: station dictionary. Output: `edge_index` and `pos` dictionary for plotting.
- `plot_graph(N, edge_index, pos, output_path=None, show=True)`: renders a station graph using NetworkX and the shared plot style. Returns the output path when saving.
- `distance_graph(stations, criterion=120)`: builds a graph by connecting stations within a distance threshold.

## Input and Output Patterns

- GLSTM models usually receive the `edge_index` returned by `knn_topology(...)`; `GLSTM_v2` converts it to one dense adjacency reused by every recurrent layer. The maintained runner also supplies station-similarity weights as the graph prior.
- `LearnableAdjacency` in `src/Models/model.py` converts `edge_index` to a dense adjacency through `adjacency_matrix(...)`.
- Plotting functions assume `pos` maps node ids to coordinate-like pairs.
- Distance helpers expect latitude and longitude arrays in station order.

## Notes for Future Changes

- Keep graph construction deterministic unless an experiment explicitly requires random topology.
- If a graph function changes station ordering, update the caller documentation and any run metadata that depends on station names.
- Prefer returning both topology and plotting positions for new station-graph builders.
