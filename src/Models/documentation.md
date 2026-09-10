# Models Package Documentation

`src/Models` defines neural architectures for station-level precipitation forecasting. The package currently contains GLSTM variants and a graph-temporal Transformer.

## Package-Level Conventions

- Model inputs usually follow `[batch, history, station, feature]`.
- Single-sample model helpers may accept `[history, station, feature]` and internally add a batch dimension.
- Multi-step outputs follow `[batch, horizon, station]` when `target_dim == 1`.
- With the maintained GLSTM's optional `learn_std=True`, `forward` returns a
  tuple containing that forecast plus `[batch, horizon]` learned population
  standard deviations across stations.
- `edge_index` follows PyTorch Geometric convention `[2, E]`.
- `out_channels` means forecast horizon in the maintained runner, not number of stations.

## Libraries Used

- `torch` and `torch.nn` for all neural-network modules.
- `math` for sinusoidal time encoding.
- `warnings` for defensive model-configuration warnings.
- `Graph.graph_related_utils.adjacency_matrix` and `max_graph_aggregate` for graph operations.

## `model.py`

### `GLSTMCell_v1`

First graph-aware LSTM cell variant.

- `__init__(self, N, input_size, hidden_size, edge_index, edge_weight=None, learn_adj=True)`: stores node count, feature sizes, graph topology, and learnable/fixed adjacency settings.
- `forward(self, X, H_prev, C_prev)`: updates hidden and cell states for one time step. Inputs are current node features and previous recurrent states. Output is `(C, H)`.

### `GLSTM_v1`

Stacked recurrent model using `GLSTMCell_v1`.

- `__init__(...)`: builds one or more recurrent graph-LSTM layers and a final projection head.
- `forward(self, x_seq)`: consumes a sequence shaped `[B, T, N, F]` and returns forecasts shaped by the configured output horizon/stations.

### `GLSTMCell_v2`

Numerically safer graph-aware LSTM cell with adjacency normalization and optional learned edge weights. As a standalone module it owns its graph state; inside the maintained `GLSTM_v2`, that state is tied to `cell_0` so every recurrent layer shares one adjacency.

- `__init__(..., edge_weight=None, learn_adj=True, lock_topology=True, cell_clip=5.0, eps=1e-6, learn_self_att=False)`: initializes gates, adjacency parameters, clipping limits, and epsilon safeguards. With an explicit `edge_weight`, each selected edge is `edge_weight_prior * exp(a_logits)`, with initial residual zero; this anchors calibration to geographic/climatological weights. Without explicit weights, the historical sigmoid initialization is retained for checkpoint compatibility. `learn_adj` controls the off-diagonal graph weights. `learn_self_att=False` preserves the historical unit diagonal, while `True` learns one positive self-attention weight per station as `exp(diag(a_logits))` and therefore requires `learn_adj=True`.
- `_normalize_adjacency(self, A)`: applies symmetric degree normalization.
- `current_adjacency(self, normalized=True)`: returns the current learned/fixed adjacency matrix. With learned self-attention, the raw matrix contains the positive, individually calibrated diagonal weights before symmetric degree normalization.
- `adjacency_anchor_loss()`: returns one-half of the squared deviation from the saved graph-prior residuals for existing edges.
- `reset_parameters(self)`: resets recurrent and adjacency parameters.
- `forward(self, X, H_prev, C_prev, A=None)`: one recurrent update. The maintained model always passes its single normalized adjacency explicitly, reusing the same tensor across all times and layers.

### `GLSTM_v2`

Maintained GLSTM model used by `src/run_experiment.py`.

- `__init__(self, N, edge_index, in_channels, hidden_size, out_channels, edge_weight=None, lstm_layers=1, aggr="vanilla", learn_adj=True, lock_topology=True, dropout=0.2, cell_clip=5.0, share_adjacency=True, learn_std=False, learn_self_att=False)`: builds the recurrent stack and forecast head. The maintained/default path ties every cell to one graph parameter and one set of topology buffers. With `lock_topology=True`, off-diagonal learning changes only initial edge weights; with `False`, new non-self edges may be learned. Explicit `edge_weight` values establish the shared graph prior. `learn_self_att` is GLSTM-only, requires `learn_adj=True`, and controls the diagonal separately. `share_adjacency=False` exists only to reproduce historical checkpoints that learned a different graph per layer. When `learn_std=True`, an auxiliary MLP receives the concatenated cross-node mean and population standard deviation of the last hidden representation and predicts one non-negative spread value per lead day. No auxiliary module is created when the option is false.
- `_tie_adjacency_state(self)`: aliases the first cell's graph parameter and buffers into every upper cell while retaining historical checkpoint key names, including the diagonal logits used when self-attention is learned.
- `current_adjacency(self, normalized=True)`: exposes the single adjacency effectively used by the whole recurrent stack and by plots/diagnostics.
- `adjacency_anchor_loss()`: delegates the shared graph-prior penalty once, irrespective of recurrent depth.
- `reset_parameters(self)`: resets recurrent cells, the shared adjacency, and projection layers.
- `forward(self, x_seq)`: accepts `[B, T, N, F]`, computes the normalized adjacency once, and reuses that exact tensor in every time/layer update. It returns `[B, horizon, N]` with `learn_std=False`, preserving the historical contract exactly, or `(forecast [B, horizon, N], predicted_node_std [B, horizon])` with `learn_std=True`.

### `NodewiseLSTM`

Independent non-graph baseline used by `run_experiment.py` when `EMPTY_GRAPH=True`.

- `__init__(self, N, in_channels, hidden_size, out_channels, lstm_layers=1, dropout=0.2)`: creates one distinct `nn.LSTM` and one distinct forecast head for every station. The parameters are not shared between stations and the model has no edges or graph aggregation.
- `forward(self, x_seq)`: accepts `[B, T, N, F]` and returns `[B, horizon, N]`. Altering one station's input cannot alter another station's prediction.
- `current_adjacency(...)`: returns the identity matrix solely so generic topology diagnostics can explicitly record the absence of non-self connections.

### `LearnableAdjacency`

Reusable adjacency module shared by graph-temporal models.

- `__init__(self, N, edge_index, edge_weight=None, learn_adj=True, lock_topology=True, eps=1e-6)`: creates an identity-plus-edge adjacency. Explicit weights use the same prior-times-log-residual parameterization as GLSTM. If learning is enabled, `lock_topology=True` masks logits to the base topology; `False` makes all non-self station pairs learnable, initialized near zero when absent initially. This Transformer-oriented module keeps its identity diagonal fixed and does not expose the GLSTM-only `learn_self_att` option.
- `_normalize_adjacency(self, A)`: symmetric degree normalization.
- `current_adjacency(self, normalized=True)`: returns normalized or raw adjacency.
- `adjacency_anchor_loss()`: returns the graph-prior penalty used by stable training.
- `reset_parameters(self)`: restores initial adjacency logits.

### `GraphResidualMixer`

Residual spatial mixer over station nodes.

- `__init__(self, hidden_size, dropout=0.1)`: creates layer norm, self projection, graph-context projection, output projection, activation, and dropout.
- `reset_parameters(self)`: resets all linear/layer-normalization modules.
- `forward(self, x, A)`: accepts `[..., N, H]`, computes graph context with dense adjacency `A`, and returns a residual update with the same shape.

### `GraphTemporalTransformer_v1`

Graph-temporal Transformer for precipitation forecasting.

- `__init__(...)`: configures station count, topology, feature count, hidden size, forecast horizon, target dimension, attention heads, encoder/decoder depth, dropout, max input lengths, optional node embeddings, optional precipitation residuals, and output activation.
- `reset_parameters(self)`: resets adjacency, projections, mixers, transformer layers, layer norms, node embeddings, horizon queries, and output head.
- `current_adjacency(self, normalized=True)`: exposes the adjacency used by plotting and diagnostics.
- `_resolve_four_dim_input(self, x)`: disambiguates 4D tensors as `[B, history, N, F]` or `[window, days, N, F]`.
- `_prepare_input(self, x)`: validates accepted input ranks: `[T, N, F]`, `[B, T, N, F]`, or `[B, window, days, N, F]`; returns a batched 4D tensor plus a squeeze flag.
- `_time_encoding(self, length, device, dtype)`: builds sinusoidal temporal encodings shaped for `[B, T, N, H]`.
- `_apply_output_activation(self, pred)`: applies optional `softplus` or `relu` output activation.
- `forward(self, x)`: projects features, adds time/node encodings, applies graph mixing, encodes temporal memory per station, decodes learned horizon queries, and returns `[B, horizon, N]` or the single-sample equivalent.

## `models_utils.py`

Utility functions:

- `reset_weights(module)`: recursively calls `reset_parameters()` on modules that expose it. Useful for reinitializing models between experiments.
