import torch
from Graph.graph_related_utils import adjacency_matrix
import torch.nn as nn
import math
import sys
import os
import warnings
sys.path.append(os.path.join("..", "src"))
from Graph.graph_related_utils import max_graph_aggregate


device = 'cuda' if torch.cuda.is_available() else 'cpu'




class GLSTMCell_v1(nn.Module):
    def __init__(self, N, input_size, hidden_size, edge_index, edge_weight=None, learn_adj=True):
        super().__init__()
        self.edge_index = edge_index
        self.learn_adj = learn_adj
        
        
        if edge_weight is None:
            self.edge_weight = None if not learn_adj else torch.ones(edge_index.shape[1], device=edge_index.device)
        else:
            self.edge_weight = edge_weight.to(edge_index.device) if torch.is_tensor(edge_weight) else torch.tensor(edge_weight, device=edge_index.device)

        A_init = adjacency_matrix(N, edge_index, self.edge_weight)
        
        if learn_adj:
            self.A = nn.Parameter(A_init)
        else:
            self.register_buffer("A", A_init)
            
        
        self.W_i = nn.Linear(input_size, hidden_size).to(device)
        self.W_f = nn.Linear(input_size, hidden_size).to(device)
        self.W_o = nn.Linear(input_size, hidden_size).to(device)
        self.W_u = nn.Linear(input_size, hidden_size).to(device)

        self.U_i = nn.Linear(hidden_size, hidden_size).to(device)
        self.U_f = nn.Linear(hidden_size, hidden_size).to(device)
        self.U_o = nn.Linear(hidden_size, hidden_size).to(device)
        self.U_u = nn.Linear(hidden_size, hidden_size).to(device)

    def forward(self, X, H_prev, C_prev):
        # X: [B, N, F], H/C: [B, H, N]
        T, N, F = X.shape
        if self.learn_adj:
                    
            H_graph = torch.einsum("bhm,mn->bhn", H_prev, self.A).to(device)
            C_graph = torch.einsum("bhm,mn->bhn", C_prev, self.A).to(device)


            Hg = H_graph.permute(0, 2, 1).to(device)
            Hp = H_prev.permute(0, 2, 1).to(device)
        else:

            H_graph = torch.einsum("bhm,mn->bhn", H_prev, self.A).to(device)
            C_graph = torch.einsum("bhm,mn->bhn", C_prev, self.A).to(device)


            Hg = H_graph.permute(0, 2, 1).to(device)
            Hp = H_prev.permute(0, 2, 1).to(device)
            
       
        Xp = X.to(device)

        I = torch.sigmoid(self.W_i(Xp) + self.U_i(Hg)).permute(0, 2, 1)
        Fg = torch.sigmoid(self.W_f(Xp) + self.U_f(Hp)).permute(0, 2, 1)
        O = torch.sigmoid(self.W_o(Xp) + self.U_o(Hg)).permute(0, 2, 1)
        U = torch.tanh(self.W_u(Xp) + self.U_u(Hg)).permute(0, 2, 1)

        C_t = I * U + Fg * C_graph
        H_t = O * torch.tanh(C_t)
        return C_t, H_t


class GLSTM_v1(nn.Module):
    def __init__(self, N, edge_index, in_channels, hidden_size, out_channels, lstm_layers=1, learn_adj=True, dropout=0.2):
        super().__init__()
        self.N = N
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.window = out_channels
        
        #LSTM blocks
        self.cell_0 = GLSTMCell_v1(N, in_channels, hidden_size, edge_index, learn_adj=learn_adj)
        self.cells = nn.ModuleList([GLSTMCell_v1(N=N, input_size=hidden_size, hidden_size=hidden_size,edge_index=edge_index, learn_adj=learn_adj) for _ in range(self.lstm_layers-1)])
        
        #fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_channels)
        ).to(device)
        

    def forward(self, x_seq):
        # x_seq: [B, T, N, F]
        
        B, T, N, _ = x_seq.shape

        H = torch.zeros((B, self.hidden_size, self.N), device=x_seq.device)
        C = torch.zeros((B, self.hidden_size, self.N), device=x_seq.device)
        
        
        H_layers = torch.zeros((B, T, self.N, self.hidden_size), device=x_seq.device, dtype=x_seq.dtype)
        
        
        for t in range(T):
            C, H = self.cell_0(x_seq[:,t], H, C)
            H_layers[:,t] = H.permute(0,2,1)
        

        for j in range(self.lstm_layers-1):
                H = torch.zeros((B, self.hidden_size, self.N), device=x_seq.device)
                C = torch.zeros((B, self.hidden_size, self.N), device=x_seq.device)
                for t in range(T):

                    C, H = self.cells[j](H_layers[:,t], H, C)
                    H_layers[:,t] = H.permute(0,2,1)

        out = H_layers[:,-1].to(device)

        return self.fc(out).permute(0, 2, 1)   # [B, OUT, N]


class GLSTMCell_v2(nn.Module):
    def __init__(self, N, input_size, hidden_size, edge_index, edge_weight=None, learn_adj=True, lock_topology=True, cell_clip=5.0, eps=1e-6, learn_self_att=False):
        super().__init__()
        self.N = N
        self.hidden_size = hidden_size
        self.learn_adj = bool(learn_adj)
        self.learn_self_att = bool(learn_self_att)
        if self.learn_self_att and not self.learn_adj:
            raise ValueError("learn_self_att=True requires learn_adj=True.")
        self.lock_topology = lock_topology
        self.cell_clip = cell_clip
        self.eps = eps

        uses_edge_weight_prior = edge_weight is not None
        A_base = adjacency_matrix(N, edge_index, edge_weight).float().to(edge_index.device)
        I = torch.eye(N, device=A_base.device, dtype=A_base.dtype)
        offdiag_init = A_base.clone()
        offdiag_init.fill_diagonal_(0.0)
        edge_mask = (offdiag_init > 0).float()
        topology_mask = edge_mask if lock_topology else (torch.ones_like(edge_mask) - I)

        self.register_buffer("I", I)
        self.register_buffer("edge_mask", edge_mask)
        self.register_buffer("topology_mask", topology_mask)
        self.register_buffer("edge_weight_prior", offdiag_init)
        self.register_buffer(
            "uses_edge_weight_prior",
            torch.tensor(uses_edge_weight_prior, dtype=torch.bool, device=A_base.device),
        )

        if learn_adj:
            if topology_mask.any():
                if uses_edge_weight_prior:
                    # A learnable log-residual is anchored at zero. The raw
                    # edge is prior * exp(residual), so regularization keeps
                    # the selected prior instead of pulling weights to 0.5.
                    init_logits = torch.where(
                        edge_mask > 0,
                        torch.zeros_like(offdiag_init),
                        torch.full_like(offdiag_init, -9.21024036697585),
                    ) * topology_mask
                else:
                    # Preserve the historical default when no explicit
                    # station-similarity prior was supplied.
                    max_val = offdiag_init.max().clamp_min(1.0)
                    init_probs = (offdiag_init / max_val).clamp(0.05, 0.95)
                    init_probs = torch.where(edge_mask > 0, init_probs, torch.full_like(init_probs, 1e-4))
                    init_logits = torch.logit(init_probs, eps=1e-4) * topology_mask
            else:
                init_logits = torch.zeros_like(offdiag_init)

            self.register_buffer("a_logits_init", init_logits.clone())
            self.a_logits = nn.Parameter(init_logits.clone())
        else:
            A_fixed_raw = I + offdiag_init
            A_fixed = self._normalize_adjacency(A_fixed_raw)
            self.register_buffer("A_fixed_raw", A_fixed_raw)
            self.register_buffer("A_fixed", A_fixed)

        self.W_i = nn.Linear(input_size, hidden_size)
        self.W_f = nn.Linear(input_size, hidden_size)
        self.W_o = nn.Linear(input_size, hidden_size)
        self.W_u = nn.Linear(input_size, hidden_size)

        self.U_i = nn.Linear(hidden_size, hidden_size)
        self.U_f = nn.Linear(hidden_size, hidden_size)
        self.U_o = nn.Linear(hidden_size, hidden_size)
        self.U_u = nn.Linear(hidden_size, hidden_size)

    def _normalize_adjacency(self, A):
        deg = A.sum(dim=1).clamp_min(self.eps)
        deg_inv_sqrt = deg.rsqrt()
        return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]

    def current_adjacency(self, normalized=True):
        if not self.learn_adj:
            return self.A_fixed if normalized else self.A_fixed_raw

        if bool(self.uses_edge_weight_prior.item()):
            existing_edges = self.edge_weight_prior * torch.exp(self.a_logits) * self.edge_mask
            candidate_edges = torch.sigmoid(self.a_logits) * (self.topology_mask - self.edge_mask)
            offdiag = existing_edges + candidate_edges
        else:
            offdiag = torch.sigmoid(self.a_logits) * self.topology_mask
        offdiag = 0.5 * (offdiag + offdiag.T)
        self_attention = self.I
        if self.learn_self_att:
            # Diagonal logits represent log self-loop weights. Zero therefore
            # preserves the initial identity diagonal while allowing calibration.
            self_attention = torch.diag_embed(torch.exp(torch.diagonal(self.a_logits)))
        A = self_attention + offdiag
        return self._normalize_adjacency(A) if normalized else A

    def _reset_recurrent_parameters(self):
        for layer in (self.W_i, self.W_f, self.W_o, self.W_u, self.U_i, self.U_f, self.U_o, self.U_u):
            layer.reset_parameters()

    def _reset_adjacency_parameters(self):
        if self.learn_adj:
            with torch.no_grad():
                self.a_logits.copy_(self.a_logits_init)

    def adjacency_anchor_loss(self):
        """L2 penalty that anchors learned edge residuals to their prior."""
        if not self.learn_adj:
            return self.I.new_zeros(())
        anchor_mask = self.edge_mask
        if self.learn_self_att:
            anchor_mask = anchor_mask + self.I
        delta = (self.a_logits - self.a_logits_init) * anchor_mask
        return 0.5 * delta.square().sum()

    def reset_parameters(self):
        self._reset_recurrent_parameters()
        self._reset_adjacency_parameters()

    def forward(self, X, H_prev, C_prev, A=None):
        if A is None:
            A = self.current_adjacency(normalized=True)
        H_graph = torch.matmul(H_prev, A)
        C_graph = torch.matmul(C_prev, A)

        Xp = X
        Hg = H_graph.transpose(1, 2)
        Hp = H_prev.transpose(1, 2)

        I_gate = torch.sigmoid(self.W_i(Xp) + self.U_i(Hg)).transpose(1, 2)
        F_gate = torch.sigmoid(self.W_f(Xp) + self.U_f(Hp)).transpose(1, 2)
        O_gate = torch.sigmoid(self.W_o(Xp) + self.U_o(Hg)).transpose(1, 2)
        U_gate = torch.tanh(self.W_u(Xp) + self.U_u(Hg)).transpose(1, 2)

        C_t = I_gate * U_gate + F_gate * C_graph
        if self.cell_clip is not None:
            C_t = C_t.clamp(-self.cell_clip, self.cell_clip)
        H_t = O_gate * torch.tanh(C_t)
        return C_t, H_t



    
class GLSTM_v2(nn.Module):
    def __init__(self, N, edge_index, in_channels, hidden_size, out_channels, edge_weight=None, lstm_layers=1, aggr='vanilla', learn_adj=True, lock_topology=True, dropout=0.2, cell_clip=5.0, share_adjacency=True, learn_std=False, learn_self_att=False):
        super().__init__()
        self.N = N
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.window = out_channels
        #self.fc_layers = fc_layers
        self.dropout = dropout
        self.aggr = aggr
        self.share_adjacency = bool(share_adjacency)
        self.learn_std = bool(learn_std)
        self.learn_self_att = bool(learn_self_att)
        self.cell_0 = GLSTMCell_v2(
            N=N,
            input_size=in_channels,
            hidden_size=hidden_size,
            edge_index=edge_index,
            edge_weight=edge_weight,
            learn_adj=learn_adj,
            learn_self_att=learn_self_att,
            lock_topology=lock_topology,
            cell_clip=cell_clip,
        )
        self.cells = nn.ModuleList([
            GLSTMCell_v2(
                N=N,
                input_size=hidden_size,
                hidden_size=hidden_size,
                edge_index=edge_index,
                edge_weight=edge_weight,
                learn_adj=learn_adj,
                learn_self_att=learn_self_att,
                lock_topology=lock_topology,
                cell_clip=cell_clip,
            )
            for _ in range(self.lstm_layers - 1)
        ])
        if self.share_adjacency:
            self._tie_adjacency_state()

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, self.window)
        )
        if self.learn_std:
            self.std_head = nn.Sequential(
                nn.Linear(2 * hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, self.window),
                nn.Softplus(),
            )

    def _tie_adjacency_state(self):
        """Make every recurrent cell reference the first cell's graph state."""
        adjacency_buffers = (
            "I",
            "edge_mask",
            "topology_mask",
            "edge_weight_prior",
            "uses_edge_weight_prior",
            "a_logits_init",
            "A_fixed_raw",
            "A_fixed",
        )
        for cell in self.cells:
            for name in adjacency_buffers:
                if hasattr(self.cell_0, name):
                    setattr(cell, name, getattr(self.cell_0, name))
            if self.cell_0.learn_adj:
                cell.a_logits = self.cell_0.a_logits

    def current_adjacency(self, normalized=True):
        """Return the adjacency effectively reused by the recurrent stack."""
        return self.cell_0.current_adjacency(normalized=normalized)

    def adjacency_anchor_loss(self):
        """Return the anchor penalty once for the shared model adjacency."""
        return self.cell_0.adjacency_anchor_loss()

    def reset_parameters(self):
        self.cell_0.reset_parameters()
        for cell in self.cells:
            if self.share_adjacency:
                cell._reset_recurrent_parameters()
            else:
                cell.reset_parameters()
        for layer in self.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        if self.learn_std:
            for layer in self.std_head:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()

    def _decode_outputs(self, out):
        forecast = self.fc(out).transpose(1, 2)
        if not self.learn_std:
            return forecast

        node_mean = out.mean(dim=1)
        node_std = out.std(dim=1, correction=0)
        std_features = torch.cat((node_mean, node_std), dim=-1)
        return forecast, self.std_head(std_features)

    def forward(self, x_seq):
        B, T, N, _ = x_seq.shape

        H = x_seq.new_zeros((B, self.hidden_size, self.N))
        C = x_seq.new_zeros((B, self.hidden_size, self.N))
        A0 = self.current_adjacency(normalized=True)

        if self.lstm_layers == 1:
            for t in range(T):
                C, H = self.cell_0(x_seq[:, t], H, C, A=A0)
            out = H.transpose(1, 2)
            return self._decode_outputs(out)

        H_layers = x_seq.new_zeros((B, T, self.N, self.hidden_size))

        for t in range(T):
            C, H = self.cell_0(x_seq[:, t], H, C, A=A0)
            H_layers[:, t] = H.transpose(1, 2)

        for j, cell in enumerate(self.cells):
            H = x_seq.new_zeros((B, self.hidden_size, self.N))
            C = x_seq.new_zeros((B, self.hidden_size, self.N))
            A_layer = A0 if self.share_adjacency else cell.current_adjacency(normalized=True)
            for t in range(T):
                C, H = cell(H_layers[:, t], H, C, A=A_layer)
                H_layers[:, t] = H.transpose(1, 2)

        out = H_layers[:, -1]
        #pred_dim = self.hidden_size
        #for i, dim, in enumerate(self.fc_layers):
        #    out = nn.Linear(pred_dim,dim).to(out.device)(out)
        #    out = nn.Softplus()(out)
        #    out = nn.Dropout(self.dropout)(out)
        #    pred_dim = dim
        #nn.Linear(self.fc_layers[-1], self.window).to(out.device)(out)

        return self._decode_outputs(out)


class NodewiseLSTM(nn.Module):
    """One completely independent LSTM and forecast head for each station.

    The model consumes the same ``[batch, time, station, feature]`` tensor as
    :class:`GLSTM_v2`, but each station is processed by a separate recurrent
    module. It therefore has neither message passing nor shared recurrent or
    decoder parameters across stations.
    """

    def __init__(
        self,
        N,
        in_channels,
        hidden_size,
        out_channels,
        lstm_layers=1,
        dropout=0.2,
    ):
        super().__init__()
        if N < 1 or in_channels < 1 or out_channels < 1 or hidden_size < 2:
            raise ValueError(
                "N, in_channels, and out_channels must be positive; hidden_size must be at least 2."
            )
        if lstm_layers < 1:
            raise ValueError("lstm_layers must be at least 1.")

        self.N = int(N)
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.window = int(out_channels)
        self.lstm_layers = int(lstm_layers)
        self.dropout = float(dropout)
        recurrent_dropout = self.dropout if self.lstm_layers > 1 else 0.0
        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=self.in_channels,
                    hidden_size=self.hidden_size,
                    num_layers=self.lstm_layers,
                    dropout=recurrent_dropout,
                    batch_first=True,
                )
                for _ in range(self.N)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.hidden_size),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                    nn.Linear(self.hidden_size, self.hidden_size // 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_size // 2, self.window),
                )
                for _ in range(self.N)
            ]
        )
        self.register_buffer("identity_adjacency", torch.eye(self.N))

    def current_adjacency(self, normalized=True):
        """Expose identity for generic topology diagnostics; no edges exist."""
        return self.identity_adjacency

    def adjacency_anchor_loss(self):
        """Match the graph-model interface without introducing graph loss."""
        return self.identity_adjacency.new_zeros(())

    def forward(self, x_seq):
        if x_seq.ndim != 4:
            raise ValueError("Expected x_seq with shape [batch, time, station, feature].")
        if x_seq.shape[2] != self.N or x_seq.shape[3] != self.in_channels:
            raise ValueError(
                "Input station/feature dimensions disagree with the trained NodewiseLSTM: "
                f"expected ({self.N}, {self.in_channels}), got ({x_seq.shape[2]}, {x_seq.shape[3]})."
            )

        station_forecasts = []
        for station_index, (lstm, head) in enumerate(zip(self.lstms, self.heads)):
            sequence, _ = lstm(x_seq[:, :, station_index, :])
            station_forecasts.append(head(sequence[:, -1, :]))
        return torch.stack(station_forecasts, dim=2)


class LearnableAdjacency(nn.Module):
    """
    Matriz de adjacencia aprendivel no mesmo espirito do GLSTM_v2.

    A topologia base e dada por edge_index, enquanto os pesos das arestas
    existentes podem ser ajustados por gradiente via `a_logits`.
    """

    def __init__(self, N, edge_index, edge_weight=None, learn_adj=True, lock_topology=True, eps=1e-6):
        super().__init__()
        self.N = N
        self.learn_adj = learn_adj
        self.lock_topology = lock_topology
        self.eps = eps

        uses_edge_weight_prior = edge_weight is not None
        A_base = adjacency_matrix(N, edge_index, edge_weight).float().to(edge_index.device)
        I = torch.eye(N, device=A_base.device, dtype=A_base.dtype)
        offdiag_init = A_base.clone()
        offdiag_init.fill_diagonal_(0.0)
        edge_mask = (offdiag_init > 0).float()
        topology_mask = edge_mask if lock_topology else (torch.ones_like(edge_mask) - I)

        self.register_buffer("I", I)
        self.register_buffer("edge_mask", edge_mask)
        self.register_buffer("topology_mask", topology_mask)
        self.register_buffer("edge_weight_prior", offdiag_init)
        self.register_buffer(
            "uses_edge_weight_prior",
            torch.tensor(uses_edge_weight_prior, dtype=torch.bool, device=A_base.device),
        )

        if learn_adj:
            if topology_mask.any():
                if uses_edge_weight_prior:
                    # A learnable log-residual is anchored at zero. The raw
                    # edge is prior * exp(residual), so regularization keeps
                    # the selected prior instead of pulling weights to 0.5.
                    init_logits = torch.where(
                        edge_mask > 0,
                        torch.zeros_like(offdiag_init),
                        torch.full_like(offdiag_init, -9.21024036697585),
                    ) * topology_mask
                else:
                    # Preserve the historical default when no explicit
                    # station-similarity prior was supplied.
                    max_val = offdiag_init.max().clamp_min(1.0)
                    init_probs = (offdiag_init / max_val).clamp(0.05, 0.95)
                    init_probs = torch.where(edge_mask > 0, init_probs, torch.full_like(init_probs, 1e-4))
                    init_logits = torch.logit(init_probs, eps=1e-4) * topology_mask
            else:
                init_logits = torch.zeros_like(offdiag_init)

            self.register_buffer("a_logits_init", init_logits.clone())
            self.a_logits = nn.Parameter(init_logits.clone())
        else:
            A_fixed_raw = I + offdiag_init
            self.register_buffer("A_fixed_raw", A_fixed_raw)
            self.register_buffer("A_fixed", self._normalize_adjacency(A_fixed_raw))

    def _normalize_adjacency(self, A):
        deg = A.sum(dim=1).clamp_min(self.eps)
        deg_inv_sqrt = deg.rsqrt()
        return deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]

    def current_adjacency(self, normalized=True):
        if not self.learn_adj:
            return self.A_fixed if normalized else self.A_fixed_raw

        if bool(self.uses_edge_weight_prior.item()):
            existing_edges = self.edge_weight_prior * torch.exp(self.a_logits) * self.edge_mask
            candidate_edges = torch.sigmoid(self.a_logits) * (self.topology_mask - self.edge_mask)
            offdiag = existing_edges + candidate_edges
        else:
            offdiag = torch.sigmoid(self.a_logits) * self.topology_mask
        offdiag = 0.5 * (offdiag + offdiag.T)
        A = self.I + offdiag
        return self._normalize_adjacency(A) if normalized else A

    def reset_parameters(self):
        if self.learn_adj:
            with torch.no_grad():
                self.a_logits.copy_(self.a_logits_init)

    def adjacency_anchor_loss(self):
        """L2 penalty that anchors learned edge residuals to their prior."""
        if not self.learn_adj:
            return self.I.new_zeros(())
        delta = (self.a_logits - self.a_logits_init) * self.edge_mask
        return 0.5 * delta.square().sum()


class GraphResidualMixer(nn.Module):
    """
    Mistura espacial sobre os nos usando a matriz de adjacencia atual.
    """

    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.self_proj = nn.Linear(hidden_size, hidden_size)
        self.graph_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.self_proj.reset_parameters()
        self.graph_proj.reset_parameters()
        self.out_proj.reset_parameters()

    def forward(self, x, A):
        # x: [..., N, H]
        *leading, N, hidden = x.shape
        x_norm = self.norm(x)
        x_flat = x_norm.reshape(-1, N, hidden)
        graph_ctx = torch.einsum("bnh,nm->bmh", x_flat, A).reshape(*leading, N, hidden)
        update = self.self_proj(x_norm) + self.graph_proj(graph_ctx)
        update = self.out_proj(self.activation(update))
        return x + self.dropout(update)


class GraphTemporalTransformer_v1(nn.Module):
    """
    Transformer espaco-temporal para previsao de precipitacao em grafos.

    Espera, por padrao:
      - entrada simples: [historico, nos, features]
      - entrada batelada: [B, historico, nos, features]
      - entrada hierarquica explicita: [B, window, dias, nos, features]

    Observacao:
      - para manter compatibilidade com o pipeline atual do repositorio,
        tensors 4D sao interpretados por padrao como [B, historico, nos, features].
      - se voce quiser tratar 4D como [window, dias, nos, features], use
        `four_dim_mode="window_days"`.
      - entradas 5D sao achatadas em uma unica sequencia temporal
        preservando a ordem window -> dias.

    A saida segue:
      - [horizonte, nos] para target_dim=1 e entrada simples
      - [B, horizonte, nos] para target_dim=1 e entrada batelada
      - mantem a dimensao final target_dim quando target_dim > 1

    Ideia da arquitetura:
      1. projecao das features meteorologicas para um espaco latente;
      2. embeddings de estacao e posicao temporal;
      3. mistura espacial entre estacoes usando a adjacencia do grafo;
      4. Transformer temporal usando todo o historico passado como memoria;
      5. decoder Transformer com queries do horizonte para prever precipitacao.
    """

    def __init__(
        self,
        N,
        edge_index,
        in_channels,
        hidden_size,
        out_channels,
        edge_weight=None,
        learn_adj=True,
        lock_topology=True,
        target_dim=1,
        nhead=4,
        num_day_layers=2,
        num_window_layers=2,
        num_decoder_layers=1,
        dim_feedforward=None,
        dropout=0.1,
        max_window=64,
        max_days=64,
        squeeze_output=True,
        four_dim_mode="batch_days",
        precip_col=0,
        output_activation=None,
        use_node_embeddings=True,
        use_precip_residual=False,
    ):
        super().__init__()
        if hidden_size % nhead != 0:
            raise ValueError("hidden_size deve ser divisivel por nhead.")
        if four_dim_mode not in {"batch_days", "window_days", "auto"}:
            raise ValueError("four_dim_mode deve ser um de {'batch_days', 'window_days', 'auto'}.")
        if num_day_layers < 1:
            raise ValueError("num_day_layers deve ser >= 1.")
        if num_window_layers < 0:
            raise ValueError("num_window_layers deve ser >= 0.")
        if num_decoder_layers < 1:
            raise ValueError("num_decoder_layers deve ser >= 1.")
        if output_activation == "identity":
            output_activation = None
        if output_activation not in {None, "softplus", "relu"}:
            raise ValueError("output_activation deve ser None, 'identity', 'softplus' ou 'relu'.")
        if not (0 <= precip_col < in_channels):
            raise ValueError(f"precip_col={precip_col} fora do intervalo para F={in_channels}.")

        self.N = N
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.window = out_channels
        self.target_dim = target_dim
        self.max_window = max_window
        self.max_days = max_days
        self.squeeze_output = squeeze_output
        self.four_dim_mode = four_dim_mode
        self.precip_col = precip_col
        self.output_activation = output_activation
        self.use_node_embeddings = use_node_embeddings
        self.use_precip_residual = use_precip_residual

        if out_channels == N and target_dim == 1 and N > 1:
            warnings.warn(
                "GraphTemporalTransformer_v1 recebeu out_channels == N. "
                "Neste modelo, out_channels representa o horizonte de previsao, "
                "nao o numero de nos. Verifique se o correto seria out_channels=horizon.",
                stacklevel=2,
            )

        ff_dim = dim_feedforward if dim_feedforward is not None else hidden_size * 4

        self.adj = LearnableAdjacency(
            N=N,
            edge_index=edge_index,
            edge_weight=edge_weight,
            learn_adj=learn_adj,
            lock_topology=lock_topology,
        )

        self.input_proj = nn.Linear(in_channels, hidden_size)
        self.input_dropout = nn.Dropout(dropout)

        if use_node_embeddings:
            self.node_embedding = nn.Parameter(torch.zeros(1, 1, N, hidden_size))
        else:
            self.register_parameter("node_embedding", None)

        self.horizon_queries = nn.Parameter(torch.zeros(1, out_channels, hidden_size))

        self.day_graph_mixer = GraphResidualMixer(hidden_size, dropout=dropout)
        self.window_graph_mixer = GraphResidualMixer(hidden_size, dropout=dropout)
        self.extra_graph_mixers = nn.ModuleList(
            [GraphResidualMixer(hidden_size, dropout=dropout) for _ in range(max(num_window_layers - 1, 0))]
        )

        day_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.day_encoder = nn.TransformerEncoder(day_layer, num_layers=num_day_layers)
        self.day_norm = nn.LayerNorm(hidden_size)

        self.window_norm = nn.LayerNorm(hidden_size)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.decoder_norm = nn.LayerNorm(hidden_size)

        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, target_dim),
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.adj.reset_parameters()
        self.input_proj.reset_parameters()
        self.day_graph_mixer.reset_parameters()
        self.window_graph_mixer.reset_parameters()
        for mixer in self.extra_graph_mixers:
            mixer.reset_parameters()
        self.day_norm.reset_parameters()
        self.window_norm.reset_parameters()
        self.decoder_norm.reset_parameters()

        for module in self.output_head:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        if self.node_embedding is not None:
            nn.init.normal_(self.node_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.horizon_queries, mean=0.0, std=0.02)

        for stack in (self.day_encoder, self.decoder):
            for layer in stack.layers:
                for submodule in layer.modules():
                    if submodule is layer:
                        continue
                    if hasattr(submodule, "reset_parameters"):
                        submodule.reset_parameters()

    def current_adjacency(self, normalized=True):
        return self.adj.current_adjacency(normalized=normalized)

    def adjacency_anchor_loss(self):
        """Return the L2 anchor penalty for the graph prior."""
        return self.adj.adjacency_anchor_loss()

    def _resolve_four_dim_input(self, x):
        # x: [A, B, N, F], onde (A, B) pode ser (batch, historico) ou (window, dias)
        A, B, N, F = x.shape

        if self.four_dim_mode == "batch_days":
            return x, False

        if self.four_dim_mode == "window_days":
            if A > self.max_window:
                raise ValueError(f"window={A} excede max_window={self.max_window}.")
            if B > self.max_days:
                raise ValueError(f"dias={B} excede max_days={self.max_days}.")
            return x.reshape(1, A * B, N, F), True

        # Modo automatico:
        # 4D e ambiguo quando batch e window sao pequenos. Para proteger o
        # DataLoader do repositorio, o fallback seguro e [B, historico, N, F].
        return x, False

    def _prepare_input(self, x):
        squeeze_batch = False
        hierarchical_input = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            squeeze_batch = True
        elif x.ndim == 4:
            x, squeeze_batch = self._resolve_four_dim_input(x)
            hierarchical_input = self.four_dim_mode == "window_days"
        elif x.ndim == 5:
            hierarchical_input = True
            B, W, D, N, F = x.shape
            if W > self.max_window:
                raise ValueError(f"window={W} excede max_window={self.max_window}.")
            if D > self.max_days:
                raise ValueError(f"dias={D} excede max_days={self.max_days}.")
            x = x.reshape(B, W * D, N, F)
        else:
            raise ValueError(
                "GraphTemporalTransformer_v1 espera X com shape "
                "[historico, nos, features], [B, historico, nos, features] "
                "ou [B, window, dias, nos, features]."
            )

        B, T, N, F = x.shape
        if N != self.N:
            raise ValueError(f"Numero de nos invalido. Esperado N={self.N}, recebido N={N}.")
        if F != self.in_channels:
            raise ValueError(
                f"Numero de features invalido. Esperado F={self.in_channels}, recebido F={F}."
            )
        if T > self.max_days and not hierarchical_input:
            raise ValueError(f"historico={T} excede max_days={self.max_days}.")

        return x, squeeze_batch

    def _time_encoding(self, length, device, dtype):
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.hidden_size, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.hidden_size)
        )
        pe = torch.zeros(1, length, 1, self.hidden_size, device=device, dtype=torch.float32)
        pe[..., 0::2] = torch.sin(position * div_term).unsqueeze(0).unsqueeze(2)
        pe[..., 1::2] = torch.cos(position * div_term[: pe[..., 1::2].shape[-1]]).unsqueeze(0).unsqueeze(2)
        return pe.to(dtype=dtype)

    def _apply_output_activation(self, pred):
        if self.output_activation == "softplus":
            return torch.nn.functional.softplus(pred)
        if self.output_activation == "relu":
            return torch.relu(pred)
        return pred

    def forward(self, x):
        x, squeeze_batch = self._prepare_input(x)
        B, T, N, _ = x.shape

        A = self.current_adjacency(normalized=True)
        last_precip = x[:, -1, :, self.precip_col] if self.use_precip_residual else None

        x = self.input_proj(x)
        x = x + self._time_encoding(T, x.device, x.dtype)
        if self.node_embedding is not None:
            x = x + self.node_embedding.to(dtype=x.dtype)
        x = self.input_dropout(x)

        # Mistura espacial em cada passo passado antes da atencao temporal.
        x = self.day_graph_mixer(x, A)

        # Transformer temporal por estacao usando todo o historico como memoria.
        day_tokens = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, self.hidden_size)
        day_tokens = self.day_encoder(day_tokens)
        day_tokens = self.day_norm(day_tokens)

        # Volta para [B, T, N, H] e refina a memoria temporal com o grafo.
        memory_grid = day_tokens.reshape(B, N, T, self.hidden_size).permute(0, 2, 1, 3).contiguous()
        memory_grid = self.window_graph_mixer(memory_grid, A)
        for mixer in self.extra_graph_mixers:
            memory_grid = mixer(memory_grid, A)
        memory_grid = self.window_norm(memory_grid)

        memory_seq = memory_grid.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, self.hidden_size)

        # Queries aprendidas do horizonte fazem cross-attention contra todo o passado.
        horizon_queries = self.horizon_queries.expand(B * N, -1, -1)
        horizon_queries = horizon_queries + memory_seq[:, -1:].expand(-1, self.window, -1)
        decoded = self.decoder(tgt=horizon_queries, memory=memory_seq)
        decoded = self.decoder_norm(decoded)

        pred = self.output_head(decoded)
        pred = pred.reshape(B, N, self.window, self.target_dim).permute(0, 2, 1, 3)

        if self.squeeze_output and self.target_dim == 1:
            pred = pred.squeeze(-1)
            if last_precip is not None:
                pred = pred + last_precip.unsqueeze(1).expand(-1, self.window, -1)
        elif last_precip is not None:
            base = last_precip.unsqueeze(1).unsqueeze(-1).expand(-1, self.window, -1, 1)
            pred = torch.cat((pred[..., :1] + base, pred[..., 1:]), dim=-1)

        pred = self._apply_output_activation(pred)

        if squeeze_batch:
            pred = pred.squeeze(0)

        return pred
