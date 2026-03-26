import torch
from Graph.graph_related_utils import adjacency_matrix
import torch.nn as nn
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
    def __init__(self, N, input_size, hidden_size, edge_index, edge_weight=None, learn_adj=True, cell_clip=5.0, eps=1e-6):
        super().__init__()
        self.N = N
        self.hidden_size = hidden_size
        self.learn_adj = learn_adj
        self.cell_clip = cell_clip
        self.eps = eps

        A_base = adjacency_matrix(N, edge_index, edge_weight).float().to(edge_index.device)
        I = torch.eye(N, device=A_base.device, dtype=A_base.dtype)
        offdiag_init = A_base.clone()
        offdiag_init.fill_diagonal_(0.0)
        edge_mask = (offdiag_init > 0).float()

        self.register_buffer("I", I)
        self.register_buffer("edge_mask", edge_mask)

        if learn_adj:
            if edge_mask.any():
                max_val = offdiag_init.max().clamp_min(1.0)
                init_probs = (offdiag_init / max_val).clamp(0.05, 0.95)
                init_logits = torch.logit(init_probs, eps=1e-4) * edge_mask
            else:
                init_logits = torch.zeros_like(offdiag_init)

            self.register_buffer("a_logits_init", init_logits.clone())
            self.a_logits = nn.Parameter(init_logits.clone())
        else:
            A_fixed = self._normalize_adjacency(I + edge_mask)
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
            return self.A_fixed if normalized else (self.I + self.edge_mask)

        offdiag = torch.sigmoid(self.a_logits) * self.edge_mask
        offdiag = 0.5 * (offdiag + offdiag.T)
        A = self.I + offdiag
        return self._normalize_adjacency(A) if normalized else A

    def reset_parameters(self):
        for layer in (self.W_i, self.W_f, self.W_o, self.W_u, self.U_i, self.U_f, self.U_o, self.U_u):
            layer.reset_parameters()

        if self.learn_adj:
            with torch.no_grad():
                self.a_logits.copy_(self.a_logits_init)

    def forward(self, X, H_prev, C_prev):
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
    def __init__(self, N, edge_index, in_channels, hidden_size, out_channels, lstm_layers=1, learn_adj=True, dropout=0.2, cell_clip=5.0):
        super().__init__()
        self.N = N
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.window = out_channels

        self.cell_0 = GLSTMCell_v2(
            N=N,
            input_size=in_channels,
            hidden_size=hidden_size,
            edge_index=edge_index,
            learn_adj=learn_adj,
            cell_clip=cell_clip,
        )
        self.cells = nn.ModuleList([
            GLSTMCell_v2(
                N=N,
                input_size=hidden_size,
                hidden_size=hidden_size,
                edge_index=edge_index,
                learn_adj=learn_adj,
                cell_clip=cell_clip,
            )
            for _ in range(self.lstm_layers - 1)
        ])

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Softplus(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.window)
        )

    def reset_parameters(self):
        self.cell_0.reset_parameters()
        for cell in self.cells:
            cell.reset_parameters()
        for layer in self.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x_seq):
        B, T, N, _ = x_seq.shape

        H = x_seq.new_zeros((B, self.hidden_size, self.N))
        C = x_seq.new_zeros((B, self.hidden_size, self.N))
        H_layers = x_seq.new_zeros((B, T, self.N, self.hidden_size))

        for t in range(T):
            C, H = self.cell_0(x_seq[:, t], H, C)
            H_layers[:, t] = H.transpose(1, 2)

        for j in range(self.lstm_layers - 1):
            H = x_seq.new_zeros((B, self.hidden_size, self.N))
            C = x_seq.new_zeros((B, self.hidden_size, self.N))
            for t in range(T):
                C, H = self.cells[j](H_layers[:, t], H, C)
                H_layers[:, t] = H.transpose(1, 2)

        out = H_layers[:, -1]
        return self.fc(out).transpose(1, 2)