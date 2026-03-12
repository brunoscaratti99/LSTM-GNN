import torch
from Data_Library import adjacency_matrix
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class GLSTMCellBatch(nn.Module):
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


class GLSTMBatch(nn.Module):
    def __init__(self, N, edge_index, in_channels, hidden_size, out_channels, lstm_layers=1, learn_adj=True):
        super().__init__()
        self.N = N
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.window = out_channels
        
        #LSTM blocks
        self.cell_0 = GLSTMCellBatch(N, in_channels, hidden_size, edge_index, learn_adj=learn_adj)
        self.cells = nn.ModuleList([GLSTMCellBatch(N=N, input_size=hidden_size, hidden_size=hidden_size,edge_index=edge_index, learn_adj=learn_adj) for _ in range(self.lstm_layers-1)])
        
        #fully connected layer
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
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