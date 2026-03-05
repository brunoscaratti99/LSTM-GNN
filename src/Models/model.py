import torch
from Data_Library import adjacency_matrix
import torch.nn as nn
device = 'cuda' if torch.cuda.is_available() else 'cpu'



class GLSTMCellBatch(nn.Module):
    def __init__(self, N, input_size, hidden_size, edge_index, edge_weight=None, learn_adj=True):
        super().__init__()
        self.edge_index = edge_index
        self.learn_adj = learn_adj
        
        
        if edge_weight is None and learn_adj==True:
            self.edge_weight = torch.ones(edge_index.shape[1])
        
        if edge_weight is None and learn_adj==False:
            self.edge_weight = None
            
        else:
            self.edge_weight = self.edge_weight
            


        if learn_adj:
            self.A = nn.Parameter(adjacency_matrix(N, edge_index, self.edge_weight))
        else:
            self.A = adjacency_matrix(N, edge_index, self.edge_weight)
            
        """
        else:
            self.gcn_h = GCNConv(hidden_size, hidden_size)
            self.gcn_c = GCNConv(hidden_size, hidden_size)
        """


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
        if self.learn_adj:
            
            #A_eff = self.A
            #self.A = nn.Parameter(self.A+self.A.T)
            #A_raw = self.A
            #A_pos = F.softplus(A_raw)
            #A_eff = A_pos / (A_pos.sum(dim=1, keepdim=True)+1e-6)
  
                    
            H_graph = torch.einsum("bhm,mn->bhn", H_prev, self.A).to(device)
            C_graph = torch.einsum("bhm,mn->bhn", C_prev, self.A).to(device)


            Hg = H_graph.permute(0, 2, 1).to(device)
            Hp = H_prev.permute(0, 2, 1).to(device)
        else:
            A_eff = self.A
            
            H_graph = torch.einsum("bhm,mn->bhn", H_prev, A_eff).to(device)
            C_graph = torch.einsum("bhm,mn->bhn", C_prev, A_eff).to(device)


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
    def __init__(self, N, edge_index, in_channels, hidden_size, out_channels=1, learn_adj=True):
        super().__init__()
        self.N = N
        self.hidden_size = hidden_size
        self.cell = GLSTMCellBatch(N, in_channels, hidden_size, edge_index, learn_adj=learn_adj)
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 1)
            
            #nn.Softplus(beta=0.5),
            #nn.Linear(hidden_size, hidden_size),
            #nn.Softplus(beta=0.5),
            #nn.Linear(hidden_size, hidden_size//2),
            #nn.Softplus(beta=0.5),
            #nn.Linear(hidden_size//2, out_channels)
            #nn.ReLU(),
            #nn.Linear(hidden_size//4, out_channels)
            #nn.Sigmoid()
        ).to(device)

    def forward(self, x_seq):
        #print(x_seq.shape)
        # x_seq: [B, T, N, F]
        B, T, _, _ = x_seq.shape
        H = torch.zeros((B, self.hidden_size, self.N), device=x_seq.device)
        C = torch.zeros((B, self.hidden_size, self.N), device=x_seq.device)

        for t in range(T):
            C, H = self.cell(x_seq[:, t], H, C)
            #print(C.norm(), H.norm())
        out = H.permute(0, 2, 1).to(device)  # [B, N, H]
        return self.fc(out).squeeze(-1)  # [B, N]