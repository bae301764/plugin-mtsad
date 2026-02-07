import torch.nn as nn
import torch
import numpy as np



class plugin_module(nn.Module):
    def __init__(self, num_nodes, hidden_dim=128, bottleneck_dim=64, dropout=0.1):
        super().__init__()
        input_dim = num_nodes * num_nodes
        self.input_dim = input_dim
        #  Autoencoder for a flattened N¡¿N dependency matrix.
        #  Goal: learn to reconstruct the dependency matrix so reconstruction error can indicate structural changes in dependencies.

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):  # x: [B, N*N]
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon
    

def extract_dependency_matrix(index, att_weight, B, N, K, device):
    # Convert GNN message-passing edges (K neighbors per node + self-loop) and their attention weights
    # into a dense dependency/attention matrix A ¡ô R^{B¡¿N¡¿N}, then flatten to [B, N*N].

    if isinstance(index, np.ndarray):
        index = torch.from_numpy(index).to(device)  

    num_neighbors = B * N * K
    neighbors_att = att_weight[:num_neighbors].view(B * N, K)
    selfloop_att = att_weight[num_neighbors:].view(B * N, 1)

    neighbors_src = index[0, :num_neighbors].view(B * N, K)
    selfloop_src = index[0, num_neighbors:].view(B * N, 1)

    neighbors_dst = index[1, :num_neighbors].view(B * N, K)
    selfloop_dst = index[1, num_neighbors:].view(B * N, 1)
    
    full_att = torch.cat([neighbors_att, selfloop_att], dim=1).view(B, N, K+1)
    full_src = torch.cat([neighbors_src, selfloop_src], dim=1).view(B, N, K+1)
    full_dst = torch.cat([neighbors_dst, selfloop_dst], dim=1).view(B, N, K+1)

    offset = (torch.arange(B, device=device).view(B, 1, 1) * N)
    src_local = full_src - offset
    dst_local = full_dst - offset
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(-1, N, K+1)
    att_matrix = torch.zeros(B, N, N, device=device)
    att_matrix[batch_idx, dst_local, src_local] = full_att
    att_matrix_flat = att_matrix.view(B, -1)  # [B, N*N]
    
    return att_matrix_flat, att_matrix
            