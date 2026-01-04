import torch
import torch.nn as nn
import torch.nn.functional as F

class SinkhornProjection(nn.Module):
    def __init__(self, n_streams, iterations=20):
        super().__init__()
        self.n_streams = n_streams
        self.iterations = iterations

    def forward(self, x):
        M = torch.exp(x)
        for _ in range(self.iterations):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-9)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-9)
        return M

class MHCBlock(nn.Module):
    def __init__(self, d_model, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        
        # Linear layer instead of layer_fn passed from outside for simplicity
        self.temporal_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )

        self.theta_pre = nn.Parameter(torch.randn(n_streams))
        self.theta_post = nn.Parameter(torch.randn(n_streams))
        self.theta_res = nn.Parameter(torch.randn(n_streams, n_streams))
        
        self.sinkhorn = SinkhornProjection(n_streams)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_stream):
        # x_stream: [batch, n_streams, seq_len, d_model]
        h_pre = torch.sigmoid(self.theta_pre)
        h_post = 2 * torch.sigmoid(self.theta_post)
        h_res = self.sinkhorn(self.theta_res)

        # Pre-mapping
        layer_input = torch.einsum('n,bnld->bld', h_pre, x_stream)
        layer_input = self.norm(layer_input)

        # Temporal computation
        layer_output = self.temporal_layer(layer_input)

        # Update streams
        res_part = torch.einsum('mn,bnld->bmld', h_res, x_stream)
        post_part = torch.einsum('n,bld->bnld', h_post, layer_output)
        
        return res_part + post_part

class MHCTimeSeriesModel(nn.Module):
    def __init__(self, input_dim, d_model, seq_len, pred_len, n_streams=4, n_layers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.n_streams = n_streams
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.blocks = nn.ModuleList([MHCBlock(d_model, n_streams) for _ in range(n_layers)])
        
        # Final head to project from [B, n, L, D] to [B, pred_len, 1]
        self.head = nn.Linear(d_model * n_streams * seq_len, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.shape[0]
        h = self.embedding(x) # [B, L, D]
        h_stream = h.unsqueeze(1).repeat(1, self.n_streams, 1, 1) # [B, n, L, D]
        
        for block in self.blocks:
            h_stream = block(h_stream)
            
        # Flatten and project
        # h_stream: [B, n, L, D] -> [B, n*L*D]
        out = h_stream.reshape(batch_size, -1)
        out = self.head(out) # [B, pred_len]
        return out.unsqueeze(-1) # [B, pred_len, 1]
