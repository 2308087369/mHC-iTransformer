import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.RevIN import RevIN

class SinkhornProjection(nn.Module):
    def __init__(self, n_streams, iterations=3):
        super().__init__()
        self.iterations = iterations

    def forward(self, A):
        M = torch.exp(A)
        for _ in range(self.iterations):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-6)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-6)
        return M

class MHCBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_streams=4, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.d_model = d_model
        
        # Sublayers
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MHC Manifold Parameters for Attention residual
        self.theta_attn = nn.Parameter(torch.eye(n_streams) * 2.0)
        self.phi_attn = nn.Parameter(torch.ones(n_streams) * 0.5)
        
        # MHC Manifold Parameters for FFN residual
        self.theta_ffn = nn.Parameter(torch.eye(n_streams) * 2.0)
        self.phi_ffn = nn.Parameter(torch.ones(n_streams) * 0.5)
        
        self.sinkhorn = SinkhornProjection(n_streams)
        
        # Aggregation weight for sublayer input
        self.agg_weight = nn.Parameter(torch.ones(n_streams) / n_streams)

    def forward(self, x_stream):
        # x_stream: [B, N, C, D]
        B, N, C, D = x_stream.shape
        
        # --- 1. MHC Attention Layer ---
        # Aggregate streams for sublayer input
        w_agg = torch.softmax(self.agg_weight, dim=0)
        sub_in = torch.einsum('n,bncd->bcd', w_agg, x_stream) # [B, C, D]
        
        # Attention sublayer
        sub_in_norm = self.norm1(sub_in)
        attn_out, _ = self.attention(sub_in_norm, sub_in_norm, sub_in_norm)
        
        # MHC Residual Connection
        W_attn = self.sinkhorn(self.theta_attn) # [N, N]
        phi_attn = torch.sigmoid(self.phi_attn).view(1, N, 1, 1)
        
        # H_{l+1} = H_l * W + phi * Sublayer(H_agg)
        x_stream = torch.einsum('mn,bncd->bmcd', W_attn, x_stream) + phi_attn * attn_out.unsqueeze(1)
        
        # --- 2. MHC FFN Layer ---
        w_agg_ffn = torch.softmax(self.agg_weight, dim=0)
        sub_in_ffn = torch.einsum('n,bncd->bcd', w_agg_ffn, x_stream)
        
        sub_in_ffn_norm = self.norm2(sub_in_ffn)
        ffn_out = self.ffn(sub_in_ffn_norm)
        
        # MHC Residual Connection
        W_ffn = self.sinkhorn(self.theta_ffn)
        phi_ffn = torch.sigmoid(self.phi_ffn).view(1, N, 1, 1)
        
        x_stream = torch.einsum('mn,bncd->bmcd', W_ffn, x_stream) + phi_ffn * ffn_out.unsqueeze(1)
        
        return x_stream

class MHC_iTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, n_heads=8, n_layers=3, n_streams=4, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        self.n_streams = n_streams
        
        self.revin = RevIN(input_dim, affine=True)
        
        # iTransformer: Project temporal dimension (seq_len) to hidden dimension (d_model)
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        self.layers = nn.ModuleList([
            MHCBlock(d_model, n_heads, n_streams, dropout) for _ in range(n_layers)
        ])
        
        # Final projection: d_model back to pred_len
        self.predict_linear = nn.Linear(d_model, pred_len)
        
        # Stream aggregation at the end
        self.final_agg = nn.Parameter(torch.ones(n_streams) / n_streams)

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        
        # 1. RevIN
        x = self.revin(x, 'norm') # [B, L, C]
        
        # 2. Inverted Embedding (iTransformer)
        # Permute to [B, C, L] to treat each variate independently
        x = x.permute(0, 2, 1) # [B, C, L]
        enc_out = self.enc_embedding(x) # [B, C, D]
        
        # 3. Initialize MHC Streams
        # [B, N, C, D]
        h_stream = enc_out.unsqueeze(1).repeat(1, self.n_streams, 1, 1)
        
        # 4. MHC-Transformer Layers
        for layer in self.layers:
            h_stream = layer(h_stream)
            
        # 5. Final Stream Aggregation
        w_final = torch.softmax(self.final_agg, dim=0)
        out = torch.einsum('n,bncd->bcd', w_final, h_stream) # [B, C, D]
        
        # 6. Projection to pred_len
        dec_out = self.predict_linear(out) # [B, C, pred_len]
        
        # 7. Permute back to [B, pred_len, C]
        dec_out = dec_out.permute(0, 2, 1) # [B, pred_len, C]
        
        # 8. Denorm
        dec_out = self.revin(dec_out, 'denorm')
        
        # Return all channels for correct multivariate processing
        return dec_out
