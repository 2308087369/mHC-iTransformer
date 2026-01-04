import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        return x

    def _get_statistics(self, x):
        # x: [B, L, C]
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + 1e-9)
        x = x * self.stdev
        x = x + self.mean
        return x

class SinkhornProjection(nn.Module):
    def __init__(self, n_streams, iterations=3): # 减少迭代次数提高稳定性
        super().__init__()
        self.iterations = iterations

    def forward(self, A):
        # A: [n, n]
        M = torch.exp(A)
        for _ in range(self.iterations):
            M = M / (M.sum(dim=-1, keepdim=True) + 1e-6)
            M = M / (M.sum(dim=-2, keepdim=True) + 1e-6)
        return M

class MHCPatchLayer(nn.Module):
    def __init__(self, d_model, n_heads, n_streams=4, dropout=0.1):
        super().__init__()
        self.n_streams = n_streams
        self.d_model = d_model
        
        # 核心计算单元
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MHC 机制重新设计：作为流之间的交互控制器
        # theta 控制流形变换矩阵
        self.theta = nn.Parameter(torch.eye(n_streams) * 2.0) # 初始化强自反馈
        self.sinkhorn = SinkhornProjection(n_streams)
        
        # 混合权重
        self.alpha = nn.Parameter(torch.ones(n_streams) * 0.5)

    def forward(self, x_stream):
        # x_stream: [B, n_streams, P, D]
        B, N, P, D = x_stream.shape
        
        # 1. 产生多流交互矩阵 (Birkhoff Polytope)
        W = self.sinkhorn(self.theta) # [N, N]
        
        # 2. 多流混合 (Stream Mixing)
        # 每一个流都是其他流的线性组合，受双稳态矩阵约束
        x_mixed = torch.einsum('mn,bnid->bmid', W, x_stream)
        
        # 3. 对混合后的流进行独立的特征提取
        # 为了效率，我们合并 batch 和 stream 维度
        x_flat = x_mixed.reshape(B * N, P, D)
        
        # Attention
        res = x_flat
        x_flat = self.norm1(x_flat)
        attn_out, _ = self.attention(x_flat, x_flat, x_flat)
        x_flat = res + attn_out
        
        # MLP
        res = x_flat
        x_flat = self.norm2(x_flat)
        x_flat = res + self.mlp(x_flat)
        
        out = x_flat.view(B, N, P, D)
        
        # 4. 这里的残差连接也受 alpha 控制
        alpha = torch.sigmoid(self.alpha).view(1, N, 1, 1)
        return alpha * out + (1 - alpha) * x_stream

class MHCPatchTST(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, patch_len=16, stride=8, 
                 d_model=128, n_heads=8, n_layers=3, n_streams=4, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_streams = n_streams
        
        self.revin = RevIN(input_dim)
        
        # Patching
        self.n_patches = (max(seq_len, patch_len) - patch_len) // stride + 1
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.n_patches, d_model))
        
        # MHC Layers
        self.layers = nn.ModuleList([
            MHCPatchLayer(d_model, n_heads, n_streams, dropout) for _ in range(n_layers)
        ])
        
        # Head
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model * self.n_patches, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pred_len)
        )
        
        # 多流聚合权重
        self.agg_weight = nn.Parameter(torch.ones(n_streams))

    def forward(self, x):
        # x: [B, L, C]
        B, L, C = x.shape
        
        # 1. Norm
        x = self.revin(x, 'norm')
        
        # 2. Target (Channel 0)
        z = x[:, :, 0] # [B, L]
        
        # 3. Patching
        # [B, P, PL]
        patches = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        
        # 4. Embedding
        enc = self.patch_embed(patches) # [B, P, D]
        
        # 5. Init Streams
        # [B, N, P, D]
        h = enc.unsqueeze(1).repeat(1, self.n_streams, 1, 1)
        h = h + self.pos_embed
        
        # 6. Layers
        for layer in self.layers:
            h = layer(h)
            
        # 7. Aggregate Streams
        w = torch.softmax(self.agg_weight, dim=0)
        out = torch.einsum('n,bnid->bid', w, h) # [B, P, D]
        
        # 8. Head
        out = self.head(out) # [B, pred_len]
        
        # 9. Denorm
        # out: [B, pred_len] -> [B, pred_len, 1]
        out = out.unsqueeze(-1)
        # 使用 RevIN 存储的第0个通道的统计量
        out = out * self.revin.stdev[:, :, 0:1] + self.revin.mean[:, :, 0:1]
        
        return out
