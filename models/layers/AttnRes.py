import torch
import torch.nn as nn
import torch.nn.functional as F

class BlockAttnRes(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Learned pseudo-query vector (parameter)
        # We use a Linear layer to represent the projection w_l
        self.proj = nn.Linear(d_model, 1, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, blocks: list[torch.Tensor], partial_block: torch.Tensor) -> torch.Tensor:
        """
        Inter-block attention: attend over block reps + partial sum.
        blocks:
            List of tensors [B, T, D]: completed block representations
        partial_block:
            [B, T, D] or None: intra-block partial sum
        """
        candidates = list(blocks)
        if partial_block is not None:
            candidates.append(partial_block)
            
        if not candidates:
            # This should only happen at the very first step if no blocks and no partial input?
            # Ideally, the first layer input is the first 'partial_block' or 'block'.
            # If candidates is empty, return zeros or handle error.
            # Assuming typical usage, there is always at least input embedding.
            return partial_block 
            
        V = torch.stack(candidates, dim=0) # [N_cand, B, T, D]
        
        # Compute keys
        K = self.norm(V) # [N_cand, B, T, D]
        
        # Compute attention scores
        # proj.weight: [1, D] -> squeeze -> [D]
        # K: [N, B, T, D]
        # logits: [N, B, T]
        logits = torch.einsum('d, n b t d -> n b t', self.proj.weight.squeeze(0), K)
        
        # Softmax over depth (N dimension)
        attn_weights = F.softmax(logits, dim=0) # [N_cand, B, T]
        
        # Weighted aggregation
        h = torch.einsum('n b t, n b t d -> b t d', attn_weights, V)
        
        return h

class AttnResTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="relu"):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # Self-Attention
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(d_model)
        
        # Feed Forward
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.mlp_norm = nn.LayerNorm(d_model)
        
        # AttnRes modules
        self.attn_res_proj = BlockAttnRes(d_model)
        self.mlp_res_proj = BlockAttnRes(d_model)
        
        self.attn_res_norm = nn.LayerNorm(d_model) # Used inside BlockAttnRes actually? 
        # Wait, BlockAttnRes has its own norm.
        # The pseudocode passed `self.attn_res_norm` to `block_attn_res`.
        # In my implementation, `BlockAttnRes` has `self.norm`.
        # So I don't need to pass it explicitly.

    def forward(self, blocks: list[torch.Tensor], partial_block: torch.Tensor):
        # 1. Block AttnRes before Attention
        # Note: In standard ResNet, x + Sublayer(Norm(x)) (PreNorm) or Norm(x + Sublayer(x)) (PostNorm)
        # The pseudocode suggests:
        # h = block_attn_res(blocks, partial_block)
        # attn_out = attn(norm(h))
        # partial_block += attn_out
        
        h_attn = self.attn_res_proj(blocks, partial_block)
        
        # Pre-Norm for Attention
        # Standard PreNorm: x + Attn(Norm(x))
        # Here we use h_attn as the "input" to the layer's computation
        h_norm = self.attn_norm(h_attn)
        attn_out, _ = self.attn(h_norm, h_norm, h_norm)
        
        # Accumulate residual
        if partial_block is not None:
            partial_block = partial_block + attn_out
        else:
            partial_block = attn_out
            
        # 2. Block AttnRes before MLP
        h_mlp = self.mlp_res_proj(blocks, partial_block)
        
        h_norm_mlp = self.mlp_norm(h_mlp)
        mlp_out = self.mlp(h_norm_mlp)
        
        # Accumulate residual
        partial_block = partial_block + mlp_out
        
        return partial_block
