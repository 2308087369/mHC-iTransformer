import torch
import torch.nn as nn
from models.layers.RevIN import RevIN
from models.layers.AttnRes import AttnResTransformerLayer

class AttnRes_iTransformer(nn.Module):
    """
    iTransformer with Attention Residuals (AttnRes)
    """
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, n_heads=8, n_layers=3, dropout=0.1, activation='gelu'):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        
        self.revin = RevIN(input_dim, affine=True)
        
        # Inverted Embedding
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        # AttnRes Layers
        # We use Full AttnRes (each layer is a block) for simplicity and performance
        self.layers = nn.ModuleList([
            AttnResTransformerLayer(d_model, n_heads, dropout=dropout, activation=activation)
            for _ in range(n_layers)
        ])
        
        self.predict_linear = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        x = self.revin(x, 'norm')
        
        # Invert dimensions: treat L as features
        x = x.permute(0, 2, 1) # [B, C, L]
        enc_out = self.enc_embedding(x) # [B, C, D]
        
        # AttnRes Logic
        # blocks: list of completed block representations [B, C, D]
        # We treat the embedding as the first block (Layer 0 output)
        blocks = [enc_out] 
        partial_block = None
        
        for layer in self.layers:
            # Apply layer with AttnRes
            # Returns the accumulated partial_block (which effectively becomes the layer output)
            partial_block = layer(blocks, partial_block)
            
            # Complete the block (Full AttnRes: every layer is a block)
            blocks.append(partial_block)
            partial_block = None
            
        # The final output is the last block
        # blocks[-1] is [B, C, D]
        enc_out = blocks[-1]
        
        # Project back to pred_len
        dec_out = self.predict_linear(enc_out) # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1) # [B, pred_len, C]
        
        dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out
