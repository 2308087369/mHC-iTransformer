import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, seq_len, pred_len):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pred_len = pred_len
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Use last hidden state
        out = self.fc(out[:, -1, :]) # [batch, pred_len]
        return out.unsqueeze(-1) # [batch, pred_len, 1]

class PatchTST(nn.Module):
    """
    Simplified PatchTST inspired model
    """
    def __init__(self, input_dim, seq_len, pred_len, patch_len=16, stride=8, d_model=128, n_heads=4, n_layers=3):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Number of patches
        self.n_patches = (max(seq_len, patch_len) - patch_len) // stride + 1
        
        self.patch_embed = nn.Linear(patch_len * input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model * self.n_patches, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.shape[0]
        
        # Simple patching
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :] # [batch, patch_len, input_dim]
            patches.append(patch.reshape(batch_size, -1))
        
        patches = torch.stack(patches, dim=1) # [batch, n_patches, patch_len*input_dim]
        
        enc_out = self.patch_embed(patches) # [batch, n_patches, d_model]
        enc_out = self.transformer(enc_out) # [batch, n_patches, d_model]
        
        flattened = enc_out.reshape(batch_size, -1)
        out = self.head(flattened) # [batch, pred_len]
        
        return out.unsqueeze(-1)

class iTransformer(nn.Module):
    """
    Original iTransformer: Inverted dimensions + Standard Residual Connections
    """
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.predict_linear = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        # Invert dimensions: treat L as features
        x = x.permute(0, 2, 1) # [B, C, L]
        enc_out = self.enc_embedding(x) # [B, C, D]
        
        # Standard Transformer Encoder (with standard residual connections)
        enc_out = self.transformer(enc_out) # [B, C, D]
        
        # Project back to pred_len
        dec_out = self.predict_linear(enc_out) # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1) # [B, pred_len, C]
        
        return dec_out[:, :, 0:1] # Return target channel
