import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.RevIN import RevIN

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim, seq_len, pred_len):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.pred_len = pred_len
        
        self.revin = RevIN(input_dim, affine=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, pred_len * input_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        x = self.revin(x, 'norm')
        
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Use last hidden state
        out = self.fc(out[:, -1, :]) # [batch, pred_len * input_dim]
        out = out.reshape(x.size(0), self.pred_len, self.input_dim)
        
        out = self.revin(out, 'denorm')
        return out

class PatchTST(nn.Module):
    """
    Full PatchTST with Channel Independence and RevIN
    """
    def __init__(self, input_dim, seq_len, pred_len, patch_len=16, stride=8, d_model=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim
        
        self.revin = RevIN(input_dim, affine=True)
        
        # Patching
        self.n_patches = (max(seq_len, patch_len) - patch_len) // stride + 1
        
        # Channel Independence: Input dim to patch embedding is patch_len (1 channel per patch)
        self.patch_embed = nn.Linear(patch_len, d_model)
        
        # Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.n_patches, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.head = nn.Linear(d_model * self.n_patches, pred_len)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        batch_size = x.shape[0]
        
        # RevIN
        x = self.revin(x, 'norm')
        
        # Channel Independence: Reshape to [batch * input_dim, seq_len, 1]
        x = x.permute(0, 2, 1).reshape(batch_size * self.input_dim, self.seq_len, 1)
        
        # Patching
        patches = []
        for i in range(0, self.seq_len - self.patch_len + 1, self.stride):
            patch = x[:, i:i+self.patch_len, :] # [B*C, patch_len, 1]
            patches.append(patch.squeeze(-1))   # [B*C, patch_len]
        
        patches = torch.stack(patches, dim=1) # [B*C, n_patches, patch_len]
        
        # Embedding
        enc_out = self.patch_embed(patches) # [B*C, n_patches, d_model]
        enc_out = enc_out + self.pos_embedding
        
        # Transformer
        enc_out = self.transformer(enc_out) # [B*C, n_patches, d_model]
        
        # Head
        flattened = enc_out.reshape(batch_size * self.input_dim, -1)
        out = self.head(flattened) # [B*C, pred_len]
        
        # Reshape back
        out = out.reshape(batch_size, self.input_dim, self.pred_len).permute(0, 2, 1) # [B, pred_len, C]
        
        # RevIN Denorm
        out = self.revin(out, 'denorm')
        
        return out

class iTransformer(nn.Module):
    """
    Full iTransformer with RevIN and correct multivariate handling
    """
    def __init__(self, input_dim, seq_len, pred_len, d_model=128, n_heads=8, n_layers=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_dim = input_dim
        
        self.revin = RevIN(input_dim, affine=True)
        
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True, activation='gelu', norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.predict_linear = nn.Linear(d_model, pred_len)

    def forward(self, x):
        # x: [B, L, C]
        x = self.revin(x, 'norm')
        
        # Invert dimensions: treat L as features
        x = x.permute(0, 2, 1) # [B, C, L]
        enc_out = self.enc_embedding(x) # [B, C, D]
        
        # Transformer Encoder
        enc_out = self.transformer(enc_out) # [B, C, D]
        
        # Project back to pred_len
        dec_out = self.predict_linear(enc_out) # [B, C, pred_len]
        dec_out = dec_out.permute(0, 2, 1) # [B, pred_len, C]
        
        dec_out = self.revin(dec_out, 'denorm')
        
        return dec_out
