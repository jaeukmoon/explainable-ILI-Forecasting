import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

from einops import rearrange


class Transformer(nn.Module):
    def __init__(self, configs, device):
        super(Transformer, self).__init__()
        self.device = device
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.input_dim = configs.enc_in
        self.output_dim = configs.c_out
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = F.gelu if configs.activation == 'gelu' else F.relu

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        # Embedding layer
        self.input_embedding = nn.Linear(self.patch_size, self.d_model)

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.n_heads,
                dim_feedforward=self.d_ff,
                dropout=self.dropout,
                activation=configs.activation
            ) for _ in range(self.e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(self.d_model)

        # Output layer
        self.output_layer = nn.Linear(self.d_model * self.patch_num, self.pred_len)

        # Initialization
        for layer in (self.input_embedding, self.output_layer):
            layer.to(device=self.device)
            layer.train()

    def forward(self, x, itr):
        B, L, M = x.shape

        # Normalize input
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        x = (x - means) / stdev

        # Patch embedding
        x = rearrange(x, 'b l m -> b m l')  # Shape: [B, M, L]
        x = self.padding_patch_layer(x)  # Add padding
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)  # Create patches
        x = rearrange(x, 'b m n p -> (b m) n p')  # Shape: [B*M, patch_num, patch_size]
        x = self.input_embedding(x)  # Map to d_model dimension

        # Transformer Encoding
        for layer in self.encoder_layers:
            x = layer(x)

        x = self.encoder_norm(x)

        # Output projection
        x = x[:, :self.patch_num, :]  # Limit to patch_num
        x = x.reshape(B * M, -1)  # Flatten
        x = self.output_layer(x)  # Project to prediction dimension
        x = rearrange(x, '(b m) l -> b l m', b=B)  # Reshape back to original format

        # Denormalize output
        x = x * stdev
        x = x + means

        return x