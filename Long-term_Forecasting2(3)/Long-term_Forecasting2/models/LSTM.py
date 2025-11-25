import torch
import torch.nn as nn
from einops import rearrange


class LSTM(nn.Module):
    def __init__(self, configs, device):
        super(LSTM, self).__init__()
        self.device = device
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.hidden_dim = configs.d_ff  # Hidden dimension of LSTM
        self.num_layers = configs.e_layers  # Number of LSTM layers
        self.input_dim = configs.enc_in
        self.output_dim = configs.c_out
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        # Embedding layer
        self.input_embedding = nn.Linear(self.patch_size, self.d_model)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=configs.dropout
        )

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim * self.patch_num, self.pred_len)

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

        # Ensure LSTM is in training mode during forward
        self.lstm.train()  # Explicitly set LSTM to training mode
        x, (h_n, c_n) = self.lstm(x)  # x: [B*M, patch_num, hidden_dim]

        # Output projection
        x = x[:, :self.patch_num, :]  # Limit to patch_num
        x = x.reshape(B * M, -1)  # Flatten
        x = self.output_layer(x)  # Project to prediction dimension
        x = rearrange(x, '(b m) l -> b l m', b=B)  # Reshape back to original format

        # Denormalize output
        x = x * stdev
        x = x + means

        return x
