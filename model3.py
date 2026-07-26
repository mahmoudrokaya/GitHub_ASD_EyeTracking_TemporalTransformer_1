import torch
import torch.nn as nn

class TemporalTransformerModel(nn.Module):
    def __init__(self, input_dim=2, d_model=128, nhead=4, num_layers=4, num_classes=2, dropout=0.1):
        super(TemporalTransformerModel, self).__init__()

        # Fix: match input dimension to your data (was 100, now 2)
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # -> (batch_size, seq_len, d_model)
        x = self.transformer(x)  # -> (batch_size, seq_len, d_model)

        # Pool across temporal dimension
        x = x.permute(0, 2, 1)  # (batch_size, d_model, seq_len)
        x = self.global_pool(x)  # -> (batch_size, d_model, 1)
        x = x.squeeze(-1)        # -> (batch_size, d_model)

        out = self.classifier(x)  # -> (batch_size, num_classes)
        return out
