import torch
import torch.nn as nn

class TemporalTransformerWithRegularization(nn.Module):
    def __init__(self, input_dim, model_dim=128, num_heads=4, num_layers=2, dropout=0.3):
        super(TemporalTransformerWithRegularization, self).__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.semantic_token = nn.Parameter(torch.randn(1, 1, model_dim))  # learnable token
        self.output_layer = nn.Linear(model_dim, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, F = x.size()
        x = self.input_proj(x)  # [B, T, D]
        token = self.semantic_token.repeat(B, 1, 1)  # [B, 1, D]
        x = torch.cat([token, x], dim=1)  # [B, T+1, D]
        x = self.transformer(x)  # [B, T+1, D]
        self.last_attn = x.detach()  # Save for semantic reentry tracking if needed
        x = x[:, 0]  # semantic token output
        x = self.dropout(x)
        return self.output_layer(x)
