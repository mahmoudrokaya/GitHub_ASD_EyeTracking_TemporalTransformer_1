import torch
import torch.nn as nn

class TemporalTransformerVariant(nn.Module):
    def __init__(self, input_dim=6, d_model=128, nhead=4, num_layers=4, num_classes=2, dropout=0.1):
        super(TemporalTransformerVariant, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.classifier(x)
