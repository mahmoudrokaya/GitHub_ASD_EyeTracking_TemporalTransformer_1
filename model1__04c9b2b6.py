import torch
import torch.nn as nn

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=2, embed_dim=64, num_heads=4, num_layers=2, dropout=0.4, num_classes=2):
        super(TemporalTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)
        return self.fc(x)
