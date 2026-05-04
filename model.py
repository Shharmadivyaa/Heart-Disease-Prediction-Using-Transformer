"""
Heart Disease Prediction using Transformer (Tabular Transformer)
================================================================
Architecture:
  - Feature Embedding layer (each feature → embedding vector)
  - Positional encoding (optional for tabular)
  - Multi-Head Self-Attention layers
  - Feed-Forward layers
  - Classification head

Dataset: UCI Heart Disease (Cleveland) - 14 features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeatureEmbedding(nn.Module):
    """Embeds each tabular feature into a d_model-dimensional vector."""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        # One linear projection per feature (scalar → d_model)
        self.embeddings = nn.ModuleList([
            nn.Linear(1, d_model) for _ in range(num_features)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features)
        tokens = []
        for i, embed in enumerate(self.embeddings):
            feat = x[:, i].unsqueeze(-1)          # (batch, 1)
            tokens.append(embed(feat).unsqueeze(1)) # (batch, 1, d_model)
        out = torch.cat(tokens, dim=1)              # (batch, num_features, d_model)
        return self.layer_norm(out)


class TransformerBlock(nn.Module):
    """Single Transformer encoder block."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        # Feed-forward with residual
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class HeartDiseaseTransformer(nn.Module):
    """
    Tabular Transformer for Heart Disease Binary Classification.

    Parameters
    ----------
    num_features   : number of input features (13 for Cleveland dataset)
    d_model        : embedding dimension (default 64)
    nhead          : attention heads (default 4)
    num_layers     : transformer blocks (default 3)
    dim_feedforward: inner FF dimension (default 256)
    dropout        : dropout rate
    num_classes    : 2 for binary classification
    """

    def __init__(
        self,
        num_features: int = 13,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_classes: int = 2,
    ):
        super().__init__()
        self.feature_embedding = FeatureEmbedding(num_features, d_model)

        # [CLS] token — aggregates sequence info for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)

        # Embed features: (batch, num_features, d_model)
        tokens = self.feature_embedding(x)

        # Prepend CLS token
        cls = self.cls_token.expand(batch, -1, -1)  # (batch, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)     # (batch, 1+num_features, d_model)

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Use CLS token output for classification
        cls_out = tokens[:, 0, :]  # (batch, d_model)
        return self.classifier(cls_out)

    def get_attention_weights(self, x: torch.Tensor):
        """Returns attention weights from first transformer block for explainability."""
        batch = x.size(0)
        tokens = self.feature_embedding(x)
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        _, weights = self.transformer_blocks[0].attention(tokens, tokens, tokens)
        return weights.detach()


if __name__ == "__main__":
    model = HeartDiseaseTransformer()
    dummy = torch.randn(8, 13)
    out = model(dummy)
    print(f"Model output shape: {out.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Model architecture:")
    print(model)
