"""
Common Components

Modules shared between Phase 1 and Phase 2
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding

    Since Transformers do not have sequence order information,
    positional information needs to be explicitly added

    Ported from Phase 1's simple_transformer.py
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encoding
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not trained)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer

    Attention where query (Q) references key-value (K, V)

    Usage example:
        DNA->RNA (transcription):
            query: RNA representation
            key, value: DNA representation
            -> RNA incorporates information from DNA
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = None,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of multi-heads
            dim_feedforward: FFN intermediate layer dimension (d_model * 4 if None)
            dropout: Dropout rate
        """
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # [batch, seq, feature] order
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer Normalization (2 layers)
        self.norm1 = nn.LayerNorm(d_model)  # After Attention
        self.norm2 = nn.LayerNorm(d_model)  # After FFN

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Standard Transformer Block structure:
        1. Multi-Head Attention
        2. Add & Norm
        3. Feed-Forward Network
        4. Add & Norm

        Args:
            query: [batch, query_len, d_model]
            key: [batch, key_len, d_model]
            value: [batch, value_len, d_model]
            key_padding_mask: [batch, key_len] padding positions=True

        Returns:
            output: [batch, query_len, d_model]
            attention_weights: [batch, num_heads, query_len, key_len]
        """
        # 1. Multi-Head Cross-Attention
        attn_output, attn_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # Return per-head weights
        )

        # 2. Add & Norm (after Attention)
        x = self.norm1(query + self.dropout(attn_output))

        # 3. Feed-Forward Network
        ffn_output = self.ffn(x)

        # 4. Add & Norm (after FFN)
        output = self.norm2(x + ffn_output)

        return output, attn_weights


# Usage example (only runs when this file is executed directly)
if __name__ == "__main__":
    print("=" * 70)
    print("Common Components Test")
    print("=" * 70)
    print()

    # PositionalEncoding test
    print("[PositionalEncoding Test]")
    d_model = 128
    seq_len = 50
    batch_size = 4

    pos_enc = PositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_enc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == x.shape
    print("PositionalEncoding verification passed")
    print()

    # CrossAttentionLayer test
    print("[CrossAttentionLayer Test]")
    cross_attn = CrossAttentionLayer(d_model=128, nhead=4)

    query = torch.randn(batch_size, 30, 128)  # RNA
    key = torch.randn(batch_size, 90, 128)    # DNA
    value = key  # Usually key == value

    output, attn_weights = cross_attn(query, key, value)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    assert output.shape == query.shape  # Same shape as query
    assert attn_weights.shape == (batch_size, 4, 30, 90)  # [B, nhead, Q_len, K_len]
    print("CrossAttentionLayer verification passed")
    print()

    print("=" * 70)
    print("All components are working correctly.")
    print("=" * 70)
