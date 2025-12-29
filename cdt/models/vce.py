"""
Virtual Cell Embedder (VCE)

Integrates three modalities (DNA, RNA, Protein) to generate
a single cell representation (cell embedding)
"""

import torch
import torch.nn as nn


class VirtualCellEmbedder(nn.Module):
    """
    Virtual Cell Embedder (Mean Pooling version)

    Integrates sequence representations from each modality
    to generate a cell-level single vector representation

    Architecture:
    1. Mean pooling (compress each modality from sequence to vector)
    2. Concatenation (concatenate three vectors)
    3. MLP (integration processing)
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = None,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            hidden_dim: Intermediate layer dimension (d_model * 2 if None)
            dropout: Dropout rate
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model * 2

        # Integration MLP
        # Input: d_model * 3 (DNA, RNA, Protein concatenated)
        # Output: d_model
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(
        self,
        dna_encoded: torch.Tensor,
        rna_encoded: torch.Tensor,
        protein_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dna_encoded: [batch, dna_len, d_model]
            rna_encoded: [batch, rna_len, d_model]
            protein_encoded: [batch, protein_len, d_model]

        Returns:
            cell_embedding: [batch, d_model]
        """
        # Step 1: Mean pooling (sequence -> vector)
        dna_pooled = dna_encoded.mean(dim=1)        # [batch, d_model]
        rna_pooled = rna_encoded.mean(dim=1)        # [batch, d_model]
        protein_pooled = protein_encoded.mean(dim=1)  # [batch, d_model]

        # Step 2: Concatenation
        concat = torch.cat([dna_pooled, rna_pooled, protein_pooled], dim=-1)
        # [batch, d_model * 3]

        # Step 3: Integration
        cell_embedding = self.fusion(concat)  # [batch, d_model]

        return cell_embedding


class VirtualCellEmbedderWithAttention(nn.Module):
    """
    Attention-based Virtual Cell Embedder

    Uses Attention pooling instead of Mean pooling
    Can focus on more important positions
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()

        # Learnable query vectors for Attention pooling
        self.dna_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.rna_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.protein_query = nn.Parameter(torch.randn(1, 1, d_model))

        # Attention layers
        self.dna_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.rna_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )
        self.protein_attn = nn.MultiheadAttention(
            d_model, num_heads=4, dropout=dropout, batch_first=True
        )

        # Integration MLP
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(
        self,
        dna_encoded: torch.Tensor,
        rna_encoded: torch.Tensor,
        protein_encoded: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            dna_encoded: [batch, dna_len, d_model]
            rna_encoded: [batch, rna_len, d_model]
            protein_encoded: [batch, protein_len, d_model]

        Returns:
            cell_embedding: [batch, d_model]
        """
        batch_size = dna_encoded.size(0)

        # Attention pooling
        # Aggregate information from important positions using query vectors
        dna_query = self.dna_query.expand(batch_size, -1, -1)
        dna_pooled, _ = self.dna_attn(dna_query, dna_encoded, dna_encoded)
        dna_pooled = dna_pooled.squeeze(1)  # [batch, d_model]

        rna_query = self.rna_query.expand(batch_size, -1, -1)
        rna_pooled, _ = self.rna_attn(rna_query, rna_encoded, rna_encoded)
        rna_pooled = rna_pooled.squeeze(1)

        protein_query = self.protein_query.expand(batch_size, -1, -1)
        protein_pooled, _ = self.protein_attn(protein_query, protein_encoded, protein_encoded)
        protein_pooled = protein_pooled.squeeze(1)

        # Concatenation and integration
        concat = torch.cat([dna_pooled, rna_pooled, protein_pooled], dim=-1)
        cell_embedding = self.fusion(concat)

        return cell_embedding


# Usage example (only runs when this file is executed directly)
if __name__ == "__main__":
    print("=" * 70)
    print("Virtual Cell Embedder Test")
    print("=" * 70)
    print()

    batch_size = 4
    d_model = 128

    # Dummy encoded representations
    dna_encoded = torch.randn(batch_size, 90, d_model)
    rna_encoded = torch.randn(batch_size, 30, d_model)
    protein_encoded = torch.randn(batch_size, 10, d_model)

    # VCE (Mean pooling version)
    print("[VCE (Mean pooling)]")
    vce = VirtualCellEmbedder(d_model)
    cell_emb = vce(dna_encoded, rna_encoded, protein_encoded)

    print(f"DNA encoded: {dna_encoded.shape}")
    print(f"RNA encoded: {rna_encoded.shape}")
    print(f"Protein encoded: {protein_encoded.shape}")
    print(f"Cell embedding: {cell_emb.shape}")
    assert cell_emb.shape == (batch_size, d_model)
    print("VCE verification passed")
    print()

    # VCE (Attention pooling version)
    print("[VCE (Attention pooling)]")
    vce_attn = VirtualCellEmbedderWithAttention(d_model)
    cell_emb_attn = vce_attn(dna_encoded, rna_encoded, protein_encoded)

    print(f"Cell embedding (Attention): {cell_emb_attn.shape}")
    assert cell_emb_attn.shape == (batch_size, d_model)
    print("VCE (Attention) verification passed")
    print()

    # Parameter count comparison
    vce_params = sum(p.numel() for p in vce.parameters())
    vce_attn_params = sum(p.numel() for p in vce_attn.parameters())

    print(f"VCE (Mean) parameter count: {vce_params:,}")
    print(f"VCE (Attention) parameter count: {vce_attn_params:,}")
    print()

    print("=" * 70)
    print("All VCE variants are working correctly.")
    print("=" * 70)
