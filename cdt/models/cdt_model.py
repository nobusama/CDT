"""
CDT Model (Central Dogma Transformer)

Integrates three modalities (DNA, RNA, Protein) to generate
cell-level representations (Cell Embedding).
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

from .components import PositionalEncoding, CrossAttentionLayer
from .vce import VirtualCellEmbedder


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder for each modality.

    Architecture:
    1. Embedding layer
    2. Positional Encoding
    3. TransformerEncoder (multiple Self-Attention layers)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        """
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: FFN intermediate dimension (defaults to d_model * 4)
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        # Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # [batch, seq, feature] order
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len] Token IDs
            src_key_padding_mask: [batch, seq_len] Padding positions = True

        Returns:
            [batch, seq_len, d_model] Encoded representations
        """
        # 1. Embedding + sqrt(d_model) scaling (standard Transformer practice)
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        # [batch, seq_len, d_model]

        # 2. Positional Encoding
        x = self.pos_encoder(x)

        # 3. TransformerEncoder (Self-Attention)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x


class CDTModel(nn.Module):
    """
    Central Dogma Transformer

    Integrates three modalities (DNA, RNA, Protein) to generate
    cell-level representations.

    Architecture:
    1. Encode each modality with TransformerEncoder
    2. Learn inter-modality interactions via Cross-Attention
    3. Integrate with Virtual Cell Embedder
    """

    def __init__(
        self,
        dna_vocab_size: int,
        rna_vocab_size: int,
        protein_vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        max_len: int = 5000
    ):
        """
        Args:
            dna_vocab_size: DNA vocabulary size
            rna_vocab_size: RNA vocabulary size
            protein_vocab_size: Protein vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of layers per encoder
            dim_feedforward: FFN intermediate dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        # Encoders (one per modality)
        self.dna_encoder = TransformerEncoder(
            dna_vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, max_len
        )
        self.rna_encoder = TransformerEncoder(
            rna_vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, max_len
        )
        self.protein_encoder = TransformerEncoder(
            protein_vocab_size, d_model, nhead, num_layers,
            dim_feedforward, dropout, max_len
        )

        # Cross-Attention Layers
        # DNA -> RNA (Transcription)
        self.dna_to_rna = CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)

        # RNA -> Protein (Translation)
        self.rna_to_protein = CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)

        # Protein -> DNA (Feedback)
        self.protein_to_dna = CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)

        # Virtual Cell Embedder
        self.vce = VirtualCellEmbedder(d_model, dropout=dropout)

    def forward(
        self,
        dna_tokens: torch.Tensor,
        rna_tokens: torch.Tensor,
        protein_tokens: torch.Tensor,
        dna_padding_mask: Optional[torch.Tensor] = None,
        rna_padding_mask: Optional[torch.Tensor] = None,
        protein_padding_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            dna_tokens: [batch, dna_len] DNA tokens
            rna_tokens: [batch, rna_len] RNA tokens
            protein_tokens: [batch, protein_len] Protein tokens
            dna_padding_mask: [batch, dna_len] Padding positions = True
            rna_padding_mask: [batch, rna_len] Padding positions = True
            protein_padding_mask: [batch, protein_len] Padding positions = True
            return_attention_weights: Whether to return attention weights

        Returns:
            cell_embedding: [batch, d_model] Cell representation
            or
            (cell_embedding, attention_weights_dict) tuple
        """
        # Step 1: Self-Attention Encoding (learn intra-modality relationships)
        dna_encoded = self.dna_encoder(dna_tokens, dna_padding_mask)
        # [batch, dna_len, d_model]

        rna_encoded = self.rna_encoder(rna_tokens, rna_padding_mask)
        # [batch, rna_len, d_model]

        protein_encoded = self.protein_encoder(protein_tokens, protein_padding_mask)
        # [batch, protein_len, d_model]

        # Step 2: Cross-Attention (learn inter-modality interactions)

        # DNA -> RNA: RNA queries DNA information (Transcription)
        rna_fused, dna_to_rna_weights = self.dna_to_rna(
            query=rna_encoded,
            key=dna_encoded,
            value=dna_encoded,
            key_padding_mask=dna_padding_mask
        )
        # [batch, rna_len, d_model]

        # RNA -> Protein: Protein queries RNA information (Translation)
        protein_fused, rna_to_protein_weights = self.rna_to_protein(
            query=protein_encoded,
            key=rna_fused,  # RNA with DNA information
            value=rna_fused,
            key_padding_mask=rna_padding_mask
        )
        # [batch, protein_len, d_model]

        # Protein -> DNA: DNA queries Protein information (Feedback)
        dna_fused, protein_to_dna_weights = self.protein_to_dna(
            query=dna_encoded,
            key=protein_fused,
            value=protein_fused,
            key_padding_mask=protein_padding_mask
        )
        # [batch, dna_len, d_model]

        # Step 3: Virtual Cell Embedding (Integration)
        cell_embedding = self.vce(dna_fused, rna_fused, protein_fused)
        # [batch, d_model]

        if return_attention_weights:
            attention_weights = {
                'dna_to_rna': dna_to_rna_weights,
                'rna_to_protein': rna_to_protein_weights,
                'protein_to_dna': protein_to_dna_weights
            }
            return cell_embedding, attention_weights

        return cell_embedding


# Usage example (runs only when this file is executed directly)
if __name__ == "__main__":
    print("=" * 70)
    print("CDT Model Test")
    print("=" * 70)
    print()

    # Hyperparameters
    batch_size = 4
    dna_len = 90
    rna_len = 30
    protein_len = 10

    dna_vocab_size = 5
    rna_vocab_size = 5
    protein_vocab_size = 25
    d_model = 128

    # Initialize model
    model = CDTModel(
        dna_vocab_size=dna_vocab_size,
        rna_vocab_size=rna_vocab_size,
        protein_vocab_size=protein_vocab_size,
        d_model=d_model,
        nhead=4,
        num_layers=2
    )

    # Dummy data
    dna_tokens = torch.randint(0, dna_vocab_size, (batch_size, dna_len))
    rna_tokens = torch.randint(0, rna_vocab_size, (batch_size, rna_len))
    protein_tokens = torch.randint(0, protein_vocab_size, (batch_size, protein_len))

    print("[Input]")
    print(f"DNA tokens: {dna_tokens.shape}")
    print(f"RNA tokens: {rna_tokens.shape}")
    print(f"Protein tokens: {protein_tokens.shape}")
    print()

    # Forward pass
    print("[Forward pass (without attention weights)]")
    cell_emb = model(dna_tokens, rna_tokens, protein_tokens)
    print(f"Cell embedding: {cell_emb.shape}")
    assert cell_emb.shape == (batch_size, d_model)
    print("✓ Cell embedding shape verified")
    print()

    # Get attention weights
    print("[Forward pass (with attention weights)]")
    cell_emb, attn_weights = model(
        dna_tokens, rna_tokens, protein_tokens,
        return_attention_weights=True
    )

    print(f"Cell embedding: {cell_emb.shape}")
    print(f"DNA->RNA weights: {attn_weights['dna_to_rna'].shape}")
    print(f"RNA->Protein weights: {attn_weights['rna_to_protein'].shape}")
    print(f"Protein->DNA weights: {attn_weights['protein_to_dna'].shape}")
    print()

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("[Parameter count]")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()

    print("=" * 70)
    print("CDT Model is working correctly.")
    print("=" * 70)
