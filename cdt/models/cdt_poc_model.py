"""
CDT POC Model (for pre-computed embeddings)

Predicts enhancer-gene regulatory relationships using
pre-computed Enformer, ESM-2, and scGPT embeddings

Architecture (following existing CDT model):
1. Project each modality's embeddings to common dimension
2. Self-Attention (intra-modality processing)
3. Cross-Attention (inter-modality interactions)
4. VCE (integration) + classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EmbeddingProjector(nn.Module):
    """Project each modality's embeddings to common dimension"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class SelfAttentionBlock(nn.Module):
    """
    Self-Attention Block (intra-modality processing)

    Pre-computed embeddings are single vectors,
    but processed as sequence length 1 to maintain Self-Attention structure
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, d_model] or [batch, seq_len, d_model]

        Returns:
            [batch, d_model] or [batch, seq_len, d_model]
        """
        # Expand to 3D if input is 2D
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, d_model]
            squeeze_output = True

        # Self-Attention + Residual + LayerNorm
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if squeeze_output:
            x = x.squeeze(1)  # [batch, d_model]

        return x


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention Block (inter-modality interactions)

    query_modality references key_modality's information
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        # Multi-Head Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, d_model]
            key: [batch, d_model]
            value: [batch, d_model]

        Returns:
            output: [batch, d_model]
            attention_weights: attention weights
        """
        # Expand to 3D
        q = query.unsqueeze(1) if query.dim() == 2 else query
        k = key.unsqueeze(1) if key.dim() == 2 else key
        v = value.unsqueeze(1) if value.dim() == 2 else value

        # Cross-Attention
        attn_out, attn_weights = self.cross_attn(q, k, v)
        x = self.norm1(q + self.dropout(attn_out))

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x.squeeze(1), attn_weights


class CDTPOCModel(nn.Module):
    """
    CDT POC Model

    Following existing CDT model structure:
    1. Projection (dimension unification)
    2. Self-Attention (intra-modality)
    3. Cross-Attention (inter-modality)
    4. VCE + Classification

    Uses pre-computed embeddings:
    - DNA: Enformer (3072 dimensions)
    - Protein: ESM-2 (1280 dimensions)
    - Cell: scGPT (512 dimensions)
    """

    def __init__(
        self,
        dna_dim: int = 3072,
        protein_dim: int = 1280,
        cell_dim: int = 512,
        hidden_dim: int = 256,
        nhead: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # ========================================
        # 1. Projectors (dimension unification)
        # ========================================
        self.dna_projector = EmbeddingProjector(dna_dim, hidden_dim, dropout)
        self.protein_projector = EmbeddingProjector(protein_dim, hidden_dim, dropout)
        self.cell_projector = EmbeddingProjector(cell_dim, hidden_dim, dropout)

        # ========================================
        # 2. Self-Attention (intra-modality)
        # ========================================
        self.dna_self_attn = SelfAttentionBlock(hidden_dim, nhead, dropout)
        self.protein_self_attn = SelfAttentionBlock(hidden_dim, nhead, dropout)
        self.cell_self_attn = SelfAttentionBlock(hidden_dim, nhead, dropout)

        # ========================================
        # 3. Cross-Attention (inter-modality) - biological cycle
        # ========================================
        # DNA -> Cell: Cell references DNA information (transcription)
        self.dna_to_cell = CrossAttentionBlock(hidden_dim, nhead, dropout)

        # Cell -> Protein: Protein references Cell information (translation)
        self.cell_to_protein = CrossAttentionBlock(hidden_dim, nhead, dropout)

        # Protein -> DNA: DNA references Protein information (TF feedback)
        self.protein_to_dna = CrossAttentionBlock(hidden_dim, nhead, dropout)

        # ========================================
        # 4. VCE (integration) + Classification
        # ========================================
        self.vce_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        cell_emb: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            dna_emb: [batch, 3072] Enformer embeddings
            protein_emb: [batch, 1280] ESM-2 embeddings
            cell_emb: [batch, 512] scGPT embeddings
            return_attention: whether to return attention weights

        Returns:
            logits: [batch, 1] prediction logits
        """
        # ========================================
        # Step 1: Projection (dimension unification)
        # ========================================
        dna = self.dna_projector(dna_emb)          # [batch, hidden_dim]
        protein = self.protein_projector(protein_emb)  # [batch, hidden_dim]
        cell = self.cell_projector(cell_emb)       # [batch, hidden_dim]

        # ========================================
        # Step 2: Self-Attention (intra-modality)
        # ========================================
        dna = self.dna_self_attn(dna)              # [batch, hidden_dim]
        protein = self.protein_self_attn(protein)  # [batch, hidden_dim]
        cell = self.cell_self_attn(cell)           # [batch, hidden_dim]

        # ========================================
        # Step 3: Cross-Attention (inter-modality) - biological cycle
        # ========================================
        # DNA -> Cell: Cell references DNA (transcription)
        cell_fused, dna_to_cell_attn = self.dna_to_cell(cell, dna, dna)

        # Cell -> Protein: Protein references Cell (translation)
        protein_fused, cell_to_prot_attn = self.cell_to_protein(protein, cell_fused, cell_fused)

        # Protein -> DNA: DNA references Protein (TF feedback)
        dna_fused, prot_to_dna_attn = self.protein_to_dna(dna, protein_fused, protein_fused)

        # ========================================
        # Step 4: VCE (integration) + Classification
        # ========================================
        # Concatenate
        concat = torch.cat([dna_fused, cell_fused, protein_fused], dim=1)
        # [batch, hidden_dim * 3]

        # VCE fusion
        vce = self.vce_fusion(concat)  # [batch, hidden_dim]

        # Classification
        logits = self.classifier(vce)  # [batch, 1]

        if return_attention:
            attention = {
                'dna_to_cell': dna_to_cell_attn,      # Transcription
                'cell_to_protein': cell_to_prot_attn,  # Translation
                'protein_to_dna': prot_to_dna_attn     # Feedback
            }
            return logits, attention

        return logits

    def get_num_params(self) -> int:
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineModel(nn.Module):
    """
    Baseline model (without Self/Cross-Attention)

    Simple Concatenation + MLP for comparison
    """

    def __init__(
        self,
        dna_dim: int = 3072,
        protein_dim: int = 1280,
        cell_dim: int = 512,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        total_dim = dna_dim + protein_dim + cell_dim

        self.mlp = nn.Sequential(
            nn.Linear(total_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        cell_emb: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        # Simply Concatenate
        x = torch.cat([dna_emb, protein_emb, cell_emb], dim=1)
        logits = self.mlp(x)

        if return_attention:
            return logits, {}
        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DNAOnlyModel(nn.Module):
    """
    DNA-only Baseline (seq2cells-like approach)

    Predicts using only Enformer embeddings
    For comparison with standard real-world approaches
    """

    def __init__(
        self,
        dna_dim: int = 3072,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(dna_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor = None,
        cell_emb: torch.Tensor = None,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Uses only DNA embeddings (protein_emb, cell_emb are ignored)
        Maintains same interface as other models
        """
        logits = self.mlp(dna_emb)

        if return_attention:
            return logits, {}
        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("CDT POC Model Test")
    print("=" * 60)

    batch_size = 8

    # Dummy data
    dna_emb = torch.randn(batch_size, 3072)
    protein_emb = torch.randn(batch_size, 1280)
    cell_emb = torch.randn(batch_size, 512)

    # CDT POC Model
    print("\n[CDT POC Model]")
    print("  Structure: Projection -> Self-Attention -> Cross-Attention -> VCE -> Classifier")
    model = CDTPOCModel()
    print(f"  Parameter count: {model.get_num_params():,}")

    logits = model(dna_emb, protein_emb, cell_emb)
    print(f"  Output shape: {logits.shape}")

    logits, attn = model(dna_emb, protein_emb, cell_emb, return_attention=True)
    print(f"  Attention keys: {list(attn.keys())}")

    # Baseline Model (Concat MLP)
    print("\n[Baseline Model (3-modal Concatenation)]")
    baseline = BaselineModel()
    print(f"  Parameter count: {baseline.get_num_params():,}")

    logits = baseline(dna_emb, protein_emb, cell_emb)
    print(f"  Output shape: {logits.shape}")

    # DNA-only Model (seq2cells-like)
    print("\n[DNA-only Model (seq2cells-like approach)]")
    dna_only = DNAOnlyModel()
    print(f"  Parameter count: {dna_only.get_num_params():,}")

    logits = dna_only(dna_emb, protein_emb, cell_emb)
    print(f"  Output shape: {logits.shape}")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
