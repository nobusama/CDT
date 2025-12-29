"""
CDT v3.2 Model: Fusion MLP with Cyclic Cross-Attention

NOTE: Using __future__ annotations to maintain Python 3.9 compatibility

Inputs:
  - DNA: [batch, 896, 3072] - Enformer trunk output (896 bins x 128bp)
  - Protein: [n_proteins, 768] - ProteomeLM embeddings (all proteins, shared across batches)
  - RNA: [batch, n_genes, 512] - scGPT gene embeddings (gene expression profile)

Outputs:
  - logits: [batch, n_proteins] - beta value prediction for each (enhancer, protein) pair

Architecture:
  1. Projection (dimension unification):
     DNA [batch, 896, 3072]     -> [batch, 896, hidden]
     RNA [batch, n_genes, 512]  -> [batch, n_genes, hidden]
     Protein [n_proteins, 768]  -> [batch, n_proteins, hidden]

  2. Self-Attention:
     - DNA Self-Attention: interactions between 896 positions
     - RNA Self-Attention: gene co-expression patterns
     - Protein Self-Attention: protein-protein interactions (PPI)

  3. Cyclic Cross-Attention (DNA -> RNA -> Protein -> DNA):
     Step 1: DNA -> RNA (transcription)
       Q=RNA, K/V=DNA -> Attention [batch, nhead, n_genes, 896]
     Step 2: RNA -> Protein (translation)
       Q=Protein, K/V=RNA -> Attention [batch, nhead, n_proteins, n_genes]
     Step 3: Protein -> DNA (TF feedback)
       Q=DNA, K/V=Protein -> Attention [batch, nhead, 896, n_proteins]

  4. Fusion MLP (no pooling, preserving positional information):
     concat([dna_fused, rna_fused, protein_fused], dim=1)
     -> [batch, 896 + n_genes + n_proteins, hidden]
     -> MLP processing
     -> Extract Protein part [batch, n_proteins, hidden]

  5. Task Layer:
     [batch, n_proteins, hidden] -> [batch, n_proteins] (regression output)

Interpretable Attention Maps:
  - dna_to_rna: which DNA positions affect which genes
  - rna_to_protein: which genes are related to which proteins
  - protein_to_dna: which proteins bind to which DNA positions
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# v3.2: VCE -> Fusion MLP (no pooling, preserving positional information)


def attention_sparsity_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Minimize attention distribution entropy to promote sparsity

    Args:
        attn_weights: [batch, nhead, Q, K] (after softmax, 0-1 probability distribution)

    Returns:
        sparsity_loss: scalar (smaller = sparser attention)
    """
    # Entropy: H = -sum(p * log(p))
    # Low entropy = concentrated on few positions
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
    return entropy.mean()


@dataclass
class CDTv2Config:
    """CDT v2 Model Configuration"""
    # Input dimensions
    dna_dim: int = 3072      # Enformer trunk output
    dna_seq_len: int = 896   # Enformer bins (128bp resolution)
    protein_dim: int = 768   # ProteomeLM output (or 1280 for ESM-2)
    rna_dim: int = 512       # scGPT output
    n_proteins: int = 2360   # Number of proteins (output dimension)

    # Model dimensions
    hidden_dim: int = 256
    nhead: int = 4
    dropout: float = 0.1

    # Self-Attention configuration
    dna_self_attn_layers: int = 2      # Number of DNA Self-Attention layers
    rna_self_attn_layers: int = 1      # Number of RNA Self-Attention layers
    protein_self_attn_layers: int = 1  # Number of Protein Self-Attention layers

    # Cross-Attention configuration (cyclic: DNA->RNA->Protein->DNA)
    # Step 1: DNA -> RNA (transcription): Q=RNA, K=DNA
    # Step 2: RNA -> Protein (translation): Q=Protein, K=RNA
    # Step 3: Protein -> DNA (TF feedback): Q=DNA, K=Protein


class SequenceProjector(nn.Module):
    """Project sequence embeddings to common dimension (preserving sequence length)"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] or [seq_len, input_dim] or [batch, input_dim]
        Returns:
            Same shape projected to output_dim
        """
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class DNASelfAttentionBlock(nn.Module):
    """
    Self-Attention within DNA sequence

    Learns interactions between 896 positions
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, seq_len, d_model]

        Returns:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, nhead, seq_len, seq_len] if return_attention
        """
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attention,
            average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights if return_attention else None


class RNASelfAttentionBlock(nn.Module):
    """
    Self-Attention for RNA gene expression

    Learns gene co-expression patterns
    n_genes: number of genes in RNA expression profile
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, n_genes, d_model]

        Returns:
            output: [batch, n_genes, d_model]
            attn_weights: [batch, nhead, n_genes, n_genes] if return_attention
        """
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attention,
            average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights if return_attention else None


class ProteinSelfAttentionBlock(nn.Module):
    """
    Self-Attention between proteins

    Learns protein-protein interaction (PPI) patterns
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [batch, n_proteins, d_model]

        Returns:
            output: [batch, n_proteins, d_model]
            attn_weights: [batch, nhead, n_proteins, n_proteins] if return_attention
        """
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            need_weights=return_attention,
            average_attn_weights=False
        )
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights if return_attention else None


class DNAToRNACrossAttention(nn.Module):
    """
    DNA -> RNA Cross-Attention (Cyclic Step 1: Transcription)

    RNA references DNA information (transcription process)
    Q = RNA [batch, n_genes, hidden]  # n_genes: number of RNA genes
    K, V = DNA [batch, 896, hidden]
    -> Attention [batch, nhead, n_genes, 896] <- DNA position importance is interpretable!
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        rna: torch.Tensor,
        dna: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rna: [batch, n_genes, hidden]
            dna: [batch, 896, hidden]

        Returns:
            output: [batch, n_genes, hidden]
            attn_weights: [batch, nhead, n_genes, 896]
        """
        attn_out, attn_weights = self.cross_attn(
            rna, dna, dna,
            need_weights=True,
            average_attn_weights=False
        )

        x = self.norm1(rna + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class RNAToProteinCrossAttention(nn.Module):
    """
    RNA -> Protein Cross-Attention (Cyclic Step 2: Translation)

    Protein references RNA information (translation process)
    Q = Protein [batch, n_proteins, hidden]
    K, V = RNA [batch, n_genes, hidden]
    -> Attention [batch, n_proteins, n_genes]
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        protein: torch.Tensor,
        rna: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            protein: [batch, n_proteins, hidden]
            rna: [batch, n_genes, hidden]

        Returns:
            output: [batch, n_proteins, hidden]
            attn_weights: [batch, nhead, n_proteins, n_genes]
        """
        attn_out, attn_weights = self.cross_attn(
            protein, rna, rna,
            need_weights=True,
            average_attn_weights=False
        )

        x = self.norm1(protein + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class ProteinToDNACrossAttention(nn.Module):
    """
    Protein -> DNA Cross-Attention (Cyclic Step 3: TF Feedback)

    DNA references Protein information (TF feedback)
    Q = DNA [batch, 896, hidden]
    K, V = Protein [batch, n_proteins, hidden]
    -> Attention [batch, 896, n_proteins]
    """

    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dna: torch.Tensor,
        protein: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            dna: [batch, 896, hidden]
            protein: [batch, n_proteins, hidden]

        Returns:
            output: [batch, 896, hidden]
            attn_weights: [batch, nhead, 896, n_proteins]
        """
        attn_out, attn_weights = self.cross_attn(
            dna, protein, protein,
            need_weights=True,
            average_attn_weights=False
        )

        x = self.norm1(dna + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attn_weights


class RNAPooling(nn.Module):
    """
    Pool RNA gene expression

    [batch, n_genes, hidden] -> [batch, hidden]
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # Attention pooling
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, rna: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna: [batch, n_genes, hidden]
        Returns:
            [batch, hidden]
        """
        batch_size = rna.size(0)
        query = self.query.expand(batch_size, -1, -1)  # [batch, 1, hidden]

        pooled, _ = self.attn(query, rna, rna)  # [batch, 1, hidden]
        pooled = self.norm(pooled)

        return pooled.squeeze(1)  # [batch, hidden]


class CDTv2Model(nn.Module):
    """
    CDT v2 Model: Full Proteome Attention

    Inputs:
      - DNA: [batch, 896, 3072] - batch enhancer regions
      - Protein: [n_proteins, 768] - all protein embeddings (shared across batches)
      - RNA: [batch, n_genes, 512] - gene expression for each sample

    Outputs:
      - logits: [batch, n_proteins] - binding prediction for each (enhancer, protein) pair

    Attention:
      - protein_to_dna: [batch, nhead, n_proteins, 896] - which DNA positions each TF attends to
    """

    def __init__(self, config: Optional[CDTv2Config] = None):
        super().__init__()

        if config is None:
            config = CDTv2Config()

        self.config = config

        # ========================================
        # 1. Projectors
        # ========================================
        # DNA: [batch, 896, 3072] -> [batch, 896, hidden]
        self.dna_projector = SequenceProjector(
            config.dna_dim, config.hidden_dim, config.dropout
        )
        # Protein: [n_proteins, 768] -> [n_proteins, hidden]
        self.protein_projector = SequenceProjector(
            config.protein_dim, config.hidden_dim, config.dropout
        )
        # RNA: [batch, n_genes, 512] -> [batch, n_genes, hidden]
        self.rna_projector = SequenceProjector(
            config.rna_dim, config.hidden_dim, config.dropout
        )

        # ========================================
        # 2. DNA Self-Attention (896 positions)
        # ========================================
        self.dna_self_attn_layers = nn.ModuleList([
            DNASelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.dna_self_attn_layers)
        ])

        # ========================================
        # 3. RNA Self-Attention (gene co-expression patterns)
        # ========================================
        self.rna_self_attn_layers = nn.ModuleList([
            RNASelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.rna_self_attn_layers)
        ])

        # ========================================
        # 4. Protein Self-Attention (PPI patterns)
        # ========================================
        self.protein_self_attn_layers = nn.ModuleList([
            ProteinSelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.protein_self_attn_layers)
        ])

        # ========================================
        # Cyclic Cross-Attention (DNA -> RNA -> Protein -> DNA)
        # ========================================
        # Step 1: DNA -> RNA (transcription): Q=RNA, K=DNA
        self.dna_to_rna = DNAToRNACrossAttention(
            config.hidden_dim, config.nhead, config.dropout
        )

        # Step 2: RNA -> Protein (translation): Q=Protein, K=RNA
        self.rna_to_protein = RNAToProteinCrossAttention(
            config.hidden_dim, config.nhead, config.dropout
        )

        # Step 3: Protein -> DNA (TF feedback): Q=DNA, K=Protein
        self.protein_to_dna = ProteinToDNACrossAttention(
            config.hidden_dim, config.nhead, config.dropout
        )

        # ========================================
        # 8. Fusion MLP (VCE replacement)
        # ========================================
        # Concatenate DNA, RNA, Protein and process with MLP
        # [batch, 896 + n_genes + n_proteins, hidden] -> process -> [batch, n_proteins]
        #
        # This achieves:
        # - All three embeddings contribute equally
        # - Suppresses ProteomeLM dominance
        # - "Comprehensive judgment" for prediction

        # Fusion MLP: process concatenated representations
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # ========================================
        # 9. Task Layer
        # ========================================
        # protein part hidden representation -> 1 scalar
        self.task_layer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )

    def forward(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        rna_emb: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            dna_emb: [batch, 896, 3072] Enformer sequence-level embeddings
            protein_emb: [n_proteins, 768] ProteomeLM embeddings (all proteins)
            rna_emb: [batch, n_genes, 512] scGPT gene embeddings
            return_attention: whether to return attention weights

        Returns:
            logits: [batch, n_proteins] - binding prediction for each (enhancer, protein)
            attention (if return_attention):
                dna_self_attn: list of [batch, nhead, 896, 896]
                rna_self_attn: list of [batch, nhead, n_genes, n_genes]
                protein_self_attn: list of [batch, nhead, n_proteins, n_proteins]
                Cyclic Cross-Attention:
                dna_to_rna: [batch, nhead, n_genes, 896]    # Step1: DNA->RNA (transcription)
                rna_to_protein: [batch, nhead, n_proteins, n_genes]  # Step2: RNA->Protein (translation)
                protein_to_dna: [batch, nhead, 896, n_proteins]  # Step3: Protein->DNA (TF feedback)
        """
        batch_size = dna_emb.size(0)
        n_proteins = protein_emb.size(0)
        attention_maps = {}

        # ========================================
        # Step 1: Projection
        # ========================================
        # DNA: [batch, 896, 3072] -> [batch, 896, hidden]
        dna = self.dna_projector(dna_emb)

        # Protein: [n_proteins, 768] -> [n_proteins, hidden]
        protein = self.protein_projector(protein_emb)
        # Expand for batch: [n_proteins, hidden] -> [batch, n_proteins, hidden]
        protein = protein.unsqueeze(0).expand(batch_size, -1, -1)

        # RNA: [batch, n_genes, 512] -> [batch, n_genes, hidden]
        rna = self.rna_projector(rna_emb)

        # ========================================
        # Step 2: DNA Self-Attention
        # ========================================
        dna_self_attns = []
        for layer in self.dna_self_attn_layers:
            dna, attn = layer(dna, return_attention=return_attention)
            if return_attention and attn is not None:
                dna_self_attns.append(attn)

        if return_attention and dna_self_attns:
            attention_maps['dna_self_attn'] = dna_self_attns

        # ========================================
        # Step 3: RNA Self-Attention (gene co-expression)
        # ========================================
        rna_self_attns = []
        for layer in self.rna_self_attn_layers:
            rna, attn = layer(rna, return_attention=return_attention)
            if return_attention and attn is not None:
                rna_self_attns.append(attn)

        if return_attention and rna_self_attns:
            attention_maps['rna_self_attn'] = rna_self_attns

        # ========================================
        # Step 4: Protein Self-Attention (PPI)
        # ========================================
        protein_self_attns = []
        for layer in self.protein_self_attn_layers:
            protein, attn = layer(protein, return_attention=return_attention)
            if return_attention and attn is not None:
                protein_self_attns.append(attn)

        if return_attention and protein_self_attns:
            attention_maps['protein_self_attn'] = protein_self_attns

        # ========================================
        # Cyclic Cross-Attention (DNA -> RNA -> Protein -> DNA)
        # ========================================

        # Step 5: DNA -> RNA (transcription)
        # RNA references DNA information
        # Attention [batch, nhead, n_genes, 896] <- DNA position importance is interpretable!
        rna_fused, dna_to_rna_attn = self.dna_to_rna(
            rna=rna,  # Q: [batch, n_genes, hidden]
            dna=dna   # K,V: [batch, 896, hidden]
        )
        if return_attention:
            attention_maps['dna_to_rna'] = dna_to_rna_attn

        # Step 6: RNA -> Protein (translation)
        # Protein references RNA information
        protein_fused, rna_to_protein_attn = self.rna_to_protein(
            protein=protein,  # Q: [batch, n_proteins, hidden]
            rna=rna_fused     # K,V: [batch, n_genes, hidden]
        )
        if return_attention:
            attention_maps['rna_to_protein'] = rna_to_protein_attn

        # Step 7: Protein -> DNA (TF feedback)
        # DNA references Protein information
        dna_fused, protein_to_dna_attn = self.protein_to_dna(
            dna=dna,              # Q: [batch, 896, hidden]
            protein=protein_fused  # K,V: [batch, n_proteins, hidden]
        )
        if return_attention:
            attention_maps['protein_to_dna'] = protein_to_dna_attn

        # ========================================
        # Step 8: Fusion MLP (VCE replacement)
        # ========================================
        # Concatenate three embeddings along sequence dimension
        # [batch, 896 + n_genes + n_proteins, hidden]
        combined = torch.cat([dna_fused, rna_fused, protein_fused], dim=1)

        # Process with Fusion MLP
        combined = self.fusion_mlp(combined)

        # Extract Protein part (last n_proteins positions)
        n_proteins_dim = protein_fused.size(1)
        protein_repr = combined[:, -n_proteins_dim:, :]  # [batch, n_proteins, hidden]

        # ========================================
        # Step 9: Task Layer
        # ========================================
        # protein_repr: [batch, n_proteins, hidden]
        # -> task_layer -> [batch, n_proteins, 1]
        # -> squeeze -> [batch, n_proteins]
        logits = self.task_layer(protein_repr).squeeze(-1)  # [batch, n_proteins]

        if return_attention:
            return logits, attention_maps

        return logits

    def get_num_params(self) -> int:
        """Return number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_protein_representations(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        rna_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Get protein representations after Cross-Attention

        Obtain hidden representation for each protein:
        - Transfer learning
        - Protein similarity comparison
        - Clustering and visualization

        Args:
            dna_emb: [batch, 896, 3072] Enformer sequence-level embeddings
            protein_emb: [n_proteins, 768] ProteomeLM embeddings
            rna_emb: [batch, n_genes, 512] scGPT gene embeddings

        Returns:
            protein_fused: [batch, n_proteins, hidden] protein representations
        """
        batch_size = dna_emb.size(0)

        # Projection
        dna = self.dna_projector(dna_emb)
        protein = self.protein_projector(protein_emb)
        protein = protein.unsqueeze(0).expand(batch_size, -1, -1)
        rna = self.rna_projector(rna_emb)

        # DNA Self-Attention
        for layer in self.dna_self_attn_layers:
            dna, _ = layer(dna, return_attention=False)

        # RNA Self-Attention
        for layer in self.rna_self_attn_layers:
            rna, _ = layer(rna, return_attention=False)

        # Protein Self-Attention
        for layer in self.protein_self_attn_layers:
            protein, _ = layer(protein, return_attention=False)

        # Cyclic Cross-Attention
        rna_fused, _ = self.dna_to_rna(rna=rna, dna=dna)
        protein_fused, _ = self.rna_to_protein(protein=protein, rna=rna_fused)

        return protein_fused

    @staticmethod
    def compute_sparsity_loss(attention_maps: Dict[str, torch.Tensor], target_keys: list = None) -> torch.Tensor:
        """
        Compute sparsity loss for Attention Maps

        Args:
            attention_maps: output from forward(return_attention=True)
            target_keys: keys to apply sparsity (default: ['dna_to_rna'])

        Returns:
            sparsity_loss: scalar
        """
        if target_keys is None:
            target_keys = ['dna_to_rna']  # DNA->RNA is the most interpretable part

        total_loss = 0.0
        count = 0

        for key in target_keys:
            if key in attention_maps:
                attn = attention_maps[key]
                total_loss += attention_sparsity_loss(attn)
                count += 1

        return total_loss / max(count, 1)

    def get_attention_to_genomic_coords(
        self,
        attention: torch.Tensor,
        center_pos: int,
        bin_size: int = 128
    ) -> Dict[str, any]:
        """
        Convert attention weights to genomic coordinates

        Args:
            attention: [batch, nhead, n_proteins, 896]
            center_pos: genomic coordinate of enhancer center
            bin_size: bp per bin (default: 128)

        Returns:
            dict with 'start', 'end', 'weights' per bin
        """
        n_bins = attention.shape[-1]
        half_len = (n_bins * bin_size) // 2

        start_pos = center_pos - half_len
        coords = []

        for i in range(n_bins):
            bin_start = start_pos + i * bin_size
            bin_end = bin_start + bin_size
            coords.append({
                'bin': i,
                'start': bin_start,
                'end': bin_end
            })

        return {
            'coords': coords,
            'attention': attention,
            'center': center_pos,
            'bin_size': bin_size
        }


if __name__ == "__main__":
    print("=" * 60)
    print("CDT v2 Model Test (Full Proteome Version)")
    print("=" * 60)

    batch_size = 4
    n_proteins = 100  # For testing (actual: ~20,000)
    n_genes = 500     # For testing (actual: ~20,000)
    dna_seq_len = 896

    # Dummy data
    dna_emb = torch.randn(batch_size, dna_seq_len, 3072)  # [batch, 896, 3072]
    protein_emb = torch.randn(n_proteins, 768)            # [n_proteins, 768]
    rna_emb = torch.randn(batch_size, n_genes, 512)       # [batch, n_genes, 512]

    print(f"\nInput shapes:")
    print(f"  DNA:     {dna_emb.shape} (batch x 896 bins x 3072)")
    print(f"  Protein: {protein_emb.shape} (n_proteins x 768) - all proteins")
    print(f"  RNA:     {rna_emb.shape} (batch x n_genes x 512)")

    # Create model
    config = CDTv2Config()
    model = CDTv2Model(config)

    print(f"\nModel configuration:")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  nhead: {config.nhead}")
    print(f"  dna_self_attn_layers: {config.dna_self_attn_layers}")
    print(f"  Number of parameters: {model.get_num_params():,}")

    # Forward pass
    print("\n" + "-" * 40)
    print("Forward pass (return_attention=False)")
    logits = model(dna_emb, protein_emb, rna_emb)
    print(f"  Output shape: {logits.shape}")
    print(f"  -> [batch={batch_size}, n_proteins={n_proteins}]")
    print(f"  -> Binding prediction for all proteins for each enhancer")

    print("\n" + "-" * 40)
    print("Forward pass (return_attention=True)")
    logits, attention = model(dna_emb, protein_emb, rna_emb, return_attention=True)
    print(f"  Output shape: {logits.shape}")
    print(f"  Attention keys: {list(attention.keys())}")

    print("\n  Attention shapes (interpretable!):")
    for key, value in attention.items():
        if isinstance(value, list):
            print(f"    {key}: list of {len(value)} tensors")
            for i, v in enumerate(value):
                print(f"      [{i}]: {v.shape}")
        else:
            print(f"    {key}: {value.shape}")

    # Check attention distribution (Cyclic Cross-Attention)
    print("\n" + "-" * 40)
    print("Cyclic Cross-Attention Map verification:")

    # Step 1: DNA -> RNA (transcription) - Most interpretable!
    print("\n  [Step 1] DNA -> RNA (transcription)")
    dna_to_rna_attn = attention['dna_to_rna']  # [batch, nhead, n_genes, 896]
    print(f"    Shape: {dna_to_rna_attn.shape}")
    print(f"    -> How each gene attends to DNA 896 positions")
    attn_dist = dna_to_rna_attn[0, 0, 0, :]  # gene0 distribution
    print(f"    gene0 distribution: sum={attn_dist.sum().item():.4f}, max_bin={attn_dist.argmax().item()}")

    # Step 2: RNA -> Protein (translation)
    print("\n  [Step 2] RNA -> Protein (translation)")
    rna_to_prot_attn = attention['rna_to_protein']  # [batch, nhead, n_proteins, n_genes]
    print(f"    Shape: {rna_to_prot_attn.shape}")
    print(f"    -> Which genes each protein attends to")

    # Step 3: Protein -> DNA (TF feedback)
    print("\n  [Step 3] Protein -> DNA (TF feedback)")
    prot_to_dna_attn = attention['protein_to_dna']  # [batch, nhead, 896, n_proteins]
    print(f"    Shape: {prot_to_dna_attn.shape}")
    print(f"    -> Which proteins each DNA position attends to")

    # Convert to probability with sigmoid
    print("\n" + "-" * 40)
    print("Binding probability verification")
    probs = torch.sigmoid(logits)
    print(f"  logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"  probability range: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
    print(f"  Sample 0 predictions:")
    print(f"    Number of proteins with binding probability > 0.5: {(probs[0] > 0.5).sum().item()}/{n_proteins}")

    print("\n" + "=" * 60)
    print("CDT v2 Model (Cyclic Cross-Attention) test complete!")
    print(f"  -> Output: [batch, n_proteins] = [{batch_size}, {n_proteins}]")
    print(f"  -> Cycle: DNA -> RNA -> Protein -> DNA")
    print(f"  -> dna_to_rna [batch, nhead, n_genes, 896] shows")
    print(f"    'which DNA positions are important for transcription'")
    print("=" * 60)
