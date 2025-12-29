"""
CDT v3.2 Model: Fusion MLP with Cyclic Cross-Attention

NOTE: __future__ annotations を使用してPython 3.9互換を維持

入力:
  - DNA: [batch, 896, 3072] - Enformer trunk output (896 bins × 128bp)
  - Protein: [n_proteins, 768] - ProteomeLM embeddings (全タンパク質、バッチ間で共有)
  - RNA: [batch, n_genes, 512] - scGPT gene embeddings (遺伝子発現プロファイル)

出力:
  - logits: [batch, n_proteins] - 各(エンハンサー, タンパク質)ペアのbeta値予測

アーキテクチャ:
  1. Projection (次元統一):
     DNA [batch, 896, 3072]     → [batch, 896, hidden]
     RNA [batch, n_genes, 512]  → [batch, n_genes, hidden]
     Protein [n_proteins, 768]  → [batch, n_proteins, hidden]

  2. Self-Attention:
     - DNA Self-Attention: 896位置間の相互作用
     - RNA Self-Attention: 遺伝子間の共発現パターン
     - Protein Self-Attention: タンパク質間相互作用(PPI)

  3. 循環Cross-Attention (DNA → RNA → Protein → DNA):
     Step 1: DNA → RNA (転写)
       Q=RNA, K/V=DNA → Attention [batch, nhead, n_genes, 896]
     Step 2: RNA → Protein (翻訳)
       Q=Protein, K/V=RNA → Attention [batch, nhead, n_proteins, n_genes]
     Step 3: Protein → DNA (TFフィードバック)
       Q=DNA, K/V=Protein → Attention [batch, nhead, 896, n_proteins]

  4. Fusion MLP (プーリングなし、位置情報保持):
     concat([dna_fused, rna_fused, protein_fused], dim=1)
     → [batch, 896 + n_genes + n_proteins, hidden]
     → MLP処理
     → Protein部分を抽出 [batch, n_proteins, hidden]

  5. Task Layer:
     [batch, n_proteins, hidden] → [batch, n_proteins] (回帰出力)

解釈可能なAttention Map:
  - dna_to_rna: どのDNA位置がどの遺伝子に影響するか
  - rna_to_protein: どの遺伝子がどのタンパク質に関連するか
  - protein_to_dna: どのタンパク質がどのDNA位置に結合するか
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# v3.2: VCE → Fusion MLP に変更（プーリングなし、位置情報保持）


def attention_sparsity_loss(attn_weights: torch.Tensor) -> torch.Tensor:
    """
    Attention分布のエントロピーを最小化してスパース性を促進

    Args:
        attn_weights: [batch, nhead, Q, K] (softmax後、0-1の確率分布)

    Returns:
        sparsity_loss: スカラー（小さい = 疎なAttention）
    """
    # エントロピー: H = -Σ p * log(p)
    # 低いエントロピー = 少数の位置に集中
    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
    return entropy.mean()


@dataclass
class CDTv2Config:
    """CDT v2 モデル設定"""
    # 入力次元
    dna_dim: int = 3072      # Enformer trunk output
    dna_seq_len: int = 896   # Enformer bins (128bp resolution)
    protein_dim: int = 768   # ProteomeLM output (or 1280 for ESM-2)
    rna_dim: int = 512       # scGPT output
    n_proteins: int = 2360   # タンパク質数（出力次元）

    # モデル次元
    hidden_dim: int = 256
    nhead: int = 4
    dropout: float = 0.1

    # Self-Attention設定
    dna_self_attn_layers: int = 2      # DNA自身のSelf-Attention層数
    rna_self_attn_layers: int = 1      # RNA Self-Attention層数
    protein_self_attn_layers: int = 1  # Protein Self-Attention層数

    # Cross-Attention設定 (循環: DNA→RNA→Protein→DNA)
    # Step 1: DNA → RNA (転写): Q=RNA, K=DNA
    # Step 2: RNA → Protein (翻訳): Q=Protein, K=RNA
    # Step 3: Protein → DNA (TFフィードバック): Q=DNA, K=Protein


class SequenceProjector(nn.Module):
    """シーケンス埋め込みを共通次元に投影 (シーケンス長を保持)"""

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, input_dim] または [seq_len, input_dim] または [batch, input_dim]
        Returns:
            同じ形状で output_dim に投影
        """
        x = self.linear(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x


class DNASelfAttentionBlock(nn.Module):
    """
    DNA配列内のSelf-Attention

    896ポジション間の相互作用を学習
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
    RNA遺伝子(RNA)発現のSelf-Attention

    遺伝子(RNA)間の共発現パターンを学習
    n_genes: 遺伝子(RNA)発現プロファイルの遺伝子数
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
    タンパク質間のSelf-Attention

    タンパク質間相互作用（PPI）パターンを学習
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
    DNA → RNA のCross-Attention (循環Step 1: 転写)

    RNAがDNA情報を参照（転写プロセス）
    Q = RNA [batch, n_genes, hidden]  # n_genes: 遺伝子(RNA)数
    K, V = DNA [batch, 896, hidden]
    → Attention [batch, nhead, n_genes, 896] ← DNA位置ごとの重要度が解釈可能！
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
    RNA → Protein のCross-Attention (循環Step 2: 翻訳)

    ProteinがRNA情報を参照（翻訳プロセス）
    Q = Protein [batch, n_proteins, hidden]
    K, V = RNA [batch, n_genes, hidden]
    → Attention [batch, n_proteins, n_genes]
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
    Protein → DNA のCross-Attention (循環Step 3: TFフィードバック)

    DNAがProtein情報を参照（TFフィードバック）
    Q = DNA [batch, 896, hidden]
    K, V = Protein [batch, n_proteins, hidden]
    → Attention [batch, 896, n_proteins]
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
    RNA遺伝子発現をプーリング

    [batch, n_genes, hidden] → [batch, hidden]
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

    入力:
      - DNA: [batch, 896, 3072] - batch個のエンハンサー領域
      - Protein: [n_proteins, 768] - 全タンパク質埋め込み（バッチ間で共有）
      - RNA: [batch, n_genes, 512] - 各サンプルの遺伝子発現

    出力:
      - logits: [batch, n_proteins] - 各(エンハンサー, タンパク質)ペアの結合予測

    Attention:
      - protein_to_dna: [batch, nhead, n_proteins, 896] - 各TFがどのDNA位置に注目するか
    """

    def __init__(self, config: Optional[CDTv2Config] = None):
        super().__init__()

        if config is None:
            config = CDTv2Config()

        self.config = config

        # ========================================
        # 1. Projectors
        # ========================================
        # DNA: [batch, 896, 3072] → [batch, 896, hidden]
        self.dna_projector = SequenceProjector(
            config.dna_dim, config.hidden_dim, config.dropout
        )
        # Protein: [n_proteins, 768] → [n_proteins, hidden]
        self.protein_projector = SequenceProjector(
            config.protein_dim, config.hidden_dim, config.dropout
        )
        # RNA: [batch, n_genes, 512] → [batch, n_genes, hidden]
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
        # 3. RNA Self-Attention (遺伝子間共発現パターン)
        # ========================================
        self.rna_self_attn_layers = nn.ModuleList([
            RNASelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.rna_self_attn_layers)
        ])

        # ========================================
        # 4. Protein Self-Attention (PPI パターン)
        # ========================================
        self.protein_self_attn_layers = nn.ModuleList([
            ProteinSelfAttentionBlock(config.hidden_dim, config.nhead, config.dropout)
            for _ in range(config.protein_self_attn_layers)
        ])

        # ========================================
        # 循環 Cross-Attention (DNA → RNA → Protein → DNA)
        # ========================================
        # Step 1: DNA → RNA (転写): Q=RNA, K=DNA
        self.dna_to_rna = DNAToRNACrossAttention(
            config.hidden_dim, config.nhead, config.dropout
        )

        # Step 2: RNA → Protein (翻訳): Q=Protein, K=RNA
        self.rna_to_protein = RNAToProteinCrossAttention(
            config.hidden_dim, config.nhead, config.dropout
        )

        # Step 3: Protein → DNA (TFフィードバック): Q=DNA, K=Protein
        self.protein_to_dna = ProteinToDNACrossAttention(
            config.hidden_dim, config.nhead, config.dropout
        )

        # ========================================
        # 8. Fusion MLP (VCE代替)
        # ========================================
        # DNA, RNA, Protein を連結してMLP
        # [batch, 896 + n_genes + n_proteins, hidden] → 処理 → [batch, n_proteins]
        #
        # これにより:
        # - 3つの埋め込みが均等に貢献
        # - ProteomeLMの支配を抑制
        # - 「総合判断」で予測

        # Fusion MLP: 連結された表現を処理
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )

        # ========================================
        # 9. Task Layer
        # ========================================
        # protein部分の hidden 表現 → 1スカラー
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
            protein_emb: [n_proteins, 768] ProteomeLM embeddings (全タンパク質)
            rna_emb: [batch, n_genes, 512] scGPT gene embeddings
            return_attention: Attention weightsを返すか

        Returns:
            logits: [batch, n_proteins] - 各(エンハンサー, タンパク質)の結合予測
            attention (if return_attention):
                dna_self_attn: list of [batch, nhead, 896, 896]
                rna_self_attn: list of [batch, nhead, n_genes, n_genes]
                protein_self_attn: list of [batch, nhead, n_proteins, n_proteins]
                循環Cross-Attention:
                dna_to_rna: [batch, nhead, n_genes, 896]    # Step1: DNA→RNA (転写)
                rna_to_protein: [batch, nhead, n_proteins, n_genes]  # Step2: RNA→Protein (翻訳)
                protein_to_dna: [batch, nhead, 896, n_proteins]  # Step3: Protein→DNA (TFフィードバック)
        """
        batch_size = dna_emb.size(0)
        n_proteins = protein_emb.size(0)
        attention_maps = {}

        # ========================================
        # Step 1: Projection
        # ========================================
        # DNA: [batch, 896, 3072] → [batch, 896, hidden]
        dna = self.dna_projector(dna_emb)

        # Protein: [n_proteins, 768] → [n_proteins, hidden]
        protein = self.protein_projector(protein_emb)
        # Expand for batch: [n_proteins, hidden] → [batch, n_proteins, hidden]
        protein = protein.unsqueeze(0).expand(batch_size, -1, -1)

        # RNA: [batch, n_genes, 512] → [batch, n_genes, hidden]
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
        # Step 3: RNA Self-Attention (遺伝子間共発現)
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
        # 循環 Cross-Attention (DNA → RNA → Protein → DNA)
        # ========================================

        # Step 5: DNA → RNA (転写)
        # RNAがDNA情報を参照
        # Attention [batch, nhead, n_genes, 896] ← DNA位置の重要度が解釈可能!
        rna_fused, dna_to_rna_attn = self.dna_to_rna(
            rna=rna,  # Q: [batch, n_genes, hidden]
            dna=dna   # K,V: [batch, 896, hidden]
        )
        if return_attention:
            attention_maps['dna_to_rna'] = dna_to_rna_attn

        # Step 6: RNA → Protein (翻訳)
        # ProteinがRNA情報を参照
        protein_fused, rna_to_protein_attn = self.rna_to_protein(
            protein=protein,  # Q: [batch, n_proteins, hidden]
            rna=rna_fused     # K,V: [batch, n_genes, hidden]
        )
        if return_attention:
            attention_maps['rna_to_protein'] = rna_to_protein_attn

        # Step 7: Protein → DNA (TFフィードバック)
        # DNAがProtein情報を参照
        dna_fused, protein_to_dna_attn = self.protein_to_dna(
            dna=dna,              # Q: [batch, 896, hidden]
            protein=protein_fused  # K,V: [batch, n_proteins, hidden]
        )
        if return_attention:
            attention_maps['protein_to_dna'] = protein_to_dna_attn

        # ========================================
        # Step 8: Fusion MLP (VCE代替)
        # ========================================
        # 3つの埋め込みをシーケンス方向に連結
        # [batch, 896 + n_genes + n_proteins, hidden]
        combined = torch.cat([dna_fused, rna_fused, protein_fused], dim=1)

        # Fusion MLP で処理
        combined = self.fusion_mlp(combined)

        # Protein 部分を抽出（末尾 n_proteins 位置）
        n_proteins_dim = protein_fused.size(1)
        protein_repr = combined[:, -n_proteins_dim:, :]  # [batch, n_proteins, hidden]

        # ========================================
        # Step 9: Task Layer
        # ========================================
        # protein_repr: [batch, n_proteins, hidden]
        # → task_layer → [batch, n_proteins, 1]
        # → squeeze → [batch, n_proteins]
        logits = self.task_layer(protein_repr).squeeze(-1)  # [batch, n_proteins]

        if return_attention:
            return logits, attention_maps

        return logits

    def get_num_params(self) -> int:
        """学習可能パラメータ数を返す"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_protein_representations(
        self,
        dna_emb: torch.Tensor,
        protein_emb: torch.Tensor,
        rna_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-Attention後のタンパク質表現を取得

        各タンパク質の hidden 表現を取得:
        - 転移学習
        - タンパク質間の類似性比較
        - クラスタリング・可視化

        Args:
            dna_emb: [batch, 896, 3072] Enformer sequence-level embeddings
            protein_emb: [n_proteins, 768] ProteomeLM embeddings
            rna_emb: [batch, n_genes, 512] scGPT gene embeddings

        Returns:
            protein_fused: [batch, n_proteins, hidden] タンパク質表現
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

        # 循環 Cross-Attention
        rna_fused, _ = self.dna_to_rna(rna=rna, dna=dna)
        protein_fused, _ = self.rna_to_protein(protein=protein, rna=rna_fused)

        return protein_fused

    @staticmethod
    def compute_sparsity_loss(attention_maps: Dict[str, torch.Tensor], target_keys: list = None) -> torch.Tensor:
        """
        Attention Mapのスパース性損失を計算

        Args:
            attention_maps: forward(return_attention=True) の出力
            target_keys: スパース化対象のキー（デフォルト: ['dna_to_rna']）

        Returns:
            sparsity_loss: スカラー
        """
        if target_keys is None:
            target_keys = ['dna_to_rna']  # DNA→RNA が最も解釈したい部分

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
        Attention weightsをゲノム座標に変換

        Args:
            attention: [batch, nhead, n_proteins, 896]
            center_pos: エンハンサー中心のゲノム座標
            bin_size: 1 binあたりのbp (default: 128)

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
    print("CDT v2 Model テスト (Full Proteome Version)")
    print("=" * 60)

    batch_size = 4
    n_proteins = 100  # テスト用（実際は ~20,000）
    n_genes = 500     # テスト用（実際は ~20,000）
    dna_seq_len = 896

    # ダミーデータ
    dna_emb = torch.randn(batch_size, dna_seq_len, 3072)  # [batch, 896, 3072]
    protein_emb = torch.randn(n_proteins, 768)            # [n_proteins, 768]
    rna_emb = torch.randn(batch_size, n_genes, 512)       # [batch, n_genes, 512]

    print(f"\n入力形状:")
    print(f"  DNA:     {dna_emb.shape} (batch × 896 bins × 3072)")
    print(f"  Protein: {protein_emb.shape} (n_proteins × 768) - 全タンパク質")
    print(f"  RNA:     {rna_emb.shape} (batch × n_genes × 512)")

    # モデル作成
    config = CDTv2Config()
    model = CDTv2Model(config)

    print(f"\nモデル設定:")
    print(f"  hidden_dim: {config.hidden_dim}")
    print(f"  nhead: {config.nhead}")
    print(f"  dna_self_attn_layers: {config.dna_self_attn_layers}")
    print(f"  パラメータ数: {model.get_num_params():,}")

    # Forward pass
    print("\n" + "-" * 40)
    print("Forward pass (return_attention=False)")
    logits = model(dna_emb, protein_emb, rna_emb)
    print(f"  出力形状: {logits.shape}")
    print(f"  → [batch={batch_size}, n_proteins={n_proteins}]")
    print(f"  → 各エンハンサーに対する全タンパク質の結合予測")

    print("\n" + "-" * 40)
    print("Forward pass (return_attention=True)")
    logits, attention = model(dna_emb, protein_emb, rna_emb, return_attention=True)
    print(f"  出力形状: {logits.shape}")
    print(f"  Attention keys: {list(attention.keys())}")

    print("\n  Attention shapes (解釈可能!):")
    for key, value in attention.items():
        if isinstance(value, list):
            print(f"    {key}: list of {len(value)} tensors")
            for i, v in enumerate(value):
                print(f"      [{i}]: {v.shape}")
        else:
            print(f"    {key}: {value.shape}")

    # Attention分布の確認 (循環Cross-Attention)
    print("\n" + "-" * 40)
    print("循環Cross-Attention Map確認:")

    # Step 1: DNA → RNA (転写) - 最も解釈可能！
    print("\n  [Step 1] DNA → RNA (転写)")
    dna_to_rna_attn = attention['dna_to_rna']  # [batch, nhead, n_genes, 896]
    print(f"    形状: {dna_to_rna_attn.shape}")
    print(f"    → 各遺伝子がDNA 896位置にどう注目するか")
    attn_dist = dna_to_rna_attn[0, 0, 0, :]  # gene0の分布
    print(f"    gene0の分布: sum={attn_dist.sum().item():.4f}, max_bin={attn_dist.argmax().item()}")

    # Step 2: RNA → Protein (翻訳)
    print("\n  [Step 2] RNA → Protein (翻訳)")
    rna_to_prot_attn = attention['rna_to_protein']  # [batch, nhead, n_proteins, n_genes]
    print(f"    形状: {rna_to_prot_attn.shape}")
    print(f"    → 各タンパク質がどの遺伝子(RNA)に注目するか")

    # Step 3: Protein → DNA (TFフィードバック)
    print("\n  [Step 3] Protein → DNA (TFフィードバック)")
    prot_to_dna_attn = attention['protein_to_dna']  # [batch, nhead, 896, n_proteins]
    print(f"    形状: {prot_to_dna_attn.shape}")
    print(f"    → 各DNA位置がどのタンパク質に注目するか")

    # シグモイドで確率に変換
    print("\n" + "-" * 40)
    print("結合確率の確認")
    probs = torch.sigmoid(logits)
    print(f"  logits範囲: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print(f"  確率範囲: [{probs.min().item():.3f}, {probs.max().item():.3f}]")
    print(f"  サンプル0の予測:")
    print(f"    結合確率 > 0.5 のタンパク質数: {(probs[0] > 0.5).sum().item()}/{n_proteins}")

    print("\n" + "=" * 60)
    print("✓ CDT v2 Model (循環Cross-Attention) テスト完了!")
    print(f"  → 出力: [batch, n_proteins] = [{batch_size}, {n_proteins}]")
    print(f"  → 循環: DNA → RNA → Protein → DNA")
    print(f"  → dna_to_rna [batch, nhead, n_genes, 896] で")
    print(f"    「どのDNA位置が転写に重要か」が解釈可能")
    print("=" * 60)
