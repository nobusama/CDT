"""
CDT POC Model (事前計算済み埋め込み用)

事前計算済みのEnformer, ESM-2, scGPT埋め込みを使用して
エンハンサー→遺伝子の調節関係を予測する

アーキテクチャ（既存CDTモデルに準拠）:
1. 各モダリティの埋め込みを共通次元に投影
2. Self-Attention（各モダリティ内の処理）
3. Cross-Attention（モダリティ間の相互作用）
4. VCE（統合）+ 分類ヘッド
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class EmbeddingProjector(nn.Module):
    """各モダリティの埋め込みを共通次元に投影"""

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
    Self-Attention Block（各モダリティ内の処理）

    事前計算済み埋め込みは1ベクトルだが、
    Self-Attentionの構造を保持するためシーケンス長1として処理
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
            x: [batch, d_model] または [batch, seq_len, d_model]

        Returns:
            [batch, d_model] または [batch, seq_len, d_model]
        """
        # 入力が2次元の場合、3次元に拡張
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
    Cross-Attention Block（モダリティ間の相互作用）

    query_modality が key_modality の情報を参照する
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
        # 3次元に拡張
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

    既存のCDTモデル構造に準拠:
    1. Projection（次元統一）
    2. Self-Attention（各モダリティ内）
    3. Cross-Attention（モダリティ間）
    4. VCE + Classification

    事前計算済み埋め込みを使用:
    - DNA: Enformer (3072次元)
    - Protein: ESM-2 (1280次元)
    - Cell: scGPT (512次元)
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
        # 1. Projectors（次元統一）
        # ========================================
        self.dna_projector = EmbeddingProjector(dna_dim, hidden_dim, dropout)
        self.protein_projector = EmbeddingProjector(protein_dim, hidden_dim, dropout)
        self.cell_projector = EmbeddingProjector(cell_dim, hidden_dim, dropout)

        # ========================================
        # 2. Self-Attention（各モダリティ内）
        # ========================================
        self.dna_self_attn = SelfAttentionBlock(hidden_dim, nhead, dropout)
        self.protein_self_attn = SelfAttentionBlock(hidden_dim, nhead, dropout)
        self.cell_self_attn = SelfAttentionBlock(hidden_dim, nhead, dropout)

        # ========================================
        # 3. Cross-Attention（モダリティ間）- 生物学的循環
        # ========================================
        # DNA → Cell: CellがDNAの情報を参照（転写）
        self.dna_to_cell = CrossAttentionBlock(hidden_dim, nhead, dropout)

        # Cell → Protein: ProteinがCellの情報を参照（翻訳）
        self.cell_to_protein = CrossAttentionBlock(hidden_dim, nhead, dropout)

        # Protein → DNA: DNAがProteinの情報を参照（TFフィードバック）
        self.protein_to_dna = CrossAttentionBlock(hidden_dim, nhead, dropout)

        # ========================================
        # 4. VCE（統合）+ Classification
        # ========================================
        self.vce_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        # 二値分類ヘッド
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
            dna_emb: [batch, 3072] Enformer埋め込み
            protein_emb: [batch, 1280] ESM-2埋め込み
            cell_emb: [batch, 512] scGPT埋め込み
            return_attention: Attention weightsを返すか

        Returns:
            logits: [batch, 1] 予測ロジット
        """
        # ========================================
        # Step 1: Projection（次元統一）
        # ========================================
        dna = self.dna_projector(dna_emb)          # [batch, hidden_dim]
        protein = self.protein_projector(protein_emb)  # [batch, hidden_dim]
        cell = self.cell_projector(cell_emb)       # [batch, hidden_dim]

        # ========================================
        # Step 2: Self-Attention（各モダリティ内）
        # ========================================
        dna = self.dna_self_attn(dna)              # [batch, hidden_dim]
        protein = self.protein_self_attn(protein)  # [batch, hidden_dim]
        cell = self.cell_self_attn(cell)           # [batch, hidden_dim]

        # ========================================
        # Step 3: Cross-Attention（モダリティ間）- 生物学的循環
        # ========================================
        # DNA → Cell: CellがDNAを参照（転写）
        cell_fused, dna_to_cell_attn = self.dna_to_cell(cell, dna, dna)

        # Cell → Protein: ProteinがCellを参照（翻訳）
        protein_fused, cell_to_prot_attn = self.cell_to_protein(protein, cell_fused, cell_fused)

        # Protein → DNA: DNAがProteinを参照（TFフィードバック）
        dna_fused, prot_to_dna_attn = self.protein_to_dna(dna, protein_fused, protein_fused)

        # ========================================
        # Step 4: VCE（統合）+ Classification
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
                'dna_to_cell': dna_to_cell_attn,      # 転写
                'cell_to_protein': cell_to_prot_attn,  # 翻訳
                'protein_to_dna': prot_to_dna_attn     # フィードバック
            }
            return logits, attention

        return logits

    def get_num_params(self) -> int:
        """学習可能パラメータ数を返す"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BaselineModel(nn.Module):
    """
    ベースラインモデル（Self/Cross-Attentionなし）

    単純なConcatenation + MLPで比較用
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
        # 単純にConcatenate
        x = torch.cat([dna_emb, protein_emb, cell_emb], dim=1)
        logits = self.mlp(x)

        if return_attention:
            return logits, {}
        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class DNAOnlyModel(nn.Module):
    """
    DNA-only Baseline (seq2cells的アプローチ)

    Enformer埋め込みのみを使用して予測
    実世界での標準的アプローチとの比較用
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
        DNA埋め込みのみを使用（protein_emb, cell_embは無視）
        他のモデルと同じインターフェースを維持
        """
        logits = self.mlp(dna_emb)

        if return_attention:
            return logits, {}
        return logits

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("CDT POC Model テスト")
    print("=" * 60)

    batch_size = 8

    # ダミーデータ
    dna_emb = torch.randn(batch_size, 3072)
    protein_emb = torch.randn(batch_size, 1280)
    cell_emb = torch.randn(batch_size, 512)

    # CDT POC Model
    print("\n【CDT POC Model】")
    print("  構造: Projection → Self-Attention → Cross-Attention → VCE → Classifier")
    model = CDTPOCModel()
    print(f"  パラメータ数: {model.get_num_params():,}")

    logits = model(dna_emb, protein_emb, cell_emb)
    print(f"  出力形状: {logits.shape}")

    logits, attn = model(dna_emb, protein_emb, cell_emb, return_attention=True)
    print(f"  Attention keys: {list(attn.keys())}")

    # Baseline Model (Concat MLP)
    print("\n【Baseline Model (3-modal Concatenation)】")
    baseline = BaselineModel()
    print(f"  パラメータ数: {baseline.get_num_params():,}")

    logits = baseline(dna_emb, protein_emb, cell_emb)
    print(f"  出力形状: {logits.shape}")

    # DNA-only Model (seq2cells-like)
    print("\n【DNA-only Model (seq2cells-like approach)】")
    dna_only = DNAOnlyModel()
    print(f"  パラメータ数: {dna_only.get_num_params():,}")

    logits = dna_only(dna_emb, protein_emb, cell_emb)
    print(f"  出力形状: {logits.shape}")

    print("\n" + "=" * 60)
    print("✓ テスト完了!")
    print("=" * 60)
