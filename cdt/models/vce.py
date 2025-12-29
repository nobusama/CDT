"""
Virtual Cell Embedder (VCE)

3つのモダリティ（DNA, RNA, Protein）を統合して
単一のセル表現（cell embedding）を生成する
"""

import torch
import torch.nn as nn


class VirtualCellEmbedder(nn.Module):
    """
    Virtual Cell Embedder (Mean Pooling版)

    各モダリティの配列表現を統合して、
    セルレベルの単一ベクトル表現を生成

    アーキテクチャ:
    1. Mean pooling（各モダリティを配列→ベクトルに圧縮）
    2. Concatenation（3つのベクトルを連結）
    3. MLP（統合処理）
    """

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = None,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: モデルの次元数
            hidden_dim: 中間層の次元数（Noneの場合 d_model * 2）
            dropout: ドロップアウト率
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = d_model * 2

        # 統合MLP
        # 入力: d_model * 3（DNA, RNA, Proteinを連結）
        # 出力: d_model
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
        # ステップ1: Mean pooling（配列 → ベクトル）
        dna_pooled = dna_encoded.mean(dim=1)        # [batch, d_model]
        rna_pooled = rna_encoded.mean(dim=1)        # [batch, d_model]
        protein_pooled = protein_encoded.mean(dim=1)  # [batch, d_model]

        # ステップ2: 連結
        concat = torch.cat([dna_pooled, rna_pooled, protein_pooled], dim=-1)
        # [batch, d_model * 3]

        # ステップ3: 統合
        cell_embedding = self.fusion(concat)  # [batch, d_model]

        return cell_embedding


class VirtualCellEmbedderWithAttention(nn.Module):
    """
    Attention-based Virtual Cell Embedder

    Mean poolingの代わりにAttention poolingを使用
    より重要な位置に注目できる
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: モデルの次元数
            dropout: ドロップアウト率
        """
        super().__init__()

        # Attention pooling用のクエリベクトル（学習可能）
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

        # 統合MLP
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
        # クエリベクトルを使って重要な位置から情報を集約
        dna_query = self.dna_query.expand(batch_size, -1, -1)
        dna_pooled, _ = self.dna_attn(dna_query, dna_encoded, dna_encoded)
        dna_pooled = dna_pooled.squeeze(1)  # [batch, d_model]

        rna_query = self.rna_query.expand(batch_size, -1, -1)
        rna_pooled, _ = self.rna_attn(rna_query, rna_encoded, rna_encoded)
        rna_pooled = rna_pooled.squeeze(1)

        protein_query = self.protein_query.expand(batch_size, -1, -1)
        protein_pooled, _ = self.protein_attn(protein_query, protein_encoded, protein_encoded)
        protein_pooled = protein_pooled.squeeze(1)

        # 連結・統合
        concat = torch.cat([dna_pooled, rna_pooled, protein_pooled], dim=-1)
        cell_embedding = self.fusion(concat)

        return cell_embedding


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    print("=" * 70)
    print("Virtual Cell Embedderのテスト")
    print("=" * 70)
    print()

    batch_size = 4
    d_model = 128

    # ダミーエンコード済み表現
    dna_encoded = torch.randn(batch_size, 90, d_model)
    rna_encoded = torch.randn(batch_size, 30, d_model)
    protein_encoded = torch.randn(batch_size, 10, d_model)

    # VCE（Mean pooling版）
    print("【VCE (Mean pooling)】")
    vce = VirtualCellEmbedder(d_model)
    cell_emb = vce(dna_encoded, rna_encoded, protein_encoded)

    print(f"DNA encoded: {dna_encoded.shape}")
    print(f"RNA encoded: {rna_encoded.shape}")
    print(f"Protein encoded: {protein_encoded.shape}")
    print(f"Cell embedding: {cell_emb.shape}")
    assert cell_emb.shape == (batch_size, d_model)
    print("✓ VCE動作確認")
    print()

    # VCE（Attention pooling版）
    print("【VCE (Attention pooling)】")
    vce_attn = VirtualCellEmbedderWithAttention(d_model)
    cell_emb_attn = vce_attn(dna_encoded, rna_encoded, protein_encoded)

    print(f"Cell embedding (Attention): {cell_emb_attn.shape}")
    assert cell_emb_attn.shape == (batch_size, d_model)
    print("✓ VCE (Attention) 動作確認")
    print()

    # パラメータ数比較
    vce_params = sum(p.numel() for p in vce.parameters())
    vce_attn_params = sum(p.numel() for p in vce_attn.parameters())

    print(f"VCE (Mean) パラメータ数: {vce_params:,}")
    print(f"VCE (Attention) パラメータ数: {vce_attn_params:,}")
    print()

    print("=" * 70)
    print("すべてのVCEバリアントが正常に動作しています。")
    print("=" * 70)
