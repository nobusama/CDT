"""
共通コンポーネント

Phase 1とPhase 2で共通利用するモジュール
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置エンコーディング

    Transformerは配列の順序情報を持たないため、
    位置情報を明示的に追加する必要がある

    Phase 1のsimple_transformer.pyから移植
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: モデルの次元数（埋め込みサイズ）
            max_len: 最大配列長
            dropout: ドロップアウト率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 位置エンコーディングを事前計算
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数次元
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数次元

        # バッチ次元を追加 [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # バッファとして登録（学習されない）
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        # 位置エンコーディングを追加
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer

    query（Q）が key-value（K, V）を参照するAttention

    使用例:
        DNA→RNA（転写）:
            query: RNA表現
            key, value: DNA表現
            → RNAがDNAの情報を取り込む
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
            d_model: モデルの次元数
            nhead: マルチヘッド数
            dim_feedforward: FFNの中間層次元数（Noneの場合 d_model * 4）
            dropout: ドロップアウト率
        """
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Multi-Head Attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True  # [batch, seq, feature]の順
        )

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

        # Layer Normalization (2つ)
        self.norm1 = nn.LayerNorm(d_model)  # Attention後
        self.norm2 = nn.LayerNorm(d_model)  # FFN後

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        標準的なTransformer Blockの構造:
        1. Multi-Head Attention
        2. Add & Norm
        3. Feed-Forward Network
        4. Add & Norm

        Args:
            query: [batch, query_len, d_model]
            key: [batch, key_len, d_model]
            value: [batch, value_len, d_model]
            key_padding_mask: [batch, key_len] パディング位置=True

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
            average_attn_weights=False  # ヘッドごとのweightsを返す
        )

        # 2. Add & Norm (after Attention)
        x = self.norm1(query + self.dropout(attn_output))

        # 3. Feed-Forward Network
        ffn_output = self.ffn(x)

        # 4. Add & Norm (after FFN)
        output = self.norm2(x + ffn_output)

        return output, attn_weights


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    print("=" * 70)
    print("共通コンポーネントのテスト")
    print("=" * 70)
    print()

    # PositionalEncodingテスト
    print("【PositionalEncodingテスト】")
    d_model = 128
    seq_len = 50
    batch_size = 4

    pos_enc = PositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_enc(x)

    print(f"入力 shape: {x.shape}")
    print(f"出力 shape: {output.shape}")
    assert output.shape == x.shape
    print("✓ PositionalEncoding動作確認")
    print()

    # CrossAttentionLayerテスト
    print("【CrossAttentionLayerテスト】")
    cross_attn = CrossAttentionLayer(d_model=128, nhead=4)

    query = torch.randn(batch_size, 30, 128)  # RNA
    key = torch.randn(batch_size, 90, 128)    # DNA
    value = key  # 通常 key == value

    output, attn_weights = cross_attn(query, key, value)

    print(f"Query shape: {query.shape}")
    print(f"Key shape: {key.shape}")
    print(f"出力 shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")

    assert output.shape == query.shape  # queryと同じ形状
    assert attn_weights.shape == (batch_size, 4, 30, 90)  # [B, nhead, Q_len, K_len]
    print("✓ CrossAttentionLayer動作確認")
    print()

    print("=" * 70)
    print("すべてのコンポーネントが正常に動作しています。")
    print("=" * 70)
