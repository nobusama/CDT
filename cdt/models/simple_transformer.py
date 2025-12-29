"""
Simple Transformer Model

Phase 1用のシンプルなTransformer
RNA配列からProtein配列を予測する
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    位置エンコーディング

    Transformerは配列の順序情報を持たないため、
    位置情報を明示的に追加する必要がある
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


class SimpleTransformer(nn.Module):
    """
    Phase 1用のシンプルなTransformer

    RNA配列（入力） → Protein配列（出力）を予測

    アーキテクチャ:
    1. RNA埋め込み
    2. 位置エンコーディング
    3. Transformer Encoder
    4. 線形層（分類）
    """

    def __init__(
        self,
        rna_vocab_size: int = 5,      # RNAの語彙サイズ（A,U,C,G,N）
        protein_vocab_size: int = 25,  # Proteinの語彙サイズ
        d_model: int = 128,            # モデルの次元数
        nhead: int = 4,                # マルチヘッドアテンションのヘッド数
        num_layers: int = 2,           # Transformerレイヤー数
        dim_feedforward: int = 512,    # フィードフォワード層の次元数
        dropout: float = 0.1,          # ドロップアウト率
        max_len: int = 1000            # 最大配列長
    ):
        """
        Args:
            rna_vocab_size: RNAの語彙サイズ
            protein_vocab_size: Proteinの語彙サイズ
            d_model: モデルの次元数（埋め込みサイズ）
            nhead: アテンションヘッド数
            num_layers: Transformerレイヤー数
            dim_feedforward: フィードフォワード層の次元数
            dropout: ドロップアウト率
            max_len: 最大配列長
        """
        super().__init__()

        self.d_model = d_model

        # RNA埋め込み層（トークンID → ベクトル）
        self.rna_embedding = nn.Embedding(rna_vocab_size, d_model)

        # 位置エンコーディング
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # [batch, seq, feature]の順
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # 出力層（各位置でProteinアミノ酸を予測）
        self.fc_out = nn.Linear(d_model, protein_vocab_size)

        # パラメータの初期化
        self._init_weights()

    def _init_weights(self):
        """パラメータの初期化"""
        initrange = 0.1
        self.rna_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        rna_tokens: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        順伝播

        Args:
            rna_tokens: RNA配列のトークン [batch_size, seq_len]
            src_key_padding_mask: パディングマスク [batch_size, seq_len]
                                  Trueの位置はマスクされる（無視される）

        Returns:
            logits: [batch_size, seq_len, protein_vocab_size]
        """
        # 1. 埋め込み
        # [batch_size, seq_len] → [batch_size, seq_len, d_model]
        x = self.rna_embedding(rna_tokens) * math.sqrt(self.d_model)

        # 2. 位置エンコーディング
        x = self.pos_encoder(x)

        # 3. Transformer Encoder
        # [batch_size, seq_len, d_model] → [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 4. 出力層
        # [batch_size, seq_len, d_model] → [batch_size, seq_len, protein_vocab_size]
        logits = self.fc_out(x)

        return logits

    def predict(self, rna_tokens: torch.Tensor) -> torch.Tensor:
        """
        予測（推論モード）

        Args:
            rna_tokens: RNA配列のトークン [batch_size, seq_len]

        Returns:
            predicted_tokens: 予測されたProteinトークン [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(rna_tokens)
            # 各位置で最も確率の高いトークンを選択
            predicted_tokens = torch.argmax(logits, dim=-1)
        return predicted_tokens


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    print("=" * 70)
    print("SimpleTransformerのテスト")
    print("=" * 70)
    print()

    # モデル作成
    print("【モデル作成】")
    model = SimpleTransformer(
        rna_vocab_size=5,
        protein_vocab_size=25,
        d_model=128,
        nhead=4,
        num_layers=2
    )

    # パラメータ数を計算
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    print()

    # ダミーデータで動作確認
    print("【順伝播テスト】")
    batch_size = 4
    seq_len = 30

    # ダミーのRNA配列（0-4のランダムな整数）
    rna_tokens = torch.randint(0, 5, (batch_size, seq_len))
    print(f"入力 RNA tokens shape: {rna_tokens.shape}")

    # 順伝播
    logits = model(rna_tokens)
    print(f"出力 logits shape: {logits.shape}")
    print(f"  → [バッチサイズ={batch_size}, 配列長={seq_len}, "
          f"Protein語彙数={logits.shape[-1]}]")
    print()

    # 予測
    print("【予測テスト】")
    predicted = model.predict(rna_tokens)
    print(f"予測 Protein tokens shape: {predicted.shape}")
    print(f"最初のサンプルの予測: {predicted[0, :10]}")
    print()

    # パディングマスクのテスト
    print("【パディングマスクのテスト】")
    # 最後の10要素をパディングとしてマスク
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -10:] = True  # 最後の10要素をマスク

    logits_masked = model(rna_tokens, src_key_padding_mask=padding_mask)
    print(f"マスク付き出力 shape: {logits_masked.shape}")
    print("→ パディング部分は無視されて計算される")
    print()

    print("=" * 70)
    print("テスト完了！モデルは正常に動作しています。")
    print("=" * 70)
