"""
CDT Model (Central Dogma Transformer)

DNA, RNA, Proteinの3つのモダリティを統合して
セルレベルの表現（Cell Embedding）を生成する
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Dict

from .components import PositionalEncoding, CrossAttentionLayer
from .vce import VirtualCellEmbedder


class TransformerEncoder(nn.Module):
    """
    各モダリティ用のTransformer Encoder

    構成:
    1. Embedding層
    2. Positional Encoding
    3. TransformerEncoder（複数層のSelf-Attention）
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
            vocab_size: 語彙サイズ
            d_model: モデルの次元数
            nhead: マルチヘッド数
            num_layers: Transformer層の数
            dim_feedforward: FFNの中間層次元数（Noneの場合 d_model * 4）
            dropout: ドロップアウト率
            max_len: 最大配列長
        """
        super().__init__()

        if dim_feedforward is None:
            dim_feedforward = d_model * 4

        # Embedding層
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
            batch_first=True  # [batch, seq, feature]の順
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len] トークンID
            src_key_padding_mask: [batch, seq_len] パディング位置=True

        Returns:
            [batch, seq_len, d_model] エンコードされた表現
        """
        # 1. Embedding + sqrt(d_model)スケーリング（Transformerの標準）
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        # [batch, seq_len, d_model]

        # 2. Positional Encoding
        x = self.pos_encoder(x)

        # 3. TransformerEncoder（Self-Attention）
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        return x


class CDTModel(nn.Module):
    """
    Central Dogma Transformer

    DNA, RNA, Proteinの3つのモダリティを統合して
    セルレベルの表現を生成する

    アーキテクチャ:
    1. 各モダリティをTransformerEncoderでエンコード
    2. Cross-Attentionでモダリティ間の相互作用を学習
    3. Virtual Cell Embedderで統合
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
            dna_vocab_size: DNA語彙サイズ
            rna_vocab_size: RNA語彙サイズ
            protein_vocab_size: Protein語彙サイズ
            d_model: モデルの次元数
            nhead: マルチヘッド数
            num_layers: 各Encoderの層数
            dim_feedforward: FFNの中間層次元数
            dropout: ドロップアウト率
            max_len: 最大配列長
        """
        super().__init__()

        # Encoders（各モダリティ用）
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
        # DNA → RNA (転写)
        self.dna_to_rna = CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)

        # RNA → Protein (翻訳)
        self.rna_to_protein = CrossAttentionLayer(d_model, nhead, dim_feedforward, dropout)

        # Protein → DNA (フィードバック)
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
            dna_tokens: [batch, dna_len] DNAトークン
            rna_tokens: [batch, rna_len] RNAトークン
            protein_tokens: [batch, protein_len] Proteinトークン
            dna_padding_mask: [batch, dna_len] パディング位置=True
            rna_padding_mask: [batch, rna_len] パディング位置=True
            protein_padding_mask: [batch, protein_len] パディング位置=True
            return_attention_weights: Attention weightsを返すか

        Returns:
            cell_embedding: [batch, d_model] セル表現
            または
            (cell_embedding, attention_weights_dict) のタプル
        """
        # ステップ1: Self-Attention Encoding（各モダリティ内部の関係を学習）
        dna_encoded = self.dna_encoder(dna_tokens, dna_padding_mask)
        # [batch, dna_len, d_model]

        rna_encoded = self.rna_encoder(rna_tokens, rna_padding_mask)
        # [batch, rna_len, d_model]

        protein_encoded = self.protein_encoder(protein_tokens, protein_padding_mask)
        # [batch, protein_len, d_model]

        # ステップ2: Cross-Attention（モダリティ間の相互作用を学習）

        # DNA → RNA: RNAがDNAの情報を参照（転写）
        rna_fused, dna_to_rna_weights = self.dna_to_rna(
            query=rna_encoded,
            key=dna_encoded,
            value=dna_encoded,
            key_padding_mask=dna_padding_mask
        )
        # [batch, rna_len, d_model]

        # RNA → Protein: ProteinがRNAの情報を参照（翻訳）
        protein_fused, rna_to_protein_weights = self.rna_to_protein(
            query=protein_encoded,
            key=rna_fused,  # DNA情報を含んだRNA
            value=rna_fused,
            key_padding_mask=rna_padding_mask
        )
        # [batch, protein_len, d_model]

        # Protein → DNA: DNAがProteinの情報を参照（フィードバック）
        dna_fused, protein_to_dna_weights = self.protein_to_dna(
            query=dna_encoded,
            key=protein_fused,
            value=protein_fused,
            key_padding_mask=protein_padding_mask
        )
        # [batch, dna_len, d_model]

        # ステップ3: Virtual Cell Embedding（統合）
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


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    print("=" * 70)
    print("CDT Modelのテスト")
    print("=" * 70)
    print()

    # ハイパーパラメータ
    batch_size = 4
    dna_len = 90
    rna_len = 30
    protein_len = 10

    dna_vocab_size = 5
    rna_vocab_size = 5
    protein_vocab_size = 25
    d_model = 128

    # モデル初期化
    model = CDTModel(
        dna_vocab_size=dna_vocab_size,
        rna_vocab_size=rna_vocab_size,
        protein_vocab_size=protein_vocab_size,
        d_model=d_model,
        nhead=4,
        num_layers=2
    )

    # ダミーデータ
    dna_tokens = torch.randint(0, dna_vocab_size, (batch_size, dna_len))
    rna_tokens = torch.randint(0, rna_vocab_size, (batch_size, rna_len))
    protein_tokens = torch.randint(0, protein_vocab_size, (batch_size, protein_len))

    print("【入力】")
    print(f"DNA tokens: {dna_tokens.shape}")
    print(f"RNA tokens: {rna_tokens.shape}")
    print(f"Protein tokens: {protein_tokens.shape}")
    print()

    # 順伝播
    print("【順伝播（Attention weightsなし）】")
    cell_emb = model(dna_tokens, rna_tokens, protein_tokens)
    print(f"Cell embedding: {cell_emb.shape}")
    assert cell_emb.shape == (batch_size, d_model)
    print("✓ Cell embedding形状確認")
    print()

    # Attention weightsを取得
    print("【順伝播（Attention weightsあり）】")
    cell_emb, attn_weights = model(
        dna_tokens, rna_tokens, protein_tokens,
        return_attention_weights=True
    )

    print(f"Cell embedding: {cell_emb.shape}")
    print(f"DNA→RNA weights: {attn_weights['dna_to_rna'].shape}")
    print(f"RNA→Protein weights: {attn_weights['rna_to_protein'].shape}")
    print(f"Protein→DNA weights: {attn_weights['protein_to_dna'].shape}")
    print()

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("【パラメータ数】")
    print(f"総パラメータ数: {total_params:,}")
    print(f"学習可能パラメータ数: {trainable_params:,}")
    print()

    print("=" * 70)
    print("CDT Modelが正常に動作しています。")
    print("=" * 70)
