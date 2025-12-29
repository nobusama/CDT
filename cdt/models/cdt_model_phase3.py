"""
CDT Model Phase 3

Pre-trained foundation modelsを統合したバージョン:
- DNA: Nucleotide Transformer v2 (97.9M params, 512d) ✅ Week 2
- RNA: RNA-FM (99M params, 640d) ✅ Week 3
- Protein: ESM-2 (150M params, 640d) ✅ Week 4
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from transformers import AutoTokenizer, AutoModelForMaskedLM
import fm
import esm

from .components import CrossAttentionLayer
from .vce import VirtualCellEmbedder


class SimpleTransformerEncoder(nn.Module):
    """
    Phase 2と同じシンプルなTransformer
    Phase 4でRNA-FM/ESM-2に置き換える予定
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tokens: [batch, seq_len]
            src_key_padding_mask: [batch, seq_len]

        Returns:
            [batch, seq_len, d_model]
        """
        import math
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return x


class CDTModelPhase3(nn.Module):
    """
    CDT Model Phase 3: Pre-trained Models統合版

    アーキテクチャ:
    1. Pre-trained Encoders
       - DNA: Nucleotide Transformer v2 (97.9M params)
       - RNA: Simple Transformer (placeholder)
       - Protein: Simple Transformer (placeholder)
    2. Projection Layers (統一次元へ変換)
    3. Cross-Attention Layers
    4. Virtual Cell Embedder

    Week 2: DNA (NT-v2) 統合 ✅
    Week 3: RNA (RNA-FM) に置き換え ✅
    Week 4: Protein (ESM-2) に置き換え ✅
    """

    def __init__(
        self,
        # DNA Encoder設定
        dna_model_name: str = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        freeze_dna: bool = True,

        # RNA Encoder設定
        freeze_rna: bool = True,

        # Protein Encoder設定
        protein_model_name: str = "esm2_t30_150M_UR50D",
        freeze_protein: bool = True,

        # モデル全体の設定
        d_model: int = 768,
        nhead: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            dna_model_name: Hugging FaceのDNAモデル名
            freeze_dna: DNAエンコーダーを固定するか
            freeze_rna: RNAエンコーダーを固定するか
            protein_model_name: ESM-2モデル名
            freeze_protein: Proteinエンコーダーを固定するか
            d_model: 統一モデル次元
            nhead: Cross-Attentionヘッド数
            dropout: ドロップアウト率
        """
        super().__init__()

        self.d_model = d_model

        # =================================================================
        # 1. DNA Encoder: Nucleotide Transformer v2 (Pre-trained)
        # =================================================================
        print(f"Loading DNA encoder: {dna_model_name}...")

        self.dna_tokenizer = AutoTokenizer.from_pretrained(
            dna_model_name,
            trust_remote_code=True
        )

        self.dna_encoder = AutoModelForMaskedLM.from_pretrained(
            dna_model_name,
            trust_remote_code=True
        )

        # DNAエンコーダーを固定
        if freeze_dna:
            for param in self.dna_encoder.parameters():
                param.requires_grad = False
            print(f"  DNA encoder frozen (97.9M params)")

        # DNA出力次元: 512 → d_model (768)
        dna_hidden_size = self.dna_encoder.config.hidden_size  # 512
        self.dna_projection = nn.Linear(dna_hidden_size, d_model)

        print(f"  DNA: {dna_hidden_size}d → {d_model}d (projection)")

        # =================================================================
        # 2. RNA Encoder: RNA-FM (Pre-trained)
        # =================================================================
        print("Loading RNA encoder: RNA-FM...")

        self.rna_model, self.rna_alphabet = fm.pretrained.rna_fm_t12()
        self.rna_batch_converter = self.rna_alphabet.get_batch_converter()

        # RNAエンコーダーを固定
        if freeze_rna:
            for param in self.rna_model.parameters():
                param.requires_grad = False
            print(f"  RNA encoder frozen (99M params)")

        # RNA出力次元: 640 → d_model (768)
        rna_hidden_size = 640  # RNA-FMの出力次元
        self.rna_projection = nn.Linear(rna_hidden_size, d_model)

        print(f"  RNA: {rna_hidden_size}d → {d_model}d (projection)")

        # =================================================================
        # 3. Protein Encoder: ESM-2 (Pre-trained)
        # =================================================================
        print("Loading Protein encoder: ESM-2...")

        self.protein_model, self.protein_alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.protein_batch_converter = self.protein_alphabet.get_batch_converter()

        # Proteinエンコーダーを固定
        if freeze_protein:
            for param in self.protein_model.parameters():
                param.requires_grad = False
            print(f"  Protein encoder frozen (150M params)")

        # Protein出力次元: 640 → d_model (768)
        protein_hidden_size = 640  # ESM-2の出力次元
        self.protein_projection = nn.Linear(protein_hidden_size, d_model)

        print(f"  Protein: {protein_hidden_size}d → {d_model}d (projection)")

        # =================================================================
        # 4. Cross-Attention Layers (Phase 2と同じ)
        # =================================================================
        self.dna_to_rna = CrossAttentionLayer(d_model, nhead, dropout=dropout)
        self.rna_to_protein = CrossAttentionLayer(d_model, nhead, dropout=dropout)
        self.protein_to_dna = CrossAttentionLayer(d_model, nhead, dropout=dropout)

        print("Cross-Attention layers initialized")

        # =================================================================
        # 5. Virtual Cell Embedder (Phase 2と同じ)
        # =================================================================
        self.vce = VirtualCellEmbedder(d_model, dropout=dropout)

        print("Virtual Cell Embedder initialized")
        print(f"\nCDTModelPhase3 ready! Total dimension: {d_model}")

    def forward(
        self,
        dna_sequences: List[str],
        rna_sequences: List[str],
        protein_sequences: List[str],
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            dna_sequences: List[str] DNA配列の文字列リスト
                例: ["ATGCGATCG", "ATGGCTA"]

            rna_sequences: List[str] RNA配列の文字列リスト
                例: ["AUGCUAGCU", "AUGGCUA"]

            protein_sequences: List[str] Protein配列の文字列リスト
                例: ["MKTAYIAK", "MKTAY"]

            return_attention: Attention mapを返すか

        Returns:
            cell_embedding: [batch, d_model] セルレベルの埋め込み

            (return_attention=Trueの場合)
            attention_maps: Dict[str, torch.Tensor]
        """

        # =================================================================
        # 1. DNA Encoding (Nucleotide Transformer v2)
        # =================================================================

        # DNAをトークナイズ
        dna_inputs = self.dna_tokenizer(
            dna_sequences,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        # デバイスを合わせる（CPUで動作）
        device = torch.device('cpu')  # すべてCPUで統一
        dna_inputs = {k: v.to(device) for k, v in dna_inputs.items()}

        # Nucleotide Transformer v2で特徴抽出
        with torch.no_grad():
            dna_outputs = self.dna_encoder(
                **dna_inputs,
                output_hidden_states=True
            )

        # 最後の層のhidden statesを取得
        dna_features = dna_outputs.hidden_states[-1]  # [batch, dna_seq_len, 512]

        # Projection: 512 → 768
        dna_emb = self.dna_projection(dna_features)  # [batch, dna_seq_len, 768]

        # DNA padding mask (transformersの1=valid, 0=paddingを反転)
        dna_key_padding_mask = (1 - dna_inputs['attention_mask']).bool()

        # =================================================================
        # 2. RNA Encoding (RNA-FM)
        # =================================================================

        # RNA配列をトークナイズ
        rna_batch = [(f'rna{i}', seq) for i, seq in enumerate(rna_sequences)]
        _, _, rna_tokens_converted = self.rna_batch_converter(rna_batch)

        # デバイスを合わせる
        rna_tokens_converted = rna_tokens_converted.to(device)

        # RNA-FMで特徴抽出
        with torch.no_grad():
            rna_results = self.rna_model(rna_tokens_converted, repr_layers=[12])
            rna_features = rna_results['representations'][12]  # [batch, rna_seq_len, 640]

        # Projection: 640 → 768
        rna_emb = self.rna_projection(rna_features)  # [batch, rna_seq_len, 768]

        # RNA padding mask (RNA-FMは自動的にpadding処理)
        rna_padding_mask = None  # RNA-FMが内部で処理

        # =================================================================
        # 3. Protein Encoding (ESM-2)
        # =================================================================

        # Protein配列をトークナイズ
        protein_batch = [(f'protein{i}', seq) for i, seq in enumerate(protein_sequences)]
        _, _, protein_tokens_converted = self.protein_batch_converter(protein_batch)

        # デバイスを合わせる
        protein_tokens_converted = protein_tokens_converted.to(device)

        # ESM-2で特徴抽出
        with torch.no_grad():
            protein_results = self.protein_model(protein_tokens_converted, repr_layers=[30])
            protein_features = protein_results['representations'][30]  # [batch, protein_seq_len, 640]

        # Projection: 640 → 768
        protein_emb = self.protein_projection(protein_features)  # [batch, protein_seq_len, 768]

        # Protein padding mask (ESM-2が内部で処理)
        protein_padding_mask = None  # ESM-2が内部で処理

        # =================================================================
        # 4. Cross-Attention
        # =================================================================

        # DNA → RNA
        rna_fused, attn_dna_to_rna = self.dna_to_rna(
            query=rna_emb,
            key=dna_emb,
            value=dna_emb,
            key_padding_mask=dna_key_padding_mask
        )

        # RNA → Protein
        protein_fused, attn_rna_to_protein = self.rna_to_protein(
            query=protein_emb,
            key=rna_fused,
            value=rna_fused,
            key_padding_mask=rna_padding_mask
        )

        # Protein → DNA
        dna_fused, attn_protein_to_dna = self.protein_to_dna(
            query=dna_emb,
            key=protein_fused,
            value=protein_fused,
            key_padding_mask=protein_padding_mask
        )

        # =================================================================
        # 5. Virtual Cell Embedder
        # =================================================================
        cell_embedding = self.vce(
            dna_fused,
            rna_fused,
            protein_fused
        )  # [batch, 768]

        if return_attention:
            attention_maps = {
                'dna_to_rna': attn_dna_to_rna,
                'rna_to_protein': attn_rna_to_protein,
                'protein_to_dna': attn_protein_to_dna
            }
            return cell_embedding, attention_maps

        return cell_embedding

    def get_trainable_params(self) -> int:
        """学習可能なパラメータ数を取得"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_params(self) -> int:
        """全パラメータ数を取得"""
        return sum(p.numel() for p in self.parameters())


def print_model_info(model: CDTModelPhase3):
    """モデル情報を表示"""
    print("\n" + "="*70)
    print("CDT Model Phase 3 - Model Information")
    print("="*70)

    total_params = model.get_total_params()
    trainable_params = model.get_trainable_params()
    frozen_params = total_params - trainable_params

    print(f"\nParameters:")
    print(f"  Total:      {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"  Trainable:  {trainable_params:,} ({trainable_params/1e6:.1f}M)")
    print(f"  Frozen:     {frozen_params:,} ({frozen_params/1e6:.1f}M)")
    print(f"  Frozen %:   {frozen_params/total_params*100:.1f}%")

    print(f"\nArchitecture:")
    print(f"  DNA Encoder:     Nucleotide Transformer v2 (97.9M, frozen)")
    print(f"  RNA Encoder:     RNA-FM (99M, frozen)")
    print(f"  Protein Encoder: ESM-2 (150M, frozen)")
    print(f"  Model dimension: {model.d_model}")

    print(f"\nComponents:")
    print(f"  DNA projection:      512 → {model.d_model}")
    print(f"  RNA projection:      640 → {model.d_model}")
    print(f"  Protein projection:  640 → {model.d_model}")
    print(f"  Cross-Attention:     3 layers")
    print(f"  VCE:                 Virtual Cell Embedder")

    print("="*70 + "\n")
