"""
CDT Dataset

PyTorchのDatasetクラス
合成データをトークン化してモデルに渡す
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
from cdt.data.synthetic_data_generator import SyntheticDataGenerator
from cdt.tokenizers.dna_tokenizer import DNATokenizer
from cdt.tokenizers.rna_tokenizer import RNATokenizer
from cdt.tokenizers.protein_tokenizer import ProteinTokenizer


class CDTDataset(Dataset):
    """
    CDTモデル用のPyTorchデータセット

    合成データを生成し、トークン化してTensorに変換

    Example:
        >>> dataset = CDTDataset(num_samples=100)
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['dna', 'rna', 'protein', 'dna_tokens', 'rna_tokens', 'protein_tokens'])
    """

    def __init__(
        self,
        num_samples: int = 1000,
        min_length: int = 30,
        max_length: int = 300,
        seed: int = None
    ):
        """
        データセットの初期化

        Args:
            num_samples: 生成するサンプル数
            min_length: DNA配列の最小長（塩基数）
            max_length: DNA配列の最大長（塩基数）
            seed: 乱数シード（再現性のため）
        """
        # トークナイザーの初期化
        self.dna_tokenizer = DNATokenizer()
        self.rna_tokenizer = RNATokenizer()
        self.protein_tokenizer = ProteinTokenizer()

        # データ生成器の初期化
        self.generator = SyntheticDataGenerator(seed=seed)

        # 合成データの生成
        self.data = self.generator.generate(
            num_samples=num_samples,
            min_length=min_length,
            max_length=max_length
        )

    def __len__(self) -> int:
        """データセットのサイズを返す"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        指定されたインデックスのサンプルを取得

        Args:
            idx: サンプルのインデックス

        Returns:
            dict: {
                'dna': str,
                'rna': str,
                'protein': str,
                'dna_tokens': Tensor,
                'rna_tokens': Tensor,
                'protein_tokens': Tensor
            }
        """
        # 生の配列データを取得
        sample = self.data[idx]

        # トークン化
        dna_tokens = self.dna_tokenizer.encode(sample['dna'])
        rna_tokens = self.rna_tokenizer.encode(sample['rna'])
        protein_tokens = self.protein_tokenizer.encode(sample['protein'])

        # PyTorch Tensorに変換
        return {
            # 生の配列（文字列）
            'dna': sample['dna'],
            'rna': sample['rna'],
            'protein': sample['protein'],
            # トークン化された配列（Tensor）
            'dna_tokens': torch.tensor(dna_tokens, dtype=torch.long),
            'rna_tokens': torch.tensor(rna_tokens, dtype=torch.long),
            'protein_tokens': torch.tensor(protein_tokens, dtype=torch.long),
        }

    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        各トークナイザーの語彙サイズを返す

        Returns:
            dict: {'dna': int, 'rna': int, 'protein': int}
        """
        return {
            'dna': len(self.dna_tokenizer),
            'rna': len(self.rna_tokenizer),
            'protein': len(self.protein_tokenizer),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    バッチをパディングして統一する

    PyTorchのDataLoaderで使用するカスタムcollate関数
    異なる長さの配列をパディングして同じ長さに揃える

    Args:
        batch: サンプルのリスト

    Returns:
        dict: パディング済みのバッチ
    """
    # バッチサイズ
    batch_size = len(batch)

    # 各配列の最大長を計算
    max_dna_len = max(len(sample['dna_tokens']) for sample in batch)
    max_rna_len = max(len(sample['rna_tokens']) for sample in batch)
    max_protein_len = max(len(sample['protein_tokens']) for sample in batch)

    # パディング値
    dna_pad_id = 4  # 'N'
    rna_pad_id = 4  # 'N'
    protein_pad_id = 20  # 'X'

    # パディング済みのテンソルを作成
    dna_tokens = torch.full((batch_size, max_dna_len), dna_pad_id, dtype=torch.long)
    rna_tokens = torch.full((batch_size, max_rna_len), rna_pad_id, dtype=torch.long)
    protein_tokens = torch.full((batch_size, max_protein_len), protein_pad_id, dtype=torch.long)

    # 生の配列を格納するリスト
    dna_seqs = []
    rna_seqs = []
    protein_seqs = []

    # 各サンプルをパディング済みテンソルにコピー
    for i, sample in enumerate(batch):
        # トークン
        dna_len = len(sample['dna_tokens'])
        rna_len = len(sample['rna_tokens'])
        protein_len = len(sample['protein_tokens'])

        dna_tokens[i, :dna_len] = sample['dna_tokens']
        rna_tokens[i, :rna_len] = sample['rna_tokens']
        protein_tokens[i, :protein_len] = sample['protein_tokens']

        # 生の配列
        dna_seqs.append(sample['dna'])
        rna_seqs.append(sample['rna'])
        protein_seqs.append(sample['protein'])

    return {
        'dna': dna_seqs,
        'rna': rna_seqs,
        'protein': protein_seqs,
        'dna_tokens': dna_tokens,
        'rna_tokens': rna_tokens,
        'protein_tokens': protein_tokens,
    }


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("CDTDatasetのテスト")
    print("=" * 60)
    print()

    # データセット作成
    print("【データセット作成】")
    dataset = CDTDataset(num_samples=100, min_length=30, max_length=60, seed=42)
    print(f"データセットサイズ: {len(dataset)}")
    print(f"語彙サイズ: {dataset.get_vocab_sizes()}")
    print()

    # 1サンプル取得
    print("【1サンプル取得】")
    sample = dataset[0]
    print(f"DNA配列:     {sample['dna'][:30]}... (全{len(sample['dna'])}塩基)")
    print(f"RNA配列:     {sample['rna'][:30]}... (全{len(sample['rna'])}塩基)")
    print(f"Protein配列: {sample['protein']}")
    print()
    print(f"DNAトークン:     {sample['dna_tokens'][:10]}... (shape: {sample['dna_tokens'].shape})")
    print(f"RNAトークン:     {sample['rna_tokens'][:10]}... (shape: {sample['rna_tokens'].shape})")
    print(f"Proteinトークン: {sample['protein_tokens']} (shape: {sample['protein_tokens'].shape})")
    print()

    # DataLoader作成
    print("【DataLoader作成（バッチ処理）】")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 1バッチ取得
    batch = next(iter(dataloader))
    print(f"バッチサイズ: {len(batch['dna'])}")
    print(f"DNAトークン shape:     {batch['dna_tokens'].shape}")
    print(f"RNAトークン shape:     {batch['rna_tokens'].shape}")
    print(f"Proteinトークン shape: {batch['protein_tokens'].shape}")
    print()

    # バッチの詳細
    print("【バッチの詳細】")
    for i in range(len(batch['dna'])):
        print(f"サンプル {i+1}:")
        print(f"  DNA: {batch['dna'][i][:20]}...")
        print(f"  Protein: {batch['protein'][i]}")
