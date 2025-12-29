"""
Synthetic Data Generator

DNA → RNA → Protein の合成データを生成
Phase 1のPOC（Proof of Concept）用
"""

import random
from typing import List, Dict


class SyntheticDataGenerator:
    """
    生物学的に正しいDNA→RNA→Protein関係を持つ合成データを生成

    Central Dogma（セントラルドグマ）に従う：
    1. DNA配列をランダム生成
    2. DNA → RNA に転写（T → U）
    3. RNA → Protein に翻訳（コドン表使用）

    Example:
        >>> generator = SyntheticDataGenerator(seed=42)
        >>> data = generator.generate(num_samples=10, min_length=30, max_length=60)
        >>> print(data[0])
        {'dna': 'ATGCGA...', 'rna': 'AUGCGA...', 'protein': 'MR...'}
    """

    def __init__(self, seed: int = None):
        """
        データ生成器の初期化

        Args:
            seed: 乱数シード（再現性のため）
        """
        if seed is not None:
            random.seed(seed)

        # 遺伝暗号表（標準コドン表）
        # RNA配列（3塩基）→ アミノ酸（1文字）
        self.codon_table = {
            # Phenylalanine
            'UUU': 'F', 'UUC': 'F',
            # Leucine
            'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
            # Isoleucine
            'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
            # Methionine (開始コドン)
            'AUG': 'M',
            # Valine
            'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
            # Serine
            'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
            # Proline
            'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            # Threonine
            'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            # Alanine
            'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            # Tyrosine
            'UAU': 'Y', 'UAC': 'Y',
            # Histidine
            'CAU': 'H', 'CAC': 'H',
            # Glutamine
            'CAA': 'Q', 'CAG': 'Q',
            # Asparagine
            'AAU': 'N', 'AAC': 'N',
            # Lysine
            'AAA': 'K', 'AAG': 'K',
            # Aspartic acid
            'GAU': 'D', 'GAC': 'D',
            # Glutamic acid
            'GAA': 'E', 'GAG': 'E',
            # Cysteine
            'UGU': 'C', 'UGC': 'C',
            # Tryptophan
            'UGG': 'W',
            # Arginine
            'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
            # Glycine
            'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
            # 終止コドン
            'UAA': '*', 'UAG': '*', 'UGA': '*',
        }

        # DNA塩基
        self.dna_bases = ['A', 'T', 'C', 'G']

    def generate_dna(self, length: int) -> str:
        """
        ランダムなDNA配列を生成

        Args:
            length: DNA配列の長さ（塩基数）

        Returns:
            str: DNA配列（必ずATGで開始、3の倍数の長さ）
        """
        # 開始コドン（ATG）で始める
        dna = 'ATG'

        # 残りの長さを3の倍数に調整
        remaining = length - 3
        remaining = (remaining // 3) * 3

        # ランダムな塩基を追加
        dna += ''.join(random.choices(self.dna_bases, k=remaining))

        return dna

    def transcribe(self, dna: str) -> str:
        """
        DNA配列をRNA配列に転写

        Args:
            dna: DNA配列

        Returns:
            str: RNA配列（T → U に置換）
        """
        # T（チミン）を U（ウラシル）に置換
        return dna.replace('T', 'U')

    def translate(self, rna: str) -> str:
        """
        RNA配列をタンパク質配列に翻訳

        Args:
            rna: RNA配列

        Returns:
            str: タンパク質配列（アミノ酸の1文字表記）
        """
        protein = ""

        # 3塩基ずつ読む（コドン）
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i:i+3]

            # コドン表で変換
            if codon in self.codon_table:
                amino_acid = self.codon_table[codon]

                # 終止コドンで翻訳終了
                if amino_acid == '*':
                    break

                protein += amino_acid
            else:
                # 未知のコドンは 'X' として扱う
                protein += 'X'

        return protein

    def generate_sample(self, length: int) -> Dict[str, str]:
        """
        1つのサンプル（DNA, RNA, Protein）を生成

        Args:
            length: DNA配列の長さ

        Returns:
            dict: {'dna': str, 'rna': str, 'protein': str}
        """
        # DNA配列を生成
        dna = self.generate_dna(length)

        # DNA → RNA に転写
        rna = self.transcribe(dna)

        # RNA → Protein に翻訳
        protein = self.translate(rna)

        return {
            'dna': dna,
            'rna': rna,
            'protein': protein
        }

    def generate(
        self,
        num_samples: int = 100,
        min_length: int = 30,
        max_length: int = 300
    ) -> List[Dict[str, str]]:
        """
        複数のサンプルを生成

        Args:
            num_samples: 生成するサンプル数
            min_length: DNA配列の最小長（塩基数）
            max_length: DNA配列の最大長（塩基数）

        Returns:
            list: サンプルのリスト
        """
        samples = []

        for _ in range(num_samples):
            # ランダムな長さを選択（3の倍数に調整される）
            length = random.randint(min_length, max_length)

            # サンプル生成
            sample = self.generate_sample(length)
            samples.append(sample)

        return samples


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    # データ生成器のインスタンス作成
    generator = SyntheticDataGenerator(seed=42)

    print("=" * 60)
    print("合成データ生成器のテスト")
    print("=" * 60)
    print()

    # 1サンプル生成
    print("【1サンプル生成例】")
    sample = generator.generate_sample(length=30)
    print(f"DNA:     {sample['dna']}")
    print(f"RNA:     {sample['rna']}")
    print(f"Protein: {sample['protein']}")
    print()

    # 配列の長さを確認
    print("【長さの確認】")
    print(f"DNA長:     {len(sample['dna'])} 塩基")
    print(f"RNA長:     {len(sample['rna'])} 塩基")
    print(f"Protein長: {len(sample['protein'])} アミノ酸")
    print(f"比率:      DNA/RNA : Protein = 3 : 1")
    print()

    # 複数サンプル生成
    print("【10サンプル生成】")
    samples = generator.generate(num_samples=10, min_length=30, max_length=60)
    print(f"生成したサンプル数: {len(samples)}")
    print()

    # 最初の3サンプルを表示
    print("【最初の3サンプル】")
    for i, sample in enumerate(samples[:3]):
        print(f"サンプル {i+1}:")
        print(f"  DNA:     {sample['dna'][:30]}... (全{len(sample['dna'])}塩基)")
        print(f"  RNA:     {sample['rna'][:30]}... (全{len(sample['rna'])}塩基)")
        print(f"  Protein: {sample['protein']} (全{len(sample['protein'])}アミノ酸)")
        print()

    # 統計情報
    print("【統計情報】")
    dna_lengths = [len(s['dna']) for s in samples]
    protein_lengths = [len(s['protein']) for s in samples]
    print(f"DNA長の範囲: {min(dna_lengths)} - {max(dna_lengths)} 塩基")
    print(f"Protein長の範囲: {min(protein_lengths)} - {max(protein_lengths)} アミノ酸")
    print(f"平均DNA長: {sum(dna_lengths)/len(dna_lengths):.1f} 塩基")
    print(f"平均Protein長: {sum(protein_lengths)/len(protein_lengths):.1f} アミノ酸")
