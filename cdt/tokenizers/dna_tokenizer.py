"""
DNA Tokenizer

DNA配列を数値に変換するトークナイザー
"""


class DNATokenizer:
    """
    DNA配列（ATCG）をトークン（整数）に変換

    Example:
        >>> tokenizer = DNATokenizer()
        >>> tokens = tokenizer.encode("ATCG")
        >>> print(tokens)  # [0, 1, 2, 3]
        >>> seq = tokenizer.decode(tokens)
        >>> print(seq)  # "ATCG"
    """

    def __init__(self):
        """
        トークナイザーの初期化

        vocab: 塩基 → 整数の辞書
        """
        # 語彙（vocabulary）: 各塩基に番号を割り当て
        self.vocab = {
            'A': 0,  # Adenine（アデニン）
            'T': 1,  # Thymine（チミン）
            'C': 2,  # Cytosine（シトシン）
            'G': 3,  # Guanine（グアニン）
            'N': 4,  # Unknown（不明な塩基）
        }

        # 逆引き辞書: 整数 → 塩基
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # 特殊トークン
        self.pad_token = 'N'  # パディング用
        self.pad_token_id = self.vocab[self.pad_token]

        # 語彙サイズ
        self.vocab_size = len(self.vocab)

    def encode(self, sequence):
        """
        DNA配列を整数リストに変換

        Args:
            sequence (str): DNA配列（例: "ATCGAT"）

        Returns:
            list[int]: トークンIDのリスト（例: [0, 1, 2, 3, 0, 1]）
        """
        # 大文字に統一
        sequence = sequence.upper()

        # 各塩基を整数に変換
        tokens = []
        for base in sequence:
            if base in self.vocab:
                tokens.append(self.vocab[base])
            else:
                # 未知の塩基は 'N' として扱う
                tokens.append(self.vocab['N'])

        return tokens

    def decode(self, tokens):
        """
        整数リストをDNA配列に変換

        Args:
            tokens (list[int]): トークンIDのリスト（例: [0, 1, 2, 3]）

        Returns:
            str: DNA配列（例: "ATCG"）
        """
        # 各整数を塩基に変換
        sequence = ''.join([self.id_to_token[token_id] for token_id in tokens])
        return sequence

    def __len__(self):
        """語彙サイズを返す"""
        return self.vocab_size

    def __repr__(self):
        """トークナイザーの文字列表現"""
        return f"DNATokenizer(vocab_size={self.vocab_size})"


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    # トークナイザーのインスタンス作成
    tokenizer = DNATokenizer()

    print(f"トークナイザー: {tokenizer}")
    print(f"語彙サイズ: {len(tokenizer)}")
    print(f"語彙: {tokenizer.vocab}")
    print()

    # テスト配列
    dna_sequence = "ATCGATCG"
    print(f"元の配列: {dna_sequence}")

    # エンコード（配列 → 数値）
    tokens = tokenizer.encode(dna_sequence)
    print(f"トークン化: {tokens}")

    # デコード（数値 → 配列）
    decoded = tokenizer.decode(tokens)
    print(f"デコード: {decoded}")
    print(f"一致: {dna_sequence == decoded}")
    print()

    # 未知の塩基を含む配列
    unknown_sequence = "ATXCG"  # X は未知
    print(f"未知の塩基を含む配列: {unknown_sequence}")
    tokens = tokenizer.encode(unknown_sequence)
    print(f"トークン化: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"デコード: {decoded}")  # X → N に変換される
