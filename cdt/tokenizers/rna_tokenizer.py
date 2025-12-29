"""
RNA Tokenizer

RNA配列を数値に変換するトークナイザー
DNAとの違い：T（チミン）の代わりにU（ウラシル）を使用
"""


class RNATokenizer:
    """
    RNA配列（AUCG）をトークン（整数）に変換

    Example:
        >>> tokenizer = RNATokenizer()
        >>> tokens = tokenizer.encode("AUCG")
        >>> print(tokens)  # [0, 1, 2, 3]
        >>> seq = tokenizer.decode(tokens)
        >>> print(seq)  # "AUCG"
    """

    def __init__(self):
        """
        トークナイザーの初期化

        vocab: 塩基 → 整数の辞書
        """
        # 語彙（vocabulary）: 各塩基に番号を割り当て
        self.vocab = {
            'A': 0,  # Adenine（アデニン）
            'U': 1,  # Uracil（ウラシル）- RNAではTの代わりにU
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
        RNA配列を整数リストに変換

        Args:
            sequence (str): RNA配列（例: "AUCGAU"）

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
        整数リストをRNA配列に変換

        Args:
            tokens (list[int]): トークンIDのリスト（例: [0, 1, 2, 3]）

        Returns:
            str: RNA配列（例: "AUCG"）
        """
        # 各整数を塩基に変換
        sequence = ''.join([self.id_to_token[token_id] for token_id in tokens])
        return sequence

    def __len__(self):
        """語彙サイズを返す"""
        return self.vocab_size

    def __repr__(self):
        """トークナイザーの文字列表現"""
        return f"RNATokenizer(vocab_size={self.vocab_size})"


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    # トークナイザーのインスタンス作成
    tokenizer = RNATokenizer()

    print(f"トークナイザー: {tokenizer}")
    print(f"語彙サイズ: {len(tokenizer)}")
    print(f"語彙: {tokenizer.vocab}")
    print()

    # テスト配列
    rna_sequence = "AUCGAUCG"
    print(f"元の配列: {rna_sequence}")

    # エンコード（配列 → 数値）
    tokens = tokenizer.encode(rna_sequence)
    print(f"トークン化: {tokens}")

    # デコード（数値 → 配列）
    decoded = tokenizer.decode(tokens)
    print(f"デコード: {decoded}")
    print(f"一致: {rna_sequence == decoded}")
    print()

    # 未知の塩基を含む配列
    unknown_sequence = "AUXCG"  # X は未知
    print(f"未知の塩基を含む配列: {unknown_sequence}")
    tokens = tokenizer.encode(unknown_sequence)
    print(f"トークン化: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"デコード: {decoded}")  # X → N に変換される
