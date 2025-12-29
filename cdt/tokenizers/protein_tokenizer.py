"""
Protein Tokenizer

タンパク質配列（アミノ酸配列）を数値に変換するトークナイザー
20種類の標準アミノ酸 + 特殊文字
"""


class ProteinTokenizer:
    """
    タンパク質配列をトークン（整数）に変換

    20種類の標準アミノ酸:
    A(Ala), C(Cys), D(Asp), E(Glu), F(Phe), G(Gly), H(His), I(Ile), K(Lys), L(Leu),
    M(Met), N(Asn), P(Pro), Q(Gln), R(Arg), S(Ser), T(Thr), V(Val), W(Trp), Y(Tyr)

    特殊文字:
    X: 未知のアミノ酸
    B: Aspartic acid (D) または Asparagine (N)
    Z: Glutamic acid (E) または Glutamine (Q)
    U: Selenocysteine（セレノシステイン）
    O: Pyrrolysine（ピロリジン）

    Example:
        >>> tokenizer = ProteinTokenizer()
        >>> tokens = tokenizer.encode("ACDEFG")
        >>> print(tokens)  # [0, 1, 2, 3, 4, 5]
        >>> seq = tokenizer.decode(tokens)
        >>> print(seq)  # "ACDEFG"
    """

    def __init__(self):
        """
        トークナイザーの初期化

        vocab: アミノ酸 → 整数の辞書
        """
        # 語彙（vocabulary）: 各アミノ酸に番号を割り当て
        # 20種類の標準アミノ酸（アルファベット順）
        self.vocab = {
            'A': 0,   # Alanine（アラニン）
            'C': 1,   # Cysteine（システイン）
            'D': 2,   # Aspartic acid（アスパラギン酸）
            'E': 3,   # Glutamic acid（グルタミン酸）
            'F': 4,   # Phenylalanine（フェニルアラニン）
            'G': 5,   # Glycine（グリシン）
            'H': 6,   # Histidine（ヒスチジン）
            'I': 7,   # Isoleucine（イソロイシン）
            'K': 8,   # Lysine（リジン）
            'L': 9,   # Leucine（ロイシン）
            'M': 10,  # Methionine（メチオニン）
            'N': 11,  # Asparagine（アスパラギン）
            'P': 12,  # Proline（プロリン）
            'Q': 13,  # Glutamine（グルタミン）
            'R': 14,  # Arginine（アルギニン）
            'S': 15,  # Serine（セリン）
            'T': 16,  # Threonine（スレオニン）
            'V': 17,  # Valine（バリン）
            'W': 18,  # Tryptophan（トリプトファン）
            'Y': 19,  # Tyrosine（チロシン）
            # 特殊文字
            'X': 20,  # Unknown（未知のアミノ酸）
            'B': 21,  # Aspartic acid or Asparagine
            'Z': 22,  # Glutamic acid or Glutamine
            'U': 23,  # Selenocysteine
            'O': 24,  # Pyrrolysine
        }

        # 逆引き辞書: 整数 → アミノ酸
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # 特殊トークン
        self.pad_token = 'X'  # パディング用（未知のアミノ酸）
        self.pad_token_id = self.vocab[self.pad_token]

        # 語彙サイズ
        self.vocab_size = len(self.vocab)

    def encode(self, sequence):
        """
        タンパク質配列を整数リストに変換

        Args:
            sequence (str): タンパク質配列（例: "ACDEFG"）

        Returns:
            list[int]: トークンIDのリスト（例: [0, 1, 2, 3, 4, 5]）
        """
        # 大文字に統一
        sequence = sequence.upper()

        # 各アミノ酸を整数に変換
        tokens = []
        for aa in sequence:
            if aa in self.vocab:
                tokens.append(self.vocab[aa])
            else:
                # 未知のアミノ酸は 'X' として扱う
                tokens.append(self.vocab['X'])

        return tokens

    def decode(self, tokens):
        """
        整数リストをタンパク質配列に変換

        Args:
            tokens (list[int]): トークンIDのリスト（例: [0, 1, 2, 3, 4, 5]）

        Returns:
            str: タンパク質配列（例: "ACDEFG"）
        """
        # 各整数をアミノ酸に変換
        sequence = ''.join([self.id_to_token[token_id] for token_id in tokens])
        return sequence

    def __len__(self):
        """語彙サイズを返す"""
        return self.vocab_size

    def __repr__(self):
        """トークナイザーの文字列表現"""
        return f"ProteinTokenizer(vocab_size={self.vocab_size})"


# 使用例（このファイルを直接実行した時のみ動く）
if __name__ == "__main__":
    # トークナイザーのインスタンス作成
    tokenizer = ProteinTokenizer()

    print(f"トークナイザー: {tokenizer}")
    print(f"語彙サイズ: {len(tokenizer)}")
    print(f"20種類の標準アミノ酸 + 5種類の特殊文字 = {len(tokenizer)} 文字")
    print()

    # テスト配列（標準的なタンパク質配列）
    protein_sequence = "ACDEFGHIKLMNPQRSTVWY"  # 20種類すべて
    print(f"元の配列（20種類のアミノ酸）: {protein_sequence}")

    # エンコード（配列 → 数値）
    tokens = tokenizer.encode(protein_sequence)
    print(f"トークン化: {tokens}")

    # デコード（数値 → 配列）
    decoded = tokenizer.decode(tokens)
    print(f"デコード: {decoded}")
    print(f"一致: {protein_sequence == decoded}")
    print()

    # 実際のタンパク質配列例（インスリンのA鎖の一部）
    insulin_a = "GIVEQCCTSICSLYQLENYCN"
    print(f"インスリンA鎖の一部: {insulin_a}")
    tokens = tokenizer.encode(insulin_a)
    print(f"トークン化: {tokens}")
    print()

    # 特殊文字を含む配列
    special_sequence = "ACXBZ"
    print(f"特殊文字を含む配列: {special_sequence}")
    tokens = tokenizer.encode(special_sequence)
    print(f"トークン化: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"デコード: {decoded}")
