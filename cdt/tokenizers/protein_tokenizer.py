"""
Protein Tokenizer

Tokenizer that converts protein sequences (amino acid sequences) to numerical values
20 standard amino acids + special characters
"""


class ProteinTokenizer:
    """
    Converts protein sequences to tokens (integers)

    20 standard amino acids:
    A(Ala), C(Cys), D(Asp), E(Glu), F(Phe), G(Gly), H(His), I(Ile), K(Lys), L(Leu),
    M(Met), N(Asn), P(Pro), Q(Gln), R(Arg), S(Ser), T(Thr), V(Val), W(Trp), Y(Tyr)

    Special characters:
    X: Unknown amino acid
    B: Aspartic acid (D) or Asparagine (N)
    Z: Glutamic acid (E) or Glutamine (Q)
    U: Selenocysteine
    O: Pyrrolysine

    Example:
        >>> tokenizer = ProteinTokenizer()
        >>> tokens = tokenizer.encode("ACDEFG")
        >>> print(tokens)  # [0, 1, 2, 3, 4, 5]
        >>> seq = tokenizer.decode(tokens)
        >>> print(seq)  # "ACDEFG"
    """

    def __init__(self):
        """
        Initialize the tokenizer

        vocab: Amino acid -> integer dictionary
        """
        # Vocabulary: assign a number to each amino acid
        # 20 standard amino acids (alphabetical order)
        self.vocab = {
            'A': 0,   # Alanine
            'C': 1,   # Cysteine
            'D': 2,   # Aspartic acid
            'E': 3,   # Glutamic acid
            'F': 4,   # Phenylalanine
            'G': 5,   # Glycine
            'H': 6,   # Histidine
            'I': 7,   # Isoleucine
            'K': 8,   # Lysine
            'L': 9,   # Leucine
            'M': 10,  # Methionine
            'N': 11,  # Asparagine
            'P': 12,  # Proline
            'Q': 13,  # Glutamine
            'R': 14,  # Arginine
            'S': 15,  # Serine
            'T': 16,  # Threonine
            'V': 17,  # Valine
            'W': 18,  # Tryptophan
            'Y': 19,  # Tyrosine
            # Special characters
            'X': 20,  # Unknown amino acid
            'B': 21,  # Aspartic acid or Asparagine
            'Z': 22,  # Glutamic acid or Glutamine
            'U': 23,  # Selenocysteine
            'O': 24,  # Pyrrolysine
        }

        # Reverse lookup dictionary: integer -> amino acid
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Special tokens
        self.pad_token = 'X'  # For padding (unknown amino acid)
        self.pad_token_id = self.vocab[self.pad_token]

        # Vocabulary size
        self.vocab_size = len(self.vocab)

    def encode(self, sequence):
        """
        Convert protein sequence to integer list

        Args:
            sequence (str): Protein sequence (e.g., "ACDEFG")

        Returns:
            list[int]: List of token IDs (e.g., [0, 1, 2, 3, 4, 5])
        """
        # Convert to uppercase
        sequence = sequence.upper()

        # Convert each amino acid to integer
        tokens = []
        for aa in sequence:
            if aa in self.vocab:
                tokens.append(self.vocab[aa])
            else:
                # Treat unknown amino acids as 'X'
                tokens.append(self.vocab['X'])

        return tokens

    def decode(self, tokens):
        """
        Convert integer list to protein sequence

        Args:
            tokens (list[int]): List of token IDs (e.g., [0, 1, 2, 3, 4, 5])

        Returns:
            str: Protein sequence (e.g., "ACDEFG")
        """
        # Convert each integer to amino acid
        sequence = ''.join([self.id_to_token[token_id] for token_id in tokens])
        return sequence

    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size

    def __repr__(self):
        """String representation of the tokenizer"""
        return f"ProteinTokenizer(vocab_size={self.vocab_size})"


# Usage example (only runs when this file is executed directly)
if __name__ == "__main__":
    # Create tokenizer instance
    tokenizer = ProteinTokenizer()

    print(f"Tokenizer: {tokenizer}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"20 standard amino acids + 5 special characters = {len(tokenizer)} characters")
    print()

    # Test sequence (standard protein sequence)
    protein_sequence = "ACDEFGHIKLMNPQRSTVWY"  # All 20 types
    print(f"Original sequence (20 amino acids): {protein_sequence}")

    # Encode (sequence -> numbers)
    tokens = tokenizer.encode(protein_sequence)
    print(f"Tokenized: {tokens}")

    # Decode (numbers -> sequence)
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    print(f"Match: {protein_sequence == decoded}")
    print()

    # Real protein sequence example (part of insulin A chain)
    insulin_a = "GIVEQCCTSICSLYQLENYCN"
    print(f"Insulin A chain (partial): {insulin_a}")
    tokens = tokenizer.encode(insulin_a)
    print(f"Tokenized: {tokens}")
    print()

    # Sequence with special characters
    special_sequence = "ACXBZ"
    print(f"Sequence with special characters: {special_sequence}")
    tokens = tokenizer.encode(special_sequence)
    print(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
