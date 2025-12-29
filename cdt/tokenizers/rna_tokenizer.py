"""
RNA Tokenizer

Tokenizer that converts RNA sequences to numerical values
Difference from DNA: Uses U (Uracil) instead of T (Thymine)
"""


class RNATokenizer:
    """
    Converts RNA sequences (AUCG) to tokens (integers)

    Example:
        >>> tokenizer = RNATokenizer()
        >>> tokens = tokenizer.encode("AUCG")
        >>> print(tokens)  # [0, 1, 2, 3]
        >>> seq = tokenizer.decode(tokens)
        >>> print(seq)  # "AUCG"
    """

    def __init__(self):
        """
        Initialize the tokenizer

        vocab: Base -> integer dictionary
        """
        # Vocabulary: assign a number to each base
        self.vocab = {
            'A': 0,  # Adenine
            'U': 1,  # Uracil - In RNA, U replaces T
            'C': 2,  # Cytosine
            'G': 3,  # Guanine
            'N': 4,  # Unknown base
        }

        # Reverse lookup dictionary: integer -> base
        self.id_to_token = {v: k for k, v in self.vocab.items()}

        # Special tokens
        self.pad_token = 'N'  # For padding
        self.pad_token_id = self.vocab[self.pad_token]

        # Vocabulary size
        self.vocab_size = len(self.vocab)

    def encode(self, sequence):
        """
        Convert RNA sequence to integer list

        Args:
            sequence (str): RNA sequence (e.g., "AUCGAU")

        Returns:
            list[int]: List of token IDs (e.g., [0, 1, 2, 3, 0, 1])
        """
        # Convert to uppercase
        sequence = sequence.upper()

        # Convert each base to integer
        tokens = []
        for base in sequence:
            if base in self.vocab:
                tokens.append(self.vocab[base])
            else:
                # Treat unknown bases as 'N'
                tokens.append(self.vocab['N'])

        return tokens

    def decode(self, tokens):
        """
        Convert integer list to RNA sequence

        Args:
            tokens (list[int]): List of token IDs (e.g., [0, 1, 2, 3])

        Returns:
            str: RNA sequence (e.g., "AUCG")
        """
        # Convert each integer to base
        sequence = ''.join([self.id_to_token[token_id] for token_id in tokens])
        return sequence

    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size

    def __repr__(self):
        """String representation of the tokenizer"""
        return f"RNATokenizer(vocab_size={self.vocab_size})"


# Usage example (only runs when this file is executed directly)
if __name__ == "__main__":
    # Create tokenizer instance
    tokenizer = RNATokenizer()

    print(f"Tokenizer: {tokenizer}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Vocabulary: {tokenizer.vocab}")
    print()

    # Test sequence
    rna_sequence = "AUCGAUCG"
    print(f"Original sequence: {rna_sequence}")

    # Encode (sequence -> numbers)
    tokens = tokenizer.encode(rna_sequence)
    print(f"Tokenized: {tokens}")

    # Decode (numbers -> sequence)
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    print(f"Match: {rna_sequence == decoded}")
    print()

    # Sequence with unknown base
    unknown_sequence = "AUXCG"  # X is unknown
    print(f"Sequence with unknown base: {unknown_sequence}")
    tokens = tokenizer.encode(unknown_sequence)
    print(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")  # X is converted to N
