"""
DNA Tokenizer

Tokenizer that converts DNA sequences to numerical values
"""


class DNATokenizer:
    """
    Converts DNA sequences (ATCG) to tokens (integers)

    Example:
        >>> tokenizer = DNATokenizer()
        >>> tokens = tokenizer.encode("ATCG")
        >>> print(tokens)  # [0, 1, 2, 3]
        >>> seq = tokenizer.decode(tokens)
        >>> print(seq)  # "ATCG"
    """

    def __init__(self):
        """
        Initialize the tokenizer

        vocab: Base -> integer dictionary
        """
        # Vocabulary: assign a number to each base
        self.vocab = {
            'A': 0,  # Adenine
            'T': 1,  # Thymine
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
        Convert DNA sequence to integer list

        Args:
            sequence (str): DNA sequence (e.g., "ATCGAT")

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
        Convert integer list to DNA sequence

        Args:
            tokens (list[int]): List of token IDs (e.g., [0, 1, 2, 3])

        Returns:
            str: DNA sequence (e.g., "ATCG")
        """
        # Convert each integer to base
        sequence = ''.join([self.id_to_token[token_id] for token_id in tokens])
        return sequence

    def __len__(self):
        """Return vocabulary size"""
        return self.vocab_size

    def __repr__(self):
        """String representation of the tokenizer"""
        return f"DNATokenizer(vocab_size={self.vocab_size})"


# Usage example (only runs when this file is executed directly)
if __name__ == "__main__":
    # Create tokenizer instance
    tokenizer = DNATokenizer()

    print(f"Tokenizer: {tokenizer}")
    print(f"Vocabulary size: {len(tokenizer)}")
    print(f"Vocabulary: {tokenizer.vocab}")
    print()

    # Test sequence
    dna_sequence = "ATCGATCG"
    print(f"Original sequence: {dna_sequence}")

    # Encode (sequence -> numbers)
    tokens = tokenizer.encode(dna_sequence)
    print(f"Tokenized: {tokens}")

    # Decode (numbers -> sequence)
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")
    print(f"Match: {dna_sequence == decoded}")
    print()

    # Sequence with unknown base
    unknown_sequence = "ATXCG"  # X is unknown
    print(f"Sequence with unknown base: {unknown_sequence}")
    tokens = tokenizer.encode(unknown_sequence)
    print(f"Tokenized: {tokens}")
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: {decoded}")  # X is converted to N
