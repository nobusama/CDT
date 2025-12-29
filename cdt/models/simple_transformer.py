"""
Simple Transformer Model

Simple Transformer for Phase 1
Predicts Protein sequence from RNA sequence
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional Encoding

    Since Transformers do not have sequence order information,
    positional information needs to be explicitly added
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension (embedding size)
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Pre-compute positional encoding
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pe[:, 0::2] = torch.sin(position * div_term)  # Even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd dimensions

        # Add batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # Register as buffer (not trained)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]

        Returns:
            [batch_size, seq_len, d_model]
        """
        # Add positional encoding
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SimpleTransformer(nn.Module):
    """
    Simple Transformer for Phase 1

    Predicts RNA sequence (input) -> Protein sequence (output)

    Architecture:
    1. RNA embedding
    2. Positional encoding
    3. Transformer Encoder
    4. Linear layer (classification)
    """

    def __init__(
        self,
        rna_vocab_size: int = 5,      # RNA vocabulary size (A,U,C,G,N)
        protein_vocab_size: int = 25,  # Protein vocabulary size
        d_model: int = 128,            # Model dimension
        nhead: int = 4,                # Number of multi-head attention heads
        num_layers: int = 2,           # Number of Transformer layers
        dim_feedforward: int = 512,    # Feed-forward layer dimension
        dropout: float = 0.1,          # Dropout rate
        max_len: int = 1000            # Maximum sequence length
    ):
        """
        Args:
            rna_vocab_size: RNA vocabulary size
            protein_vocab_size: Protein vocabulary size
            d_model: Model dimension (embedding size)
            nhead: Number of attention heads
            num_layers: Number of Transformer layers
            dim_feedforward: Feed-forward layer dimension
            dropout: Dropout rate
            max_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        # RNA embedding layer (token ID -> vector)
        self.rna_embedding = nn.Embedding(rna_vocab_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # [batch, seq, feature] order
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output layer (predict Protein amino acid at each position)
        self.fc_out = nn.Linear(d_model, protein_vocab_size)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize parameters"""
        initrange = 0.1
        self.rna_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(
        self,
        rna_tokens: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            rna_tokens: RNA sequence tokens [batch_size, seq_len]
            src_key_padding_mask: Padding mask [batch_size, seq_len]
                                  True positions are masked (ignored)

        Returns:
            logits: [batch_size, seq_len, protein_vocab_size]
        """
        # 1. Embedding
        # [batch_size, seq_len] -> [batch_size, seq_len, d_model]
        x = self.rna_embedding(rna_tokens) * math.sqrt(self.d_model)

        # 2. Positional encoding
        x = self.pos_encoder(x)

        # 3. Transformer Encoder
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # 4. Output layer
        # [batch_size, seq_len, d_model] -> [batch_size, seq_len, protein_vocab_size]
        logits = self.fc_out(x)

        return logits

    def predict(self, rna_tokens: torch.Tensor) -> torch.Tensor:
        """
        Prediction (inference mode)

        Args:
            rna_tokens: RNA sequence tokens [batch_size, seq_len]

        Returns:
            predicted_tokens: Predicted Protein tokens [batch_size, seq_len]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(rna_tokens)
            # Select token with highest probability at each position
            predicted_tokens = torch.argmax(logits, dim=-1)
        return predicted_tokens


# Usage example (only runs when this file is executed directly)
if __name__ == "__main__":
    print("=" * 70)
    print("SimpleTransformer Test")
    print("=" * 70)
    print()

    # Create model
    print("[Model Creation]")
    model = SimpleTransformer(
        rna_vocab_size=5,
        protein_vocab_size=25,
        d_model=128,
        nhead=4,
        num_layers=2
    )

    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameter count: {total_params:,}")
    print(f"Trainable parameter count: {trainable_params:,}")
    print()

    # Verify operation with dummy data
    print("[Forward Pass Test]")
    batch_size = 4
    seq_len = 30

    # Dummy RNA sequence (random integers 0-4)
    rna_tokens = torch.randint(0, 5, (batch_size, seq_len))
    print(f"Input RNA tokens shape: {rna_tokens.shape}")

    # Forward pass
    logits = model(rna_tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"  -> [batch_size={batch_size}, seq_len={seq_len}, "
          f"Protein vocab size={logits.shape[-1]}]")
    print()

    # Prediction
    print("[Prediction Test]")
    predicted = model.predict(rna_tokens)
    print(f"Predicted Protein tokens shape: {predicted.shape}")
    print(f"First sample prediction: {predicted[0, :10]}")
    print()

    # Padding mask test
    print("[Padding Mask Test]")
    # Mask last 10 elements as padding
    padding_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    padding_mask[:, -10:] = True  # Mask last 10 elements

    logits_masked = model(rna_tokens, src_key_padding_mask=padding_mask)
    print(f"Masked output shape: {logits_masked.shape}")
    print("-> Padding portions are ignored in computation")
    print()

    print("=" * 70)
    print("Test complete! Model is working correctly.")
    print("=" * 70)
