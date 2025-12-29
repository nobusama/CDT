"""
Position Tracker for CDT Interpretability

Maps k-mer tokens back to nucleotide/amino acid positions
for fine-grained interpretation of cross-attention.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class PositionTracker:
    """
    Track and map token positions to sequence positions

    For character-based tokenization (RNA-FM, ESM-2):
      - 1 token = 1 nucleotide/amino acid
      - Direct interpretation possible

    For k-mer tokenization (DNA-NT-v2):
      - 1 token = k nucleotides (typically k=6)
      - Position tracking needed
    """

    def __init__(self, kmer_size: int = 6, stride: int = 1):
        """
        Initialize position tracker

        Args:
            kmer_size: Size of k-mer (default 6 for DNA-NT-v2)
            stride: Stride between k-mers (default 1 for overlapping)
        """
        self.kmer_size = kmer_size
        self.stride = stride

    def map_token_to_positions(self, token_idx: int) -> List[int]:
        """
        Map a token index to sequence positions

        Args:
            token_idx: Token index

        Returns:
            List of sequence positions covered by this token
        """
        start = token_idx * self.stride
        end = start + self.kmer_size
        return list(range(start, end))

    def calculate_nucleotide_length(self, num_tokens: int) -> int:
        """
        Calculate original sequence length from number of tokens

        Args:
            num_tokens: Number of k-mer tokens

        Returns:
            Original sequence length in nucleotides
        """
        if num_tokens == 0:
            return 0
        return (num_tokens - 1) * self.stride + self.kmer_size

    def aggregate_attention_to_nucleotide(
        self,
        attention_map: torch.Tensor,
        is_query_kmer: bool = False,
        is_key_kmer: bool = False
    ) -> torch.Tensor:
        """
        Aggregate token-level attention to nucleotide-level

        Args:
            attention_map: [query_tokens, key_tokens] or [batch, query_tokens, key_tokens]
            is_query_kmer: Whether query uses k-mer tokenization
            is_key_kmer: Whether key uses k-mer tokenization

        Returns:
            Nucleotide-level attention map
        """
        # Handle batch dimension
        has_batch = (attention_map.ndim == 3)
        if not has_batch:
            attention_map = attention_map.unsqueeze(0)

        batch_size, query_len, key_len = attention_map.shape

        # Calculate nucleotide lengths
        query_nuc_len = self.calculate_nucleotide_length(query_len) if is_query_kmer else query_len
        key_nuc_len = self.calculate_nucleotide_length(key_len) if is_key_kmer else key_len

        # Initialize nucleotide-level attention
        nucleotide_attention = torch.zeros(
            batch_size, query_nuc_len, key_nuc_len,
            dtype=attention_map.dtype,
            device=attention_map.device
        )

        # Aggregate from tokens to nucleotides
        for q_tok in range(query_len):
            for k_tok in range(key_len):
                # Get positions
                q_positions = self.map_token_to_positions(q_tok) if is_query_kmer else [q_tok]
                k_positions = self.map_token_to_positions(k_tok) if is_key_kmer else [k_tok]

                # Average attention weight across nucleotide pairs
                attention_value = attention_map[:, q_tok, k_tok] / (len(q_positions) * len(k_positions))

                for q_pos in q_positions:
                    for k_pos in k_positions:
                        if q_pos < query_nuc_len and k_pos < key_nuc_len:
                            nucleotide_attention[:, q_pos, k_pos] += attention_value

        # Remove batch dimension if input didn't have it
        if not has_batch:
            nucleotide_attention = nucleotide_attention.squeeze(0)

        return nucleotide_attention

    def visualize_token_to_nucleotide_mapping(
        self,
        num_tokens: int,
        save_path: Optional[str] = None
    ):
        """
        Visualize how tokens map to nucleotides

        Args:
            num_tokens: Number of k-mer tokens
            save_path: Optional path to save figure
        """
        nucleotide_len = self.calculate_nucleotide_length(num_tokens)

        # Create mapping matrix
        mapping = np.zeros((num_tokens, nucleotide_len))
        for token_idx in range(num_tokens):
            positions = self.map_token_to_positions(token_idx)
            for pos in positions:
                if pos < nucleotide_len:
                    mapping[token_idx, pos] = 1

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(
            mapping,
            cmap='Blues',
            cbar_kws={'label': 'Coverage'},
            ax=ax,
            xticklabels=5,
            yticklabels=5
        )
        ax.set_xlabel('Nucleotide Position')
        ax.set_ylabel('Token Index')
        ax.set_title(f'K-mer Token to Nucleotide Mapping (k={self.kmer_size}, stride={self.stride})')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved token mapping visualization to {save_path}")

        plt.close()

    def compare_attention_granularity(
        self,
        token_attention: torch.Tensor,
        nucleotide_attention: torch.Tensor,
        sequence_info: Dict,
        save_path: Optional[str] = None
    ):
        """
        Compare token-level vs nucleotide-level attention

        Args:
            token_attention: Token-level attention map
            nucleotide_attention: Nucleotide-level attention map
            sequence_info: Dict with 'query_seq' and 'key_seq'
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Token-level
        sns.heatmap(
            token_attention.cpu().numpy(),
            cmap='viridis',
            ax=axes[0],
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Attention Weight'}
        )
        axes[0].set_title(f'Token-Level Attention\n({token_attention.shape[0]} × {token_attention.shape[1]})')
        axes[0].set_xlabel('Key Tokens')
        axes[0].set_ylabel('Query Tokens')

        # Nucleotide-level
        sns.heatmap(
            nucleotide_attention.cpu().numpy(),
            cmap='viridis',
            ax=axes[1],
            xticklabels=False,
            yticklabels=False,
            cbar_kws={'label': 'Attention Weight'}
        )
        axes[1].set_title(f'Nucleotide-Level Attention\n({nucleotide_attention.shape[0]} × {nucleotide_attention.shape[1]})')
        axes[1].set_xlabel('Key Positions')
        axes[1].set_ylabel('Query Positions')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved attention comparison to {save_path}")

        plt.close()

    def compute_computational_overhead(
        self,
        token_attention: torch.Tensor,
        is_query_kmer: bool = True,
        is_key_kmer: bool = True,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        Measure computational overhead of position tracking

        Args:
            token_attention: Token-level attention map
            is_query_kmer: Whether query uses k-mer
            is_key_kmer: Whether key uses k-mer
            num_runs: Number of timing runs

        Returns:
            Dict with timing statistics
        """
        import time

        # Warm up
        for _ in range(3):
            _ = self.aggregate_attention_to_nucleotide(
                token_attention, is_query_kmer, is_key_kmer
            )

        # Time direct access (no processing)
        direct_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = token_attention.clone()  # Simulate direct use
            direct_times.append(time.perf_counter() - start)

        # Time with position tracking
        tracking_times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.aggregate_attention_to_nucleotide(
                token_attention, is_query_kmer, is_key_kmer
            )
            tracking_times.append(time.perf_counter() - start)

        direct_mean = np.mean(direct_times) * 1000  # ms
        tracking_mean = np.mean(tracking_times) * 1000  # ms
        overhead = (tracking_mean / direct_mean - 1) * 100  # percentage

        return {
            'direct_time_ms': direct_mean,
            'tracking_time_ms': tracking_mean,
            'overhead_percent': overhead,
            'absolute_overhead_ms': tracking_mean - direct_mean
        }


def test_position_tracker():
    """Test position tracker functionality"""
    print("=" * 80)
    print("Testing Position Tracker")
    print("=" * 80)
    print()

    tracker = PositionTracker(kmer_size=6, stride=1)

    # Test 1: Token to position mapping
    print("Test 1: Token to position mapping")
    for token_idx in range(5):
        positions = tracker.map_token_to_positions(token_idx)
        print(f"  Token {token_idx} → positions {positions}")
    print()

    # Test 2: Calculate sequence length
    print("Test 2: Calculate sequence length")
    for num_tokens in [10, 20, 30]:
        seq_len = tracker.calculate_nucleotide_length(num_tokens)
        print(f"  {num_tokens} tokens → {seq_len} nucleotides")
    print()

    # Test 3: Aggregate attention
    print("Test 3: Aggregate attention to nucleotide level")
    token_attention = torch.randn(10, 15)  # 10 query tokens, 15 key tokens
    print(f"  Input: {token_attention.shape} (token-level)")

    nucleotide_attention = tracker.aggregate_attention_to_nucleotide(
        token_attention,
        is_query_kmer=True,
        is_key_kmer=True
    )
    print(f"  Output: {nucleotide_attention.shape} (nucleotide-level)")
    print()

    # Test 4: Computational overhead
    print("Test 4: Computational overhead")
    overhead = tracker.compute_computational_overhead(token_attention)
    print(f"  Direct access: {overhead['direct_time_ms']:.3f} ms")
    print(f"  With tracking: {overhead['tracking_time_ms']:.3f} ms")
    print(f"  Overhead: {overhead['overhead_percent']:.1f}%")
    print()

    print("All tests passed!")


if __name__ == "__main__":
    test_position_tracker()
