#!/usr/bin/env python3
"""
Generate Enformer sequence-level embeddings (896 bins × 3072 features).

This script extracts DNA sequences from hg38 and runs Enformer trunk
WITHOUT pooling to preserve spatial information for attention analysis.

Requirements:
    - hg38.fa in data/raw/reference/
    - enformer-pytorch package
    - pyfaidx for FASTA reading

Usage:
    python scripts/data_prep/generate_enformer_seqlevel.py --n_samples 1000

Output:
    data/processed/embeddings/enformer_seqlevel/pilot_{n}.h5
    Shape per sample: (896, 3072)
"""

import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch

# Enformer constants
ENFORMER_SEQ_LENGTH = 196_608  # Input sequence length (196kb)
ENFORMER_OUTPUT_BINS = 896     # Output bins (128bp resolution)
ENFORMER_TRUNK_DIM = 3072      # Trunk embedding dimension


def load_enformer_model(device: str = "mps"):
    """Load Enformer model (trunk only, no heads).

    Uses default target_length=896 to get the central ~114kb region
    (896 bins × 128bp = 114,688bp).
    """
    try:
        from enformer_pytorch import Enformer
    except ImportError:
        raise ImportError("Please install: pip install enformer-pytorch")

    print(f"Loading Enformer model on {device}...")
    model = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
    model = model.to(device)
    model.eval()
    return model


def load_genome(fasta_path: str):
    """Load reference genome with pyfaidx."""
    try:
        from pyfaidx import Fasta
    except ImportError:
        raise ImportError("Please install: pip install pyfaidx")

    print(f"Loading genome from {fasta_path}...")
    return Fasta(fasta_path)


def one_hot_encode(sequence: str) -> np.ndarray:
    """One-hot encode DNA sequence (A, C, G, T -> 4 channels)."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
    seq_upper = sequence.upper()
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)

    for i, base in enumerate(seq_upper):
        if base in 'ACGT':
            encoded[i, mapping[base]] = 1.0
        # N and other characters remain as zeros (ambiguous)

    return encoded


def extract_sequence(genome, chrom: str, center: int, seq_length: int = ENFORMER_SEQ_LENGTH) -> str:
    """Extract DNA sequence centered at position."""
    half_len = seq_length // 2
    start = max(0, center - half_len)
    end = center + half_len

    # Handle chromosome naming (chr1 vs 1)
    if chrom not in genome.keys():
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        else:
            chrom = f"chr{chrom}"

    if chrom not in genome.keys():
        raise ValueError(f"Chromosome {chrom} not found in genome")

    chrom_len = len(genome[chrom])
    if end > chrom_len:
        # Pad with N if sequence extends beyond chromosome
        seq = str(genome[chrom][start:chrom_len])
        seq += 'N' * (end - chrom_len)
    else:
        seq = str(genome[chrom][start:end])

    # Pad at start if needed
    if start < 0:
        seq = 'N' * (-start) + seq

    return seq


def run_enformer_trunk(model, sequences: list, device: str = "mps", batch_size: int = 1) -> np.ndarray:
    """
    Run Enformer and extract trunk embeddings (896 bins × 3072).

    Uses return_only_embeddings=True to get embeddings before the target heads.
    """
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]

            # One-hot encode
            batch_encoded = np.stack([one_hot_encode(seq) for seq in batch_seqs])
            batch_tensor = torch.from_numpy(batch_encoded).to(device)

            # Get trunk output using return_only_embeddings=True
            # Output shape: (batch, 896, 3072)
            trunk_output = model(batch_tensor, return_only_embeddings=True)

            # Move to CPU and convert to numpy
            embeddings = trunk_output.cpu().numpy()
            all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def load_enhancer_coordinates(h5_path: str, n_samples: int = None) -> list:
    """
    Load enhancer coordinates from Gasperini training data.

    Returns list of (chrom, center, pair_idx) tuples.
    """
    coords = []

    with h5py.File(h5_path, 'r') as f:
        # Check available keys
        print(f"H5 keys: {list(f.keys())}")

        # Get enhancer coordinates (handle both key naming conventions)
        chrom_key = 'enhancer_chr' if 'enhancer_chr' in f else 'enhancer_chrom'
        if chrom_key not in f:
            raise KeyError("enhancer_chr/enhancer_chrom not found in H5 file")

        chroms = f[chrom_key][:]
        starts = f['enhancer_start'][:]
        ends = f['enhancer_end'][:]

        for i, (chrom, start, end) in enumerate(zip(chroms, starts, ends)):
            if isinstance(chrom, bytes):
                chrom = chrom.decode('utf-8')
            center = (start + end) // 2
            coords.append((chrom, center, i))

            if n_samples and len(coords) >= n_samples:
                break

    return coords


def main():
    parser = argparse.ArgumentParser(description="Generate Enformer sequence-level embeddings")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples to process (default: 1000 for pilot)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for Enformer (default: 1, memory-intensive)")
    parser.add_argument("--device", type=str, default="mps",
                        help="Device: mps, cuda, or cpu")
    parser.add_argument("--output_dir", type=str,
                        default="data/processed/embeddings/enformer_seqlevel",
                        help="Output directory")
    args = parser.parse_args()

    # Paths
    genome_path = Path("data/raw/reference/hg38.fa")
    train_h5_path = Path("data/processed/training/gasperini_train.h5")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check prerequisites
    if not genome_path.exists():
        print(f"Error: {genome_path} not found!")
        print("Run: python scripts/data_prep/download_hg38.py")
        return

    if not train_h5_path.exists():
        print(f"Error: {train_h5_path} not found!")
        return

    # Load resources
    genome = load_genome(str(genome_path))
    model = load_enformer_model(args.device)
    coords = load_enhancer_coordinates(str(train_h5_path), args.n_samples)

    print(f"\nProcessing {len(coords)} enhancer regions...")

    # Process in batches and save
    output_file = output_dir / f"pilot_{len(coords)}.h5"

    # Pre-allocate output array
    embeddings = np.zeros((len(coords), ENFORMER_OUTPUT_BINS, ENFORMER_TRUNK_DIM), dtype=np.float32)
    metadata = []

    for i, (chrom, center, pair_idx) in enumerate(tqdm(coords, desc="Generating embeddings")):
        try:
            # Extract sequence
            seq = extract_sequence(genome, chrom, center)

            # Run Enformer trunk
            emb = run_enformer_trunk(model, [seq], args.device, batch_size=1)
            embeddings[i] = emb[0]

            metadata.append({
                'pair_idx': pair_idx,
                'chrom': chrom,
                'center': center
            })

        except Exception as e:
            print(f"\nError processing {chrom}:{center}: {e}")
            # Leave as zeros

    # Save to H5
    print(f"\nSaving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('embeddings', data=embeddings, compression='gzip')
        f.create_dataset('pair_indices', data=[m['pair_idx'] for m in metadata])
        f.create_dataset('chroms', data=[m['chrom'].encode() for m in metadata])
        f.create_dataset('centers', data=[m['center'] for m in metadata])

        # Metadata
        f.attrs['n_samples'] = len(coords)
        f.attrs['n_bins'] = ENFORMER_OUTPUT_BINS
        f.attrs['embedding_dim'] = ENFORMER_TRUNK_DIM
        f.attrs['seq_length'] = ENFORMER_SEQ_LENGTH

    print(f"\nDone! Output shape: {embeddings.shape}")
    print(f"File size: {output_file.stat().st_size / (1024**2):.1f} MB")


if __name__ == "__main__":
    main()
