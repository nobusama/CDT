#!/usr/bin/env python3
"""
Generate Enformer embeddings for missing enhancers.

Compute Enformer embeddings for the missing 4,441 enhancers
and merge with existing pilot_1000.h5.

Usage:
    python scripts/data_prep/generate_enformer_missing.py

Output:
    colab_data_v3/pilot_full.h5 (all 5,303 enhancers)
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
    """Load Enformer model (trunk only)."""
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
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_upper = sequence.upper()
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)

    for i, base in enumerate(seq_upper):
        if base in 'ACGT':
            encoded[i, mapping[base]] = 1.0

    return encoded


def extract_sequence(genome, chrom: str, center: int, seq_length: int = ENFORMER_SEQ_LENGTH) -> str:
    """Extract DNA sequence centered at position."""
    half_len = seq_length // 2
    start = max(0, center - half_len)
    end = center + half_len

    # Handle chromosome naming
    if chrom not in genome.keys():
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        else:
            chrom = f"chr{chrom}"

    if chrom not in genome.keys():
        raise ValueError(f"Chromosome {chrom} not found in genome")

    chrom_len = len(genome[chrom])
    if end > chrom_len:
        seq = str(genome[chrom][start:chrom_len])
        seq += 'N' * (end - chrom_len)
    else:
        seq = str(genome[chrom][start:end])

    if start < 0:
        seq = 'N' * (-start) + seq

    return seq


def run_enformer_trunk(model, sequence: str, device: str = "mps") -> np.ndarray:
    """Run Enformer and extract trunk embedding."""
    with torch.no_grad():
        encoded = one_hot_encode(sequence)
        batch_tensor = torch.from_numpy(encoded).unsqueeze(0).to(device)
        trunk_output = model(batch_tensor, return_only_embeddings=True)
        return trunk_output.cpu().numpy()[0]


def load_missing_enhancers(tsv_path: str) -> list:
    """Load missing enhancer coordinates from TSV."""
    coords = []
    with open(tsv_path, 'r') as f:
        header = f.readline()  # Skip header
        for line in f:
            parts = line.strip().split('\t')
            chrom, start, end, center = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            coords.append({'chr': chrom, 'start': start, 'end': end, 'center': center})
    return coords


def load_existing_embeddings(h5_path: str) -> dict:
    """Load existing embeddings from pilot_1000.h5."""
    with h5py.File(h5_path, 'r') as f:
        return {
            'embeddings': f['embeddings'][:],
            'centers': f['centers'][:],
            'chroms': [c.decode() if isinstance(c, bytes) else c for c in f['chroms'][:]],
        }


def main():
    parser = argparse.ArgumentParser(description="Generate Enformer embeddings for missing enhancers")
    parser.add_argument("--device", type=str, default="mps", help="Device: mps, cuda, or cpu")
    parser.add_argument("--batch_start", type=int, default=0, help="Start index (for resuming)")
    parser.add_argument("--batch_end", type=int, default=None, help="End index (for partial run)")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Save checkpoint every N samples")
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent.parent
    genome_path = project_root / "data/raw/reference/hg38.fa"
    missing_tsv = project_root / "data/processed/missing_enhancers.tsv"
    existing_h5 = project_root / "colab_data_v3/pilot_1000.h5"
    output_h5 = project_root / "colab_data_v3/pilot_full.h5"
    checkpoint_h5 = project_root / "colab_data_v3/pilot_missing_checkpoint.h5"

    # Check prerequisites
    if not genome_path.exists():
        print(f"Error: {genome_path} not found!")
        print("Download hg38.fa first.")
        return

    if not missing_tsv.exists():
        print(f"Error: {missing_tsv} not found!")
        print("Run coordinate mapping test first.")
        return

    # Load resources
    print("Loading resources...")
    genome = load_genome(str(genome_path))
    model = load_enformer_model(args.device)

    # Load missing enhancers
    missing_coords = load_missing_enhancers(str(missing_tsv))
    print(f"Missing enhancers to process: {len(missing_coords)}")

    # Handle batch range
    start_idx = args.batch_start
    end_idx = args.batch_end if args.batch_end else len(missing_coords)
    batch_coords = missing_coords[start_idx:end_idx]
    print(f"Processing range: {start_idx} to {end_idx} ({len(batch_coords)} samples)")

    # Load checkpoint if exists
    if checkpoint_h5.exists() and start_idx == 0:
        print(f"Found checkpoint at {checkpoint_h5}")
        with h5py.File(checkpoint_h5, 'r') as f:
            processed_count = f.attrs.get('processed_count', 0)
        if processed_count > 0:
            print(f"Resuming from checkpoint (processed: {processed_count})")
            start_idx = processed_count
            batch_coords = missing_coords[start_idx:end_idx]

    # Pre-allocate arrays for new embeddings
    new_embeddings = np.zeros((len(batch_coords), ENFORMER_OUTPUT_BINS, ENFORMER_TRUNK_DIM), dtype=np.float32)
    new_centers = []
    new_chroms = []

    # Process missing enhancers
    print(f"\nProcessing {len(batch_coords)} missing enhancers...")

    for i, coord in enumerate(tqdm(batch_coords, desc="Computing embeddings")):
        try:
            seq = extract_sequence(genome, coord['chr'], coord['center'])
            emb = run_enformer_trunk(model, seq, args.device)
            new_embeddings[i] = emb
            new_centers.append(coord['center'])
            new_chroms.append(coord['chr'])

        except Exception as e:
            print(f"\nError at {coord['chr']}:{coord['center']}: {e}")
            new_centers.append(coord['center'])
            new_chroms.append(coord['chr'])
            # Leave embedding as zeros

        # Save checkpoint
        if (i + 1) % args.checkpoint_interval == 0:
            print(f"\nSaving checkpoint at {i + 1}...")
            with h5py.File(checkpoint_h5, 'w') as f:
                f.create_dataset('embeddings', data=new_embeddings[:i+1])
                f.create_dataset('centers', data=new_centers)
                f.create_dataset('chroms', data=[c.encode() for c in new_chroms])
                f.attrs['processed_count'] = start_idx + i + 1

    # Load existing embeddings
    print("\nLoading existing embeddings...")
    existing = load_existing_embeddings(str(existing_h5))

    # Merge
    print("Merging embeddings...")
    all_embeddings = np.concatenate([existing['embeddings'], new_embeddings], axis=0)
    all_centers = np.concatenate([existing['centers'], new_centers])
    all_chroms = existing['chroms'] + new_chroms

    # Save merged file
    print(f"Saving to {output_h5}...")
    with h5py.File(output_h5, 'w') as f:
        f.create_dataset('embeddings', data=all_embeddings, compression='gzip')
        f.create_dataset('centers', data=all_centers)
        f.create_dataset('chroms', data=[c.encode() for c in all_chroms])

        f.attrs['n_samples'] = len(all_centers)
        f.attrs['n_bins'] = ENFORMER_OUTPUT_BINS
        f.attrs['embedding_dim'] = ENFORMER_TRUNK_DIM

    print(f"\nDone!")
    print(f"  Total embeddings: {len(all_centers)}")
    print(f"  Existing: {len(existing['centers'])}")
    print(f"  New: {len(new_centers)}")
    print(f"  Output: {output_h5}")

    # Clean up checkpoint
    if checkpoint_h5.exists():
        checkpoint_h5.unlink()
        print("Checkpoint removed.")


if __name__ == "__main__":
    main()
