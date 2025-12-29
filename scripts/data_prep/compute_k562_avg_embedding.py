#!/usr/bin/env python3
"""
K562 Average scGPT Embedding Calculation Script

Computes the average scGPT embedding from Morris K562 cells,
to be used as a common cell embedding for Gasperini POC.

Usage:
    python scripts/compute_k562_avg_embedding.py
"""

import numpy as np
import h5py
from pathlib import Path

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
SCGPT_PATH = PROJECT_ROOT / "data/processed/embeddings/scgpt_embeddings_v1.h5"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/embeddings/k562_avg_scgpt.npy"


def main():
    print("=" * 60)
    print("K562 Average scGPT Embedding Calculation")
    print("=" * 60)

    # Load scGPT embeddings
    print(f"\nLoading scGPT embeddings from: {SCGPT_PATH}")

    if not SCGPT_PATH.exists():
        print(f"ERROR: File not found: {SCGPT_PATH}")
        print("Please run generate_scgpt_embeddings_local.py first.")
        return

    with h5py.File(SCGPT_PATH, 'r') as f:
        embeddings = f['embeddings'][:]
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dtype: {embeddings.dtype}")

        # Verify 512 dimensions
        if embeddings.shape[1] != 512:
            print(f"  WARNING: Expected 512 dims, got {embeddings.shape[1]}")
            print("  This might not be correct cell embeddings!")
            return

    # Compute average
    print("\nComputing mean embedding...")
    avg_embedding = embeddings.mean(axis=0).astype(np.float32)
    print(f"  Average shape: {avg_embedding.shape}")
    print(f"  Average dtype: {avg_embedding.dtype}")
    print(f"  Stats: min={avg_embedding.min():.4f}, max={avg_embedding.max():.4f}, mean={avg_embedding.mean():.4f}")

    # Save
    print(f"\nSaving to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, avg_embedding)

    # Verify
    loaded = np.load(OUTPUT_PATH)
    assert loaded.shape == (512,), f"Expected (512,), got {loaded.shape}"
    assert loaded.dtype == np.float32, f"Expected float32, got {loaded.dtype}"
    print(f"  Verified: {loaded.shape}, {loaded.dtype}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
