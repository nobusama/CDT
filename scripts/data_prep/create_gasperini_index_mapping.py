#!/usr/bin/env python3
"""
Create index mapping for Gasperini-aligned (v3.3.1 format)

v3.3.1 format:
  old_to_new = [[old_idx, new_idx], ...]
  - old_idx: Index in original ProteinLM (20420)
  - new_idx: Index in new Gasperini-aligned (11018)

Usage:
    python scripts/data_prep/create_gasperini_index_mapping.py
"""

import numpy as np
import h5py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input files
ORIGINAL_PROTEOMELM = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings.h5"
GASPERINI_PROTEOMELM = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings_gasperini_aligned.h5"

# Output
OUTPUT_PATH = DATA_DIR / "processed/embeddings/protein_index_mapping_gasperini_aligned.npz"


def main():
    print("Creating Gasperini-aligned index mapping (v3.3.1 format)...")
    print()

    # Load original ProteinLM gene names
    print("Loading original ProteinLM...")
    with h5py.File(ORIGINAL_PROTEOMELM, 'r') as f:
        orig_genes = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
    orig_gene_to_idx = {g: i for i, g in enumerate(orig_genes)}
    print(f"  Genes: {len(orig_genes)}")

    # Load new Gasperini-aligned gene names
    print("Loading Gasperini-aligned ProteinLM...")
    with h5py.File(GASPERINI_PROTEOMELM, 'r') as f:
        new_genes = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
    print(f"  Genes: {len(new_genes)}")

    # Create old_to_new mapping
    print("\nCreating old_to_new mapping...")
    old_to_new = []
    old_indices = []

    for new_idx, gene in enumerate(new_genes):
        if gene in orig_gene_to_idx:
            old_idx = orig_gene_to_idx[gene]
            old_to_new.append([old_idx, new_idx])
            old_indices.append(old_idx)

    old_to_new = np.array(old_to_new, dtype=np.int64)
    old_indices = np.array(old_indices, dtype=np.int64)

    print(f"  Mappings created: {len(old_to_new)}")

    # Verify
    print("\nVerification:")
    print(f"  old_to_new shape: {old_to_new.shape}")
    print(f"  old_indices range: {old_indices.min()} - {old_indices.max()}")
    print(f"  new_indices range: {old_to_new[:, 1].min()} - {old_to_new[:, 1].max()}")

    # Spot check
    print("\n  Spot check (first 5):")
    for i in range(5):
        old_idx, new_idx = old_to_new[i]
        old_gene = orig_genes[old_idx]
        new_gene = new_genes[new_idx]
        match = "OK" if old_gene == new_gene else "MISMATCH"
        print(f"    old_idx={old_idx} ({old_gene}) -> new_idx={new_idx} ({new_gene}) {match}")

    # Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    np.savez(
        OUTPUT_PATH,
        old_to_new=old_to_new,
        old_indices=old_indices
    )

    print("\nDone!")
    print(f"  Format: old_to_new = [[old_idx, new_idx], ...]")
    print(f"  Use: old_to_new_idx = {{int(pair[0]): int(pair[1]) for pair in data['old_to_new']}}")


if __name__ == "__main__":
    main()
