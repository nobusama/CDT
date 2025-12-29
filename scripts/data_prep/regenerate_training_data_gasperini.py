#!/usr/bin/env python3
"""
Regenerate training data with Gasperini-aligned gene indices

Maps original training data (esm2_idx: 0-20419) to new Gasperini-aligned
indices (0-11017)

Usage:
    python scripts/data_prep/regenerate_training_data_gasperini.py
"""

import numpy as np
import h5py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input files
ORIGINAL_PROTEOMELM = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings.h5"
GASPERINI_PROTEOMELM = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings_gasperini_aligned.h5"
TRAIN_DATA = DATA_DIR / "processed/training/gasperini_train.h5"
VAL_DATA = DATA_DIR / "processed/training/gasperini_val.h5"

# Output files
OUTPUT_DIR = DATA_DIR / "processed/training_gasperini_aligned"


def load_gene_mappings():
    """Create gene name mappings"""
    print("Loading gene mappings...")

    # Original ProteinLM: idx -> gene_name
    with h5py.File(ORIGINAL_PROTEOMELM, 'r') as f:
        orig_genes = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
    orig_idx_to_gene = {i: g for i, g in enumerate(orig_genes)}
    print(f"  Original ProteinLM: {len(orig_genes)} genes")

    # New Gasperini-aligned: gene_name -> new_idx
    with h5py.File(GASPERINI_PROTEOMELM, 'r') as f:
        new_genes = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
    gene_to_new_idx = {g: i for i, g in enumerate(new_genes)}
    print(f"  Gasperini-aligned: {len(new_genes)} genes")

    return orig_idx_to_gene, gene_to_new_idx


def convert_training_file(input_path: Path, output_path: Path,
                          orig_idx_to_gene: dict, gene_to_new_idx: dict):
    """Convert training data file"""
    print(f"\nConverting {input_path.name}...")

    with h5py.File(input_path, 'r') as f:
        # Load all data
        esm2_idx = f['esm2_idx'][:]
        enformer_idx = f['enformer_idx'][:]
        labels = f['labels'][:]
        beta = f['beta'][:] if 'beta' in f else np.zeros_like(labels, dtype=np.float32)
        enhancer_chr = f['enhancer_chr'][:]
        enhancer_start = f['enhancer_start'][:]
        enhancer_end = f['enhancer_end'][:]
        gene_ids = f['gene_ids'][:]

    n_samples = len(labels)
    print(f"  Original samples: {n_samples}")

    # Convert esm2_idx to new indices
    new_protein_idx = []
    valid_mask = []

    for i, old_idx in enumerate(esm2_idx):
        gene_name = orig_idx_to_gene.get(int(old_idx))
        if gene_name and gene_name in gene_to_new_idx:
            new_idx = gene_to_new_idx[gene_name]
            new_protein_idx.append(new_idx)
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    n_valid = valid_mask.sum()
    print(f"  Valid samples (gene in Gasperini-aligned): {n_valid} ({n_valid/n_samples*100:.1f}%)")

    # Filter data
    new_protein_idx = np.array(new_protein_idx, dtype=np.int32)
    enformer_idx_filtered = enformer_idx[valid_mask]
    labels_filtered = labels[valid_mask]
    beta_filtered = beta[valid_mask]
    enhancer_chr_filtered = enhancer_chr[valid_mask]
    enhancer_start_filtered = enhancer_start[valid_mask]
    enhancer_end_filtered = enhancer_end[valid_mask]
    gene_ids_filtered = gene_ids[valid_mask]

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('protein_idx', data=new_protein_idx)  # New name
        f.create_dataset('esm2_idx', data=new_protein_idx)  # Keep old name for compatibility
        f.create_dataset('enformer_idx', data=enformer_idx_filtered)
        f.create_dataset('labels', data=labels_filtered)
        f.create_dataset('beta', data=beta_filtered)
        f.create_dataset('enhancer_chr', data=enhancer_chr_filtered)
        f.create_dataset('enhancer_start', data=enhancer_start_filtered)
        f.create_dataset('enhancer_end', data=enhancer_end_filtered)
        f.create_dataset('gene_ids', data=gene_ids_filtered)

        f.attrs['n_samples'] = n_valid
        f.attrs['n_proteins'] = len(gene_to_new_idx)
        f.attrs['alignment'] = 'Gasperini-aligned (11018 genes)'

    print(f"  Saved to: {output_path}")
    print(f"  protein_idx range: {new_protein_idx.min()} - {new_protein_idx.max()}")

    return n_valid


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load mappings
    orig_idx_to_gene, gene_to_new_idx = load_gene_mappings()

    # Convert training data
    n_train = convert_training_file(
        TRAIN_DATA,
        OUTPUT_DIR / "gasperini_train.h5",
        orig_idx_to_gene,
        gene_to_new_idx
    )

    # Convert validation data
    n_val = convert_training_file(
        VAL_DATA,
        OUTPUT_DIR / "gasperini_val.h5",
        orig_idx_to_gene,
        gene_to_new_idx
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Training samples: {n_train}")
    print(f"  Validation samples: {n_val}")
    print(f"  Protein count: {len(gene_to_new_idx)}")
    print(f"  Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
