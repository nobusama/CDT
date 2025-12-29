#!/usr/bin/env python3
"""
Re-select gene list based on Gasperini and regenerate embeddings

Uses common genes from ProteinLM intersection Gasperini intersection scGPT (approximately 11,018 genes)

Output:
- k562_gene_embeddings_gasperini_aligned.h5 (RNA, 11018 x 512)
- human_proteomelm_embeddings_gasperini_aligned.h5 (Protein, 11018 x 768)
- protein_index_mapping_gasperini_aligned.npz

Usage:
    python scripts/data_prep/create_gasperini_aligned_embeddings.py
"""

import os
import sys
import json
import gzip
import numpy as np
import h5py
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed/embeddings"

# Input files
MORRIS_DATA = DATA_DIR / "processed/morris/stingseq_v2.h5"
GASPERINI_GENES = DATA_DIR / "raw/gasperini/scrna/GSE120861_at_scale_screen.genes.txt.gz"
SCGPT_VOCAB = PROJECT_ROOT / "models/scgpt/vocab.json"
ORIGINAL_PROTEOMELM = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings.h5"


def load_ensg_to_symbol_mapping() -> dict:
    """Create ENSG to gene symbol mapping from Morris STINGseq data"""
    print("Loading ENSG → symbol mapping from Morris data...")
    with h5py.File(MORRIS_DATA, 'r') as f:
        gene_ids = [g.decode() if isinstance(g, bytes) else g for g in f['gene_ids'][:]]
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]

    ensg_to_symbol = {ensg: symbol for ensg, symbol in zip(gene_ids, gene_names)}
    print(f"  Loaded {len(ensg_to_symbol)} ENSG → symbol mappings")
    return ensg_to_symbol


def get_gasperini_genes(ensg_to_symbol: dict) -> set:
    """Get genes expressed in Gasperini scRNA-seq (symbol format)"""
    print("Loading Gasperini genes...")
    with gzip.open(GASPERINI_GENES, 'rt') as f:
        gasperini_ensg = [line.strip() for line in f]

    gasperini_genes = set()
    for e in gasperini_ensg:
        if e in ensg_to_symbol:
            gasperini_genes.add(ensg_to_symbol[e])

    print(f"  Gasperini genes (symbols): {len(gasperini_genes)}")
    return gasperini_genes


def get_scgpt_genes() -> set:
    """Get genes in scGPT vocabulary"""
    print("Loading scGPT vocabulary...")
    with open(SCGPT_VOCAB, 'r') as f:
        vocab = json.load(f)
    scgpt_genes = set(vocab.keys())
    print(f"  scGPT genes: {len(scgpt_genes)}")
    return scgpt_genes


def get_proteomelm_genes() -> tuple:
    """Original ProteinLM embeddings"""
    print("Loading original ProteinLM embeddings...")
    with h5py.File(ORIGINAL_PROTEOMELM, 'r') as f:
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
        uniprot_ids = [g.decode() if isinstance(g, bytes) else g for g in f['uniprot_ids'][:]]
        embeddings = f['embeddings'][:]

    print(f"  ProteinLM genes: {len(gene_names)}")
    print(f"  Embeddings shape: {embeddings.shape}")
    return gene_names, uniprot_ids, embeddings


def create_aligned_gene_list():
    """Create common gene list from Gasperini intersection scGPT intersection ProteinLM"""
    print("\n" + "=" * 60)
    print("Creating Gasperini-aligned gene list")
    print("=" * 60)

    # Load data
    ensg_to_symbol = load_ensg_to_symbol_mapping()
    gasperini_genes = get_gasperini_genes(ensg_to_symbol)
    scgpt_genes = get_scgpt_genes()
    protein_genes, protein_uniprots, protein_embeddings = get_proteomelm_genes()

    # Create gene to protein index mapping
    protein_gene_to_idx = {g: i for i, g in enumerate(protein_genes)}
    protein_gene_set = set(protein_genes)

    # Find intersection
    print("\nComputing intersection...")
    aligned_genes = gasperini_genes & scgpt_genes & protein_gene_set
    print(f"  Gasperini ∩ scGPT ∩ ProteinLM: {len(aligned_genes)}")

    # Sort for reproducibility
    aligned_genes_list = sorted(list(aligned_genes))
    print(f"  Final aligned gene count: {len(aligned_genes_list)}")

    # Get protein embeddings for aligned genes
    print("\nExtracting ProteinLM embeddings for aligned genes...")
    aligned_protein_indices = [protein_gene_to_idx[g] for g in aligned_genes_list]
    aligned_protein_embeddings = protein_embeddings[aligned_protein_indices]
    aligned_uniprot_ids = [protein_uniprots[i] for i in aligned_protein_indices]

    print(f"  Aligned ProteinLM shape: {aligned_protein_embeddings.shape}")

    return aligned_genes_list, aligned_uniprot_ids, aligned_protein_embeddings


def save_proteomelm_embeddings(genes, uniprot_ids, embeddings):
    """Save aligned ProteinLM embeddings"""
    output_path = OUTPUT_DIR / "human_proteomelm_embeddings_gasperini_aligned.h5"
    print(f"\nSaving ProteinLM embeddings to {output_path}")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings.astype(np.float32), compression='gzip')
        f.create_dataset('gene_names', data=np.array(genes, dtype='S'))
        f.create_dataset('uniprot_ids', data=np.array(uniprot_ids, dtype='S'))
        f.attrs['n_proteins'] = len(genes)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['model'] = 'Bitbol-Lab/ProteomeLM-M'
        f.attrs['alignment'] = 'Gasperini ∩ scGPT ∩ ProteinLM'

    print(f"  Saved: {len(genes)} proteins, {embeddings.shape[1]} dims")


def create_index_mapping(aligned_genes: list):
    """Create old to new index mapping for training data compatibility"""
    # Load original training data to get the original protein indices
    # We need to map from the old ESM2 indices to new aligned indices

    # Actually, we need to create a mapping from gene name to new index
    gene_to_new_idx = {g: i for i, g in enumerate(aligned_genes)}

    output_path = OUTPUT_DIR / "protein_index_mapping_gasperini_aligned.npz"
    print(f"\nSaving index mapping to {output_path}")

    # Save gene to index mapping
    np.savez(
        output_path,
        gene_names=np.array(aligned_genes, dtype='S'),
        gene_to_idx=np.array(list(gene_to_new_idx.items()), dtype=object)
    )

    print(f"  Saved: {len(aligned_genes)} gene mappings")
    return gene_to_new_idx


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create aligned gene list and extract ProteinLM embeddings
    aligned_genes, aligned_uniprots, aligned_protein_embeddings = create_aligned_gene_list()

    # Save ProteinLM embeddings
    save_proteomelm_embeddings(aligned_genes, aligned_uniprots, aligned_protein_embeddings)

    # Create index mapping
    gene_to_new_idx = create_index_mapping(aligned_genes)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Aligned genes: {len(aligned_genes)}")
    print(f"  ProteinLM: human_proteomelm_embeddings_gasperini_aligned.h5")
    print(f"  Mapping: protein_index_mapping_gasperini_aligned.npz")
    print("\nNext steps:")
    print("  1. Generate scGPT RNA embeddings for aligned genes")
    print("  2. Regenerate training data with new gene indices")


if __name__ == "__main__":
    main()
