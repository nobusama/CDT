#!/usr/bin/env python3
"""
Top発現遺伝子でフィルタリング (v3.4用)

Gasperini pseudo-bulkの発現量に基づいて、top N遺伝子のみを選択し、
埋め込みとインデックスマッピングを再生成する。

メモリ問題の解決:
- 11,018遺伝子 → 5,000遺伝子に削減
- Self-Attention: 31GB → 6.4GB (A100で実行可能)

Usage:
    python scripts/data_prep/filter_top_expressed_genes.py --n-genes 5000
"""

import argparse
import gzip
import numpy as np
import h5py
from pathlib import Path
from scipy.io import mmread

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# Input files
GASPERINI_EXPRS = DATA_DIR / "raw/gasperini/scrna/GSE120861_at_scale_screen.exprs.mtx"
GASPERINI_GENES = DATA_DIR / "raw/gasperini/scrna/GSE120861_at_scale_screen.genes.txt.gz"
MORRIS_DATA = DATA_DIR / "processed/morris/stingseq_v2.h5"

# Existing Gasperini-aligned embeddings (11,018 genes)
PROTEIN_EMB = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings_gasperini_aligned.h5"
RNA_EMB = DATA_DIR / "processed/embeddings/k562_gene_embeddings_gasperini_aligned.h5"
INDEX_MAPPING = DATA_DIR / "processed/embeddings/protein_index_mapping_gasperini_aligned.npz"

# Original ProteinLM (for old_to_new mapping)
ORIGINAL_PROTEOMELM = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings.h5"

# Output directory
OUTPUT_DIR = DATA_DIR / "processed/embeddings"


def load_ensg_to_symbol():
    """ENSG → gene symbol mapping from Morris data"""
    print("Loading ENSG → symbol mapping...")
    with h5py.File(MORRIS_DATA, 'r') as f:
        gene_ids = [g.decode() if isinstance(g, bytes) else g for g in f['gene_ids'][:]]
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
    return {ensg: symbol for ensg, symbol in zip(gene_ids, gene_names)}


def compute_pseudobulk_expression(ensg_to_symbol: dict):
    """Gasperini scRNA-seqからpseudo-bulk発現量を計算"""
    print("\nComputing pseudo-bulk expression from Gasperini scRNA-seq...")
    print(f"  Loading {GASPERINI_EXPRS} (this takes a few minutes)...")

    # Load gene names (ENSG IDs)
    with gzip.open(GASPERINI_GENES, 'rt') as f:
        ensg_ids = [line.strip() for line in f]
    print(f"  Total genes in matrix: {len(ensg_ids)}")

    # Load expression matrix
    expression_matrix = mmread(GASPERINI_EXPRS).T.tocsr()  # (cells, genes)
    n_cells, n_genes = expression_matrix.shape
    print(f"  Matrix shape: {n_cells} cells × {n_genes} genes")

    # Compute pseudo-bulk (mean across cells)
    print("  Computing mean expression...")
    pseudobulk = np.array(expression_matrix.mean(axis=0)).flatten()

    # Map ENSG to symbols and filter
    gene_expression = {}
    for i, ensg in enumerate(ensg_ids):
        if ensg in ensg_to_symbol:
            symbol = ensg_to_symbol[ensg]
            gene_expression[symbol] = pseudobulk[i]

    print(f"  Genes with symbol mapping: {len(gene_expression)}")
    return gene_expression


def filter_embeddings(gene_expression: dict, n_genes: int):
    """Top N発現遺伝子で埋め込みをフィルタリング"""
    print(f"\nFiltering to top {n_genes} expressed genes...")

    # Load current Gasperini-aligned gene names
    with h5py.File(PROTEIN_EMB, 'r') as f:
        current_genes = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
        protein_emb = f['embeddings'][:]

    with h5py.File(RNA_EMB, 'r') as f:
        rna_emb = f['embeddings'][:]

    print(f"  Current gene count: {len(current_genes)}")

    # Get expression values for current genes
    gene_expr_list = []
    for gene in current_genes:
        expr = gene_expression.get(gene, 0)
        gene_expr_list.append((gene, expr))

    # Sort by expression (descending) and take top N
    gene_expr_list.sort(key=lambda x: x[1], reverse=True)
    top_genes = [g for g, e in gene_expr_list[:n_genes]]
    top_genes_set = set(top_genes)

    print(f"  Top {n_genes} genes selected")
    print(f"  Expression range: {gene_expr_list[0][1]:.4f} (max) to {gene_expr_list[n_genes-1][1]:.4f} (min at cutoff)")

    # Filter embeddings
    filtered_indices = [i for i, g in enumerate(current_genes) if g in top_genes_set]

    # Maintain order by expression
    gene_to_new_idx = {g: i for i, g in enumerate(top_genes)}
    filtered_indices_ordered = []
    for gene in top_genes:
        old_idx = current_genes.index(gene)
        filtered_indices_ordered.append(old_idx)

    filtered_protein_emb = protein_emb[filtered_indices_ordered]
    filtered_rna_emb = rna_emb[filtered_indices_ordered]

    print(f"  Filtered protein embeddings: {filtered_protein_emb.shape}")
    print(f"  Filtered RNA embeddings: {filtered_rna_emb.shape}")

    return top_genes, filtered_protein_emb, filtered_rna_emb


def create_index_mapping(top_genes: list):
    """新しいインデックスマッピングを作成 (v3.3.1形式)"""
    print("\nCreating index mapping...")

    # Load original ProteinLM gene names
    with h5py.File(ORIGINAL_PROTEOMELM, 'r') as f:
        orig_genes = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]
    orig_gene_to_idx = {g: i for i, g in enumerate(orig_genes)}

    # Create old_to_new mapping
    old_to_new = []
    for new_idx, gene in enumerate(top_genes):
        if gene in orig_gene_to_idx:
            old_idx = orig_gene_to_idx[gene]
            old_to_new.append([old_idx, new_idx])

    old_to_new = np.array(old_to_new, dtype=np.int64)
    old_indices = old_to_new[:, 0]

    print(f"  Mappings: {len(old_to_new)}")
    return old_to_new, old_indices


def save_filtered_data(top_genes, protein_emb, rna_emb, old_to_new, old_indices, n_genes):
    """フィルタリングしたデータを保存"""
    suffix = f"_top{n_genes}"

    # Protein embeddings
    protein_path = OUTPUT_DIR / f"human_proteomelm_embeddings_gasperini_aligned{suffix}.h5"
    print(f"\nSaving protein embeddings to {protein_path}")
    with h5py.File(protein_path, 'w') as f:
        f.create_dataset('embeddings', data=protein_emb.astype(np.float32), compression='gzip')
        f.create_dataset('gene_names', data=np.array(top_genes, dtype='S'))
        f.attrs['n_genes'] = len(top_genes)
        f.attrs['embedding_dim'] = protein_emb.shape[1]
        f.attrs['description'] = f'Top {n_genes} expressed genes from Gasperini pseudo-bulk'

    # RNA embeddings
    rna_path = OUTPUT_DIR / f"k562_gene_embeddings_gasperini_aligned{suffix}.h5"
    print(f"Saving RNA embeddings to {rna_path}")
    with h5py.File(rna_path, 'w') as f:
        f.create_dataset('embeddings', data=rna_emb.astype(np.float32), compression='gzip')
        f.create_dataset('gene_names', data=np.array(top_genes, dtype='S'))
        f.attrs['n_genes'] = len(top_genes)
        f.attrs['embedding_dim'] = rna_emb.shape[1]
        f.attrs['description'] = f'Top {n_genes} expressed genes, context-dependent scGPT'

    # Index mapping
    mapping_path = OUTPUT_DIR / f"protein_index_mapping_gasperini_aligned{suffix}.npz"
    print(f"Saving index mapping to {mapping_path}")
    np.savez(mapping_path, old_to_new=old_to_new, old_indices=old_indices)

    return protein_path, rna_path, mapping_path


def main():
    parser = argparse.ArgumentParser(description="Filter to top N expressed genes")
    parser.add_argument("--n-genes", type=int, default=5000,
                        help="Number of top expressed genes to keep (default: 5000)")
    args = parser.parse_args()

    n_genes = args.n_genes

    print("=" * 60)
    print(f"Filtering to Top {n_genes} Expressed Genes")
    print("=" * 60)

    # Load ENSG mapping
    ensg_to_symbol = load_ensg_to_symbol()

    # Compute pseudo-bulk expression
    gene_expression = compute_pseudobulk_expression(ensg_to_symbol)

    # Filter embeddings
    top_genes, protein_emb, rna_emb = filter_embeddings(gene_expression, n_genes)

    # Create index mapping
    old_to_new, old_indices = create_index_mapping(top_genes)

    # Save
    protein_path, rna_path, mapping_path = save_filtered_data(
        top_genes, protein_emb, rna_emb, old_to_new, old_indices, n_genes
    )

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Genes: {len(top_genes)} (filtered from 11,018)")
    print(f"  Protein embeddings: {protein_path}")
    print(f"  RNA embeddings: {rna_path}")
    print(f"  Index mapping: {mapping_path}")
    print()
    print("Memory estimate for Self-Attention:")
    attn_memory = 8 * 8 * n_genes * n_genes * 4 / 1e9
    print(f"  [batch=8, heads=8, {n_genes}, {n_genes}] × 4 bytes = {attn_memory:.1f} GB")
    print()
    print("Next steps:")
    print("  1. Upload these files to Google Drive cdt_data/")
    print("  2. Update Colab notebook to use *_top5000.h5 files")


if __name__ == "__main__":
    main()
