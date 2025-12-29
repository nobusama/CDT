#!/usr/bin/env python3
"""
UniProt ID to Ensembl ID Mapping Creation Script

Maps UniProt IDs from ESM-2 embeddings to Ensembl IDs from Enformer embeddings
via gene symbols.

Usage:
    python scripts/create_id_mapping.py
"""

import h5py
import pandas as pd
from pathlib import Path

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
ESM2_PATH = PROJECT_ROOT / "data/processed/embeddings/human_esm2_embeddings.h5"
ENFORMER_TSV = PROJECT_ROOT / "data/raw/enformer/preprocessing/tss_queries/query_gencode_v41_protein_coding_canonical_tss_hg38_nostitch.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/id_mapping.csv"


def load_esm2_ids():
    """Get UniProt IDs and gene names from ESM-2 embeddings"""
    print("Loading ESM-2 data...")

    with h5py.File(ESM2_PATH, 'r') as f:
        uniprot_ids = f['uniprot_ids'][:]
        gene_names = f['gene_names'][:]

    # bytes to str
    uniprot_ids = [u.decode('utf-8') if isinstance(u, bytes) else u for u in uniprot_ids]
    gene_names = [g.decode('utf-8') if isinstance(g, bytes) else g for g in gene_names]

    esm2_df = pd.DataFrame({
        'uniprot_id': uniprot_ids,
        'gene_symbol_esm2': gene_names
    })

    # Keep index (position in ESM-2 embedding array)
    esm2_df['esm2_idx'] = esm2_df.index

    print(f"  ESM-2 entries: {len(esm2_df)}")
    print(f"  With gene names: {(esm2_df['gene_symbol_esm2'] != '').sum()}")

    return esm2_df


def load_enformer_ids():
    """Get Ensembl IDs and gene names from Enformer TSS data"""
    print("Loading Enformer data...")

    enformer_df = pd.read_csv(ENFORMER_TSV, sep='\t')

    # Extract only necessary columns
    enformer_df = enformer_df[['group_id', 'add_id']].copy()
    enformer_df.columns = ['ensembl_id_versioned', 'gene_symbol_enformer']

    # Remove version from Ensembl ID
    enformer_df['ensembl_id'] = enformer_df['ensembl_id_versioned'].str.split('.').str[0]

    # Keep index (position in Enformer embedding array)
    enformer_df['enformer_idx'] = enformer_df.index

    print(f"  Enformer entries: {len(enformer_df)}")

    return enformer_df


def create_mapping(esm2_df, enformer_df):
    """Create mapping via gene symbols"""
    print("\nCreating mapping...")

    # Unify gene symbols to uppercase (handle case differences)
    esm2_df['gene_symbol_upper'] = esm2_df['gene_symbol_esm2'].str.upper()
    enformer_df['gene_symbol_upper'] = enformer_df['gene_symbol_enformer'].str.upper()

    # Merge (inner join: only genes present in both)
    mapping_df = pd.merge(
        esm2_df,
        enformer_df,
        on='gene_symbol_upper',
        how='inner'
    )

    # Check duplicates (cases where one UniProt ID maps to multiple Ensembl IDs)
    dup_uniprot = mapping_df.groupby('uniprot_id').size()
    dup_count = (dup_uniprot > 1).sum()

    print(f"  Matched genes: {len(mapping_df)}")
    print(f"  Unique UniProt IDs: {mapping_df['uniprot_id'].nunique()}")
    print(f"  Unique Ensembl IDs: {mapping_df['ensembl_id'].nunique()}")
    print(f"  Duplicates (1 UniProt to multiple Ensembl): {dup_count}")

    # Select only necessary columns
    result_df = mapping_df[[
        'uniprot_id',
        'ensembl_id',
        'ensembl_id_versioned',
        'gene_symbol_esm2',
        'gene_symbol_enformer',
        'esm2_idx',
        'enformer_idx'
    ]].copy()

    return result_df


def analyze_coverage(mapping_df, esm2_df, enformer_df):
    """Analyze coverage"""
    print("\n" + "=" * 50)
    print("Coverage Analysis")
    print("=" * 50)

    # ESM-2 side coverage
    esm2_matched = mapping_df['uniprot_id'].nunique()
    esm2_total = len(esm2_df)
    print(f"\nESM-2 to Enformer:")
    print(f"  Matched: {esm2_matched} / {esm2_total} ({esm2_matched/esm2_total*100:.1f}%)")

    # Enformer side coverage
    enformer_matched = mapping_df['ensembl_id'].nunique()
    enformer_total = len(enformer_df)
    print(f"\nEnformer to ESM-2:")
    print(f"  Matched: {enformer_matched} / {enformer_total} ({enformer_matched/enformer_total*100:.1f}%)")

    # Examples of unmatched genes
    esm2_symbols = set(esm2_df['gene_symbol_upper'])
    enformer_symbols = set(enformer_df['gene_symbol_upper'])

    only_esm2 = esm2_symbols - enformer_symbols
    only_enformer = enformer_symbols - esm2_symbols

    print(f"\nESM-2 only (not in Enformer): {len(only_esm2)}")
    print(f"  Examples: {list(only_esm2)[:5]}")

    print(f"\nEnformer only (not in ESM-2): {len(only_enformer)}")
    print(f"  Examples: {list(only_enformer)[:5]}")


def main():
    print("=" * 50)
    print("UniProt to Ensembl ID Mapping Creation")
    print("=" * 50)

    # Load data
    esm2_df = load_esm2_ids()
    enformer_df = load_enformer_ids()

    # Create mapping
    mapping_df = create_mapping(esm2_df, enformer_df)

    # Analyze coverage
    analyze_coverage(mapping_df, esm2_df, enformer_df)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 50)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Records: {len(mapping_df)}")
    print("=" * 50)

    return mapping_df


if __name__ == "__main__":
    main()
