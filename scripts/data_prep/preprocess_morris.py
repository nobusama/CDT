#!/usr/bin/env python3
"""
Morris STING-seq Data Preprocessing Script

Converts 10x Genomics format data to AnnData format and
performs preprocessing for CDT.

Usage:
    python scripts/preprocess_morris.py
"""

import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread
from pathlib import Path
import h5py
import anndata as ad

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data/raw/morris/extracted"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/morris"


def read_10x_mtx(matrix_path, features_path, barcodes_path):
    """Read 10x Genomics format data"""
    print(f"  Reading matrix: {matrix_path.name}")

    # Read Matrix Market format
    with gzip.open(matrix_path, 'rb') as f:
        matrix = mmread(f).T.tocsr()  # Transpose to cells x genes

    # Features (genes)
    with gzip.open(features_path, 'rt') as f:
        features = pd.read_csv(f, sep='\t', header=None,
                               names=['gene_id', 'gene_name', 'feature_type'])

    # Barcodes (cells)
    with gzip.open(barcodes_path, 'rt') as f:
        barcodes = pd.read_csv(f, sep='\t', header=None, names=['barcode'])

    # Dimension validation
    n_cells, n_genes = matrix.shape
    if n_cells != len(barcodes):
        raise ValueError(f"Matrix has {n_cells} cells but barcodes has {len(barcodes)} entries")
    if n_genes != len(features):
        raise ValueError(f"Matrix has {n_genes} genes but features has {len(features)} entries")

    return matrix, features, barcodes


def create_anndata(matrix, features, barcodes, batch_name):
    """Create AnnData object"""

    # Extract gene expression only (exclude CRISPR Guide Capture)
    gene_mask = features['feature_type'] == 'Gene Expression'
    gene_features = features[gene_mask].reset_index(drop=True)
    gene_matrix = matrix[:, gene_mask.values]

    # Create AnnData
    adata = ad.AnnData(
        X=gene_matrix,
        obs=pd.DataFrame(index=barcodes['barcode'].values),
        var=pd.DataFrame(
            {'gene_id': gene_features['gene_id'].values,
             'gene_name': gene_features['gene_name'].values},
            index=gene_features['gene_id'].values
        )
    )

    adata.obs['batch'] = batch_name

    # CRISPR guides information
    guide_mask = features['feature_type'] == 'CRISPR Guide Capture'
    if guide_mask.any():
        guide_features = features[guide_mask].reset_index(drop=True)
        guide_matrix = matrix[:, guide_mask.values]
        adata.obsm['guide_counts'] = guide_matrix.toarray()
        adata.uns['guide_names'] = guide_features['gene_name'].values

    return adata


def process_v1():
    """Process STING-seq v1"""
    print("\n" + "="*50)
    print("Processing STING-seq v1")
    print("="*50)

    # cDNA data
    matrix, features, barcodes = read_10x_mtx(
        RAW_DIR / "GSM5225857_STINGseq-v1_cDNA.matrix.mtx.gz",
        RAW_DIR / "GSM5225857_STINGseq-v1_cDNA.features.tsv.gz",
        RAW_DIR / "GSM5225857_STINGseq-v1_cDNA.barcodes.tsv.gz"
    )

    adata = create_anndata(matrix, features, barcodes, 'v1')

    # GDO (guide) data - v1 has guide information in GDO file
    _, gdo_features, _ = read_10x_mtx(
        RAW_DIR / "GSM5225859_STINGseq-v1_GDO.matrix.mtx.gz",
        RAW_DIR / "GSM5225859_STINGseq-v1_GDO.features.tsv.gz",
        RAW_DIR / "GSM5225859_STINGseq-v1_GDO.barcodes.tsv.gz"
    )

    guide_mask = gdo_features['feature_type'] == 'CRISPR Guide Capture'
    if guide_mask.any():
        adata.uns['guide_names'] = gdo_features[guide_mask]['gene_name'].values

    print(f"  Cells: {adata.n_obs}")
    print(f"  Genes: {adata.n_vars}")
    print(f"  Guides: {len(adata.uns.get('guide_names', []))}")

    return adata


def find_file(pattern, fallback_patterns=None):
    """Find file (supports multiple patterns)"""
    files = list(RAW_DIR.glob(pattern))
    if files:
        return files[0]
    if fallback_patterns:
        for fp in fallback_patterns:
            files = list(RAW_DIR.glob(fp))
            if files:
                return files[0]
    raise FileNotFoundError(f"No file matching {pattern}")


def process_v2_batch(batch_letter):
    """Process one batch of STING-seq v2"""
    print(f"\n  Processing v2 batch {batch_letter}...")

    # GSM prefix for each batch
    batch_gsm = {'A': '7108117', 'B': '7108118', 'C': '7108119', 'D': '7108120'}
    gsm = batch_gsm[batch_letter]

    # Find actual file names (supports multiple patterns)
    cDNA_matrix = find_file(
        f"*v2_cDNA-{batch_letter}_matrix*",
        [f"GSM{gsm}_matrix*", f"GSM{gsm}*matrix*"]
    )
    cDNA_features = find_file(
        f"*v2_cDNA-{batch_letter}_features*",
        [f"GSM{gsm}_features*", f"GSM{gsm}*features*"]
    )
    cDNA_barcodes = find_file(
        f"*v2_cDNA-{batch_letter}_barcodes*",
        [f"GSM{gsm}_barcodes*", f"GSM{gsm}*barcodes*"]
    )

    matrix, features, barcodes = read_10x_mtx(cDNA_matrix, cDNA_features, cDNA_barcodes)
    adata = create_anndata(matrix, features, barcodes, f'v2_{batch_letter}')

    # GDO data - read by matching barcodes
    gdo_files = list(RAW_DIR.glob(f"*v2_GDO-{batch_letter}_features*"))
    if gdo_files:
        gdo_matrix_f = list(RAW_DIR.glob(f"*v2_GDO-{batch_letter}_matrix*"))[0]
        gdo_features_f = gdo_files[0]
        gdo_barcodes_f = list(RAW_DIR.glob(f"*v2_GDO-{batch_letter}_barcodes*"))[0]

        gdo_matrix, gdo_features, gdo_barcodes = read_10x_mtx(gdo_matrix_f, gdo_features_f, gdo_barcodes_f)

        guide_mask = gdo_features['feature_type'] == 'CRISPR Guide Capture'
        if guide_mask.any():
            adata.uns['guide_names'] = gdo_features[guide_mask]['gene_name'].values

            # Match barcodes
            gdo_bc_set = set(gdo_barcodes['barcode'].values)
            cdna_bc_list = list(adata.obs.index)

            # Get index of barcodes that exist in GDO
            gdo_bc_to_idx = {bc: i for i, bc in enumerate(gdo_barcodes['barcode'].values)}

            # Assign guide counts for each cDNA cell
            n_guides = guide_mask.sum()
            guide_counts = np.zeros((len(cdna_bc_list), n_guides), dtype=np.float32)

            gdo_guide_matrix = gdo_matrix[:, guide_mask.values]
            matched = 0
            for i, bc in enumerate(cdna_bc_list):
                if bc in gdo_bc_to_idx:
                    gdo_idx = gdo_bc_to_idx[bc]
                    guide_counts[i, :] = gdo_guide_matrix[gdo_idx, :].toarray().flatten()
                    matched += 1

            adata.obsm['guide_counts'] = guide_counts
            print(f"    Matched {matched}/{len(cdna_bc_list)} cells with GDO data")

    print(f"    Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    return adata


def process_v2():
    """Process STING-seq v2 (integrate all batches)"""
    print("\n" + "="*50)
    print("Processing STING-seq v2")
    print("="*50)

    adatas = []
    for batch in ['A', 'B', 'C', 'D']:
        try:
            adata = process_v2_batch(batch)
            adatas.append(adata)
        except Exception as e:
            print(f"    Warning: Skipping batch {batch} due to error: {e}")

    if not adatas:
        raise ValueError("No batches could be processed")

    # Integrate
    print("\n  Concatenating batches...")
    adata_v2 = ad.concat(adatas, join='outer', label='batch_sub', index_unique='_')

    # Restore gene_name (may be lost during concat)
    if 'gene_name' not in adata_v2.var.columns and 'gene_name' in adatas[0].var.columns:
        # Copy gene_name from first batch
        gene_name_map = dict(zip(adatas[0].var.index, adatas[0].var['gene_name']))
        adata_v2.var['gene_name'] = [gene_name_map.get(g, g) for g in adata_v2.var.index]

    # Unify guide names
    all_guides = set()
    for a in adatas:
        if 'guide_names' in a.uns:
            all_guides.update(a.uns['guide_names'])
    adata_v2.uns['guide_names'] = np.array(sorted(all_guides))

    print(f"\n  Total v2:")
    print(f"    Cells: {adata_v2.n_obs}")
    print(f"    Genes: {adata_v2.n_vars}")
    print(f"    Unique guides: {len(adata_v2.uns['guide_names'])}")

    return adata_v2


def add_qc_metrics(adata):
    """Add QC metrics"""
    print("  Adding QC metrics...")

    # Basic QC metrics
    if sp.issparse(adata.X):
        adata.obs['n_genes'] = np.asarray((adata.X > 0).sum(axis=1)).flatten()
        adata.obs['n_counts'] = np.asarray(adata.X.sum(axis=1)).flatten()
    else:
        adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
        adata.obs['n_counts'] = adata.X.sum(axis=1)

    # Mitochondrial gene fraction
    if 'gene_name' in adata.var.columns:
        mt_genes = adata.var_names.str.startswith('MT-') | adata.var['gene_name'].str.startswith('MT-')
    else:
        mt_genes = adata.var_names.str.startswith('MT-') | adata.var_names.str.contains('-MT-')
    mt_mask = np.array(mt_genes)
    if mt_mask.any():
        mt_counts = adata.X[:, mt_mask].sum(axis=1)
        if sp.issparse(adata.X):
            mt_counts = np.asarray(mt_counts).flatten()
        adata.obs['pct_mt'] = mt_counts / adata.obs['n_counts'] * 100
    else:
        adata.obs['pct_mt'] = 0.0

    return adata


def filter_cells(adata, min_genes=200, max_genes=10000, max_pct_mt=20):
    """Filter cells"""
    print(f"  Filtering cells (min_genes={min_genes}, max_genes={max_genes}, max_pct_mt={max_pct_mt})...")

    n_before = adata.n_obs

    mask = (adata.obs['n_genes'] >= min_genes) & \
           (adata.obs['n_genes'] <= max_genes) & \
           (adata.obs['pct_mt'] <= max_pct_mt)

    adata = adata[mask].copy()

    print(f"    {n_before} -> {adata.n_obs} cells")

    return adata


def save_for_cdt(adata, output_path):
    """Save in HDF5 format for CDT"""
    print(f"  Saving to {output_path}...")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, 'w') as f:
        # Expression matrix (save as sparse matrix)
        if sp.issparse(adata.X):
            X = adata.X.tocsr()
            f.create_dataset('X_data', data=X.data, compression='gzip')
            f.create_dataset('X_indices', data=X.indices, compression='gzip')
            f.create_dataset('X_indptr', data=X.indptr, compression='gzip')
            f.attrs['X_shape'] = X.shape
            f.attrs['X_sparse'] = True
        else:
            f.create_dataset('X', data=adata.X, compression='gzip')
            f.attrs['X_sparse'] = False

        # Gene information
        f.create_dataset('gene_ids', data=np.array(adata.var.index, dtype='S'))
        if 'gene_name' in adata.var.columns:
            f.create_dataset('gene_names', data=np.array(adata.var['gene_name'], dtype='S'))
        else:
            # If no gene names from Ensembl ID, use ID as is
            f.create_dataset('gene_names', data=np.array(adata.var.index, dtype='S'))

        # Cell information
        f.create_dataset('cell_barcodes', data=np.array(adata.obs.index, dtype='S'))
        f.create_dataset('batch', data=np.array(adata.obs['batch'], dtype='S'))

        # QC metrics
        f.create_dataset('n_genes', data=adata.obs['n_genes'].values)
        f.create_dataset('n_counts', data=adata.obs['n_counts'].values)
        f.create_dataset('pct_mt', data=adata.obs['pct_mt'].values)

        # Guide information
        if 'guide_names' in adata.uns:
            f.create_dataset('guide_names', data=np.array(adata.uns['guide_names'], dtype='S'))
        if 'guide_counts' in adata.obsm:
            f.create_dataset('guide_counts', data=adata.obsm['guide_counts'], compression='gzip')

        # Metadata
        f.attrs['n_cells'] = adata.n_obs
        f.attrs['n_genes'] = adata.n_vars


def main():
    print("="*60)
    print("Morris STING-seq Data Preprocessing")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process v1
    try:
        adata_v1 = process_v1()
        adata_v1 = add_qc_metrics(adata_v1)
        adata_v1 = filter_cells(adata_v1)
        save_for_cdt(adata_v1, OUTPUT_DIR / "stingseq_v1.h5")

        # Also save in AnnData format (for compatibility)
        adata_v1.write_h5ad(OUTPUT_DIR / "stingseq_v1.h5ad")
        print(f"  v1 saved: {adata_v1.n_obs} cells, {adata_v1.n_vars} genes")
    except Exception as e:
        print(f"  Error processing v1: {e}")

    # Process v2
    try:
        adata_v2 = process_v2()
        adata_v2 = add_qc_metrics(adata_v2)
        adata_v2 = filter_cells(adata_v2)
        save_for_cdt(adata_v2, OUTPUT_DIR / "stingseq_v2.h5")

        # Also save in AnnData format
        adata_v2.write_h5ad(OUTPUT_DIR / "stingseq_v2.h5ad")
        print(f"  v2 saved: {adata_v2.n_obs} cells, {adata_v2.n_vars} genes")
    except Exception as e:
        print(f"  Error processing v2: {e}")

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


if __name__ == "__main__":
    main()
