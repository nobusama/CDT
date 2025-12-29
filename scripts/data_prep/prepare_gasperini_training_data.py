#!/usr/bin/env python3
"""
Gasperini Training Data Preparation Script

Preprocesses Gasperini CRISPRi screen data for training,
performs Train/Val/Test split and saves in HDF5 format.

Usage:
    python scripts/prepare_gasperini_training_data.py
"""

import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent
GASPERINI_PATH = PROJECT_ROOT / "data/raw/gasperini/GSE120861_all_deg_results.at_scale.txt.gz"
ID_MAPPING_PATH = PROJECT_ROOT / "data/processed/id_mapping.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/training"

def load_and_filter_data():
    """Load Gasperini data and extract DHS pairs only"""
    print("Loading Gasperini data...")
    df = pd.read_csv(GASPERINI_PATH, sep='\t', low_memory=False)
    print(f"  Total rows: {len(df)}")

    # Extract DHS only
    dhs_df = df[df['site_type'] == 'DHS'].copy()
    print(f"  DHS pairs: {len(dhs_df)}")

    # Convert p-value to numeric (some may be strings)
    p_col = 'pvalue.empirical.adjusted'
    dhs_df[p_col] = pd.to_numeric(dhs_df[p_col], errors='coerce')

    # Only pairs with valid p-values
    valid_df = dhs_df[dhs_df[p_col].notna()].copy()
    print(f"  Valid pairs (non-NA p-value): {len(valid_df)}")

    # Create labels
    valid_df['label'] = (valid_df[p_col] < 0.05).astype(int)

    return valid_df

def apply_id_mapping(df):
    """Apply ID mapping to add embedding indices"""
    print("\nApplying ID mapping...")
    id_map = pd.read_csv(ID_MAPPING_PATH)
    print(f"  ID mapping entries: {len(id_map)}")

    # Merge on ensembl_id
    merged = df.merge(
        id_map[['ensembl_id', 'enformer_idx', 'esm2_idx']],
        left_on='ENSG',
        right_on='ensembl_id',
        how='inner'
    )
    print(f"  Mapped pairs: {len(merged)}")

    # Remove NaN indices
    merged = merged.dropna(subset=['enformer_idx', 'esm2_idx'])
    merged['enformer_idx'] = merged['enformer_idx'].astype(int)
    merged['esm2_idx'] = merged['esm2_idx'].astype(int)

    print(f"  Final pairs: {len(merged)}")
    print(f"  Positive: {merged['label'].sum()}")
    print(f"  Negative: {len(merged) - merged['label'].sum()}")

    return merged

def stratified_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """Stratified sampling for Train/Val/Test split"""
    print(f"\nSplitting data ({train_ratio}/{val_ratio}/{test_ratio})...")

    # First split into Train vs (Val+Test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=random_state
    )

    # Then split into Val vs Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_df['label'],
        random_state=random_state
    )

    print(f"  Train: {len(train_df)} (Pos: {train_df['label'].sum()})")
    print(f"  Val:   {len(val_df)} (Pos: {val_df['label'].sum()})")
    print(f"  Test:  {len(test_df)} (Pos: {test_df['label'].sum()})")

    return train_df, val_df, test_df

def save_to_hdf5(df, output_path, name):
    """Save DataFrame to HDF5 format"""
    print(f"\nSaving {name} to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # Embedding indices
        f.create_dataset('enformer_idx', data=df['enformer_idx'].values, dtype='int32')
        f.create_dataset('esm2_idx', data=df['esm2_idx'].values, dtype='int32')

        # Labels
        f.create_dataset('labels', data=df['label'].values, dtype='int32')

        # Metadata (gene IDs, enhancer coordinates, etc.)
        f.create_dataset('gene_ids', data=df['ENSG'].values.astype('S'))
        f.create_dataset('enhancer_chr', data=df['target_site.chr'].astype(str).values.astype('S'))

        # Convert coordinates to numeric (may be mixed type)
        enhancer_start = pd.to_numeric(df['target_site.start'], errors='coerce').fillna(0).astype('int32')
        enhancer_end = pd.to_numeric(df['target_site.stop'], errors='coerce').fillna(0).astype('int32')
        f.create_dataset('enhancer_start', data=enhancer_start.values, dtype='int32')
        f.create_dataset('enhancer_end', data=enhancer_end.values, dtype='int32')

        # Effect size (for reference)
        beta = pd.to_numeric(df['beta'], errors='coerce').fillna(0).astype('float32')
        f.create_dataset('beta', data=beta.values, dtype='float32')

        # Attributes
        f.attrs['n_samples'] = len(df)
        f.attrs['n_positive'] = int(df['label'].sum())
        f.attrs['n_negative'] = int(len(df) - df['label'].sum())
        f.attrs['source'] = 'Gasperini et al. 2019'

    print(f"  Saved {len(df)} samples")

def main():
    print("=" * 60)
    print("Gasperini Training Data Preparation")
    print("=" * 60)

    # Load and filter data
    df = load_and_filter_data()

    # Apply ID mapping
    mapped_df = apply_id_mapping(df)

    # Train/Val/Test split
    train_df, val_df, test_df = stratified_split(mapped_df)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save in HDF5 format
    save_to_hdf5(train_df, OUTPUT_DIR / "gasperini_train.h5", "train")
    save_to_hdf5(val_df, OUTPUT_DIR / "gasperini_val.h5", "val")
    save_to_hdf5(test_df, OUTPUT_DIR / "gasperini_test.h5", "test")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    # Summary
    print(f"\nOutput files:")
    print(f"  {OUTPUT_DIR / 'gasperini_train.h5'}")
    print(f"  {OUTPUT_DIR / 'gasperini_val.h5'}")
    print(f"  {OUTPUT_DIR / 'gasperini_test.h5'}")

if __name__ == "__main__":
    main()
