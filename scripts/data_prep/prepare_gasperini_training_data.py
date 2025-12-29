#!/usr/bin/env python3
"""
Gasperini学習データ作成スクリプト

GasperiniのCRISPRi screen データを学習用に前処理し、
Train/Val/Test分割してHDF5形式で保存する。

Usage:
    python scripts/prepare_gasperini_training_data.py
"""

import os
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from sklearn.model_selection import train_test_split

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
GASPERINI_PATH = PROJECT_ROOT / "data/raw/gasperini/GSE120861_all_deg_results.at_scale.txt.gz"
ID_MAPPING_PATH = PROJECT_ROOT / "data/processed/id_mapping.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/training"

def load_and_filter_data():
    """Gasperiniデータを読み込み、DHSペアのみを抽出"""
    print("Loading Gasperini data...")
    df = pd.read_csv(GASPERINI_PATH, sep='\t', low_memory=False)
    print(f"  Total rows: {len(df)}")

    # DHSのみ抽出
    dhs_df = df[df['site_type'] == 'DHS'].copy()
    print(f"  DHS pairs: {len(dhs_df)}")

    # p値を数値に変換（一部が文字列の可能性）
    p_col = 'pvalue.empirical.adjusted'
    dhs_df[p_col] = pd.to_numeric(dhs_df[p_col], errors='coerce')

    # 有効なp値を持つペアのみ
    valid_df = dhs_df[dhs_df[p_col].notna()].copy()
    print(f"  Valid pairs (non-NA p-value): {len(valid_df)}")

    # ラベル作成
    valid_df['label'] = (valid_df[p_col] < 0.05).astype(int)

    return valid_df

def apply_id_mapping(df):
    """IDマッピングを適用して、埋め込みインデックスを追加"""
    print("\nApplying ID mapping...")
    id_map = pd.read_csv(ID_MAPPING_PATH)
    print(f"  ID mapping entries: {len(id_map)}")

    # ensembl_id でマージ
    merged = df.merge(
        id_map[['ensembl_id', 'enformer_idx', 'esm2_idx']],
        left_on='ENSG',
        right_on='ensembl_id',
        how='inner'
    )
    print(f"  Mapped pairs: {len(merged)}")

    # NaN インデックスを除外
    merged = merged.dropna(subset=['enformer_idx', 'esm2_idx'])
    merged['enformer_idx'] = merged['enformer_idx'].astype(int)
    merged['esm2_idx'] = merged['esm2_idx'].astype(int)

    print(f"  Final pairs: {len(merged)}")
    print(f"  Positive: {merged['label'].sum()}")
    print(f"  Negative: {len(merged) - merged['label'].sum()}")

    return merged

def stratified_split(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    """層化サンプリングでTrain/Val/Test分割"""
    print(f"\nSplitting data ({train_ratio}/{val_ratio}/{test_ratio})...")

    # まずTrain vs (Val+Test) に分割
    train_df, temp_df = train_test_split(
        df,
        test_size=(val_ratio + test_ratio),
        stratify=df['label'],
        random_state=random_state
    )

    # 次にVal vs Test に分割
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
    """DataFrameをHDF5形式で保存"""
    print(f"\nSaving {name} to {output_path}...")

    with h5py.File(output_path, 'w') as f:
        # 埋め込みインデックス
        f.create_dataset('enformer_idx', data=df['enformer_idx'].values, dtype='int32')
        f.create_dataset('esm2_idx', data=df['esm2_idx'].values, dtype='int32')

        # ラベル
        f.create_dataset('labels', data=df['label'].values, dtype='int32')

        # メタデータ（遺伝子ID、エンハンサー座標など）
        f.create_dataset('gene_ids', data=df['ENSG'].values.astype('S'))
        f.create_dataset('enhancer_chr', data=df['target_site.chr'].astype(str).values.astype('S'))

        # 座標は数値に変換（混合型の可能性があるため）
        enhancer_start = pd.to_numeric(df['target_site.start'], errors='coerce').fillna(0).astype('int32')
        enhancer_end = pd.to_numeric(df['target_site.stop'], errors='coerce').fillna(0).astype('int32')
        f.create_dataset('enhancer_start', data=enhancer_start.values, dtype='int32')
        f.create_dataset('enhancer_end', data=enhancer_end.values, dtype='int32')

        # 効果サイズ（参考用）
        beta = pd.to_numeric(df['beta'], errors='coerce').fillna(0).astype('float32')
        f.create_dataset('beta', data=beta.values, dtype='float32')

        # 属性
        f.attrs['n_samples'] = len(df)
        f.attrs['n_positive'] = int(df['label'].sum())
        f.attrs['n_negative'] = int(len(df) - df['label'].sum())
        f.attrs['source'] = 'Gasperini et al. 2019'

    print(f"  Saved {len(df)} samples")

def main():
    print("=" * 60)
    print("Gasperini Training Data Preparation")
    print("=" * 60)

    # データ読み込みとフィルタリング
    df = load_and_filter_data()

    # IDマッピング適用
    mapped_df = apply_id_mapping(df)

    # Train/Val/Test分割
    train_df, val_df, test_df = stratified_split(mapped_df)

    # 出力ディレクトリ作成
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # HDF5形式で保存
    save_to_hdf5(train_df, OUTPUT_DIR / "gasperini_train.h5", "train")
    save_to_hdf5(val_df, OUTPUT_DIR / "gasperini_val.h5", "val")
    save_to_hdf5(test_df, OUTPUT_DIR / "gasperini_test.h5", "test")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

    # サマリー
    print(f"\nOutput files:")
    print(f"  {OUTPUT_DIR / 'gasperini_train.h5'}")
    print(f"  {OUTPUT_DIR / 'gasperini_val.h5'}")
    print(f"  {OUTPUT_DIR / 'gasperini_test.h5'}")

if __name__ == "__main__":
    main()
