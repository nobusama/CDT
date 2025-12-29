#!/usr/bin/env python3
"""
UniProt ID ↔ Ensembl ID マッピング作成スクリプト

ESM-2埋め込みのUniProt IDとEnformer埋め込みのEnsembl IDを
遺伝子シンボルを介してマッピングする。

Usage:
    python scripts/create_id_mapping.py
"""

import h5py
import pandas as pd
from pathlib import Path

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
ESM2_PATH = PROJECT_ROOT / "data/processed/embeddings/human_esm2_embeddings.h5"
ENFORMER_TSV = PROJECT_ROOT / "data/raw/enformer/preprocessing/tss_queries/query_gencode_v41_protein_coding_canonical_tss_hg38_nostitch.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/id_mapping.csv"


def load_esm2_ids():
    """ESM-2埋め込みからUniProt IDと遺伝子名を取得"""
    print("ESM-2データを読み込み中...")

    with h5py.File(ESM2_PATH, 'r') as f:
        uniprot_ids = f['uniprot_ids'][:]
        gene_names = f['gene_names'][:]

    # bytes → str
    uniprot_ids = [u.decode('utf-8') if isinstance(u, bytes) else u for u in uniprot_ids]
    gene_names = [g.decode('utf-8') if isinstance(g, bytes) else g for g in gene_names]

    esm2_df = pd.DataFrame({
        'uniprot_id': uniprot_ids,
        'gene_symbol_esm2': gene_names
    })

    # インデックス（ESM-2埋め込み配列の位置）を保持
    esm2_df['esm2_idx'] = esm2_df.index

    print(f"  ESM-2エントリ数: {len(esm2_df)}")
    print(f"  遺伝子名あり: {(esm2_df['gene_symbol_esm2'] != '').sum()}")

    return esm2_df


def load_enformer_ids():
    """Enformer TSSデータからEnsembl IDと遺伝子名を取得"""
    print("Enformerデータを読み込み中...")

    enformer_df = pd.read_csv(ENFORMER_TSV, sep='\t')

    # 必要なカラムのみ抽出
    enformer_df = enformer_df[['group_id', 'add_id']].copy()
    enformer_df.columns = ['ensembl_id_versioned', 'gene_symbol_enformer']

    # Ensembl IDからバージョンを除去
    enformer_df['ensembl_id'] = enformer_df['ensembl_id_versioned'].str.split('.').str[0]

    # インデックス（Enformer埋め込み配列の位置）を保持
    enformer_df['enformer_idx'] = enformer_df.index

    print(f"  Enformerエントリ数: {len(enformer_df)}")

    return enformer_df


def create_mapping(esm2_df, enformer_df):
    """遺伝子シンボルを介してマッピングを作成"""
    print("\nマッピングを作成中...")

    # 遺伝子シンボルを大文字に統一（ケース違い対策）
    esm2_df['gene_symbol_upper'] = esm2_df['gene_symbol_esm2'].str.upper()
    enformer_df['gene_symbol_upper'] = enformer_df['gene_symbol_enformer'].str.upper()

    # マージ（内部結合：両方にある遺伝子のみ）
    mapping_df = pd.merge(
        esm2_df,
        enformer_df,
        on='gene_symbol_upper',
        how='inner'
    )

    # 重複チェック（1つのUniProt IDに複数のEnsembl IDがマッピングされる場合）
    dup_uniprot = mapping_df.groupby('uniprot_id').size()
    dup_count = (dup_uniprot > 1).sum()

    print(f"  マッチした遺伝子: {len(mapping_df)}")
    print(f"  ユニークUniProt ID: {mapping_df['uniprot_id'].nunique()}")
    print(f"  ユニークEnsembl ID: {mapping_df['ensembl_id'].nunique()}")
    print(f"  重複あり（1 UniProt → 複数 Ensembl）: {dup_count}")

    # 必要なカラムのみ選択
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
    """カバレッジを分析"""
    print("\n" + "=" * 50)
    print("カバレッジ分析")
    print("=" * 50)

    # ESM-2側のカバレッジ
    esm2_matched = mapping_df['uniprot_id'].nunique()
    esm2_total = len(esm2_df)
    print(f"\nESM-2 → Enformer:")
    print(f"  マッチ: {esm2_matched} / {esm2_total} ({esm2_matched/esm2_total*100:.1f}%)")

    # Enformer側のカバレッジ
    enformer_matched = mapping_df['ensembl_id'].nunique()
    enformer_total = len(enformer_df)
    print(f"\nEnformer → ESM-2:")
    print(f"  マッチ: {enformer_matched} / {enformer_total} ({enformer_matched/enformer_total*100:.1f}%)")

    # マッチしなかった遺伝子の例
    esm2_symbols = set(esm2_df['gene_symbol_upper'])
    enformer_symbols = set(enformer_df['gene_symbol_upper'])

    only_esm2 = esm2_symbols - enformer_symbols
    only_enformer = enformer_symbols - esm2_symbols

    print(f"\nESM-2のみ（Enformerになし）: {len(only_esm2)}")
    print(f"  例: {list(only_esm2)[:5]}")

    print(f"\nEnformerのみ（ESM-2になし）: {len(only_enformer)}")
    print(f"  例: {list(only_enformer)[:5]}")


def main():
    print("=" * 50)
    print("UniProt ↔ Ensembl ID マッピング作成")
    print("=" * 50)

    # データ読み込み
    esm2_df = load_esm2_ids()
    enformer_df = load_enformer_ids()

    # マッピング作成
    mapping_df = create_mapping(esm2_df, enformer_df)

    # カバレッジ分析
    analyze_coverage(mapping_df, esm2_df, enformer_df)

    # 保存
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 50)
    print(f"保存完了: {OUTPUT_PATH}")
    print(f"レコード数: {len(mapping_df)}")
    print("=" * 50)

    return mapping_df


if __name__ == "__main__":
    main()
