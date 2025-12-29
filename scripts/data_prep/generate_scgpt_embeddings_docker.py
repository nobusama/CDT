#!/usr/bin/env python3
"""
scGPT埋め込み生成スクリプト（Docker用）

Usage (inside Docker):
    python /workspace/scripts/generate_scgpt_embeddings_docker.py
"""

import os
import sys
import numpy as np
import scanpy as sc
import torch
import h5py
from pathlib import Path

# scGPTのインポート
import scgpt
from scgpt.tasks import embed_data

print(f"scGPT version: {scgpt.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

# パス設定
WORKSPACE = Path("/workspace")
DATA_DIR = WORKSPACE / "data/processed/morris"
OUTPUT_DIR = WORKSPACE / "data/processed/embeddings"
MODEL_DIR = WORKSPACE / "scgpt_model"

def download_model():
    """scGPTモデルをダウンロード"""
    import gdown

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # whole-human model
    url = 'https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y'

    if not (MODEL_DIR / 'best_model.pt').exists():
        print("Downloading scGPT model...")
        gdown.download_folder(url, output=str(MODEL_DIR), quiet=False)
    else:
        print("Model already downloaded")

def generate_embeddings(adata_path, output_name):
    """埋め込みを生成"""
    print(f"\nProcessing: {adata_path}")

    # データ読み込み
    adata = sc.read_h5ad(adata_path)
    print(f"  Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # 遺伝子名カラムを確認
    if 'gene_name' in adata.var.columns:
        gene_col = 'gene_name'
    else:
        adata.var['gene_name'] = adata.var_names
        gene_col = 'gene_name'

    # 埋め込み生成
    print("  Generating embeddings...")
    adata_embed = embed_data(
        adata,
        str(MODEL_DIR),
        gene_col=gene_col,
        batch_size=64,
    )

    # 埋め込み抽出
    embeddings = adata_embed.X
    if hasattr(embeddings, 'toarray'):
        embeddings = embeddings.toarray()

    print(f"  Embedding shape: {embeddings.shape}")

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_name

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings, compression='gzip')
        f.create_dataset('cell_barcodes', data=np.array(adata.obs.index, dtype='S'))
        f.attrs['n_cells'] = embeddings.shape[0]
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['model'] = 'scGPT-whole-human'
        f.attrs['source'] = str(adata_path)

    print(f"  Saved: {output_path}")
    return embeddings.shape

def main():
    # モデルダウンロード
    download_model()

    # v1の処理
    v1_path = DATA_DIR / "stingseq_v1.h5ad"
    if v1_path.exists():
        shape = generate_embeddings(v1_path, "scgpt_embeddings_v1.h5")
        print(f"\nv1 complete: {shape[0]} cells × {shape[1]} dims")
    else:
        print(f"v1 not found: {v1_path}")

    # v2の処理（オプション - 時間がかかる）
    v2_path = DATA_DIR / "stingseq_v2.h5ad"
    if v2_path.exists() and os.environ.get('PROCESS_V2', '0') == '1':
        shape = generate_embeddings(v2_path, "scgpt_embeddings_v2.h5")
        print(f"\nv2 complete: {shape[0]} cells × {shape[1]} dims")

    print("\nDone!")

if __name__ == "__main__":
    main()
