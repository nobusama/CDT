#!/usr/bin/env python3
"""
K562平均scGPT埋め込み計算スクリプト

Morris K562細胞のscGPT埋め込みから平均を計算し、
Gasperini POCで共通の細胞埋め込みとして使用する。

Usage:
    python scripts/compute_k562_avg_embedding.py
"""

import numpy as np
import h5py
from pathlib import Path

# パス設定
PROJECT_ROOT = Path(__file__).parent.parent
SCGPT_PATH = PROJECT_ROOT / "data/processed/embeddings/scgpt_embeddings_v1.h5"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/embeddings/k562_avg_scgpt.npy"


def main():
    print("=" * 60)
    print("K562 Average scGPT Embedding Calculation")
    print("=" * 60)

    # scGPT埋め込み読み込み
    print(f"\nLoading scGPT embeddings from: {SCGPT_PATH}")

    if not SCGPT_PATH.exists():
        print(f"ERROR: File not found: {SCGPT_PATH}")
        print("Please run generate_scgpt_embeddings_local.py first.")
        return

    with h5py.File(SCGPT_PATH, 'r') as f:
        embeddings = f['embeddings'][:]
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dtype: {embeddings.dtype}")

        # 512次元であることを確認
        if embeddings.shape[1] != 512:
            print(f"  WARNING: Expected 512 dims, got {embeddings.shape[1]}")
            print("  This might not be correct cell embeddings!")
            return

    # 平均計算
    print("\nComputing mean embedding...")
    avg_embedding = embeddings.mean(axis=0).astype(np.float32)
    print(f"  Average shape: {avg_embedding.shape}")
    print(f"  Average dtype: {avg_embedding.dtype}")
    print(f"  Stats: min={avg_embedding.min():.4f}, max={avg_embedding.max():.4f}, mean={avg_embedding.mean():.4f}")

    # 保存
    print(f"\nSaving to: {OUTPUT_PATH}")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, avg_embedding)

    # 検証
    loaded = np.load(OUTPUT_PATH)
    assert loaded.shape == (512,), f"Expected (512,), got {loaded.shape}"
    assert loaded.dtype == np.float32, f"Expected float32, got {loaded.dtype}"
    print(f"  Verified: {loaded.shape}, {loaded.dtype}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
