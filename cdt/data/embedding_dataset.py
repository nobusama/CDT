"""
CDT POC用 Embedding Dataset

事前計算済みの埋め込み（Enformer, ESM-2, scGPT）を使用するDataset
Gasperini CRISPRi screenデータで学習するためのクラス
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class CDTEmbeddingDataset(Dataset):
    """
    CDT POC用のPyTorchデータセット

    事前計算済み埋め込みを使用:
    - DNA: Enformer (3072次元)
    - Protein: ESM-2 (1280次元)
    - Cell: scGPT K562平均 (512次元) - 全サンプル共通

    Example:
        >>> dataset = CDTEmbeddingDataset('data/processed/training/gasperini_train.h5')
        >>> sample = dataset[0]
        >>> print(sample['dna_emb'].shape)  # torch.Size([3072])
    """

    def __init__(
        self,
        training_data_path: str,
        enformer_path: str = None,
        esm2_path: str = None,
        scgpt_avg_path: str = None,
        project_root: str = None
    ):
        """
        Args:
            training_data_path: 学習データ(HDF5)のパス
            enformer_path: Enformer埋め込みのパス
            esm2_path: ESM-2埋め込みのパス
            scgpt_avg_path: K562平均scGPT埋め込みのパス
            project_root: プロジェクトルート（パスの解決用）
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)

        # デフォルトパス
        if enformer_path is None:
            enformer_path = self.project_root / "data/raw/enformer/preprocessing/precomputed_embeddings/enformer_gencode_v41_protein_coding_canonical_tss_hg38_nostitch_addbin_1_emb_mean_tar_sum_aug_0.h5"
        if esm2_path is None:
            esm2_path = self.project_root / "data/processed/embeddings/human_esm2_embeddings.h5"
        if scgpt_avg_path is None:
            scgpt_avg_path = self.project_root / "data/processed/embeddings/k562_avg_scgpt.npy"

        self.training_data_path = Path(training_data_path)
        self.enformer_path = Path(enformer_path)
        self.esm2_path = Path(esm2_path)
        self.scgpt_avg_path = Path(scgpt_avg_path)

        # 学習データ読み込み
        self._load_training_data()

        # 埋め込みファイルを開く（遅延読み込み用にファイルハンドルを保持）
        self._load_embedding_files()

        # scGPT K562平均埋め込みを読み込み（全サンプル共通）
        self._load_scgpt_avg()

    def _load_training_data(self):
        """学習データを読み込み"""
        with h5py.File(self.training_data_path, 'r') as f:
            self.enformer_idx = f['enformer_idx'][:]
            self.esm2_idx = f['esm2_idx'][:]
            self.labels = f['labels'][:]
            self.n_samples = len(self.labels)
            self.n_positive = int(np.sum(self.labels))
            self.n_negative = self.n_samples - self.n_positive

    def _load_embedding_files(self):
        """埋め込みファイルを開く"""
        # Enformer埋め込み
        self.enformer_file = h5py.File(self.enformer_path, 'r')
        self.enformer_emb = self.enformer_file['emb/block0_values']
        self.enformer_dim = self.enformer_emb.shape[1]

        # ESM-2埋め込み
        self.esm2_file = h5py.File(self.esm2_path, 'r')
        self.esm2_emb = self.esm2_file['embeddings']
        self.esm2_dim = self.esm2_emb.shape[1]

    def _load_scgpt_avg(self):
        """scGPT K562平均埋め込みを読み込み"""
        if self.scgpt_avg_path.exists():
            self.scgpt_avg = np.load(self.scgpt_avg_path).astype(np.float32)
            self.scgpt_dim = len(self.scgpt_avg)
        else:
            # まだ生成されていない場合はダミー
            print(f"Warning: scGPT avg not found at {self.scgpt_avg_path}, using zeros")
            self.scgpt_avg = np.zeros(512, dtype=np.float32)
            self.scgpt_dim = 512

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        指定インデックスのサンプルを取得

        Returns:
            dict: {
                'dna_emb': Enformer埋め込み (3072,),
                'protein_emb': ESM-2埋め込み (1280,),
                'cell_emb': scGPT K562平均 (512,),
                'label': 0 or 1
            }
        """
        # インデックス取得
        enf_idx = int(self.enformer_idx[idx])
        esm_idx = int(self.esm2_idx[idx])

        # 埋め込み取得
        dna_emb = self.enformer_emb[enf_idx, :].astype(np.float32)
        protein_emb = self.esm2_emb[esm_idx, :].astype(np.float32)
        cell_emb = self.scgpt_avg  # 全サンプル共通

        # ラベル
        label = self.labels[idx]

        return {
            'dna_emb': torch.from_numpy(dna_emb),
            'protein_emb': torch.from_numpy(protein_emb),
            'cell_emb': torch.from_numpy(cell_emb),
            'label': torch.tensor(label, dtype=torch.float32),
        }

    def get_dims(self) -> Dict[str, int]:
        """各埋め込みの次元を返す"""
        return {
            'dna': self.enformer_dim,
            'protein': self.esm2_dim,
            'cell': self.scgpt_dim,
        }

    def get_class_weights(self) -> torch.Tensor:
        """クラス不均衡対策用の重みを計算"""
        # Negative / Positive の比率を重みとして使用
        weight_positive = self.n_negative / self.n_positive
        return torch.tensor([1.0, weight_positive], dtype=torch.float32)

    def close(self):
        """ファイルハンドルを閉じる"""
        try:
            if hasattr(self, 'enformer_file') and self.enformer_file:
                self.enformer_file.close()
            if hasattr(self, 'esm2_file') and self.esm2_file:
                self.esm2_file.close()
        except Exception:
            pass  # Ignore errors during cleanup

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_poc_dataloaders(
    project_root: str = None,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    CDT POC用のDataLoaderを作成

    Args:
        project_root: プロジェクトルート
        batch_size: バッチサイズ
        num_workers: DataLoaderのワーカー数

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    project_root = Path(project_root)
    training_dir = project_root / "data/processed/training"

    # データセット作成
    train_dataset = CDTEmbeddingDataset(
        training_dir / "gasperini_train.h5",
        project_root=project_root
    )
    val_dataset = CDTEmbeddingDataset(
        training_dir / "gasperini_val.h5",
        project_root=project_root
    )
    test_dataset = CDTEmbeddingDataset(
        training_dir / "gasperini_test.h5",
        project_root=project_root
    )

    # DataLoader作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 60)
    print("CDTEmbeddingDataset テスト")
    print("=" * 60)

    # プロジェクトルート
    project_root = Path(__file__).parent.parent.parent

    # データセット作成
    print("\n【データセット作成】")
    dataset = CDTEmbeddingDataset(
        project_root / "data/processed/training/gasperini_train.h5",
        project_root=project_root
    )
    print(f"サンプル数: {len(dataset)}")
    print(f"Positive: {dataset.n_positive}")
    print(f"Negative: {dataset.n_negative}")
    print(f"埋め込み次元: {dataset.get_dims()}")
    print(f"クラス重み: {dataset.get_class_weights()}")

    # 1サンプル取得
    print("\n【1サンプル取得】")
    sample = dataset[0]
    print(f"DNA embedding shape: {sample['dna_emb'].shape}")
    print(f"Protein embedding shape: {sample['protein_emb'].shape}")
    print(f"Cell embedding shape: {sample['cell_emb'].shape}")
    print(f"Label: {sample['label']}")

    # DataLoader作成
    print("\n【DataLoader作成】")
    train_loader, val_loader, test_loader = create_poc_dataloaders(
        project_root=project_root,
        batch_size=32
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 1バッチ取得
    print("\n【1バッチ取得】")
    batch = next(iter(train_loader))
    print(f"DNA embedding batch shape: {batch['dna_emb'].shape}")
    print(f"Protein embedding batch shape: {batch['protein_emb'].shape}")
    print(f"Cell embedding batch shape: {batch['cell_emb'].shape}")
    print(f"Label batch shape: {batch['label'].shape}")
    print(f"Labels: {batch['label'][:10]}")

    # クリーンアップ
    dataset.close()
    print("\n✓ テスト完了!")
