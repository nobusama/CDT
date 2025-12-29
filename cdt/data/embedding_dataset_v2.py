"""
CDT v2 Dataset: Sequence-Level Embeddings with Full Proteome

v2アーキテクチャに対応:
- DNA: [batch, 896, 3072] - Enformer sequence-level (サンプルごとに異なる)
- Protein: [n_proteins, 768] - 全プロテオーム (バッチ間で共有)
- RNA: [batch, n_genes, 512] - 遺伝子発現 (サンプルごとに異なる可能性)

出力: [batch, n_proteins] - 各(エンハンサー, タンパク質)ペアの結合予測
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List


class CDTv2Dataset(Dataset):
    """
    CDT v2用のPyTorchデータセット

    各サンプルはエンハンサー位置を表し、モデルは全タンパク質との結合を予測する。

    入力:
        - DNA: [896, 3072] (Enformer sequence-level)
        - RNA: [n_genes, 512] (遺伝子発現プロファイル)

    Protein埋め込みは全サンプル共通なので、__getitem__では返さない。
    代わりにget_protein_embeddings()で取得。

    ラベル:
        - labels: [n_proteins] (このエンハンサーと各タンパク質の結合)
    """

    def __init__(
        self,
        training_data_path: str,
        dna_seqlevel_path: str = None,
        proteomelm_path: str = None,
        rna_gene_path: str = None,
        project_root: str = None,
        n_proteins: int = None,  # デバッグ用: タンパク質数を制限
        use_training_subset: bool = False,  # 学習サブセット(9523)を使用
        use_aligned: bool = True,  # RNA-Proteinアラインメント(2360)を使用
    ):
        """
        Args:
            training_data_path: 学習データ(HDF5)のパス (enhancer-protein pairs)
            dna_seqlevel_path: Enformer sequence-level埋め込みのパス
            proteomelm_path: ProteomeLM全プロテオーム埋め込みのパス
            rna_gene_path: 遺伝子発現埋め込みのパス
            project_root: プロジェクトルート
            n_proteins: デバッグ用にタンパク質数を制限
            use_training_subset: 学習データに含まれるタンパク質のみ使用 (20420→9523)
            use_aligned: RNA-Proteinアラインメント版を使用 (2360遺伝子)
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)
        self.use_training_subset = use_training_subset
        self.use_aligned = use_aligned

        # デフォルトパス
        if dna_seqlevel_path is None:
            dna_seqlevel_path = self.project_root / "data/processed/embeddings/enformer_seqlevel/pilot_1000.h5"
        if proteomelm_path is None:
            if use_aligned:
                proteomelm_path = self.project_root / "data/processed/embeddings/human_proteomelm_embeddings_aligned.h5"
            elif use_training_subset:
                proteomelm_path = self.project_root / "data/processed/embeddings/human_proteomelm_embeddings_training_subset.h5"
            else:
                proteomelm_path = self.project_root / "data/processed/embeddings/human_proteomelm_embeddings.h5"
        if rna_gene_path is None:
            if use_aligned:
                rna_gene_path = self.project_root / "data/processed/embeddings/k562_gene_embeddings_aligned.h5"
            else:
                rna_gene_path = self.project_root / "data/processed/embeddings/k562_gene_embeddings.h5"

        self.training_data_path = Path(training_data_path)
        self.dna_seqlevel_path = Path(dna_seqlevel_path)
        self.proteomelm_path = Path(proteomelm_path)
        self.rna_gene_path = Path(rna_gene_path)
        self.n_proteins_limit = n_proteins

        # インデックスマッピングを読み込み
        if use_aligned:
            self._load_index_mapping(aligned=True)
        elif use_training_subset:
            self._load_index_mapping(aligned=False)

        # データ読み込み
        self._load_training_data()
        self._load_embeddings()

    def _load_index_mapping(self, aligned: bool = False):
        """タンパク質インデックスのマッピングを読み込み"""
        if aligned:
            mapping_path = self.project_root / "data/processed/embeddings/protein_index_mapping_aligned.npz"
        else:
            mapping_path = self.project_root / "data/processed/embeddings/protein_index_mapping.npz"
        data = np.load(mapping_path, allow_pickle=True)
        # old_to_new: [(old_idx, new_idx), ...]
        old_to_new_pairs = data['old_to_new']
        self.old_to_new_idx = {int(pair[0]): int(pair[1]) for pair in old_to_new_pairs}
        print(f"Loaded protein index mapping: {len(self.old_to_new_idx)} proteins (aligned={aligned})")

    def _load_training_data(self):
        """学習データ（enhancer-protein pairs）を読み込み"""
        with h5py.File(self.training_data_path, 'r') as f:
            # pair_indices: どのDNA埋め込みを使うか
            enformer_idx = f['enformer_idx'][:]

            # タンパク質インデックス (ESM-2/ProteomeLM)
            orig_protein_idx = f['esm2_idx'][:]

            # ラベル
            labels = f['labels'][:]

            # Beta値（回帰用）
            if 'beta' in f:
                beta_values = f['beta'][:]
            else:
                beta_values = np.zeros_like(labels, dtype=np.float32)

        # アラインモード: マッピング可能なサンプルのみフィルタ
        if self.use_aligned or self.use_training_subset:
            # 有効なサンプルをフィルタ
            valid_mask = np.array([int(idx) in self.old_to_new_idx for idx in orig_protein_idx])

            self.enformer_idx = enformer_idx[valid_mask]
            self.labels = labels[valid_mask]
            self.beta_values = beta_values[valid_mask]

            # インデックスを再マッピング
            valid_orig_idx = orig_protein_idx[valid_mask]
            self.protein_idx = np.array([
                self.old_to_new_idx[int(idx)] for idx in valid_orig_idx
            ])

            n_filtered = len(orig_protein_idx) - valid_mask.sum()
            print(f"Filtered {n_filtered} samples (no matching protein)")
            print(f"Remapped protein indices: {valid_orig_idx.max()} -> {self.protein_idx.max()}")
        else:
            self.enformer_idx = enformer_idx
            self.labels = labels
            self.beta_values = beta_values
            self.protein_idx = orig_protein_idx

        self.n_samples = len(self.labels)
        self.n_positive = int(np.sum(self.labels))
        self.n_negative = self.n_samples - self.n_positive

        print(f"Training data: {self.n_samples} samples ({self.n_positive} positive)")

    def _load_embeddings(self):
        """全埋め込みを読み込み"""
        # DNA sequence-level embeddings
        self.dna_file = h5py.File(self.dna_seqlevel_path, 'r')
        self.dna_emb = self.dna_file['embeddings']  # [N, 896, 3072]
        self.dna_pair_indices = self.dna_file['pair_indices'][:]  # マッピング
        self.dna_seq_len = self.dna_emb.shape[1]  # 896
        self.dna_dim = self.dna_emb.shape[2]  # 3072

        # Protein embeddings (全プロテオーム)
        with h5py.File(self.proteomelm_path, 'r') as f:
            self.protein_emb = f['embeddings'][:]  # [n_proteins, 768]
            self.protein_ids = [x.decode() if isinstance(x, bytes) else x
                                for x in f['uniprot_ids'][:]]
            self.protein_gene_names = [x.decode() if isinstance(x, bytes) else x
                                        for x in f['gene_names'][:]]

        # タンパク質数を制限（デバッグ用）
        if self.n_proteins_limit is not None:
            self.protein_emb = self.protein_emb[:self.n_proteins_limit]
            self.protein_ids = self.protein_ids[:self.n_proteins_limit]
            self.protein_gene_names = self.protein_gene_names[:self.n_proteins_limit]

        self.n_proteins = len(self.protein_ids)
        self.protein_dim = self.protein_emb.shape[1]  # 768

        # RNA gene embeddings
        with h5py.File(self.rna_gene_path, 'r') as f:
            self.rna_emb = f['embeddings'][:]  # [n_genes, 512]
            self.rna_gene_names = [x.decode() if isinstance(x, bytes) else x
                                   for x in f['gene_names'][:]]

        self.n_genes = self.rna_emb.shape[0]
        self.rna_dim = self.rna_emb.shape[1]  # 512

        print(f"DNA embeddings: {self.dna_emb.shape}")
        print(f"Protein embeddings: {self.protein_emb.shape}")
        print(f"RNA gene embeddings: {self.rna_emb.shape}")

        # DNA pair_index → seqlevel index マッピング作成
        self._create_dna_index_mapping()

        # タンパク質ID → インデックス マッピング
        self._create_protein_index_mapping()

    def _create_dna_index_mapping(self):
        """学習データのenformer_idxからseqlevel埋め込みへのマッピング"""
        # dna_pair_indices: seqlevel埋め込みが対応する元のpair_idx
        self.dna_idx_map = {}
        for seqlevel_idx, pair_idx in enumerate(self.dna_pair_indices):
            self.dna_idx_map[pair_idx] = seqlevel_idx

    def _create_protein_index_mapping(self):
        """タンパク質遺伝子名 → ProteomeLMインデックス マッピング"""
        self.protein_gene_to_idx = {
            gene: idx for idx, gene in enumerate(self.protein_gene_names)
        }

    def _get_dna_emb(self, enformer_idx: int) -> np.ndarray:
        """
        enformer_idxからDNA埋め込みを取得

        seqlevelデータにない場合はゼロ埋め込みを返す
        """
        if enformer_idx in self.dna_idx_map:
            seqlevel_idx = self.dna_idx_map[enformer_idx]
            return self.dna_emb[seqlevel_idx, :, :].astype(np.float32)
        else:
            # seqlevelデータがない場合はスキップ（ゼロ）
            return np.zeros((self.dna_seq_len, self.dna_dim), dtype=np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        指定インデックスのサンプルを取得

        Returns:
            dict: {
                'dna_emb': [896, 3072],
                'rna_emb': [n_genes, 512],
                'protein_idx': このペアのタンパク質インデックス,
                'label': 0 or 1,
                'beta': 効果サイズ（連続値、回帰用）
            }
        """
        # DNA埋め込み
        enf_idx = int(self.enformer_idx[idx])
        dna_emb = self._get_dna_emb(enf_idx)

        # RNA埋め込み（現状はK562の遺伝子埋め込みを全サンプル共有）
        rna_emb = self.rna_emb.astype(np.float32)

        # このペアのタンパク質インデックス
        prot_idx = int(self.protein_idx[idx])

        # ラベル（分類用）とベータ値（回帰用）
        label = self.labels[idx]
        beta = self.beta_values[idx]

        return {
            'dna_emb': torch.from_numpy(dna_emb),  # [896, 3072]
            'rna_emb': torch.from_numpy(rna_emb),  # [n_genes, 512]
            'protein_idx': torch.tensor(prot_idx, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float32),
            'beta': torch.tensor(beta, dtype=torch.float32),
        }

    def get_protein_embeddings(self) -> torch.Tensor:
        """全タンパク質埋め込みを返す（バッチ間で共有）"""
        return torch.from_numpy(self.protein_emb.astype(np.float32))

    def get_dims(self) -> Dict[str, int]:
        """各埋め込みの次元を返す"""
        return {
            'dna_seq_len': self.dna_seq_len,
            'dna_dim': self.dna_dim,
            'protein_dim': self.protein_dim,
            'rna_dim': self.rna_dim,
            'n_proteins': self.n_proteins,
            'n_genes': self.n_genes,
        }

    def get_class_weights(self) -> torch.Tensor:
        """クラス不均衡対策用の重みを計算"""
        weight_positive = self.n_negative / self.n_positive
        return torch.tensor([1.0, weight_positive], dtype=torch.float32)

    def close(self):
        """ファイルハンドルを閉じる"""
        try:
            if hasattr(self, 'dna_file') and self.dna_file:
                self.dna_file.close()
        except Exception:
            pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def collate_v2(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    v2用のカスタムcollate関数

    Returns:
        dict: {
            'dna_emb': [batch, 896, 3072],
            'rna_emb': [batch, n_genes, 512],
            'protein_indices': [batch,],
            'labels': [batch,],
            'betas': [batch,],  # 回帰用
        }
    """
    dna_embs = torch.stack([item['dna_emb'] for item in batch])
    rna_embs = torch.stack([item['rna_emb'] for item in batch])
    protein_indices = torch.stack([item['protein_idx'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    betas = torch.stack([item['beta'] for item in batch])

    return {
        'dna_emb': dna_embs,
        'rna_emb': rna_embs,
        'protein_indices': protein_indices,
        'labels': labels,
        'betas': betas,
    }


def create_v2_dataloaders(
    project_root: str = None,
    batch_size: int = 16,
    num_workers: int = 0,
    n_proteins: int = None,  # デバッグ用
    use_training_subset: bool = True,  # 学習サブセットを使用
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    CDT v2用のDataLoaderを作成

    Args:
        project_root: プロジェクトルート
        batch_size: バッチサイズ
        num_workers: DataLoaderのワーカー数
        n_proteins: デバッグ用にタンパク質数を制限
        use_training_subset: 学習データに含まれるタンパク質のみ使用

    Returns:
        (train_loader, val_loader, test_loader, protein_emb)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    project_root = Path(project_root)
    training_dir = project_root / "data/processed/training"

    # データセット作成
    train_dataset = CDTv2Dataset(
        training_dir / "gasperini_train.h5",
        project_root=project_root,
        n_proteins=n_proteins,
        use_training_subset=use_training_subset,
    )
    val_dataset = CDTv2Dataset(
        training_dir / "gasperini_val.h5",
        project_root=project_root,
        n_proteins=n_proteins,
        use_training_subset=use_training_subset,
    )
    test_dataset = CDTv2Dataset(
        training_dir / "gasperini_test.h5",
        project_root=project_root,
        n_proteins=n_proteins,
        use_training_subset=use_training_subset,
    )

    # 全タンパク質埋め込み（共有）
    protein_emb = train_dataset.get_protein_embeddings()

    # DataLoader作成
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_v2,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_v2
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_v2
    )

    return train_loader, val_loader, test_loader, protein_emb


if __name__ == "__main__":
    print("=" * 60)
    print("CDT v2 Dataset テスト")
    print("=" * 60)

    # プロジェクトルート
    project_root = Path(__file__).parent.parent.parent

    # データセット作成（タンパク質数を100に制限してテスト）
    print("\n【データセット作成】")
    dataset = CDTv2Dataset(
        project_root / "data/processed/training/gasperini_train.h5",
        project_root=project_root,
        n_proteins=100  # テスト用
    )
    print(f"サンプル数: {len(dataset)}")
    print(f"Positive: {dataset.n_positive}")
    print(f"Negative: {dataset.n_negative}")
    print(f"次元: {dataset.get_dims()}")

    # Beta値の統計
    print("\n【Beta値統計（回帰用）】")
    beta_vals = dataset.beta_values
    print(f"Beta値: min={beta_vals.min():.4f}, max={beta_vals.max():.4f}")
    print(f"        mean={beta_vals.mean():.4f}, std={beta_vals.std():.4f}")
    print(f"        非ゼロ: {(beta_vals != 0).sum()}/{len(beta_vals)}")

    # 1サンプル取得
    print("\n【1サンプル取得】")
    sample = dataset[0]
    print(f"DNA embedding shape: {sample['dna_emb'].shape}")
    print(f"RNA embedding shape: {sample['rna_emb'].shape}")
    print(f"Protein index: {sample['protein_idx']}")
    print(f"Label: {sample['label']}")
    print(f"Beta: {sample['beta']}")

    # 全タンパク質埋め込み
    print("\n【タンパク質埋め込み】")
    protein_emb = dataset.get_protein_embeddings()
    print(f"Protein embeddings shape: {protein_emb.shape}")

    # DataLoader作成
    print("\n【DataLoader作成】")
    train_loader, val_loader, test_loader, protein_emb = create_v2_dataloaders(
        project_root=project_root,
        batch_size=8,
        n_proteins=100  # テスト用
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Protein embeddings: {protein_emb.shape}")

    # 1バッチ取得
    print("\n【1バッチ取得】")
    batch = next(iter(train_loader))
    print(f"DNA embedding batch shape: {batch['dna_emb'].shape}")
    print(f"RNA embedding batch shape: {batch['rna_emb'].shape}")
    print(f"Protein indices: {batch['protein_indices']}")
    print(f"Labels: {batch['labels']}")
    print(f"Betas: {batch['betas']}")

    # クリーンアップ
    dataset.close()
    print("\n✓ テスト完了!")
