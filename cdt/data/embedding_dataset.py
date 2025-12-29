"""
CDT POC Embedding Dataset

Dataset using pre-computed embeddings (Enformer, ESM-2, scGPT)
Class for training on Gasperini CRISPRi screen data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple


class CDTEmbeddingDataset(Dataset):
    """
    PyTorch Dataset for CDT POC

    Uses pre-computed embeddings:
    - DNA: Enformer (3072 dimensions)
    - Protein: ESM-2 (1280 dimensions)
    - Cell: scGPT K562 average (512 dimensions) - shared across all samples

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
            training_data_path: Path to training data (HDF5)
            enformer_path: Path to Enformer embeddings
            esm2_path: Path to ESM-2 embeddings
            scgpt_avg_path: Path to K562 average scGPT embedding
            project_root: Project root (for path resolution)
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)

        # Default paths
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

        # Load training data
        self._load_training_data()

        # Open embedding files (keep file handles for lazy loading)
        self._load_embedding_files()

        # Load scGPT K562 average embedding (shared across all samples)
        self._load_scgpt_avg()

    def _load_training_data(self):
        """Load training data"""
        with h5py.File(self.training_data_path, 'r') as f:
            self.enformer_idx = f['enformer_idx'][:]
            self.esm2_idx = f['esm2_idx'][:]
            self.labels = f['labels'][:]
            self.n_samples = len(self.labels)
            self.n_positive = int(np.sum(self.labels))
            self.n_negative = self.n_samples - self.n_positive

    def _load_embedding_files(self):
        """Open embedding files"""
        # Enformer embeddings
        self.enformer_file = h5py.File(self.enformer_path, 'r')
        self.enformer_emb = self.enformer_file['emb/block0_values']
        self.enformer_dim = self.enformer_emb.shape[1]

        # ESM-2 embeddings
        self.esm2_file = h5py.File(self.esm2_path, 'r')
        self.esm2_emb = self.esm2_file['embeddings']
        self.esm2_dim = self.esm2_emb.shape[1]

    def _load_scgpt_avg(self):
        """Load scGPT K562 average embedding"""
        if self.scgpt_avg_path.exists():
            self.scgpt_avg = np.load(self.scgpt_avg_path).astype(np.float32)
            self.scgpt_dim = len(self.scgpt_avg)
        else:
            # Use dummy if not yet generated
            print(f"Warning: scGPT avg not found at {self.scgpt_avg_path}, using zeros")
            self.scgpt_avg = np.zeros(512, dtype=np.float32)
            self.scgpt_dim = 512

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample at specified index

        Returns:
            dict: {
                'dna_emb': Enformer embedding (3072,),
                'protein_emb': ESM-2 embedding (1280,),
                'cell_emb': scGPT K562 average (512,),
                'label': 0 or 1
            }
        """
        # Get indices
        enf_idx = int(self.enformer_idx[idx])
        esm_idx = int(self.esm2_idx[idx])

        # Get embeddings
        dna_emb = self.enformer_emb[enf_idx, :].astype(np.float32)
        protein_emb = self.esm2_emb[esm_idx, :].astype(np.float32)
        cell_emb = self.scgpt_avg  # Shared across all samples

        # Label
        label = self.labels[idx]

        return {
            'dna_emb': torch.from_numpy(dna_emb),
            'protein_emb': torch.from_numpy(protein_emb),
            'cell_emb': torch.from_numpy(cell_emb),
            'label': torch.tensor(label, dtype=torch.float32),
        }

    def get_dims(self) -> Dict[str, int]:
        """Return dimensions of each embedding"""
        return {
            'dna': self.enformer_dim,
            'protein': self.esm2_dim,
            'cell': self.scgpt_dim,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute weights for class imbalance handling"""
        # Use Negative / Positive ratio as weight
        weight_positive = self.n_negative / self.n_positive
        return torch.tensor([1.0, weight_positive], dtype=torch.float32)

    def close(self):
        """Close file handles"""
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
    Create DataLoaders for CDT POC

    Args:
        project_root: Project root
        batch_size: Batch size
        num_workers: Number of DataLoader workers

    Returns:
        (train_loader, val_loader, test_loader)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    project_root = Path(project_root)
    training_dir = project_root / "data/processed/training"

    # Create datasets
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

    # Create DataLoaders
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
    print("CDTEmbeddingDataset Test")
    print("=" * 60)

    # Project root
    project_root = Path(__file__).parent.parent.parent

    # Create dataset
    print("\n[Dataset Creation]")
    dataset = CDTEmbeddingDataset(
        project_root / "data/processed/training/gasperini_train.h5",
        project_root=project_root
    )
    print(f"Number of samples: {len(dataset)}")
    print(f"Positive: {dataset.n_positive}")
    print(f"Negative: {dataset.n_negative}")
    print(f"Embedding dimensions: {dataset.get_dims()}")
    print(f"Class weights: {dataset.get_class_weights()}")

    # Get one sample
    print("\n[Get One Sample]")
    sample = dataset[0]
    print(f"DNA embedding shape: {sample['dna_emb'].shape}")
    print(f"Protein embedding shape: {sample['protein_emb'].shape}")
    print(f"Cell embedding shape: {sample['cell_emb'].shape}")
    print(f"Label: {sample['label']}")

    # Create DataLoader
    print("\n[DataLoader Creation]")
    train_loader, val_loader, test_loader = create_poc_dataloaders(
        project_root=project_root,
        batch_size=32
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Get one batch
    print("\n[Get One Batch]")
    batch = next(iter(train_loader))
    print(f"DNA embedding batch shape: {batch['dna_emb'].shape}")
    print(f"Protein embedding batch shape: {batch['protein_emb'].shape}")
    print(f"Cell embedding batch shape: {batch['cell_emb'].shape}")
    print(f"Label batch shape: {batch['label'].shape}")
    print(f"Labels: {batch['label'][:10]}")

    # Cleanup
    dataset.close()
    print("\n[OK] Test complete!")
