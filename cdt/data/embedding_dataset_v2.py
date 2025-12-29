"""
CDT v2 Dataset: Sequence-Level Embeddings with Full Proteome

Compatible with v2 architecture:
- DNA: [batch, 896, 3072] - Enformer sequence-level (different for each sample)
- Protein: [n_proteins, 768] - Full proteome (shared across batches)
- RNA: [batch, n_genes, 512] - Gene expression (may differ per sample)

Output: [batch, n_proteins] - Binding prediction for each (enhancer, protein) pair
"""

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List


class CDTv2Dataset(Dataset):
    """
    PyTorch Dataset for CDT v2

    Each sample represents an enhancer position, and the model predicts binding
    with all proteins.

    Input:
        - DNA: [896, 3072] (Enformer sequence-level)
        - RNA: [n_genes, 512] (gene expression profile)

    Protein embeddings are shared across all samples, so not returned in __getitem__.
    Use get_protein_embeddings() to retrieve them instead.

    Labels:
        - labels: [n_proteins] (binding between this enhancer and each protein)
    """

    def __init__(
        self,
        training_data_path: str,
        dna_seqlevel_path: str = None,
        proteomelm_path: str = None,
        rna_gene_path: str = None,
        project_root: str = None,
        n_proteins: int = None,  # Debug: limit number of proteins
        use_training_subset: bool = False,  # Use training subset (9523)
        use_aligned: bool = True,  # Use RNA-Protein alignment (2360)
    ):
        """
        Args:
            training_data_path: Path to training data (HDF5) (enhancer-protein pairs)
            dna_seqlevel_path: Path to Enformer sequence-level embeddings
            proteomelm_path: Path to ProteomeLM full proteome embeddings
            rna_gene_path: Path to gene expression embeddings
            project_root: Project root
            n_proteins: Limit number of proteins for debugging
            use_training_subset: Use only proteins in training data (20420 -> 9523)
            use_aligned: Use RNA-Protein aligned version (2360 genes)
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent

        self.project_root = Path(project_root)
        self.use_training_subset = use_training_subset
        self.use_aligned = use_aligned

        # Default paths
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

        # Load index mapping
        if use_aligned:
            self._load_index_mapping(aligned=True)
        elif use_training_subset:
            self._load_index_mapping(aligned=False)

        # Load data
        self._load_training_data()
        self._load_embeddings()

    def _load_index_mapping(self, aligned: bool = False):
        """Load protein index mapping"""
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
        """Load training data (enhancer-protein pairs)"""
        with h5py.File(self.training_data_path, 'r') as f:
            # pair_indices: which DNA embedding to use
            enformer_idx = f['enformer_idx'][:]

            # Protein index (ESM-2/ProteomeLM)
            orig_protein_idx = f['esm2_idx'][:]

            # Labels
            labels = f['labels'][:]

            # Beta values (for regression)
            if 'beta' in f:
                beta_values = f['beta'][:]
            else:
                beta_values = np.zeros_like(labels, dtype=np.float32)

        # Aligned mode: filter only samples that can be mapped
        if self.use_aligned or self.use_training_subset:
            # Filter valid samples
            valid_mask = np.array([int(idx) in self.old_to_new_idx for idx in orig_protein_idx])

            self.enformer_idx = enformer_idx[valid_mask]
            self.labels = labels[valid_mask]
            self.beta_values = beta_values[valid_mask]

            # Remap indices
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
        """Load all embeddings"""
        # DNA sequence-level embeddings
        self.dna_file = h5py.File(self.dna_seqlevel_path, 'r')
        self.dna_emb = self.dna_file['embeddings']  # [N, 896, 3072]
        self.dna_pair_indices = self.dna_file['pair_indices'][:]  # Mapping
        self.dna_seq_len = self.dna_emb.shape[1]  # 896
        self.dna_dim = self.dna_emb.shape[2]  # 3072

        # Protein embeddings (full proteome)
        with h5py.File(self.proteomelm_path, 'r') as f:
            self.protein_emb = f['embeddings'][:]  # [n_proteins, 768]
            self.protein_ids = [x.decode() if isinstance(x, bytes) else x
                                for x in f['uniprot_ids'][:]]
            self.protein_gene_names = [x.decode() if isinstance(x, bytes) else x
                                        for x in f['gene_names'][:]]

        # Limit number of proteins (for debugging)
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

        # Create DNA pair_index -> seqlevel index mapping
        self._create_dna_index_mapping()

        # Create protein ID -> index mapping
        self._create_protein_index_mapping()

    def _create_dna_index_mapping(self):
        """Mapping from training data enformer_idx to seqlevel embedding"""
        # dna_pair_indices: original pair_idx that seqlevel embedding corresponds to
        self.dna_idx_map = {}
        for seqlevel_idx, pair_idx in enumerate(self.dna_pair_indices):
            self.dna_idx_map[pair_idx] = seqlevel_idx

    def _create_protein_index_mapping(self):
        """Mapping from protein gene name -> ProteomeLM index"""
        self.protein_gene_to_idx = {
            gene: idx for idx, gene in enumerate(self.protein_gene_names)
        }

    def _get_dna_emb(self, enformer_idx: int) -> np.ndarray:
        """
        Get DNA embedding from enformer_idx

        Returns zero embedding if not in seqlevel data
        """
        if enformer_idx in self.dna_idx_map:
            seqlevel_idx = self.dna_idx_map[enformer_idx]
            return self.dna_emb[seqlevel_idx, :, :].astype(np.float32)
        else:
            # Skip (zero) if seqlevel data not available
            return np.zeros((self.dna_seq_len, self.dna_dim), dtype=np.float32)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sample at specified index

        Returns:
            dict: {
                'dna_emb': [896, 3072],
                'rna_emb': [n_genes, 512],
                'protein_idx': protein index for this pair,
                'label': 0 or 1,
                'beta': effect size (continuous, for regression)
            }
        """
        # DNA embedding
        enf_idx = int(self.enformer_idx[idx])
        dna_emb = self._get_dna_emb(enf_idx)

        # RNA embedding (currently K562 gene embeddings shared across all samples)
        rna_emb = self.rna_emb.astype(np.float32)

        # Protein index for this pair
        prot_idx = int(self.protein_idx[idx])

        # Label (for classification) and beta value (for regression)
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
        """Return all protein embeddings (shared across batches)"""
        return torch.from_numpy(self.protein_emb.astype(np.float32))

    def get_dims(self) -> Dict[str, int]:
        """Return dimensions of each embedding"""
        return {
            'dna_seq_len': self.dna_seq_len,
            'dna_dim': self.dna_dim,
            'protein_dim': self.protein_dim,
            'rna_dim': self.rna_dim,
            'n_proteins': self.n_proteins,
            'n_genes': self.n_genes,
        }

    def get_class_weights(self) -> torch.Tensor:
        """Compute weights for class imbalance handling"""
        weight_positive = self.n_negative / self.n_positive
        return torch.tensor([1.0, weight_positive], dtype=torch.float32)

    def close(self):
        """Close file handles"""
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
    Custom collate function for v2

    Returns:
        dict: {
            'dna_emb': [batch, 896, 3072],
            'rna_emb': [batch, n_genes, 512],
            'protein_indices': [batch,],
            'labels': [batch,],
            'betas': [batch,],  # for regression
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
    n_proteins: int = None,  # For debugging
    use_training_subset: bool = True,  # Use training subset
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Create DataLoaders for CDT v2

    Args:
        project_root: Project root
        batch_size: Batch size
        num_workers: Number of DataLoader workers
        n_proteins: Limit number of proteins for debugging
        use_training_subset: Use only proteins in training data

    Returns:
        (train_loader, val_loader, test_loader, protein_emb)
    """
    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    project_root = Path(project_root)
    training_dir = project_root / "data/processed/training"

    # Create datasets
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

    # Full protein embeddings (shared)
    protein_emb = train_dataset.get_protein_embeddings()

    # Create DataLoaders
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
    print("CDT v2 Dataset Test")
    print("=" * 60)

    # Project root
    project_root = Path(__file__).parent.parent.parent

    # Create dataset (limit proteins to 100 for testing)
    print("\n[Dataset Creation]")
    dataset = CDTv2Dataset(
        project_root / "data/processed/training/gasperini_train.h5",
        project_root=project_root,
        n_proteins=100  # For testing
    )
    print(f"Number of samples: {len(dataset)}")
    print(f"Positive: {dataset.n_positive}")
    print(f"Negative: {dataset.n_negative}")
    print(f"Dimensions: {dataset.get_dims()}")

    # Beta value statistics
    print("\n[Beta Value Statistics (for regression)]")
    beta_vals = dataset.beta_values
    print(f"Beta values: min={beta_vals.min():.4f}, max={beta_vals.max():.4f}")
    print(f"            mean={beta_vals.mean():.4f}, std={beta_vals.std():.4f}")
    print(f"            non-zero: {(beta_vals != 0).sum()}/{len(beta_vals)}")

    # Get one sample
    print("\n[Get One Sample]")
    sample = dataset[0]
    print(f"DNA embedding shape: {sample['dna_emb'].shape}")
    print(f"RNA embedding shape: {sample['rna_emb'].shape}")
    print(f"Protein index: {sample['protein_idx']}")
    print(f"Label: {sample['label']}")
    print(f"Beta: {sample['beta']}")

    # Full protein embeddings
    print("\n[Protein Embeddings]")
    protein_emb = dataset.get_protein_embeddings()
    print(f"Protein embeddings shape: {protein_emb.shape}")

    # Create DataLoader
    print("\n[DataLoader Creation]")
    train_loader, val_loader, test_loader, protein_emb = create_v2_dataloaders(
        project_root=project_root,
        batch_size=8,
        n_proteins=100  # For testing
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Protein embeddings: {protein_emb.shape}")

    # Get one batch
    print("\n[Get One Batch]")
    batch = next(iter(train_loader))
    print(f"DNA embedding batch shape: {batch['dna_emb'].shape}")
    print(f"RNA embedding batch shape: {batch['rna_emb'].shape}")
    print(f"Protein indices: {batch['protein_indices']}")
    print(f"Labels: {batch['labels']}")
    print(f"Betas: {batch['betas']}")

    # Cleanup
    dataset.close()
    print("\n[OK] Test complete!")
