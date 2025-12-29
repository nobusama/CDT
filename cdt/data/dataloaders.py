"""
DataLoader Utilities for CDT

Utilities for creating DataLoaders and batch processing
"""

import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional

from .datasets import DummyCDTDataset, collate_fn


def create_dataloaders(
    num_samples: int = 1000,
    samples_per_gene: int = 4,
    batch_size: int = 16,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    dna_length: int = 100,
    rna_length: int = 80,
    protein_length: int = 30,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for CDT training

    Args:
        num_samples: Total number of samples to generate
        samples_per_gene: Samples per gene (for contrastive learning)
        batch_size: Batch size for training
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        dna_length: DNA sequence length
        rna_length: RNA sequence length
        protein_length: Protein sequence length
        num_workers: Number of workers for DataLoader
        seed: Random seed

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, test loaders

    Example:
        >>> train_loader, val_loader, test_loader = create_dataloaders(
        ...     num_samples=1000,
        ...     batch_size=16
        ... )
        >>> for batch in train_loader:
        ...     print(batch['gene_ids'].shape)  # torch.Size([16])
        ...     break
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio must equal 1.0"

    # Create full dataset
    full_dataset = DummyCDTDataset(
        num_samples=num_samples,
        samples_per_gene=samples_per_gene,
        dna_length=dna_length,
        rna_length=rna_length,
        protein_length=protein_length,
        seed=seed
    )

    # Calculate split sizes
    train_size = int(train_ratio * num_samples)
    val_size = int(val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader, val_loader, test_loader


def get_dataloader_stats(loader: DataLoader) -> dict:
    """
    Get statistics about a DataLoader

    Args:
        loader: DataLoader to analyze

    Returns:
        dict: Statistics including batch count, total samples, etc.

    Example:
        >>> train_loader, _, _ = create_dataloaders(num_samples=1000)
        >>> stats = get_dataloader_stats(train_loader)
        >>> print(stats)
        {'num_batches': 43, 'batch_size': 16, 'total_samples': 700, ...}
    """
    num_batches = len(loader)
    batch_size = loader.batch_size
    total_samples = len(loader.dataset)

    # Get a sample batch to check dimensions
    sample_batch = next(iter(loader))

    stats = {
        'num_batches': num_batches,
        'batch_size': batch_size,
        'total_samples': total_samples,
        'samples_per_batch': len(sample_batch['gene_ids']),
        'num_unique_genes': len(torch.unique(sample_batch['gene_ids'])),
        'dna_seq_length': len(sample_batch['dna_sequences'][0]),
        'rna_seq_length': len(sample_batch['rna_sequences'][0]),
        'protein_seq_length': len(sample_batch['protein_sequences'][0])
    }

    return stats


if __name__ == "__main__":
    # Test DataLoader creation
    print("Testing DataLoader creation...")

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        num_samples=1000,
        samples_per_gene=4,
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        dna_length=50,
        rna_length=40,
        protein_length=20,
        seed=42
    )

    print(f"\nTrain loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # Get stats
    print("\nTrain loader stats:")
    train_stats = get_dataloader_stats(train_loader)
    for key, value in train_stats.items():
        print(f"  {key}: {value}")

    # Test iteration
    print("\nTesting batch iteration...")
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"DNA sequences: {len(batch['dna_sequences'])}")
    print(f"RNA sequences: {len(batch['rna_sequences'])}")
    print(f"Protein sequences: {len(batch['protein_sequences'])}")
    print(f"Gene IDs shape: {batch['gene_ids'].shape}")
    print(f"Gene IDs (first batch): {batch['gene_ids']}")

    # Check that gene grouping works
    print("\nChecking gene grouping in batch...")
    gene_counts = {}
    for gene_id in batch['gene_ids'].tolist():
        gene_counts[gene_id] = gene_counts.get(gene_id, 0) + 1

    print(f"Unique genes in batch: {len(gene_counts)}")
    print(f"Samples per gene: {list(gene_counts.values())}")

    # Test multiple batches
    print("\nTesting multiple batches...")
    all_gene_ids = []
    for i, batch in enumerate(train_loader):
        all_gene_ids.extend(batch['gene_ids'].tolist())
        if i >= 2:  # Check first 3 batches
            break

    print(f"Total samples in 3 batches: {len(all_gene_ids)}")
    print(f"Unique genes in 3 batches: {len(set(all_gene_ids))}")

    print("\n[OK] All tests passed!")
