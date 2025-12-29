"""
CDT Data Module

Versions:
- v1: EmbeddingDataset (pooled embeddings)
- v2: CDTv2Dataset (sequence-level DNA, full proteome)
"""

from .datasets import RealisticCDTDataset, DummyCDTDataset, collate_fn
from .dataloaders import create_dataloaders, get_dataloader_stats
from .embedding_dataset import CDTEmbeddingDataset, create_poc_dataloaders  # v1 POC
from .embedding_dataset_v2 import CDTv2Dataset, collate_v2, create_v2_dataloaders  # v2

# Aliases for backwards compatibility
EmbeddingDataset = CDTEmbeddingDataset

__all__ = [
    # Legacy
    'RealisticCDTDataset',
    'DummyCDTDataset',
    'collate_fn',
    'create_dataloaders',
    'get_dataloader_stats',
    # v1 POC
    'EmbeddingDataset',
    'CDTEmbeddingDataset',
    'create_poc_dataloaders',
    # v2 (Sequence-Level)
    'CDTv2Dataset',
    'collate_v2',
    'create_v2_dataloaders',
]
