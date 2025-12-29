"""
CDT Dataset

PyTorch Dataset class
Tokenizes synthetic data and passes it to the model
"""

import torch
from torch.utils.data import Dataset
from typing import List, Dict
from cdt.data.synthetic_data_generator import SyntheticDataGenerator
from cdt.tokenizers.dna_tokenizer import DNATokenizer
from cdt.tokenizers.rna_tokenizer import RNATokenizer
from cdt.tokenizers.protein_tokenizer import ProteinTokenizer


class CDTDataset(Dataset):
    """
    PyTorch Dataset for CDT model

    Generates synthetic data, tokenizes it, and converts to Tensors

    Example:
        >>> dataset = CDTDataset(num_samples=100)
        >>> sample = dataset[0]
        >>> print(sample.keys())
        dict_keys(['dna', 'rna', 'protein', 'dna_tokens', 'rna_tokens', 'protein_tokens'])
    """

    def __init__(
        self,
        num_samples: int = 1000,
        min_length: int = 30,
        max_length: int = 300,
        seed: int = None
    ):
        """
        Initialize the dataset

        Args:
            num_samples: Number of samples to generate
            min_length: Minimum DNA sequence length (in nucleotides)
            max_length: Maximum DNA sequence length (in nucleotides)
            seed: Random seed (for reproducibility)
        """
        # Initialize tokenizers
        self.dna_tokenizer = DNATokenizer()
        self.rna_tokenizer = RNATokenizer()
        self.protein_tokenizer = ProteinTokenizer()

        # Initialize data generator
        self.generator = SyntheticDataGenerator(seed=seed)

        # Generate synthetic data
        self.data = self.generator.generate(
            num_samples=num_samples,
            min_length=min_length,
            max_length=max_length
        )

    def __len__(self) -> int:
        """Return the size of the dataset"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get the sample at the specified index

        Args:
            idx: Index of the sample

        Returns:
            dict: {
                'dna': str,
                'rna': str,
                'protein': str,
                'dna_tokens': Tensor,
                'rna_tokens': Tensor,
                'protein_tokens': Tensor
            }
        """
        # Get raw sequence data
        sample = self.data[idx]

        # Tokenize
        dna_tokens = self.dna_tokenizer.encode(sample['dna'])
        rna_tokens = self.rna_tokenizer.encode(sample['rna'])
        protein_tokens = self.protein_tokenizer.encode(sample['protein'])

        # Convert to PyTorch Tensors
        return {
            # Raw sequences (strings)
            'dna': sample['dna'],
            'rna': sample['rna'],
            'protein': sample['protein'],
            # Tokenized sequences (Tensors)
            'dna_tokens': torch.tensor(dna_tokens, dtype=torch.long),
            'rna_tokens': torch.tensor(rna_tokens, dtype=torch.long),
            'protein_tokens': torch.tensor(protein_tokens, dtype=torch.long),
        }

    def get_vocab_sizes(self) -> Dict[str, int]:
        """
        Return the vocabulary size for each tokenizer

        Returns:
            dict: {'dna': int, 'rna': int, 'protein': int}
        """
        return {
            'dna': len(self.dna_tokenizer),
            'rna': len(self.rna_tokenizer),
            'protein': len(self.protein_tokenizer),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Pad batches to uniform length

    Custom collate function for PyTorch DataLoader
    Pads sequences of different lengths to the same length

    Args:
        batch: List of samples

    Returns:
        dict: Padded batch
    """
    # Batch size
    batch_size = len(batch)

    # Calculate max length for each sequence type
    max_dna_len = max(len(sample['dna_tokens']) for sample in batch)
    max_rna_len = max(len(sample['rna_tokens']) for sample in batch)
    max_protein_len = max(len(sample['protein_tokens']) for sample in batch)

    # Padding values
    dna_pad_id = 4  # 'N'
    rna_pad_id = 4  # 'N'
    protein_pad_id = 20  # 'X'

    # Create padded tensors
    dna_tokens = torch.full((batch_size, max_dna_len), dna_pad_id, dtype=torch.long)
    rna_tokens = torch.full((batch_size, max_rna_len), rna_pad_id, dtype=torch.long)
    protein_tokens = torch.full((batch_size, max_protein_len), protein_pad_id, dtype=torch.long)

    # Lists to store raw sequences
    dna_seqs = []
    rna_seqs = []
    protein_seqs = []

    # Copy each sample to padded tensors
    for i, sample in enumerate(batch):
        # Tokens
        dna_len = len(sample['dna_tokens'])
        rna_len = len(sample['rna_tokens'])
        protein_len = len(sample['protein_tokens'])

        dna_tokens[i, :dna_len] = sample['dna_tokens']
        rna_tokens[i, :rna_len] = sample['rna_tokens']
        protein_tokens[i, :protein_len] = sample['protein_tokens']

        # Raw sequences
        dna_seqs.append(sample['dna'])
        rna_seqs.append(sample['rna'])
        protein_seqs.append(sample['protein'])

    return {
        'dna': dna_seqs,
        'rna': rna_seqs,
        'protein': protein_seqs,
        'dna_tokens': dna_tokens,
        'rna_tokens': rna_tokens,
        'protein_tokens': protein_tokens,
    }


# Usage example (runs only when this file is executed directly)
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    print("=" * 60)
    print("CDTDataset Test")
    print("=" * 60)
    print()

    # Create dataset
    print("[Dataset Creation]")
    dataset = CDTDataset(num_samples=100, min_length=30, max_length=60, seed=42)
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocabulary sizes: {dataset.get_vocab_sizes()}")
    print()

    # Get one sample
    print("[Get One Sample]")
    sample = dataset[0]
    print(f"DNA sequence:     {sample['dna'][:30]}... (total {len(sample['dna'])} nucleotides)")
    print(f"RNA sequence:     {sample['rna'][:30]}... (total {len(sample['rna'])} nucleotides)")
    print(f"Protein sequence: {sample['protein']}")
    print()
    print(f"DNA tokens:     {sample['dna_tokens'][:10]}... (shape: {sample['dna_tokens'].shape})")
    print(f"RNA tokens:     {sample['rna_tokens'][:10]}... (shape: {sample['rna_tokens'].shape})")
    print(f"Protein tokens: {sample['protein_tokens']} (shape: {sample['protein_tokens'].shape})")
    print()

    # Create DataLoader
    print("[DataLoader Creation (Batch Processing)]")
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    # Get one batch
    batch = next(iter(dataloader))
    print(f"Batch size: {len(batch['dna'])}")
    print(f"DNA tokens shape:     {batch['dna_tokens'].shape}")
    print(f"RNA tokens shape:     {batch['rna_tokens'].shape}")
    print(f"Protein tokens shape: {batch['protein_tokens'].shape}")
    print()

    # Batch details
    print("[Batch Details]")
    for i in range(len(batch['dna'])):
        print(f"Sample {i+1}:")
        print(f"  DNA: {batch['dna'][i][:20]}...")
        print(f"  Protein: {batch['protein'][i]}")
