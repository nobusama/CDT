"""
Synthetic Data Generator

Generates DNA -> RNA -> Protein synthetic data
For Phase 1 POC (Proof of Concept)
"""

import random
from typing import List, Dict


class SyntheticDataGenerator:
    """
    Generate synthetic data with biologically correct DNA->RNA->Protein relationships

    Follows the Central Dogma:
    1. Generate random DNA sequences
    2. Transcribe DNA -> RNA (T -> U)
    3. Translate RNA -> Protein (using codon table)

    Example:
        >>> generator = SyntheticDataGenerator(seed=42)
        >>> data = generator.generate(num_samples=10, min_length=30, max_length=60)
        >>> print(data[0])
        {'dna': 'ATGCGA...', 'rna': 'AUGCGA...', 'protein': 'MR...'}
    """

    def __init__(self, seed: int = None):
        """
        Initialize the data generator

        Args:
            seed: Random seed (for reproducibility)
        """
        if seed is not None:
            random.seed(seed)

        # Genetic code table (standard codon table)
        # RNA sequence (3 nucleotides) -> Amino acid (1 letter)
        self.codon_table = {
            # Phenylalanine
            'UUU': 'F', 'UUC': 'F',
            # Leucine
            'UUA': 'L', 'UUG': 'L', 'CUU': 'L', 'CUC': 'L', 'CUA': 'L', 'CUG': 'L',
            # Isoleucine
            'AUU': 'I', 'AUC': 'I', 'AUA': 'I',
            # Methionine (start codon)
            'AUG': 'M',
            # Valine
            'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V',
            # Serine
            'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'AGU': 'S', 'AGC': 'S',
            # Proline
            'CCU': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            # Threonine
            'ACU': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            # Alanine
            'GCU': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            # Tyrosine
            'UAU': 'Y', 'UAC': 'Y',
            # Histidine
            'CAU': 'H', 'CAC': 'H',
            # Glutamine
            'CAA': 'Q', 'CAG': 'Q',
            # Asparagine
            'AAU': 'N', 'AAC': 'N',
            # Lysine
            'AAA': 'K', 'AAG': 'K',
            # Aspartic acid
            'GAU': 'D', 'GAC': 'D',
            # Glutamic acid
            'GAA': 'E', 'GAG': 'E',
            # Cysteine
            'UGU': 'C', 'UGC': 'C',
            # Tryptophan
            'UGG': 'W',
            # Arginine
            'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R', 'AGA': 'R', 'AGG': 'R',
            # Glycine
            'GGU': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
            # Stop codons
            'UAA': '*', 'UAG': '*', 'UGA': '*',
        }

        # DNA bases
        self.dna_bases = ['A', 'T', 'C', 'G']

    def generate_dna(self, length: int) -> str:
        """
        Generate a random DNA sequence

        Args:
            length: DNA sequence length (in nucleotides)

        Returns:
            str: DNA sequence (always starts with ATG, length is a multiple of 3)
        """
        # Start with start codon (ATG)
        dna = 'ATG'

        # Adjust remaining length to be a multiple of 3
        remaining = length - 3
        remaining = (remaining // 3) * 3

        # Add random bases
        dna += ''.join(random.choices(self.dna_bases, k=remaining))

        return dna

    def transcribe(self, dna: str) -> str:
        """
        Transcribe DNA sequence to RNA sequence

        Args:
            dna: DNA sequence

        Returns:
            str: RNA sequence (T -> U replacement)
        """
        # Replace T (Thymine) with U (Uracil)
        return dna.replace('T', 'U')

    def translate(self, rna: str) -> str:
        """
        Translate RNA sequence to protein sequence

        Args:
            rna: RNA sequence

        Returns:
            str: Protein sequence (amino acids in 1-letter notation)
        """
        protein = ""

        # Read 3 bases at a time (codon)
        for i in range(0, len(rna) - 2, 3):
            codon = rna[i:i+3]

            # Convert using codon table
            if codon in self.codon_table:
                amino_acid = self.codon_table[codon]

                # Stop translation at stop codon
                if amino_acid == '*':
                    break

                protein += amino_acid
            else:
                # Treat unknown codon as 'X'
                protein += 'X'

        return protein

    def generate_sample(self, length: int) -> Dict[str, str]:
        """
        Generate one sample (DNA, RNA, Protein)

        Args:
            length: DNA sequence length

        Returns:
            dict: {'dna': str, 'rna': str, 'protein': str}
        """
        # Generate DNA sequence
        dna = self.generate_dna(length)

        # Transcribe DNA -> RNA
        rna = self.transcribe(dna)

        # Translate RNA -> Protein
        protein = self.translate(rna)

        return {
            'dna': dna,
            'rna': rna,
            'protein': protein
        }

    def generate(
        self,
        num_samples: int = 100,
        min_length: int = 30,
        max_length: int = 300
    ) -> List[Dict[str, str]]:
        """
        Generate multiple samples

        Args:
            num_samples: Number of samples to generate
            min_length: Minimum DNA sequence length (in nucleotides)
            max_length: Maximum DNA sequence length (in nucleotides)

        Returns:
            list: List of samples
        """
        samples = []

        for _ in range(num_samples):
            # Select random length (will be adjusted to multiple of 3)
            length = random.randint(min_length, max_length)

            # Generate sample
            sample = self.generate_sample(length)
            samples.append(sample)

        return samples


# Usage example (runs only when this file is executed directly)
if __name__ == "__main__":
    # Create data generator instance
    generator = SyntheticDataGenerator(seed=42)

    print("=" * 60)
    print("Synthetic Data Generator Test")
    print("=" * 60)
    print()

    # Generate 1 sample
    print("[Single Sample Generation Example]")
    sample = generator.generate_sample(length=30)
    print(f"DNA:     {sample['dna']}")
    print(f"RNA:     {sample['rna']}")
    print(f"Protein: {sample['protein']}")
    print()

    # Check sequence lengths
    print("[Length Verification]")
    print(f"DNA length:     {len(sample['dna'])} nucleotides")
    print(f"RNA length:     {len(sample['rna'])} nucleotides")
    print(f"Protein length: {len(sample['protein'])} amino acids")
    print(f"Ratio:          DNA/RNA : Protein = 3 : 1")
    print()

    # Generate multiple samples
    print("[Generate 10 Samples]")
    samples = generator.generate(num_samples=10, min_length=30, max_length=60)
    print(f"Number of samples generated: {len(samples)}")
    print()

    # Display first 3 samples
    print("[First 3 Samples]")
    for i, sample in enumerate(samples[:3]):
        print(f"Sample {i+1}:")
        print(f"  DNA:     {sample['dna'][:30]}... (total {len(sample['dna'])} nucleotides)")
        print(f"  RNA:     {sample['rna'][:30]}... (total {len(sample['rna'])} nucleotides)")
        print(f"  Protein: {sample['protein']} (total {len(sample['protein'])} amino acids)")
        print()

    # Statistics
    print("[Statistics]")
    dna_lengths = [len(s['dna']) for s in samples]
    protein_lengths = [len(s['protein']) for s in samples]
    print(f"DNA length range: {min(dna_lengths)} - {max(dna_lengths)} nucleotides")
    print(f"Protein length range: {min(protein_lengths)} - {max(protein_lengths)} amino acids")
    print(f"Average DNA length: {sum(dna_lengths)/len(dna_lengths):.1f} nucleotides")
    print(f"Average Protein length: {sum(protein_lengths)/len(protein_lengths):.1f} amino acids")
