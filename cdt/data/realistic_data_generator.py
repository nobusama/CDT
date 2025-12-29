"""
Realistic Data Generator for CDT Phase 5

Generates biologically realistic synthetic DNA-RNA-Protein triplets
following the Central Dogma rules:
1. DNA sequences with gene structure (promoter, start/stop codons)
2. Transcription: DNA → RNA (T → U)
3. Translation: RNA → Protein (genetic code)
"""

import random
import json
from typing import Dict, List, Tuple
from pathlib import Path


# Standard Genetic Code
GENETIC_CODE = {
    'AUG': 'M', 'UUU': 'F', 'UUC': 'F', 'UUA': 'L', 'UUG': 'L',
    'UCU': 'S', 'UCC': 'S', 'UCA': 'S', 'UCG': 'S', 'UAU': 'Y',
    'UAC': 'Y', 'UGU': 'C', 'UGC': 'C', 'UGG': 'W', 'CUU': 'L',
    'CUC': 'L', 'CUA': 'L', 'CUG': 'L', 'CCU': 'P', 'CCC': 'P',
    'CCA': 'P', 'CCG': 'P', 'CAU': 'H', 'CAC': 'H', 'CAA': 'Q',
    'CAG': 'Q', 'CGU': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'AUU': 'I', 'AUC': 'I', 'AUA': 'I', 'ACU': 'T', 'ACC': 'T',
    'ACA': 'T', 'ACG': 'T', 'AAU': 'N', 'AAC': 'N', 'AAA': 'K',
    'AAG': 'K', 'AGU': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GUU': 'V', 'GUC': 'V', 'GUA': 'V', 'GUG': 'V', 'GCU': 'A',
    'GCC': 'A', 'GCA': 'A', 'GCG': 'A', 'GAU': 'D', 'GAC': 'D',
    'GAA': 'E', 'GAG': 'E', 'GGU': 'G', 'GGC': 'G', 'GGA': 'G',
    'GGG': 'G',
    'UAA': '*', 'UAG': '*', 'UGA': '*',  # Stop codons
}

# DNA bases
DNA_BASES = ['A', 'T', 'G', 'C']
RNA_BASES = ['A', 'U', 'G', 'C']

# Common gene templates (simplified housekeeping-like genes)
GENE_TEMPLATES = [
    {
        'name': 'synthetic_gene_1',
        'function': 'Metabolic enzyme (like)',
        'typical_length': 90,  # 30 amino acids
    },
    {
        'name': 'synthetic_gene_2',
        'function': 'Ribosomal protein (like)',
        'typical_length': 60,  # 20 amino acids
    },
    {
        'name': 'synthetic_gene_3',
        'function': 'Translation factor (like)',
        'typical_length': 120,  # 40 amino acids
    },
    {
        'name': 'synthetic_gene_4',
        'function': 'Chaperone protein (like)',
        'typical_length': 75,  # 25 amino acids
    },
    {
        'name': 'synthetic_gene_5',
        'function': 'DNA binding protein (like)',
        'typical_length': 105,  # 35 amino acids
    },
    {
        'name': 'synthetic_gene_6',
        'function': 'Transport protein (like)',
        'typical_length': 90,  # 30 amino acids
    },
    {
        'name': 'synthetic_gene_7',
        'function': 'Kinase (like)',
        'typical_length': 120,  # 40 amino acids
    },
    {
        'name': 'synthetic_gene_8',
        'function': 'Transcription factor (like)',
        'typical_length': 75,  # 25 amino acids
    },
    {
        'name': 'synthetic_gene_9',
        'function': 'Cell cycle regulator (like)',
        'typical_length': 60,  # 20 amino acids
    },
    {
        'name': 'synthetic_gene_10',
        'function': 'Membrane protein (like)',
        'typical_length': 105,  # 35 amino acids
    },
]


class RealisticDataGenerator:
    """Generate biologically realistic synthetic data"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.genetic_code = GENETIC_CODE

    def generate_coding_sequence(self, num_codons: int) -> str:
        """
        Generate a random DNA coding sequence

        Args:
            num_codons: Number of codons (excludes stop codon)

        Returns:
            DNA sequence starting with ATG and ending with stop codon
        """
        # Start codon
        dna_seq = "ATG"

        # Generate random codons (excluding stop codons)
        non_stop_codons = [codon for codon in self.genetic_code.keys()
                          if self.genetic_code[codon] != '*']

        for _ in range(num_codons - 1):  # -1 because ATG is already added
            # Convert RNA codon to DNA
            rna_codon = self.rng.choice(non_stop_codons)
            dna_codon = rna_codon.replace('U', 'T')
            dna_seq += dna_codon

        # Add stop codon
        stop_codon = self.rng.choice(['TAA', 'TAG', 'TGA'])
        dna_seq += stop_codon

        return dna_seq

    def transcribe(self, dna_seq: str) -> str:
        """
        Transcribe DNA to RNA

        Args:
            dna_seq: DNA sequence

        Returns:
            RNA sequence (T → U)
        """
        return dna_seq.replace('T', 'U')

    def translate(self, rna_seq: str) -> str:
        """
        Translate RNA to Protein

        Args:
            rna_seq: RNA sequence

        Returns:
            Protein sequence (amino acids)
        """
        protein = []

        # Translate in triplets
        for i in range(0, len(rna_seq) - 2, 3):
            codon = rna_seq[i:i+3]
            if len(codon) == 3:
                aa = self.genetic_code.get(codon, 'X')  # X for unknown
                if aa == '*':  # Stop codon
                    break
                protein.append(aa)

        return ''.join(protein)

    def add_promoter_region(self, seq: str) -> str:
        """Add a simple promoter-like region (random DNA)"""
        promoter_length = self.rng.randint(10, 20)
        promoter = ''.join(self.rng.choices(DNA_BASES, k=promoter_length))
        return promoter + seq

    def introduce_mutation(self, seq: str, mutation_rate: float = 0.01) -> str:
        """
        Introduce random point mutations

        Args:
            seq: DNA/RNA sequence
            mutation_rate: Probability of mutation per base

        Returns:
            Mutated sequence
        """
        seq_list = list(seq)
        bases = DNA_BASES if 'T' in seq else RNA_BASES

        for i in range(len(seq_list)):
            if self.rng.random() < mutation_rate:
                # Exclude the current base
                other_bases = [b for b in bases if b != seq_list[i]]
                seq_list[i] = self.rng.choice(other_bases)

        return ''.join(seq_list)

    def generate_triplet(
        self,
        gene_template: Dict,
        add_promoter: bool = True,
        add_mutations: bool = False
    ) -> Dict:
        """
        Generate a single DNA-RNA-Protein triplet

        Args:
            gene_template: Gene information (name, length, etc.)
            add_promoter: Whether to add promoter region
            add_mutations: Whether to introduce mutations

        Returns:
            Dictionary with DNA, RNA, Protein sequences and metadata
        """
        # Calculate number of codons from typical length
        coding_length = gene_template['typical_length']
        num_codons = coding_length // 3

        # Generate coding sequence
        coding_dna = self.generate_coding_sequence(num_codons)

        # Optionally add promoter
        if add_promoter:
            full_dna = self.add_promoter_region(coding_dna)
            promoter_length = len(full_dna) - len(coding_dna)
        else:
            full_dna = coding_dna
            promoter_length = 0

        # Optionally add mutations
        if add_mutations:
            full_dna = self.introduce_mutation(full_dna, mutation_rate=0.005)

        # Transcribe and translate
        full_rna = self.transcribe(full_dna)
        coding_rna = full_rna[promoter_length:]  # Remove promoter region
        protein = self.translate(coding_rna)

        # Create metadata
        return {
            'gene_name': gene_template['name'],
            'function': gene_template['function'],
            'dna_sequence': full_dna,
            'rna_sequence': full_rna,
            'protein_sequence': protein,
            'annotations': {
                'promoter_length': promoter_length,
                'coding_start': promoter_length,
                'coding_end': len(full_dna) - 3,  # Before stop codon
                'start_codon_pos': promoter_length,
                'num_codons': num_codons,
                'has_mutations': add_mutations,
            }
        }

    def generate_dataset(
        self,
        num_genes: int = 10,
        variants_per_gene: int = 10,
        add_promoter: bool = True,
        mutation_rate: float = 0.5,
    ) -> List[Dict]:
        """
        Generate a full dataset

        Args:
            num_genes: Number of genes to generate
            variants_per_gene: Number of variants per gene
            add_promoter: Whether to add promoter regions
            mutation_rate: Fraction of variants with mutations

        Returns:
            List of triplets
        """
        dataset = []
        gene_templates = GENE_TEMPLATES[:num_genes]

        for gene_idx, template in enumerate(gene_templates):
            for variant_idx in range(variants_per_gene):
                # Add mutations to some variants
                add_mutations = (self.rng.random() < mutation_rate)

                triplet = self.generate_triplet(
                    template,
                    add_promoter=add_promoter,
                    add_mutations=add_mutations
                )

                # Add sample metadata
                triplet['gene_id'] = gene_idx
                triplet['variant_id'] = variant_idx
                triplet['sample_id'] = gene_idx * variants_per_gene + variant_idx

                dataset.append(triplet)

        return dataset


def save_dataset(dataset: List[Dict], output_dir: str = "data/phase5"):
    """Save dataset to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save full dataset
    dataset_file = output_path / "synthetic_realistic_triplets.json"
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Saved {len(dataset)} samples to {dataset_file}")

    # Create gene annotations summary
    gene_annotations = {}
    for sample in dataset:
        gene_name = sample['gene_name']
        if gene_name not in gene_annotations:
            gene_annotations[gene_name] = {
                'function': sample['function'],
                'num_variants': 1,
                'typical_length': len(sample['protein_sequence']),
            }
        else:
            gene_annotations[gene_name]['num_variants'] += 1

    annotations_file = output_path / "gene_annotations.json"
    with open(annotations_file, 'w') as f:
        json.dump(gene_annotations, f, indent=2)

    print(f"Saved gene annotations to {annotations_file}")

    # Print statistics
    print("\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Unique genes: {len(gene_annotations)}")
    print(f"  Avg DNA length: {sum(len(s['dna_sequence']) for s in dataset) / len(dataset):.1f}")
    print(f"  Avg RNA length: {sum(len(s['rna_sequence']) for s in dataset) / len(dataset):.1f}")
    print(f"  Avg Protein length: {sum(len(s['protein_sequence']) for s in dataset) / len(dataset):.1f}")
    print(f"  Samples with mutations: {sum(s['annotations']['has_mutations'] for s in dataset)}")


def main():
    """Generate Phase 5 dataset"""
    print("=" * 80)
    print("CDT Phase 5: Realistic Data Generation")
    print("=" * 80)
    print()

    generator = RealisticDataGenerator(seed=42)

    # Generate dataset
    print("Generating dataset...")
    dataset = generator.generate_dataset(
        num_genes=10,
        variants_per_gene=10,
        add_promoter=True,
        mutation_rate=0.5,
    )

    # Show example
    print("\nExample triplet:")
    example = dataset[0]
    print(f"  Gene: {example['gene_name']}")
    print(f"  Function: {example['function']}")
    print(f"  DNA: {example['dna_sequence'][:60]}... ({len(example['dna_sequence'])} bp)")
    print(f"  RNA: {example['rna_sequence'][:60]}... ({len(example['rna_sequence'])} nt)")
    print(f"  Protein: {example['protein_sequence']} ({len(example['protein_sequence'])} aa)")
    print(f"  Promoter length: {example['annotations']['promoter_length']}")
    print(f"  Start codon pos: {example['annotations']['start_codon_pos']}")
    print()

    # Verify translation
    print("Verifying biological correctness...")
    coding_dna = example['dna_sequence'][example['annotations']['coding_start']:]
    coding_rna = coding_dna.replace('T', 'U')
    start_codon = coding_rna[:3]
    print(f"  Start codon: {start_codon} (should be AUG)")
    assert start_codon == 'AUG', "Start codon should be AUG"
    print("  ✓ Start codon correct")

    # Check first amino acid
    first_aa = example['protein_sequence'][0]
    print(f"  First amino acid: {first_aa} (should be M for Methionine)")
    assert first_aa == 'M', "First amino acid should be M"
    print("  ✓ Translation correct")
    print()

    # Save dataset
    save_dataset(dataset, output_dir="/Users/nobuyukiota/Desktop/CDT/data/phase5")

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
