#!/usr/bin/env python3
"""
scGPT Context-Dependent Gene Embeddings Generator (Gasperini version)

Uses Gasperini et al. (2019) K562 scRNA-seq data to generate
context-dependent gene embeddings.

Changes:
- Morris STINGseq to Gasperini scRNA-seq (GSE120861)
- Uses the same source as CDT training data

Usage:
    conda activate scgpt2
    python scripts/data_prep/generate_scgpt_context_embeddings_gasperini.py
"""

import os
import sys
import json
import gzip
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp
from scipy.io import mmread

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# scGPT imports
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab

# Paths
MODEL_DIR = PROJECT_ROOT / "models/scgpt"
DATA_DIR = PROJECT_ROOT / "data"

# Gasperini scRNA-seq data (GSE120861)
GASPERINI_SCRNA_DIR = DATA_DIR / "raw/gasperini/scrna"
GASPERINI_EXPRS = GASPERINI_SCRNA_DIR / "GSE120861_at_scale_screen.exprs.mtx"
GASPERINI_GENES = GASPERINI_SCRNA_DIR / "GSE120861_at_scale_screen.genes.txt.gz"
GASPERINI_CELLS = GASPERINI_SCRNA_DIR / "GSE120861_at_scale_screen.cells.txt.gz"

# Morris data for ENSG to symbol mapping
MORRIS_DATA = DATA_DIR / "processed/morris/stingseq_v2.h5"

ALIGNED_GENES_PATH = DATA_DIR / "processed/embeddings/human_proteomelm_embeddings_gasperini_aligned.h5"
OUTPUT_DIR = DATA_DIR / "processed/embeddings"


def load_ensg_to_symbol_mapping() -> dict:
    """
    Create ENSG to gene symbol mapping from Morris STINGseq data
    """
    print("Loading ENSG → symbol mapping from Morris data...")
    with h5py.File(MORRIS_DATA, 'r') as f:
        gene_ids = [g.decode() if isinstance(g, bytes) else g for g in f['gene_ids'][:]]
        gene_names = [g.decode() if isinstance(g, bytes) else g for g in f['gene_names'][:]]

    ensg_to_symbol = {ensg: symbol for ensg, symbol in zip(gene_ids, gene_names)}
    print(f"  Loaded {len(ensg_to_symbol)} ENSG → symbol mappings")
    return ensg_to_symbol


def load_gasperini_pseudobulk(ensg_to_symbol: dict, max_cells: int = None) -> tuple:
    """
    Create pseudo-bulk expression profile from Gasperini scRNA-seq

    Returns:
        pseudobulk: [n_genes] average expression across cells
        gene_names: list of gene names (symbols, not ENSG)
    """
    print("Loading Gasperini scRNA-seq data...")
    print(f"  Expression matrix: {GASPERINI_EXPRS}")

    # Load gene names (ENSG IDs)
    print("  Loading gene names...")
    with gzip.open(GASPERINI_GENES, 'rt') as f:
        ensg_ids = [line.strip() for line in f]
    print(f"    Genes (ENSG): {len(ensg_ids)}")

    # Convert ENSG to symbols
    gene_names = []
    valid_indices = []
    for i, ensg in enumerate(ensg_ids):
        if ensg in ensg_to_symbol:
            gene_names.append(ensg_to_symbol[ensg])
            valid_indices.append(i)
        # Skip genes without symbol mapping

    print(f"    Genes with symbols: {len(gene_names)} / {len(ensg_ids)}")

    # Load cell names (for count)
    print("  Loading cell names...")
    with gzip.open(GASPERINI_CELLS, 'rt') as f:
        cell_names = [line.strip() for line in f]
    n_cells_total = len(cell_names)
    print(f"    Cells: {n_cells_total}")

    # Load expression matrix (Matrix Market format)
    print("  Loading expression matrix (this may take a few minutes)...")
    expression_matrix = mmread(GASPERINI_EXPRS).T.tocsr()  # Transpose to (cells, genes)

    print(f"    Matrix shape: {expression_matrix.shape}")

    n_cells, n_genes_total = expression_matrix.shape

    if max_cells and max_cells < n_cells:
        print(f"  Limiting to {max_cells} cells for testing...")
        expression_matrix = expression_matrix[:max_cells, :]
        n_cells = max_cells

    # Select only genes with valid symbol mapping
    print(f"  Selecting {len(valid_indices)} genes with symbol mapping...")
    expression_matrix = expression_matrix[:, valid_indices]

    # Create pseudo-bulk (mean across cells)
    print("  Computing pseudo-bulk (mean expression)...")
    pseudobulk = np.array(expression_matrix.mean(axis=0)).flatten()

    # Log-transform and normalize (CPM + log1p)
    total_counts = pseudobulk.sum()
    if total_counts > 0:
        pseudobulk = np.log1p(pseudobulk * 10000 / total_counts)

    print(f"  Non-zero genes: {(pseudobulk > 0).sum()}")
    print(f"  Expression range: [{pseudobulk.min():.2f}, {pseudobulk.max():.2f}]")

    return pseudobulk, gene_names


def load_aligned_genes(aligned_path: Path) -> list:
    """
    Load current aligned gene list (2360 genes)
    """
    with h5py.File(aligned_path, 'r') as f:
        gene_names = [g.decode() if isinstance(g, bytes) else g
                      for g in f['gene_names'][:]]
    print(f"Loaded {len(gene_names)} aligned genes")
    return gene_names


def load_scgpt_model(model_dir: Path, device: str = "cpu"):
    """
    Load scGPT model
    """
    print(f"Loading scGPT model from {model_dir}")

    vocab = GeneVocab.from_file(model_dir / "vocab.json")

    with open(model_dir / "args.json", 'r') as f:
        config = json.load(f)

    model = TransformerModel(
        ntoken=len(vocab),
        d_model=config['embsize'],
        nhead=config['nheads'],
        d_hid=config['d_hid'],
        nlayers=config['nlayers'],
        vocab=vocab,
        dropout=0.0,
        pad_token='<pad>',
        pad_value=-2,
        do_mvc=True,
        use_fast_transformer=False,
    )

    state_dict = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"  Model loaded on {device}")
    print(f"  Vocabulary size: {len(vocab)}")
    print(f"  Embedding dim: {config['embsize']}")

    return model, vocab, config


def generate_context_embeddings(
    model,
    vocab,
    pseudobulk: np.ndarray,
    all_gene_names: list,
    target_genes: list,
    device: str = "cpu",
) -> tuple:
    """
    Generate context-dependent gene embeddings
    """
    print(f"\nGenerating context-dependent embeddings...")

    # Find genes in vocabulary
    gene_to_idx = {g: i for i, g in enumerate(all_gene_names)}
    gene_to_vocab = {}
    expressed_genes = []
    expressed_values = []

    for i, gene in enumerate(all_gene_names):
        if gene in vocab and pseudobulk[i] > 0:
            gene_to_vocab[gene] = vocab[gene]
            expressed_genes.append(gene)
            expressed_values.append(pseudobulk[i])

    print(f"  Expressed genes in vocab: {len(expressed_genes)}")

    n_genes = len(expressed_genes)
    gene_ids = [gene_to_vocab[g] for g in expressed_genes]
    values = np.array(expressed_values)

    # Normalize values
    values = (values - values.mean()) / (values.std() + 1e-8)
    values = np.clip(values, -3, 3)

    print(f"  Processing {n_genes} genes in context...")

    # Use top expressed genes for context (scGPT max ~3000)
    max_context = 3000

    if n_genes > max_context:
        top_idx = np.argsort(expressed_values)[-max_context:]
        context_genes = [expressed_genes[i] for i in top_idx]
        context_ids = [gene_ids[i] for i in top_idx]
        context_values = values[top_idx]
    else:
        context_genes = expressed_genes
        context_ids = gene_ids
        context_values = values

    print(f"  Using {len(context_genes)} genes as context")

    # Create tensors
    gene_ids_tensor = torch.tensor([context_ids], dtype=torch.long, device=device)
    values_tensor = torch.tensor([context_values], dtype=torch.float, device=device)

    # Get context-dependent embeddings
    with torch.no_grad():
        token_emb = model.encoder(gene_ids_tensor)
        value_emb = model.value_encoder(values_tensor)
        combined = token_emb + value_emb

        src_key_padding_mask = torch.zeros(gene_ids_tensor.shape, dtype=torch.bool, device=device)
        output = model.transformer_encoder(
            combined.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask.T
        )
        output = output.transpose(0, 1)
        context_embeddings = output.squeeze(0).cpu().numpy()

    print(f"  Context embeddings shape: {context_embeddings.shape}")

    # Map to target genes
    context_gene_to_idx = {g: i for i, g in enumerate(context_genes)}

    target_embeddings = []
    valid_target_genes = []
    missing_genes = []

    for gene in target_genes:
        if gene in context_gene_to_idx:
            idx = context_gene_to_idx[gene]
            target_embeddings.append(context_embeddings[idx])
            valid_target_genes.append(gene)
        elif gene in vocab:
            # Fallback to token embedding
            gene_id = torch.tensor([[vocab[gene]]], dtype=torch.long, device=device)
            with torch.no_grad():
                emb = model.encoder(gene_id).squeeze().cpu().numpy()
            target_embeddings.append(emb)
            valid_target_genes.append(gene)
        else:
            missing_genes.append(gene)

    print(f"  Target genes with embeddings: {len(valid_target_genes)}")
    print(f"  Missing genes (not in vocab): {len(missing_genes)}")

    embeddings = np.array(target_embeddings)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    return embeddings, valid_target_genes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate scGPT context-dependent embeddings (Gasperini)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Max cells to use for pseudo-bulk (for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path")
    args = parser.parse_args()

    # Check if data exists
    if not GASPERINI_EXPRS.exists():
        print(f"ERROR: Expression matrix not found: {GASPERINI_EXPRS}")
        print("Please download from GEO (GSE120861)")
        sys.exit(1)

    # Set device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print("=" * 60)
    print("Using Gasperini scRNA-seq data (GSE120861)")
    print("=" * 60)

    # Load ENSG to symbol mapping
    ensg_to_symbol = load_ensg_to_symbol_mapping()

    # Load Gasperini pseudo-bulk
    pseudobulk, all_gene_names = load_gasperini_pseudobulk(ensg_to_symbol, args.max_cells)

    # Load aligned gene list
    target_genes = load_aligned_genes(ALIGNED_GENES_PATH)

    # Load scGPT model
    model, vocab, config = load_scgpt_model(MODEL_DIR, device)

    # Generate context-dependent embeddings
    embeddings, valid_genes = generate_context_embeddings(
        model, vocab, pseudobulk, all_gene_names, target_genes, device
    )

    # Check alignment
    if len(valid_genes) != len(target_genes):
        print(f"\nWARNING: Only {len(valid_genes)}/{len(target_genes)} genes have embeddings")
        print("Creating zero-padded output to maintain alignment...")

        target_to_idx = {g: i for i, g in enumerate(target_genes)}
        full_embeddings = np.zeros((len(target_genes), embeddings.shape[1]), dtype=np.float32)

        for i, gene in enumerate(valid_genes):
            if gene in target_to_idx:
                full_embeddings[target_to_idx[gene]] = embeddings[i]

        embeddings = full_embeddings
        valid_genes = target_genes

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "k562_gene_embeddings_gasperini_aligned.h5"

    print(f"\nSaving to {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings.astype(np.float32), compression='gzip')
        f.create_dataset('gene_names', data=np.array(valid_genes, dtype='S'))
        f.attrs['n_genes'] = len(valid_genes)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['model'] = 'scGPT-whole-human'
        f.attrs['description'] = 'Context-dependent gene embeddings from scGPT using Gasperini K562 pseudo-bulk'
        f.attrs['context_type'] = 'Gasperini_K562_pseudobulk'
        f.attrs['source'] = 'GSE120861'

    print(f"\nDone!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Genes: {len(valid_genes)}")
    print(f"  Output: {output_path}")

    # Compare with original token embeddings
    print("\n" + "=" * 60)
    print("Comparison with original token embeddings:")

    with h5py.File(ALIGNED_GENES_PATH, 'r') as f:
        orig_embeddings = f['embeddings'][:]

    cos_sims = []
    for i in range(min(100, len(valid_genes))):
        cos_sim = np.dot(orig_embeddings[i], embeddings[i]) / (
            np.linalg.norm(orig_embeddings[i]) * np.linalg.norm(embeddings[i]) + 1e-8
        )
        cos_sims.append(cos_sim)

    print(f"  Mean cosine similarity (first 100 genes): {np.mean(cos_sims):.4f}")
    print(f"  Std cosine similarity: {np.std(cos_sims):.4f}")
    print(f"  Min/Max: {np.min(cos_sims):.4f} / {np.max(cos_sims):.4f}")


if __name__ == "__main__":
    main()
