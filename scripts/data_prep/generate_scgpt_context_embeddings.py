#!/usr/bin/env python3
"""
scGPT Context-Dependent Gene Embeddings Generator

Generates context-dependent gene embeddings using K562 pseudo-bulk
expression profile as context.

Difference from current approach (token embeddings only):
- Token embeddings: generated from gene name only, no context
- Context-dependent: considers entire expression profile, reflects inter-gene relationships

Output format: [n_genes, 512] - same shape as current, can be used directly in CDT

Usage:
    conda activate scgpt2
    python scripts/data_prep/generate_scgpt_context_embeddings.py
"""

import os
import sys
import json
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# scGPT imports
from scgpt.model import TransformerModel
from scgpt.tokenizer import GeneVocab

# Paths
MODEL_DIR = PROJECT_ROOT / "models/scgpt"
DATA_DIR = PROJECT_ROOT / "data"
MORRIS_DATA = DATA_DIR / "processed/morris/stingseq_v2.h5"
ALIGNED_GENES_PATH = DATA_DIR / "processed/embeddings/k562_gene_embeddings_aligned.h5"
OUTPUT_DIR = DATA_DIR / "processed/embeddings"


def load_k562_pseudobulk(morris_path: Path, max_cells: int = None) -> tuple:
    """
    Create K562 pseudo-bulk expression profile

    Returns:
        pseudobulk: [n_genes] average expression across cells
        gene_names: list of gene names
    """
    print(f"Loading K562 data from {morris_path}")

    with h5py.File(morris_path, 'r') as f:
        # Sparse matrix reconstruction
        data = f['X_data'][:]
        indices = f['X_indices'][:]
        indptr = f['X_indptr'][:]

        n_cells = len(f['cell_barcodes'])
        n_genes = len(f['gene_names'])

        if max_cells and max_cells < n_cells:
            # Limit cells for testing
            indptr = indptr[:max_cells + 1]
            end_idx = indptr[-1]
            data = data[:end_idx]
            indices = indices[:end_idx]
            n_cells = max_cells

        expression_matrix = sp.csr_matrix(
            (data, indices, indptr),
            shape=(n_cells, n_genes)
        )

        gene_names = [g.decode() if isinstance(g, bytes) else g
                      for g in f['gene_names'][:]]

    print(f"  Cells: {n_cells}")
    print(f"  Genes: {n_genes}")

    # Create pseudo-bulk (mean across cells)
    print("  Computing pseudo-bulk (mean expression)...")
    pseudobulk = np.array(expression_matrix.mean(axis=0)).flatten()

    # Log-transform and normalize
    pseudobulk = np.log1p(pseudobulk * 10000 / pseudobulk.sum())  # CPM + log1p

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
        use_fast_transformer=False,  # Use PyTorch transformer for compatibility
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
    batch_size: int = 500,
) -> tuple:
    """
    Generate context-dependent gene embeddings

    Args:
        model: scGPT TransformerModel
        vocab: GeneVocab
        pseudobulk: [n_all_genes] expression values
        all_gene_names: list of all gene names in expression data
        target_genes: list of genes to output embeddings for (aligned 2360)
        device: torch device
        batch_size: genes per batch

    Returns:
        embeddings: [n_target_genes, 512]
        valid_genes: list of genes with valid embeddings
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

    # Process in batches to fit in memory
    n_genes = len(expressed_genes)
    gene_ids = [gene_to_vocab[g] for g in expressed_genes]
    values = np.array(expressed_values)

    # Normalize values to reasonable range for scGPT
    values = (values - values.mean()) / (values.std() + 1e-8)
    values = np.clip(values, -3, 3)  # Clip to reasonable range

    print(f"  Processing {n_genes} genes in context...")

    # We need to process all genes together to get proper context
    # But scGPT has a max sequence length, so we'll use top expressed genes as context
    max_context = 3000  # scGPT typically handles up to ~3000 genes

    if n_genes > max_context:
        # Select top expressed genes for context
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
        # Token embeddings
        token_emb = model.encoder(gene_ids_tensor)  # [1, n_context, 512]

        # Value embeddings
        value_emb = model.value_encoder(values_tensor)  # [1, n_context, 512]

        # Combined input
        combined = token_emb + value_emb

        # Pass through transformer
        src_key_padding_mask = torch.zeros(gene_ids_tensor.shape, dtype=torch.bool, device=device)
        output = model.transformer_encoder(
            combined.transpose(0, 1),
            src_key_padding_mask=src_key_padding_mask.T
        )
        output = output.transpose(0, 1)  # [1, n_context, 512]

        context_embeddings = output.squeeze(0).cpu().numpy()  # [n_context, 512]

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
            # Gene in vocab but not in context (low expression)
            # Use token embedding as fallback
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

    # Normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    return embeddings, valid_target_genes


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate scGPT context-dependent embeddings")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu, cuda, mps)")
    parser.add_argument("--max-cells", type=int, default=None,
                        help="Max cells to use for pseudo-bulk (for testing)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path")
    args = parser.parse_args()

    # Set device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = "mps"
    elif args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    print("=" * 60)

    # Load K562 pseudo-bulk
    pseudobulk, all_gene_names = load_k562_pseudobulk(MORRIS_DATA, args.max_cells)

    # Load aligned gene list
    target_genes = load_aligned_genes(ALIGNED_GENES_PATH)

    # Load scGPT model
    model, vocab, config = load_scgpt_model(MODEL_DIR, device)

    # Generate context-dependent embeddings
    embeddings, valid_genes = generate_context_embeddings(
        model, vocab, pseudobulk, all_gene_names, target_genes, device
    )

    # Check alignment with original
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
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "k562_gene_embeddings_context.h5"

    print(f"\nSaving to {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=embeddings.astype(np.float32), compression='gzip')
        f.create_dataset('gene_names', data=np.array(valid_genes, dtype='S'))
        f.attrs['n_genes'] = len(valid_genes)
        f.attrs['embedding_dim'] = embeddings.shape[1]
        f.attrs['model'] = 'scGPT-whole-human'
        f.attrs['description'] = 'Context-dependent gene embeddings from scGPT using K562 pseudo-bulk'
        f.attrs['context_type'] = 'K562_pseudobulk'

    print(f"\nDone!")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Genes: {len(valid_genes)}")
    print(f"  Output: {output_path}")

    # Compare with original token embeddings
    print("\n" + "=" * 60)
    print("Comparison with original token embeddings:")

    orig_path = ALIGNED_GENES_PATH
    with h5py.File(orig_path, 'r') as f:
        orig_embeddings = f['embeddings'][:]

    # Compute similarity
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
