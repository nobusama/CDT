#!/usr/bin/env python3
"""
scGPT Gene-Level Embedding Generation Script

Generates gene-level embeddings [n_genes, 512] instead of pooled cell embeddings.
For CDT v3: maintains per-gene information for cross-attention.

Usage:
    conda activate scgpt2
    python scripts/data_prep/generate_scgpt_gene_embeddings.py
"""

import os
import sys
import json
import numpy as np
import scanpy as sc
import torch
import h5py
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# scGPT imports
import scgpt
from scgpt.tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

print(f"scGPT version: {scgpt.__version__}")
print(f"PyTorch version: {torch.__version__}")

# Paths
MODEL_DIR = PROJECT_ROOT / "models/scgpt"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/embeddings"


def load_scgpt_model(model_dir: Path, device: str = "cpu"):
    """Load scGPT model using the official approach."""
    from scgpt.model import TransformerModel

    # Load config
    with open(model_dir / "args.json", "r") as f:
        model_configs = json.load(f)

    # Load vocabulary
    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)

    # Check special tokens
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Get model parameters
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    print(f"Model config: embsize={embsize}, nhead={nhead}, nlayers={nlayers}")

    # Load model with strict=False to handle architecture differences
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=embsize,
        nhead=nhead,
        d_hid=d_hid,
        nlayers=nlayers,
        vocab=vocab,
        dropout=model_configs.get("dropout", 0.2),
        pad_token="<pad>",
        pad_value=model_configs.get("pad_value", -2),
        do_mvc=True,  # Match checkpoint
        do_dab=False,
        use_batch_labels=False,
        explicit_zero_prob=False,
        use_fast_transformer=True,  # Use flash attention compatible version
    )

    # Load weights with strict=False
    state_dict = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, vocab, model_configs


def get_gene_embeddings_simple(
    adata,
    model_dir: Path,
    gene_col: str = "gene_name",
    device: str = "cpu",
):
    """
    Extract gene-level embeddings using a simplified approach.
    Uses the existing embed_data infrastructure but extracts transformer output.
    """
    from scgpt.tasks.cell_emb import get_batch_cell_embeddings
    import json

    # Load config
    with open(model_dir / "args.json", "r") as f:
        model_configs = json.load(f)

    vocab_file = model_dir / "vocab.json"
    vocab = GeneVocab.from_file(vocab_file)

    # Add special tokens if needed
    special_tokens = ["<pad>", "<cls>", "<eoc>"]
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Get gene names in vocabulary
    gene_names = adata.var[gene_col].tolist()
    genes_in_vocab = [g for g in gene_names if g in vocab]

    print(f"Genes in data: {len(gene_names)}")
    print(f"Genes in vocab: {len(genes_in_vocab)}")

    # Get gene embeddings directly from vocabulary
    # scGPT uses learned gene embeddings as part of the model
    # We can extract these from the encoder

    from scgpt.model import TransformerModel

    # Create model to extract embeddings
    model = TransformerModel(
        ntoken=len(vocab),
        d_model=model_configs["embsize"],
        nhead=model_configs["nheads"],
        d_hid=model_configs["d_hid"],
        nlayers=model_configs["nlayers"],
        vocab=vocab,
        dropout=0.0,  # No dropout for inference
        pad_token="<pad>",
        pad_value=model_configs.get("pad_value", -2),
        do_mvc=True,
        use_fast_transformer=True,
    )

    # Load with strict=False
    state_dict = torch.load(model_dir / "best_model.pt", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Get the gene token embeddings from the encoder
    # These are the base embeddings before any expression-dependent processing
    gene_embeddings_list = []
    valid_gene_names = []

    for gene in tqdm(genes_in_vocab[:3000], desc="Extracting gene embeddings"):  # Limit to 3000 for memory
        if gene in vocab:
            gene_id = vocab[gene]
            # Get embedding from the encoder
            gene_tensor = torch.tensor([[gene_id]], dtype=torch.long).to(device)
            with torch.no_grad():
                # Get the token embedding
                emb = model.encoder(gene_tensor)  # (1, 1, embsize)
                gene_embeddings_list.append(emb[0, 0].cpu().numpy())
                valid_gene_names.append(gene)

    gene_embeddings = np.array(gene_embeddings_list)

    # Normalize
    gene_embeddings = gene_embeddings / (np.linalg.norm(gene_embeddings, axis=1, keepdims=True) + 1e-8)

    print(f"Gene embeddings shape: {gene_embeddings.shape}")

    return gene_embeddings, valid_gene_names


def main():
    set_seed(42)

    # Check for K562 data
    k562_path = PROJECT_ROOT / "data/processed/morris/stingseq_v1.h5ad"

    if not k562_path.exists():
        print(f"K562 data not found at {k562_path}")
        return

    print(f"Loading K562 data from {k562_path}")
    adata = sc.read_h5ad(k562_path)
    print(f"Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Set gene name column
    if 'gene_name' not in adata.var.columns:
        adata.var['gene_name'] = adata.var_names

    # Get gene embeddings
    print("\nExtracting gene-level embeddings...")
    gene_embeddings, gene_names = get_gene_embeddings_simple(
        adata,
        MODEL_DIR,
        gene_col='gene_name',
        device='cpu',
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "k562_gene_embeddings.h5"

    print(f"\nSaving to {output_path}")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('embeddings', data=gene_embeddings, compression='gzip')
        f.create_dataset('gene_names', data=np.array(gene_names, dtype='S'))
        f.attrs['n_genes'] = len(gene_names)
        f.attrs['embed_dim'] = gene_embeddings.shape[1]
        f.attrs['model'] = 'scGPT-whole-human'
        f.attrs['description'] = 'Gene token embeddings from scGPT for K562'

    print(f"\nDone!")
    print(f"  Shape: {gene_embeddings.shape}")
    print(f"  Genes: {len(gene_names)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
