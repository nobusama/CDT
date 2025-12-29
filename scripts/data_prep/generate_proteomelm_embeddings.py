#!/usr/bin/env python3
"""
ProteomeLM Human Proteome Embedding Generation
(Using build_genome_esmc for proper ESM-C embeddings with PPI information)

Usage:
    python scripts/data_prep/generate_proteomelm_embeddings.py

Output:
    data/processed/embeddings/human_proteomelm_embeddings.h5
"""

import os
import sys
import h5py
import torch
import numpy as np
from tqdm import tqdm
import requests
from io import StringIO
from Bio import SeqIO

# Configuration
MODEL_NAME = "Bitbol-Lab/ProteomeLM-M"  # 112M params, hidden_size=768
OUTPUT_FILE = "data/processed/embeddings/human_proteomelm_embeddings.h5"
FASTA_CACHE = "data/raw/proteome/human_proteome.fasta"


def check_device():
    """Check available device"""
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return "cuda:0"
    elif torch.backends.mps.is_available():
        print("Using Apple MPS (Metal)")
        return "mps"
    else:
        print("Using CPU (this will be slow)")
        return "cpu"


def download_human_proteome():
    """Download human proteome (Swiss-Prot reviewed) from UniProt"""

    # Use cache if available
    if os.path.exists(FASTA_CACHE):
        print(f"Using cached FASTA: {FASTA_CACHE}")
        proteins = {}
        for record in SeqIO.parse(FASTA_CACHE, "fasta"):
            parts = record.id.split("|")
            if len(parts) >= 2:
                uniprot_id = parts[1]
            else:
                uniprot_id = record.id

            gene_name = ""
            if "GN=" in record.description:
                gene_name = record.description.split("GN=")[1].split()[0]

            proteins[uniprot_id] = {
                "sequence": str(record.seq),
                "gene_name": gene_name,
                "description": record.description
            }
        print(f"Loaded: {len(proteins)} proteins")
        return proteins

    print("Downloading human proteome from UniProt...")

    url = "https://rest.uniprot.org/uniprotkb/stream"
    params = {
        "format": "fasta",
        "query": "(organism_id:9606) AND (reviewed:true)",
        "compressed": "false"
    }

    response = requests.get(url, params=params, stream=True)
    response.raise_for_status()

    fasta_content = response.text
    proteins = {}

    for record in SeqIO.parse(StringIO(fasta_content), "fasta"):
        parts = record.id.split("|")
        if len(parts) >= 2:
            uniprot_id = parts[1]
        else:
            uniprot_id = record.id

        gene_name = ""
        if "GN=" in record.description:
            gene_name = record.description.split("GN=")[1].split()[0]

        proteins[uniprot_id] = {
            "sequence": str(record.seq),
            "gene_name": gene_name,
            "description": record.description
        }

    # Save to cache
    os.makedirs(os.path.dirname(FASTA_CACHE), exist_ok=True)
    with open(FASTA_CACHE, 'w') as f:
        f.write(fasta_content)

    print(f"Downloaded: {len(proteins)} proteins")
    print(f"Cached to: {FASTA_CACHE}")
    return proteins


def main():
    print("=" * 60)
    print("ProteomeLM Human Proteome Embedding Generation")
    print("(Using build_genome_esmc for proper PPI information)")
    print("=" * 60)

    device = check_device()
    torch_device = torch.device(device.replace(":0", "") if "cuda" in device else device)

    # Download/load human proteome
    print("\n[1/4] Loading human proteome...")
    proteins = download_human_proteome()

    # Prepare FASTA file
    fasta_path = "/tmp/human_proteome_for_esmc.fasta"
    print(f"\nPreparing FASTA file: {fasta_path}")
    with open(fasta_path, 'w') as f:
        for pid, pdata in proteins.items():
            f.write(f">sp|{pid}|{pdata['gene_name']}\n")
            f.write(f"{pdata['sequence']}\n")
    print(f"Wrote {len(proteins)} sequences")

    # Compute ESM-C embeddings
    # Note: Use GPU if available, otherwise CPU
    esmc_device = device if "cuda" in device else "cpu"
    print(f"\n[2/4] Computing ESM-C embeddings...")
    print(f"ESM-C device: {esmc_device}")
    if esmc_device == "cpu":
        print("This will take a while on CPU...")

    from proteomelm import build_genome_esmc
    esmc_result = build_genome_esmc(fasta_path, device=esmc_device)

    # Return value structure:
    # - inputs_embeds: Tensor [n_proteins, dim]
    # - group_embeds: Tensor [n_proteins, dim]
    # - group_labels: List[str] (protein IDs from FASTA)
    inputs_embeds = esmc_result["inputs_embeds"]
    group_embeds = esmc_result["group_embeds"]
    group_labels = esmc_result["group_labels"]

    n_proteins = len(group_labels)
    esmc_dim = inputs_embeds.shape[1] if hasattr(inputs_embeds, 'shape') else inputs_embeds.size(1)
    print(f"ESM-C embeddings computed: {n_proteins} proteins")
    print(f"ESM-C embedding dimension: {esmc_dim}")
    print(f"inputs_embeds shape: {inputs_embeds.shape}")
    print(f"group_embeds shape: {group_embeds.shape}")

    # Map protein IDs to gene names
    valid_protein_ids = []
    valid_gene_names = []
    for label in group_labels:
        # Extract UniProt ID from FASTA header (sp|XXXXX|GENE format)
        parts = label.split("|")
        if len(parts) >= 2:
            uniprot_id = parts[1]
        else:
            uniprot_id = label

        valid_protein_ids.append(uniprot_id)
        gene_name = proteins.get(uniprot_id, {}).get("gene_name", "")
        valid_gene_names.append(gene_name)

    print(f"Matched protein IDs: {len(valid_protein_ids)}")

    # Load ProteomeLM model
    print(f"\n[3/4] Loading ProteomeLM: {MODEL_NAME}")
    from proteomelm import ProteomeLMForMaskedLM
    model = ProteomeLMForMaskedLM.from_pretrained(MODEL_NAME)
    model = model.to(torch_device)
    model.eval()
    print(f"ProteomeLM hidden size: {model.config.hidden_size}")

    # Move tensors to device
    if isinstance(inputs_embeds, np.ndarray):
        inputs_embeds = torch.from_numpy(inputs_embeds)
    if isinstance(group_embeds, np.ndarray):
        group_embeds = torch.from_numpy(group_embeds)

    inputs_embeds = inputs_embeds.to(torch_device)
    group_embeds = group_embeds.to(torch_device)

    # Process with ProteomeLM (batch processing)
    print(f"\n[4/4] Processing with ProteomeLM...")
    print(f"Input: inputs_embeds {inputs_embeds.shape}, group_embeds {group_embeds.shape}")
    batch_size = 1000

    all_embeddings = []

    for i in tqdm(range(0, n_proteins, batch_size), desc="ProteomeLM"):
        batch_inputs = inputs_embeds[i:i+batch_size].unsqueeze(0)
        batch_groups = group_embeds[i:i+batch_size].unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=batch_inputs,
                group_embeds=batch_groups,  # Add ortholog group embeddings!
                output_hidden_states=True,
                output_attentions=True
            )
            hidden = outputs.hidden_states[-1].squeeze(0)
            all_embeddings.append(hidden.cpu())

    proteomelm_embeddings = torch.cat(all_embeddings, dim=0)
    print(f"ProteomeLM embeddings: {proteomelm_embeddings.shape}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    proteomelm_np = proteomelm_embeddings.numpy()

    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset('embeddings', data=proteomelm_np, compression='gzip')

        dt = h5py.special_dtype(vlen=str)
        f.create_dataset('uniprot_ids', data=valid_protein_ids, dtype=dt)
        f.create_dataset('gene_names', data=valid_gene_names, dtype=dt)

        f.attrs['model'] = MODEL_NAME
        f.attrs['source_embeddings'] = 'ESM-C (via build_genome_esmc)'
        f.attrs['embedding_dim'] = proteomelm_np.shape[1]
        f.attrs['num_proteins'] = len(valid_protein_ids)
        f.attrs['organism'] = 'Homo sapiens (9606)'
        f.attrs['has_ppi_info'] = True

    file_size = os.path.getsize(OUTPUT_FILE) / 1e6
    print(f"\nDone! File size: {file_size:.1f} MB")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Proteins: {len(valid_protein_ids)}")
    print(f"Embedding dim: {proteomelm_np.shape[1]}")


if __name__ == "__main__":
    main()
