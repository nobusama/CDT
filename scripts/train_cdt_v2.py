#!/usr/bin/env python3
"""
CDT v2 Training Script

v2 Architecture:
- DNA: [batch, 896, 3072] - Enformer sequence-level
- Protein: [n_proteins, 768] - Full proteome (shared across batches)
- RNA: [batch, n_genes, 512] - Gene expression per sample
- Output: [batch, n_proteins] - Binding prediction for each (enhancer, protein) pair

Cyclic Cross-Attention: DNA -> RNA -> Protein -> DNA

Usage:
    # For testing (limited to 100 proteins)
    python scripts/train_cdt_v2.py --n_proteins 100 --epochs 3 --batch_size 4

    # Full training (all 20420 proteins)
    python scripts/train_cdt_v2.py --epochs 20 --batch_size 8
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from cdt.data.embedding_dataset_v2 import CDTv2Dataset, collate_v2
from cdt.models.cdt_v2_seqlevel import CDTv2Config, CDTv2Model


def parse_args():
    parser = argparse.ArgumentParser(description="Train CDT v2 Model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--n_proteins", type=int, default=None, help="Limit proteins for debug")
    parser.add_argument("--use_aligned", action="store_true", default=True,
                        help="Use RNA-Protein aligned embeddings (2360 genes)")
    parser.add_argument("--no_aligned", action="store_true",
                        help="Don't use aligned embeddings")
    parser.add_argument("--device", type=str, default="mps", choices=["mps", "cuda", "cpu"])
    parser.add_argument("--output_dir", type=str, default="outputs/v2")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, dataloader, protein_emb, device, criterion):
    """Evaluate model on dataset."""
    model.eval()
    all_preds = []
    all_labels = []
    all_protein_indices = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            dna = batch['dna_emb'].to(device)
            rna = batch['rna_emb'].to(device)
            labels = batch['labels'].to(device)
            protein_indices = batch['protein_indices']

            # Forward pass
            logits = model(dna, protein_emb, rna)  # [batch, n_proteins]

            # Get predictions for target protein of each sample
            batch_size = logits.size(0)
            target_logits = []
            valid_mask = []

            for i in range(batch_size):
                prot_idx = protein_indices[i].item()
                if prot_idx < logits.size(1):
                    target_logits.append(logits[i, prot_idx])
                    valid_mask.append(True)
                else:
                    target_logits.append(torch.tensor(0.0, device=device))
                    valid_mask.append(False)

            target_logits = torch.stack(target_logits)
            valid_mask = torch.tensor(valid_mask, device=device)

            # Loss (valid samples only)
            if valid_mask.any():
                valid_logits = target_logits[valid_mask]
                valid_labels = labels[valid_mask]
                loss = criterion(valid_logits, valid_labels)
                total_loss += loss.item()
                n_batches += 1

                # Predictions
                probs = torch.sigmoid(valid_logits).cpu().numpy()
                all_preds.extend(probs.tolist())
                all_labels.extend(valid_labels.cpu().numpy().tolist())
                all_protein_indices.extend([protein_indices[i].item() for i, v in enumerate(valid_mask) if v])

    # Metrics
    if len(all_labels) > 0:
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Handle edge cases
        if len(np.unique(all_labels)) > 1:
            aupr = average_precision_score(all_labels, all_preds)
            auroc = roc_auc_score(all_labels, all_preds)
        else:
            aupr = 0.0
            auroc = 0.5

        avg_loss = total_loss / max(n_batches, 1)
    else:
        aupr = 0.0
        auroc = 0.5
        avg_loss = 0.0

    return {
        "loss": avg_loss,
        "aupr": aupr,
        "auroc": auroc,
        "n_samples": len(all_labels),
        "n_positive": int(np.sum(all_labels)) if len(all_labels) > 0 else 0,
    }


def train_epoch(model, dataloader, protein_emb, device, optimizer, criterion, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_valid_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in pbar:
        dna = batch['dna_emb'].to(device)
        rna = batch['rna_emb'].to(device)
        labels = batch['labels'].to(device)
        protein_indices = batch['protein_indices']

        # Forward pass
        logits = model(dna, protein_emb, rna)  # [batch, n_proteins]

        # Get predictions for target protein of each sample
        batch_size = logits.size(0)
        target_logits = []
        valid_mask = []

        for i in range(batch_size):
            prot_idx = protein_indices[i].item()
            if prot_idx < logits.size(1):
                target_logits.append(logits[i, prot_idx])
                valid_mask.append(True)
            else:
                # Protein outside limited range - dummy value
                target_logits.append(torch.tensor(0.0, device=device, requires_grad=True))
                valid_mask.append(False)

        target_logits = torch.stack(target_logits)
        valid_mask = torch.tensor(valid_mask, device=device)

        # Loss (valid samples only)
        if valid_mask.any():
            valid_logits = target_logits[valid_mask]
            valid_labels = labels[valid_mask]
            loss = criterion(valid_logits, valid_labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            n_valid_samples += valid_mask.sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "valid": n_valid_samples,
            })

    return total_loss / max(n_batches, 1), n_valid_samples


def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Data
    project_root = Path(__file__).parent.parent
    training_dir = project_root / "data/processed/training"

    use_aligned = not args.no_aligned
    print(f"\nLoading datasets... (use_aligned={use_aligned})")
    train_dataset = CDTv2Dataset(
        training_dir / "gasperini_train.h5",
        project_root=project_root,
        n_proteins=args.n_proteins,
        use_aligned=use_aligned,
    )
    val_dataset = CDTv2Dataset(
        training_dir / "gasperini_val.h5",
        project_root=project_root,
        n_proteins=args.n_proteins,
        use_aligned=use_aligned,
    )
    test_dataset = CDTv2Dataset(
        training_dir / "gasperini_test.h5",
        project_root=project_root,
        n_proteins=args.n_proteins,
        use_aligned=use_aligned,
    )

    # Protein embeddings (shared)
    protein_emb = train_dataset.get_protein_embeddings().to(device)
    print(f"Protein embeddings: {protein_emb.shape}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_v2, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_v2
    )

    # Model
    dims = train_dataset.get_dims()
    config = CDTv2Config(
        dna_dim=dims['dna_dim'],
        dna_seq_len=dims['dna_seq_len'],
        protein_dim=dims['protein_dim'],
        rna_dim=dims['rna_dim'],
        hidden_dim=args.hidden_dim,
        nhead=args.nhead,
    )
    model = CDTv2Model(config).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    # Save config
    config_dict = {
        **vars(config),
        "n_proteins": dims['n_proteins'],
        "n_genes": dims['n_genes'],
        "args": vars(args),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Training setup
    # Class weights for imbalanced data
    pos_weight = torch.tensor([train_dataset.n_negative / train_dataset.n_positive], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nClass weights: pos_weight={pos_weight.item():.2f}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Training loop
    best_val_aupr = 0.0
    history = []

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, n_valid = train_epoch(
            model, train_loader, protein_emb, device, optimizer, criterion, epoch
        )

        # Validate
        val_metrics = evaluate(model, val_loader, protein_emb, device, criterion)

        # Scheduler step
        scheduler.step()

        # Log
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Valid Samples: {n_valid}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}, AUPR: {val_metrics['aupr']:.4f}, AUC: {val_metrics['auroc']:.4f}")
        print(f"  Val Samples: {val_metrics['n_samples']} ({val_metrics['n_positive']} positive)")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics['loss'],
            "val_aupr": val_metrics['aupr'],
            "val_auroc": val_metrics['auroc'],
            "lr": scheduler.get_last_lr()[0],
        })

        # Save best model
        if val_metrics['aupr'] > best_val_aupr:
            best_val_aupr = val_metrics['aupr']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_aupr': val_metrics['aupr'],
            }, output_dir / "best_model.pt")
            print(f"  *** New best model saved (AUPR: {best_val_aupr:.4f})")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_dir / "final_model.pt")

    # Save history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Test evaluation
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)

    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, protein_emb, device, criterion)
    print(f"\nTest Results:")
    print(f"  Loss: {test_metrics['loss']:.4f}")
    print(f"  AUPR: {test_metrics['aupr']:.4f}")
    print(f"  AUC:  {test_metrics['auroc']:.4f}")
    print(f"  Samples: {test_metrics['n_samples']} ({test_metrics['n_positive']} positive)")

    # Save test results
    with open(output_dir / "test_results.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Cleanup
    train_dataset.close()
    val_dataset.close()
    test_dataset.close()

    print("\n" + "=" * 60)
    print(f"Training complete! Best Val AUPR: {best_val_aupr:.4f}")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
