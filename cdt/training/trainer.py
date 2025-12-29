"""
CDT Training Loop

Trainer class for CDT model training
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
import time
from pathlib import Path

from .losses import ContrastiveLoss, CodonAlignmentLoss, ContrastiveCodonLoss, ProgressiveCodonLoss


class CDTTrainer:
    """
    Trainer for CDT Model

    Handles training loop, validation, checkpointing, and logging.

    Args:
        model: CDT model to train
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        optimizer: Optimizer (e.g., AdamW)
        scheduler: Learning rate scheduler
        device: Device to train on ('cpu' or 'cuda')
        checkpoint_dir: Directory to save checkpoints
        log_interval: How often to log training stats (in batches)

    Example:
        >>> model = CDTModelPhase3()
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>> trainer = CDTTrainer(model, train_loader, val_loader, optimizer)
        >>> trainer.train(num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cpu',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        alignment_loss_weight: float = 0.0,
        alignment_loss_temperature: float = 1.0,
        alignment_loss_type: str = 'kl',  # 'kl' or 'contrastive'
        keep_last_n_checkpoints: int = 2  # Keep only last N checkpoints (in addition to best)
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_interval = log_interval
        self.keep_last_n_checkpoints = keep_last_n_checkpoints

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Track saved checkpoints
        self.saved_checkpoints = []

        # Loss functions
        self.criterion = ContrastiveLoss(temperature=0.07)

        # Alignment loss (Phase 5b/5e/5f)
        self.alignment_loss_weight = alignment_loss_weight
        self.alignment_loss_type = alignment_loss_type
        if alignment_loss_weight > 0:
            if alignment_loss_type == 'progressive':
                self.alignment_loss_fn = ProgressiveCodonLoss(margin=0.1)
                print(f"  Using Progressive Codon Loss (weight={alignment_loss_weight}, margin=0.1)")
                print(f"    Schedule: Epoch 0-4: 1 codon, 5-9: 3 codons, 10-14: 10 codons, 15+: all")
            elif alignment_loss_type == 'contrastive':
                self.alignment_loss_fn = ContrastiveCodonLoss(margin=0.1)
                print(f"  Using Contrastive Codon Loss (weight={alignment_loss_weight}, margin=0.1)")
            else:  # 'kl'
                self.alignment_loss_fn = CodonAlignmentLoss(temperature=alignment_loss_temperature)
                print(f"  Using Codon Alignment Loss (weight={alignment_loss_weight}, temp={alignment_loss_temperature})")
        else:
            self.alignment_loss_fn = None

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'contrastive_loss': [],
            'alignment_loss': [],
            'learning_rates': []
        }

        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def train_one_epoch(self, epoch: int) -> tuple:
        """
        Train for one epoch

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (avg_total_loss, avg_contrastive_loss, avg_alignment_loss)
        """
        self.model.train()
        total_loss = 0.0
        contrastive_loss_sum = 0.0
        alignment_loss_sum = 0.0
        num_batches = 0

        # Update progressive loss epoch if applicable
        if (self.alignment_loss_fn is not None and
            isinstance(self.alignment_loss_fn, ProgressiveCodonLoss)):
            self.alignment_loss_fn.set_epoch(epoch)
            num_codons = self.alignment_loss_fn.get_num_supervised_codons(30)  # Assume max 30 AA
            print(f"  [Progressive] Supervising first {num_codons} codon(s)")

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Get data
            dna_sequences = batch['dna_sequences']
            rna_sequences = batch['rna_sequences']
            protein_sequences = batch['protein_sequences']
            gene_ids = batch['gene_ids'].to(self.device)

            # Forward pass with attention if using alignment loss
            if self.alignment_loss_weight > 0:
                cell_embeddings, attention_maps = self.model(
                    dna_sequences=dna_sequences,
                    rna_sequences=rna_sequences,
                    protein_sequences=protein_sequences,
                    return_attention=True
                )
            else:
                cell_embeddings = self.model(
                    dna_sequences=dna_sequences,
                    rna_sequences=rna_sequences,
                    protein_sequences=protein_sequences
                )

            # Contrastive loss (InfoNCE)
            contrastive_loss = self.criterion(cell_embeddings, gene_ids)

            # Total loss starts with contrastive
            loss = contrastive_loss

            # Alignment loss (if enabled)
            alignment_loss_value = 0.0
            if self.alignment_loss_weight > 0 and 'codon_positions' in batch:
                # Get RNA→Protein attention
                # Shape: (batch, num_heads, num_protein_tokens, num_rna_tokens)
                rna_protein_attention = attention_maps['rna_to_protein']

                # Average over heads
                # Shape: (batch, num_protein_tokens, num_rna_tokens)
                attention = rna_protein_attention.mean(dim=1)

                # Get codon positions
                codon_positions = batch['codon_positions'].to(self.device)

                # Create protein mask (1 for valid tokens, 0 for padding)
                batch_size = len(protein_sequences)
                max_protein_len = codon_positions.size(1)
                protein_mask = torch.zeros(batch_size, max_protein_len, device=self.device)
                for i, seq in enumerate(protein_sequences):
                    protein_mask[i, :len(seq)] = 1.0

                # Compute alignment loss
                alignment_loss = self.alignment_loss_fn(
                    attention, codon_positions, protein_mask
                )

                # Add to total loss
                loss = loss + self.alignment_loss_weight * alignment_loss
                alignment_loss_value = alignment_loss.item()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            # Accumulate losses
            total_loss += loss.item()
            contrastive_loss_sum += contrastive_loss.item()
            alignment_loss_sum += alignment_loss_value
            num_batches += 1

            # Logging
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                avg_contrastive = contrastive_loss_sum / num_batches
                avg_alignment = alignment_loss_sum / num_batches
                lr = self.optimizer.param_groups[0]['lr']
                elapsed = time.time() - start_time

                if self.alignment_loss_weight > 0:
                    print(
                        f"Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                        f"Loss: {avg_loss:.4f} (InfoNCE: {avg_contrastive:.4f}, "
                        f"Align: {avg_alignment:.4f}) | LR: {lr:.6f} | "
                        f"Time: {elapsed:.2f}s"
                    )
                else:
                    print(
                        f"Epoch {epoch} [{batch_idx + 1}/{len(self.train_loader)}] "
                        f"Loss: {avg_loss:.4f} | LR: {lr:.6f} | "
                        f"Time: {elapsed:.2f}s"
                    )

        avg_total_loss = total_loss / num_batches
        avg_contrastive = contrastive_loss_sum / num_batches
        avg_alignment = alignment_loss_sum / num_batches
        return avg_total_loss, avg_contrastive, avg_alignment

    @torch.no_grad()
    def validate(self) -> float:
        """
        Validate on validation set

        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            # Get data
            dna_sequences = batch['dna_sequences']
            rna_sequences = batch['rna_sequences']
            protein_sequences = batch['protein_sequences']
            gene_ids = batch['gene_ids'].to(self.device)

            # Forward pass
            cell_embeddings = self.model(
                dna_sequences=dna_sequences,
                rna_sequences=rna_sequences,
                protein_sequences=protein_sequences
            )

            # Compute loss
            loss = self.criterion(cell_embeddings, gene_ids)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(
        self,
        epoch: int,
        filename: str,
        is_best: bool = False
    ):
        """
        Save model checkpoint

        Args:
            epoch: Current epoch
            filename: Checkpoint filename
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_loss': self.history['train_loss'][-1] if self.history['train_loss'] else None,
            'val_loss': self.history['val_loss'][-1] if self.history['val_loss'] else None,
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }

        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint: {filepath}")

        # Track this checkpoint (unless it's best_model.pt)
        if filename != 'best_model.pt':
            self.saved_checkpoints.append(filepath)

            # Remove old checkpoints if we exceed the limit
            if len(self.saved_checkpoints) > self.keep_last_n_checkpoints:
                old_checkpoint = self.saved_checkpoints.pop(0)
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    print(f"Deleted old checkpoint: {old_checkpoint}")

        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']

        # Ensure new history keys exist (for Phase 5b compatibility)
        if 'contrastive_loss' not in self.history:
            self.history['contrastive_loss'] = []
        if 'alignment_loss' not in self.history:
            self.history['alignment_loss'] = []

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}")

    def train(
        self,
        num_epochs: int,
        save_every: int = 5,
        early_stopping_patience: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Main training loop

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if val loss doesn't improve for N epochs

        Returns:
            dict: Training history

        Example:
            >>> history = trainer.train(num_epochs=50)
            >>> print(history['val_loss'][-1])  # Final validation loss
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training batches: {len(self.train_loader)}")
        if self.val_loader is not None:
            print(f"Validation batches: {len(self.val_loader)}")
        else:
            print(f"Validation batches: 0 (no validation set)")

        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*70}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*70}")

            # Train
            train_loss, contrastive_loss, alignment_loss = self.train_one_epoch(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['contrastive_loss'].append(contrastive_loss)
            self.history['alignment_loss'].append(alignment_loss)

            # Validate (if validation set exists)
            if self.val_loader is not None:
                val_loss = self.validate()
                self.history['val_loss'].append(val_loss)
            else:
                val_loss = None

            # Log learning rate
            lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(lr)

            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            if self.alignment_loss_weight > 0:
                print(f"    - InfoNCE:   {contrastive_loss:.4f}")
                print(f"    - Alignment: {alignment_loss:.4f}")
            if val_loss is not None:
                print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR:         {lr:.6f}")

            # Check for best model (based on validation loss if available, else train loss)
            check_loss = val_loss if val_loss is not None else train_loss
            is_best = check_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = check_loss
                self.best_epoch = epoch
                patience_counter = 0
                loss_type = "Val" if val_loss is not None else "Train"
                print(f"  ✓ New best model! ({loss_type} Loss: {check_loss:.4f})")
            else:
                patience_counter += 1

            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(
                    epoch=epoch,
                    filename=f'checkpoint_epoch_{epoch}.pt',
                    is_best=is_best
                )

            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                print(f"Best model at epoch {self.best_epoch} with val loss {self.best_val_loss:.4f}")
                break

        print(f"\n{'='*70}")
        print("Training complete!")
        print(f"Best model: Epoch {self.best_epoch} | Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}")

        return self.history

    def get_embedding_quality_metrics(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Compute embedding quality metrics

        Args:
            dataloader: DataLoader to evaluate

        Returns:
            dict: Metrics including intra-class similarity, inter-class distance, etc.
        """
        self.model.eval()

        all_embeddings = []
        all_gene_ids = []

        with torch.no_grad():
            for batch in dataloader:
                dna_sequences = batch['dna_sequences']
                rna_sequences = batch['rna_sequences']
                protein_sequences = batch['protein_sequences']
                gene_ids = batch['gene_ids']

                embeddings = self.model(
                    dna_sequences=dna_sequences,
                    rna_sequences=rna_sequences,
                    protein_sequences=protein_sequences
                )

                all_embeddings.append(embeddings.cpu())
                all_gene_ids.append(gene_ids)

        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)  # [N, d_model]
        all_gene_ids = torch.cat(all_gene_ids, dim=0)  # [N]

        # Normalize embeddings
        all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_embeddings, all_embeddings.T)  # [N, N]

        # Create label mask
        gene_mask = (all_gene_ids.unsqueeze(1) == all_gene_ids.unsqueeze(0)).float()

        # Intra-class similarity (same gene)
        intra_class_mask = gene_mask - torch.eye(len(all_gene_ids))
        intra_class_sim = (similarity_matrix * intra_class_mask).sum() / intra_class_mask.sum()

        # Inter-class distance (different genes)
        inter_class_mask = 1 - gene_mask
        inter_class_sim = (similarity_matrix * inter_class_mask).sum() / inter_class_mask.sum()

        metrics = {
            'intra_class_similarity': intra_class_sim.item(),
            'inter_class_similarity': inter_class_sim.item(),
            'discrimination': (intra_class_sim - inter_class_sim).item()
        }

        return metrics


if __name__ == "__main__":
    # Test Trainer with dummy data
    print("Testing CDTTrainer...")

    from cdt.models.cdt_model_phase3 import CDTModelPhase3
    from cdt.data.dataloaders import create_dataloaders

    # Create model
    model = CDTModelPhase3(
        freeze_dna=True,
        freeze_rna=True,
        freeze_protein=True
    )

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        num_samples=100,
        batch_size=8,
        dna_length=50,
        rna_length=40,
        protein_length=20
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.01
    )

    # Create trainer
    trainer = CDTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device='cpu',
        checkpoint_dir='test_checkpoints',
        log_interval=2
    )

    # Train for 2 epochs (quick test)
    print("\nRunning quick training test (2 epochs)...")
    history = trainer.train(num_epochs=2, save_every=1)

    print("\nTraining history:")
    print(f"  Train losses: {history['train_loss']}")
    print(f"  Val losses: {history['val_loss']}")

    # Test embedding quality metrics
    print("\nComputing embedding quality metrics...")
    metrics = trainer.get_embedding_quality_metrics(val_loader)
    print(f"  Intra-class similarity: {metrics['intra_class_similarity']:.4f}")
    print(f"  Inter-class similarity: {metrics['inter_class_similarity']:.4f}")
    print(f"  Discrimination: {metrics['discrimination']:.4f}")

    print("\n✓ All tests passed!")
