"""
Loss functions for CDT training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def infonce_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    InfoNCE (Noise Contrastive Estimation) Loss

    同じラベルを持つサンプル間の類似度を最大化し、
    異なるラベルを持つサンプル間の類似度を最小化する。

    Args:
        embeddings: [batch_size, d_model] モデル出力の埋め込み
        labels: [batch_size] サンプルのラベル（同じ遺伝子には同じラベル）
        temperature: スケーリングパラメータ（小さいほど鋭い分布）
        reduction: 'mean' or 'sum' or 'none'

    Returns:
        loss: スカラー（reduction='mean'の場合）

    Example:
        >>> embeddings = torch.randn(16, 768)  # batch=16, d_model=768
        >>> labels = torch.tensor([0,0,1,1,2,2,0,1,2,0,1,2,0,1,2,0])
        >>> loss = infonce_loss(embeddings, labels, temperature=0.07)
        >>> print(loss)
        tensor(2.7183)

    Note:
        - Positive pairs: 同じラベルを持つサンプルペア
        - Negative pairs: 異なるラベルを持つサンプルペア
        - Temperature小: より厳しい分類（sharp distribution）
        - Temperature大: より緩い分類（smooth distribution）
    """
    batch_size = embeddings.size(0)
    device = embeddings.device

    # Normalize embeddings to unit sphere
    # これによりコサイン類似度が内積で計算できる
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix: [batch, batch]
    # similarity[i, j] = cosine_similarity(emb[i], emb[j])
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Create label mask: mask[i, j] = 1 if labels[i] == labels[j]
    labels = labels.unsqueeze(1)  # [batch, 1]
    mask = torch.eq(labels, labels.T).float()  # [batch, batch]

    # Remove diagonal (self-similarity)
    # 自分自身とのpositive pairは除外
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask

    # Compute log probability
    # log P(positive | anchor) = log( exp(sim_pos) / Σ exp(sim_all) )
    exp_logits = torch.exp(similarity_matrix) * logits_mask
    log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True))

    # Mean of log-likelihood over positive pairs
    # 各サンプルについて、positive pairの平均log確率を計算
    mask_sum = mask.sum(dim=1)
    mask_sum = torch.clamp(mask_sum, min=1.0)  # ゼロ除算を防ぐ

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum

    # Final loss (negative log likelihood)
    loss = -mean_log_prob_pos

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss Module

    InfoNCE lossのnn.Moduleラッパー

    Args:
        temperature: スケーリングパラメータ
        reduction: 'mean' or 'sum' or 'none'

    Example:
        >>> criterion = ContrastiveLoss(temperature=0.07)
        >>> embeddings = torch.randn(16, 768)
        >>> labels = torch.randint(0, 5, (16,))
        >>> loss = criterion(embeddings, labels)
    """

    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            embeddings: [batch_size, d_model]
            labels: [batch_size]

        Returns:
            loss: スカラー or [batch_size]
        """
        return infonce_loss(
            embeddings,
            labels,
            temperature=self.temperature,
            reduction=self.reduction
        )


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Triplet Loss（参考実装）

    将来的に使用する可能性があるため実装

    Args:
        anchor: [batch, d_model] アンカー埋め込み
        positive: [batch, d_model] positive sample埋め込み
        negative: [batch, d_model] negative sample埋め込み
        margin: マージンパラメータ
        reduction: 'mean' or 'sum' or 'none'

    Returns:
        loss: スカラー

    Formula:
        L = max(0, ||anchor - positive||^2 - ||anchor - negative||^2 + margin)
    """
    distance_positive = F.pairwise_distance(anchor, positive, p=2)
    distance_negative = F.pairwise_distance(anchor, negative, p=2)

    losses = F.relu(distance_positive - distance_negative + margin)

    if reduction == 'mean':
        return losses.mean()
    elif reduction == 'sum':
        return losses.sum()
    else:
        return losses


class CodonAlignmentLoss(nn.Module):
    """
    Codon Alignment Loss for RNA→Protein attention

    Encourages each amino acid to attend to its corresponding codon (3 nucleotides).

    Args:
        temperature: Temperature for softmax normalization

    Example:
        >>> loss_fn = CodonAlignmentLoss()
        >>> attention = torch.rand(2, 30, 108)  # (batch, num_aa, num_rna)
        >>> codon_positions = torch.randint(0, 105, (2, 30, 3))  # (batch, num_aa, 3)
        >>> protein_mask = torch.ones(2, 30)
        >>> loss = loss_fn(attention, codon_positions, protein_mask)
    """

    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        attention_weights: torch.Tensor,
        codon_positions: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute codon alignment loss

        Args:
            attention_weights: (batch, num_aa, num_rna_tokens)
                RNA→Protein cross-attention weights
            codon_positions: (batch, num_aa, 3)
                For each amino acid, the 3 RNA positions of its codon
            protein_mask: (batch, num_aa)
                Mask for valid amino acids (1=valid, 0=padding)

        Returns:
            loss: Scalar tensor
        """
        batch_size, num_aa, num_rna = attention_weights.shape
        device = attention_weights.device

        # Create target attention distribution
        # Each AA should attend uniformly to its 3 codon nucleotides
        target = torch.zeros_like(attention_weights)

        # Fill target for each sample and amino acid
        for b in range(batch_size):
            for aa_idx in range(num_aa):
                # Skip if masked (check bounds first)
                if protein_mask is not None:
                    if aa_idx >= protein_mask.shape[1]:
                        continue
                    if protein_mask[b, aa_idx] == 0:
                        continue

                # Get codon positions (check bounds first)
                if aa_idx >= codon_positions.shape[1]:
                    continue

                try:
                    pos1, pos2, pos3 = codon_positions[b, aa_idx]

                    # Ensure positions are within bounds
                    if (0 <= pos1 < num_rna and
                        0 <= pos2 < num_rna and
                        0 <= pos3 < num_rna):
                        # Target: uniform distribution over 3 codon positions
                        target[b, aa_idx, pos1] = 1.0 / 3.0
                        target[b, aa_idx, pos2] = 1.0 / 3.0
                        target[b, aa_idx, pos3] = 1.0 / 3.0
                except (IndexError, RuntimeError):
                    continue

        # KL divergence: encourage predicted attention to match target
        log_pred = F.log_softmax(attention_weights / self.temperature, dim=-1)

        # Compute KL divergence
        loss = F.kl_div(
            log_pred,
            target,
            reduction='batchmean'
        )

        return loss


class ContrastiveCodonLoss(nn.Module):
    """
    Contrastive Codon Loss

    Instead of KL divergence with uniform target, use contrastive learning:
    - Positive: attention to correct codon positions should be HIGH
    - Negative: attention to other positions should be LOW

    This is much easier to learn than KL divergence.

    Args:
        margin: Margin for contrastive loss (default: 0.1)

    Example:
        >>> loss_fn = ContrastiveCodonLoss(margin=0.1)
        >>> attention = torch.rand(2, 30, 108)
        >>> codon_positions = torch.randint(0, 105, (2, 30, 3))
        >>> protein_mask = torch.ones(2, 30)
        >>> loss = loss_fn(attention, codon_positions, protein_mask)
    """

    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        attention_weights: torch.Tensor,
        codon_positions: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive codon loss

        Args:
            attention_weights: (batch, num_aa, num_rna_tokens)
            codon_positions: (batch, num_aa, 3)
            protein_mask: (batch, num_aa)

        Returns:
            loss: Scalar tensor
        """
        batch_size, num_aa, num_rna = attention_weights.shape
        device = attention_weights.device

        total_loss = 0.0
        num_valid = 0

        for b in range(batch_size):
            for aa_idx in range(num_aa):
                # Skip if masked
                if protein_mask is not None:
                    if aa_idx >= protein_mask.shape[1]:
                        continue
                    if protein_mask[b, aa_idx] == 0:
                        continue

                # Get codon positions
                if aa_idx >= codon_positions.shape[1]:
                    continue

                try:
                    pos1, pos2, pos3 = codon_positions[b, aa_idx]

                    # Ensure positions are within bounds
                    if not (0 <= pos1 < num_rna and
                           0 <= pos2 < num_rna and
                           0 <= pos3 < num_rna):
                        continue

                    # Get attention weights for this amino acid
                    attn = attention_weights[b, aa_idx]  # (num_rna,)

                    # Positive score: average attention to correct codon positions
                    pos_score = (attn[pos1] + attn[pos2] + attn[pos3]) / 3.0

                    # Negative score: average attention to all other positions
                    # Create mask for negative positions
                    neg_mask = torch.ones(num_rna, device=device, dtype=torch.bool)
                    neg_mask[pos1] = False
                    neg_mask[pos2] = False
                    neg_mask[pos3] = False

                    if neg_mask.sum() > 0:
                        neg_score = attn[neg_mask].mean()
                    else:
                        continue

                    # Contrastive loss: pos_score should be > neg_score + margin
                    # loss = max(0, margin - pos_score + neg_score)
                    loss = F.relu(self.margin - pos_score + neg_score)

                    total_loss += loss
                    num_valid += 1

                except (IndexError, RuntimeError):
                    continue

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device)


class ProgressiveCodonLoss(nn.Module):
    """
    Progressive Codon Learning Loss

    Start from the beginning (start codon) and progressively learn more codons.
    Uses biological constraint: start from AUG and read 3 nucleotides at a time.

    Strategy:
    - Early epochs: Only supervise first few codons
    - Later epochs: Supervise more codons
    - Final epochs: Supervise all codons

    This provides a curriculum learning approach with biological grounding.

    Args:
        margin: Margin for contrastive loss (default: 0.1)
        progressive_schedule: List of (epoch, num_codons) pairs
            e.g., [(0, 1), (5, 3), (10, 10), (15, -1)]
            -1 means all codons

    Example:
        >>> loss_fn = ProgressiveCodonLoss(margin=0.1)
        >>> # Epoch 1: supervise first codon
        >>> # Epoch 10: supervise first 5 codons
        >>> # Epoch 20: supervise all codons
    """

    def __init__(
        self,
        margin: float = 0.1,
        progressive_schedule: list = None
    ):
        super().__init__()
        self.margin = margin

        # Default progressive schedule
        if progressive_schedule is None:
            self.progressive_schedule = [
                (0, 1),    # Epoch 0-4: first codon only
                (5, 3),    # Epoch 5-9: first 3 codons
                (10, 10),  # Epoch 10-14: first 10 codons
                (15, -1)   # Epoch 15+: all codons
            ]
        else:
            self.progressive_schedule = progressive_schedule

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Update current epoch for progressive schedule"""
        self.current_epoch = epoch

    def get_num_supervised_codons(self, num_aa: int) -> int:
        """Get number of codons to supervise based on current epoch"""
        num_codons = 1  # Default: at least first codon

        for epoch_threshold, codons in self.progressive_schedule:
            if self.current_epoch >= epoch_threshold:
                num_codons = codons if codons > 0 else num_aa

        return min(num_codons, num_aa)

    def forward(
        self,
        attention_weights: torch.Tensor,
        codon_positions: torch.Tensor,
        protein_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute progressive codon loss

        Args:
            attention_weights: (batch, num_aa, num_rna_tokens)
            codon_positions: (batch, num_aa, 3)
            protein_mask: (batch, num_aa)

        Returns:
            loss: Scalar tensor
        """
        batch_size, num_aa, num_rna = attention_weights.shape
        device = attention_weights.device

        # Determine how many codons to supervise
        num_supervised = self.get_num_supervised_codons(num_aa)

        total_loss = 0.0
        num_valid = 0

        for b in range(batch_size):
            for aa_idx in range(num_supervised):  # Only supervise first N codons
                # Skip if masked
                if protein_mask is not None:
                    if aa_idx >= protein_mask.shape[1]:
                        continue
                    if protein_mask[b, aa_idx] == 0:
                        continue

                # Get codon positions
                if aa_idx >= codon_positions.shape[1]:
                    continue

                try:
                    pos1, pos2, pos3 = codon_positions[b, aa_idx]

                    # Ensure positions are within bounds
                    if not (0 <= pos1 < num_rna and
                           0 <= pos2 < num_rna and
                           0 <= pos3 < num_rna):
                        continue

                    # Get attention weights for this amino acid
                    attn = attention_weights[b, aa_idx]  # (num_rna,)

                    # Positive score: average attention to correct codon positions
                    pos_score = (attn[pos1] + attn[pos2] + attn[pos3]) / 3.0

                    # Negative score: average attention to all other positions
                    neg_mask = torch.ones(num_rna, device=device, dtype=torch.bool)
                    neg_mask[pos1] = False
                    neg_mask[pos2] = False
                    neg_mask[pos3] = False

                    if neg_mask.sum() > 0:
                        neg_score = attn[neg_mask].mean()
                    else:
                        continue

                    # Contrastive loss
                    loss = F.relu(self.margin - pos_score + neg_score)

                    total_loss += loss
                    num_valid += 1

                except (IndexError, RuntimeError):
                    continue

        if num_valid > 0:
            return total_loss / num_valid
        else:
            return torch.tensor(0.0, device=device)


if __name__ == "__main__":
    # Test InfoNCE loss
    print("Testing InfoNCE Loss...")

    # Create dummy data
    batch_size = 16
    d_model = 768
    embeddings = torch.randn(batch_size, d_model)

    # Labels: 4 classes, each with 4 samples
    labels = torch.tensor([0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3])

    # Compute loss
    loss = infonce_loss(embeddings, labels, temperature=0.07)
    print(f"InfoNCE Loss: {loss.item():.4f}")

    # Test with ContrastiveLoss module
    criterion = ContrastiveLoss(temperature=0.07)
    loss2 = criterion(embeddings, labels)
    print(f"ContrastiveLoss Module: {loss2.item():.4f}")

    # Test that both give same result
    assert torch.allclose(loss, loss2), "Losses should be equal"
    print("✓ Tests passed!")

    # Test with different temperatures
    print("\nTemperature sensitivity:")
    for temp in [0.01, 0.05, 0.1, 0.5, 1.0]:
        loss_temp = infonce_loss(embeddings, labels, temperature=temp)
        print(f"  temp={temp:.2f}: loss={loss_temp.item():.4f}")

    # Test Codon Alignment Loss
    print("\n\nTesting Codon Alignment Loss...")
    codon_loss_fn = CodonAlignmentLoss()
    attention = torch.rand(2, 30, 108)  # (batch=2, num_aa=30, num_rna=108)
    codon_positions = torch.randint(0, 105, (2, 30, 3))  # Random codon positions
    protein_mask = torch.ones(2, 30)

    codon_loss = codon_loss_fn(attention, codon_positions, protein_mask)
    print(f"Codon Alignment Loss: {codon_loss.item():.4f}")
    print("✓ Codon Alignment Loss test passed!")
