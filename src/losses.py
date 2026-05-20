"""
PAN 2026 — Loss functions.
CoSENT ranking loss, Focal BCE loss, R-Drop regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosent_loss(embeddings_a, embeddings_b, labels, lambda_param=20):
    """
    CoSENT ranking loss (Su 2022).

    Goal: for every (pos_pair, neg_pair) in the batch,
    push sim(pos) > sim(neg), i.e. penalise sim(neg) - sim(pos) > 0.

    Correct formulation:
        loss = log(1 + Σ exp(λ·(sim_neg - sim_pos)))
             = logsumexp([λ·(sim_j - sim_i) for all i=pos, j=neg] + [0])

    Two bugs in the original that caused inverted training:
      1. mask was (label_diff < 0)  → selected neg_i / pos_j pairs (backwards)
         Fix: mask = (label_diff > 0) → selects pos_i / neg_j pairs
      2. diff was cos_sim[i] - cos_sim[j] → penalised pos > neg (backwards)
         Fix: diff = cos_sim[j] - cos_sim[i] → penalises neg > pos

    Inputs (after L2 normalisation in the training loop):
        embeddings_a, embeddings_b : [B, hidden]  (unit vectors)
        labels                     : [B]  1=same-author, 0=diff-author
        lambda_param               : temperature (default 20)
    """
    # [B] — cosine similarity for each pair, scaled by temperature
    cos_sim = F.cosine_similarity(embeddings_a, embeddings_b) * lambda_param

    # Map labels to +1 (same-author) / -1 (diff-author)
    labels_signed = labels * 2 - 1          # 0 → -1,  1 → +1

    # [B, B] pairwise label difference
    # label_diff[i,j] > 0  ↔  i is same-author (pos),  j is diff-author (neg)
    label_diff = labels_signed[:, None] - labels_signed[None, :]  # [B, B]

    # We only care about (pos_i, neg_j) pairs
    mask = (label_diff > 0).float()         # 1 where i=pos, j=neg

    # Penalise when sim(neg_j) > sim(pos_i)
    # diff[i,j] = sim_j - sim_i  (positive = bad, neg_j beating pos_i)
    cos_sim_diff = cos_sim[None, :] - cos_sim[:, None]   # [B, B]

    # Zero out non-constraint pairs by pushing them to -inf before logsumexp
    cos_sim_diff = cos_sim_diff * mask + (mask - 1) * 1e12   # unmasked→-1e12

    # Flatten and append 0 so loss >= 0 (the +1 inside log(1 + Σexp(...)))
    cos_sim_diff = torch.cat([
        cos_sim_diff.reshape(-1),
        torch.zeros(1, device=cos_sim_diff.device),
    ])

    return torch.logsumexp(cos_sim_diff, dim=0)


class FocalBCELoss(nn.Module):
    """Focal loss for class imbalance (Lin et al. 2017)."""

    def __init__(self, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        sample_weight = targets * self.pos_weight + (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return (focal_weight * sample_weight * bce).mean()


def rdrop_loss(logits1, logits2, labels, base_loss_fn, alpha=0.7):
    """
    R-Drop regularization (Liang et al. 2021).
    Two forward passes with different dropout masks → force consistency.
    """
    loss1 = base_loss_fn(logits1, labels)
    loss2 = base_loss_fn(logits2, labels)

    p1 = torch.sigmoid(logits1).clamp(1e-7, 1 - 1e-7)
    p2 = torch.sigmoid(logits2).clamp(1e-7, 1 - 1e-7)

    kl = (
        F.kl_div(p1.log(), p2, reduction="batchmean")
        + F.kl_div(p2.log(), p1, reduction="batchmean")
    )

    return (loss1 + loss2) / 2 + alpha * kl / 2