"""
PAN 2026 — Model definitions.
StyleChangeClassifier (MLP) + SequentialCPDRefinement (Bi-LSTM).

Changes vs previous version:
  • StyleChangeClassifier: adds explicit cosine similarity scalar as feature
    input_dim = hidden_dim*4 + 1  (was hidden_dim*4)
  • SequentialCPDRefinement: optionally adds 2 positional features
    (relative position in doc, local change density in window of 5)
    input_dim += 2 when use_positional_features=True
  • Projection layers have LayerNorm for stability with larger hidden dims
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleChangeClassifier(nn.Module):
    """
    MLP classifier on frozen encoder embeddings.
    Input: [u; v; |u-v|; u⊙v; cos(u,v)] where u,v are mean-pooled embeddings.

    cos(u,v) is added as an explicit scalar — the BiLSTM and ensemble
    weight it heavily, so surfacing it directly improves gradient flow.
    """
    def __init__(self, hidden_dim, mlp_sizes=(512, 256), dropout_rates=(0.3, 0.2)):
        super().__init__()
        input_dim = hidden_dim * 4 + 1  # +1 for cosine sim scalar

        layers = []
        prev_dim = input_dim
        for size, drop in zip(mlp_sizes, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, size),
                nn.LayerNorm(size),
                nn.Tanh(),
                nn.Dropout(drop),
            ])
            prev_dim = size
        layers.append(nn.Linear(prev_dim, 1))

        self.mlp = nn.Sequential(*layers)

    def _build_features(self, emb_a, emb_b):
        cos = F.cosine_similarity(emb_a, emb_b, dim=-1, eps=1e-8).unsqueeze(-1)
        return torch.cat([
            emb_a,
            emb_b,
            torch.abs(emb_a - emb_b),
            emb_a * emb_b,
            cos,
        ], dim=-1)

    def forward(self, emb_a, emb_b):
        return self.mlp(self._build_features(emb_a, emb_b)).squeeze(-1)

    def get_features(self, emb_a, emb_b):
        """Return raw feature vector (for Bi-LSTM input)."""
        return self._build_features(emb_a, emb_b)


class SequentialCPDRefinement(nn.Module):
    """
    Bi-LSTM that processes ALL boundaries in a document sequentially.

    Input per boundary:
      - Projected features from N encoders  (feature_proj_dim * N)
      - MLP probabilities from N encoders   (N)
      - Ensemble probability                (1)
      - [optional] Positional features      (2)
          · rel_pos   : boundary index / total boundaries  ∈ [0, 1]
          · local_den : fraction of changes in ±2 window (from ensemble probs)

    Total input dim: feature_proj_dim * N + N + 1 [+ 2]
    """
    def __init__(self, encoder_dims=None, feature_proj_dim=384,
                 hidden_dim=256, num_layers=2, dropout=0.3,
                 use_positional_features=True):
        super().__init__()

        if encoder_dims is None:
            encoder_dims = {"deberta": 768, "roberta": 1024, "deberta_large": 1024}

        self.use_positional_features = use_positional_features

        # Each encoder's feature vector is hidden_dim*4+1 (MLP feature space)
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim * 4 + 1, feature_proj_dim),
                nn.LayerNorm(feature_proj_dim),
                nn.ReLU(),
            )
            for name, dim in encoder_dims.items()
        })

        n_encoders = len(encoder_dims)
        pos_dim    = 2 if use_positional_features else 0
        input_dim  = feature_proj_dim * n_encoders + n_encoders + 1 + pos_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _positional_features(self, ensemble_prob):
        """
        Args:
            ensemble_prob: [batch, seq, 1]
        Returns:
            pos_feats: [batch, seq, 2]
        """
        B, T, _ = ensemble_prob.shape
        device   = ensemble_prob.device

        # Relative position: 0 → 1 across boundaries
        rel_pos = torch.linspace(0.0, 1.0, T, device=device)
        rel_pos = rel_pos.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # [B, T, 1]

        # Local change density: mean of ensemble probs in ±2 window
        # Use avg_pool1d as a fast sliding window
        p = ensemble_prob.squeeze(-1)  # [B, T]
        # Pad edges with zeros (window=5, same length)
        padded = F.pad(p, (2, 2), mode='replicate')       # [B, T+4]
        # avg_pool1d expects [B, C, L]
        local_den = F.avg_pool1d(
            padded.unsqueeze(1), kernel_size=5, stride=1
        ).squeeze(1)  # [B, T]
        local_den = local_den.unsqueeze(-1)                # [B, T, 1]

        return torch.cat([rel_pos, local_den], dim=-1)     # [B, T, 2]

    def forward(self, encoder_features, encoder_probs, ensemble_prob):
        """
        Args:
            encoder_features : dict {name: [batch, seq, hidden*4+1]}
            encoder_probs    : dict {name: [batch, seq, 1]}
            ensemble_prob    : [batch, seq, 1]
        Returns:
            logits: [batch, seq]
        """
        projected  = []
        prob_list  = []

        for name in sorted(encoder_features.keys()):
            proj = self.projections[name](encoder_features[name])
            projected.append(proj)
            prob_list.append(encoder_probs[name])

        parts = projected + prob_list + [ensemble_prob]

        if self.use_positional_features:
            parts.append(self._positional_features(ensemble_prob))

        combined  = torch.cat(parts, dim=-1)
        lstm_out, _ = self.lstm(combined)
        logits      = self.output(lstm_out).squeeze(-1)

        return logits