"""
PAN @ CLEF 2026 — Central configuration.

Key changes vs previous version:
  • Encoders now start from pretrained SBERT checkpoints — CoSENT refines
    a working embedding space instead of building one from scratch.
      - deberta   → cross-encoder/nli-deberta-v3-base  (768d)
      - roberta   → sentence-transformers/all-roberta-large-v1  (1024d)
      - deberta_large → cross-encoder/nli-deberta-v3-large (1024d)
  • CoSENT: more pairs (max_pairs_per_group=6, max_cross=3), LR raised slightly,
    more warmup, epochs 12 with patience 5
  • BiLSTM: larger capacity (proj=384, hidden=256), positional features enabled
  • LOSS_CONFIG: hard pos_weight raised to 6.0, focal_gamma lowered to 1.0;
    medium threshold search range narrowed; easy unchanged
  • BILSTM threshold search: medium uses fine-grained range
"""

import os

_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR     = os.path.dirname(_SRC_DIR)

DATA_DIR          = os.path.join(_ROOT_DIR, "/app/data")
CHECKPOINT_DIR    = os.path.join(_ROOT_DIR, "/app/checkpoints")
FEATURE_CACHE_DIR = os.path.join(_ROOT_DIR, "/app/feature_cache")
LOG_DIR           = os.path.join(_ROOT_DIR, "/app/logs")
OUTPUT_DIR        = os.path.join(_ROOT_DIR, "/app/outputs")

for _d in [CHECKPOINT_DIR, FEATURE_CACHE_DIR, LOG_DIR, OUTPUT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================
# ENCODER MODELS
# All three now start from checkpoints that already produce
# meaningful cosine similarities (NLI / SBERT pretrained).
# CoSENT fine-tuning then re-ranks those similarities for the
# style-change task rather than building representations cold.
# ============================================================
ENCODERS = {
    "deberta": {
        # NLI-finetuned DeBERTa-v3-base — best small encoder for cosine sim tasks
        "model_name": "cross-encoder/nli-deberta-v3-base",
        "hidden_dim": 768,
    },
    "roberta": {
        # SBERT all-roberta-large — already an excellent sentence encoder
        "model_name": "sentence-transformers/all-roberta-large-v1",
        "hidden_dim": 1024,
    },
    "sentbert": {
        # NLI-finetuned DeBERTa-v3-large
        "model_name": "sentence-transformers/all-mpnet-base-v2",
        "hidden_dim": 768,
    },
}

DIFFICULTIES = ["easy", "medium", "hard"]

# ============================================================
# CoSENT ENCODER TRAINING
# More pairs: max_pairs_per_group 6 (was 3), max_cross 3 (was 1)
# Slightly higher LR — SBERT bases already warm, need bolder steps
# Epochs 12, patience 5
# ============================================================
COSENT = {
    "max_seq_len":          96,

    "batch_size":           64,
    "grad_accum_steps":     2,      # physical=32

    "lr":                   8e-6,   # slightly higher — finetuning, not cold training
    "weight_decay":         0.01,
    "max_grad_norm":        1.0,
    "warmup_ratio":         0.15,   # more warmup since we are refining, not cold

    "epochs":               12,
    "eval_every_n_epochs":  1,
    "patience":             5,

    "lambda_param":         20,

    # More pairs — key signal improvement
    "max_pairs_per_group":  6,      # was 3
    "max_cross_pairs":      3,      # was 1
    "neg_pos_ratio":        2.0,

    "train_on":             "hard",
    "val_on":               "hard",

    "use_bf16":             True,
    "use_compile":          True,
}

# ============================================================
# MLP CLASSIFIER — PER DIFFICULTY
# input_dim is now hidden_dim*4 + 1 (added cosine sim scalar)
# ============================================================
CLASSIFIER = {
    "lr":               5e-4,
    "weight_decay":     0.01,
    "epochs":           10,
    "batch_size":       512,
    "rdrop_alpha":      0.7,
    "hidden_sizes":     [512, 256],
    "dropout_rates":    [0.3, 0.2],
}

# ============================================================
# PER-DIFFICULTY LOSS CONFIG
# hard: lower focal_gamma (1.0 vs 2.0) + higher pos_weight (6.0 vs 4.0)
#   → model was underconfident at hard; BiLSTM threshold was 0.31
# medium: pos_weight raised slightly, same focal_gamma
# easy: unchanged
# ============================================================
LOSS_CONFIG = {
    "easy": {
        "focal_gamma":       2.0,
        "pos_weight":        2.5,
        "default_threshold": 0.45,
    },
    "medium": {
        "focal_gamma":       2.0,
        "pos_weight":        25.0,  # slightly up from 22.0 — 4.3% pos rate
        "default_threshold": 0.08,  # very low; fine search handled in training
    },
    "hard": {
        "focal_gamma":       1.0,   # was 2.0 — less aggressive downweighting
        "pos_weight":        6.0,   # was 4.0 — model was underconfident
        "default_threshold": 0.40,
    },
}

# ============================================================
# BI-LSTM SEQUENTIAL CPD
# Larger capacity: proj 384 (was 256), hidden 256 (was 128)
# use_positional_features: inject relative position + local density
# ============================================================
BILSTM = {
    "feature_proj_dim":        384,   # was 256
    "hidden_dim":              256,   # was 128
    "num_layers":              2,
    "dropout":                 0.3,
    "lr":                      1e-3,
    "weight_decay":            1e-4,
    "epochs":                  20,
    "batch_size":              16,
    "patience":                7,
    "use_positional_features": True,  # adds 2 extra input dims (rel_pos, local_density)

    # Medium-specific consecutive-change penalty weight
    "medium_consecutive_penalty": 0.1,

    # Per-difficulty threshold search ranges (min, max, step)
    "threshold_search": {
        "easy":   (0.30, 0.70, 0.02),
        "medium": (0.02, 0.20, 0.005),  # fine-grained low range
        "hard":   (0.20, 0.60, 0.02),
    },
}

# ============================================================
# ENSEMBLE TUNING
# ============================================================
ENSEMBLE = {
    "n_trials":      300,
    "weight_min":    0.05,
    "weight_max":    0.65,
    "threshold_min": 0.05,
    "threshold_max": 0.60,
}

# ============================================================
# DEVICE
# ============================================================
DEVICE      = "cuda"
NUM_WORKERS = 4
SEED        = 42