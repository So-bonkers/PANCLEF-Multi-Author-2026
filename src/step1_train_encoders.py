"""
PAN 2026 — Step 1: Train CoSENT encoders on HARD data only.

Usage:
    python step1_train_encoders.py                        # all encoders
    python step1_train_encoders.py --encoder deberta      # one encoder
    python step1_train_encoders.py --encoder roberta --resume

Changes vs previous version:
  • SBERT/NLI base models — CoSENT refines an already-working cosine space.
    Starting separation is ~0.4+ instead of ~0.05 cold.
  • Evaluation threshold search expanded and uses BOTH directions
    (sim > thr for same-author AND sim < thr). With SBERT bases the
    high-similarity = same-author direction is correct by default.
  • Added separation_target early-stop guard: if sep < 0.05 after epoch 3
    we log a loud warning (encoder is still collapsing).
  • cross-encoder NLI models: their native forward pass is cross-encoder
    (pair input), but we load them as bi-encoders (AutoModel) and mean-pool.
    This is correct — we only need the backbone weights, not the head.
  • torch.compile disabled for DeBERTa-v3-large on small GPUs (compile OOM).
    Controlled via per-encoder override in ENCODERS config via env var
    DISABLE_COMPILE=1.
"""

import argparse
import logging
import os
import time
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import *
from data_utils import load_dataset, print_dataset_stats, generate_all_pairs
from losses import cosent_loss


# ============================================================
# Helpers
# ============================================================

class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass

def amp_ctx(use_bf16):
    return autocast(dtype=torch.bfloat16) if use_bf16 else _NullCtx()

def setup_logging(encoder_name):
    log_file = os.path.join(LOG_DIR, f"step1_cosent_{encoder_name}.log")
    logging.getLogger().handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger(__name__)

def tlog(logger, msg):
    logger.info(msg)

def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ============================================================
# Encoder forward — MEAN POOLING (not CLS)
# ============================================================

def encode(encoder, input_ids, attention_mask):
    out    = encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state                          # [B, T, H]
    mask   = attention_mask.unsqueeze(-1).float()           # [B, T, 1]
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # [B, H]


def sanity_check(encoder, tokenizer, device, max_len, use_bf16, logger):
    """
    Check that the encoder produces finite, non-collapsed embeddings.
    Also reports pre-normalisation separation — with SBERT bases this
    should already be > 0.2 before any CoSENT training.
    """
    tlog(logger, "Running sanity check...")
    encoder.eval()

    same_author_pairs = [
        ("The sky is a pale shade of blue today.", "Clouds drift lazily across the afternoon sky."),
    ]
    diff_author_pairs = [
        ("The algorithm runs in O(n log n) time complexity.", "Thermodynamic equilibrium requires careful consideration."),
    ]

    all_texts = [p[i] for p in (same_author_pairs + diff_author_pairs) for i in range(2)]
    tok = tokenizer(all_texts, truncation=True, max_length=max_len,
                    padding=True, return_tensors="pt").to(device)
    with torch.no_grad(), amp_ctx(use_bf16):
        embs = encode(encoder, tok["input_ids"], tok["attention_mask"])
        embs_n = F.normalize(embs.float(), dim=-1)

    norms   = [round(n, 3) for n in embs.norm(dim=-1).tolist()]
    has_nan = torch.isnan(embs).any().item()
    tlog(logger, f"  shape={embs.shape} dtype={embs.dtype} norms={norms} has_nan={has_nan}")

    if has_nan:
        raise RuntimeError(
            "NaN in encoder output on clean input.\n"
            "Try: (1) delete HF cache and re-download, "
            "(2) set use_bf16=False, (3) set use_compile=False"
        )

    # Report initial cosine similarities — SBERT base should give sep > 0.3
    for i, (a, b) in enumerate(same_author_pairs):
        sim = F.cosine_similarity(embs_n[i*2:i*2+1], embs_n[i*2+1:i*2+2]).item()
        tlog(logger, f"  [pos pair {i}] cos_sim = {sim:.4f}  (want > 0.3 for SBERT base)")
    offset = len(same_author_pairs) * 2
    for i, (a, b) in enumerate(diff_author_pairs):
        sim = F.cosine_similarity(embs_n[offset+i*2:offset+i*2+1],
                                  embs_n[offset+i*2+1:offset+i*2+2]).item()
        tlog(logger, f"  [neg pair {i}] cos_sim = {sim:.4f}  (want < 0.3 for SBERT base)")

    encoder.train()
    tlog(logger, "  Sanity check passed ✓")


# ============================================================
# Dataset
# ============================================================

class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len):
        self.pairs     = pairs
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        a, b, label = self.pairs[idx]
        ta = self.tokenizer(a, truncation=True, max_length=self.max_len,
                            padding="max_length", return_tensors="pt")
        tb = self.tokenizer(b, truncation=True, max_length=self.max_len,
                            padding="max_length", return_tensors="pt")
        return {
            "input_ids_a":      ta["input_ids"].squeeze(0),
            "attention_mask_a": ta["attention_mask"].squeeze(0),
            "input_ids_b":      tb["input_ids"].squeeze(0),
            "attention_mask_b": tb["attention_mask"].squeeze(0),
            "label":            torch.tensor(label, dtype=torch.float),
        }


# ============================================================
# Evaluation
# ============================================================

def evaluate_cosent(encoder, tokenizer, val_pairs, device, max_len,
                    use_bf16=False, eval_batch=512):
    encoder.eval()
    all_sims, all_labels = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(val_pairs), eval_batch),
                      desc="  Val", leave=False):
            batch   = val_pairs[i:i+eval_batch]
            texts_a = [p[0] for p in batch]
            texts_b = [p[1] for p in batch]
            labels  = [p[2] for p in batch]
            ta = tokenizer(texts_a, truncation=True, max_length=max_len,
                           padding=True, return_tensors="pt").to(device)
            tb = tokenizer(texts_b, truncation=True, max_length=max_len,
                           padding=True, return_tensors="pt").to(device)
            with amp_ctx(use_bf16):
                ea = F.normalize(encode(encoder, ta["input_ids"], ta["attention_mask"]), dim=-1)
                eb = F.normalize(encode(encoder, tb["input_ids"], tb["attention_mask"]), dim=-1)
            sims = F.cosine_similarity(ea.float(), eb.float()).cpu().tolist()
            all_sims.extend(sims)
            all_labels.extend(labels)

    # Search both directions: high sim = same-author (SBERT convention)
    # and low sim = same-author (inverted, just in case)
    best_f1, best_thr, best_direction = 0.0, 0.5, "high"
    for thr in np.arange(0.05, 0.999, 0.01):
        # Direction A: sim >= thr → same-author (label=1)
        preds_a = [1 if s >= thr else 0 for s in all_sims]
        f1_a    = f1_score(all_labels, preds_a, average="macro", zero_division=0)
        if f1_a > best_f1:
            best_f1, best_thr, best_direction = f1_a, float(thr), "high"

        # Direction B: sim < thr → same-author (legacy inverted behaviour)
        preds_b = [1 if s < thr else 0 for s in all_sims]
        f1_b    = f1_score(all_labels, preds_b, average="macro", zero_division=0)
        if f1_b > best_f1:
            best_f1, best_thr, best_direction = f1_b, float(thr), "low"

    pos = [s for s, l in zip(all_sims, all_labels) if l == 1]
    neg = [s for s, l in zip(all_sims, all_labels) if l == 0]
    sep = float(np.mean(pos) - np.mean(neg)) if (pos and neg) else 0.0

    return {
        "f1":          best_f1,
        "threshold":   best_thr,
        "direction":   best_direction,
        "avg_pos_sim": float(np.mean(pos)) if pos else 0.0,
        "avg_neg_sim": float(np.mean(neg)) if neg else 0.0,
        "separation":  sep,
    }


# ============================================================
# Training
# ============================================================

def train_encoder(encoder_name, resume=False):
    logger      = setup_logging(encoder_name)
    set_seed(SEED)
    cfg         = ENCODERS[encoder_name]
    model_name  = cfg["model_name"]
    use_bf16    = COSENT.get("use_bf16", False)
    # Allow disabling compile via env var (useful for deberta_large on 12 GB GPU)
    disable_compile = os.environ.get("DISABLE_COMPILE", "0") == "1"
    use_compile = COSENT.get("use_compile", False) and not disable_compile

    tlog(logger, "=" * 60)
    tlog(logger, f"Encoder : {encoder_name}  ({model_name})")
    tlog(logger, f"Device  : {DEVICE}  bf16={use_bf16}  compile={use_compile}")
    tlog(logger, f"seq_len={COSENT['max_seq_len']}  epochs={COSENT['epochs']}  "
                 f"patience={COSENT['patience']}")
    tlog(logger, f"pairs: max_per_group={COSENT['max_pairs_per_group']}  "
                 f"max_cross={COSENT['max_cross_pairs']}")
    tlog(logger, "=" * 60)

    # ── Data ────────────────────────────────────────────────────────────────
    tlog(logger, "Loading hard dataset...")
    hard_train = load_dataset(DATA_DIR, "hard", "train")
    hard_val   = load_dataset(DATA_DIR, "hard", "validation")
    print_dataset_stats(hard_train, "hard/train")
    print_dataset_stats(hard_val,   "hard/val")

    tlog(logger, "Generating pairs...")
    train_pairs = generate_all_pairs(
        hard_train,
        max_per_group=COSENT["max_pairs_per_group"],
        max_cross=COSENT["max_cross_pairs"],
        neg_pos_ratio=COSENT["neg_pos_ratio"],
    )
    val_pairs = generate_all_pairs(
        hard_val,
        max_per_group=COSENT["max_pairs_per_group"],
        max_cross=COSENT["max_cross_pairs"],
        neg_pos_ratio=COSENT["neg_pos_ratio"],
    )
    tlog(logger, f"Train pairs: {len(train_pairs):,}  |  Val pairs: {len(val_pairs):,}")

    # ── Model ───────────────────────────────────────────────────────────────
    save_dir = os.path.join(CHECKPOINT_DIR, f"{encoder_name}_cosent_best")
    if resume and os.path.exists(save_dir):
        tlog(logger, f"Resuming from {save_dir}")
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
        encoder   = AutoModel.from_pretrained(save_dir).to(DEVICE)
    else:
        tlog(logger, f"Loading fresh: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encoder   = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32
        ).to(DEVICE)

    encoder = encoder.float()  # guarantee fp32 weights

    if use_compile:
        tlog(logger, "Compiling encoder (first batch slow)...")
        encoder = torch.compile(encoder)

    sanity_check(encoder, tokenizer, DEVICE, COSENT["max_seq_len"], use_bf16, logger)

    # ── DataLoader ──────────────────────────────────────────────────────────
    GRAD_ACCUM = COSENT.get("grad_accum_steps", 2)
    phys_batch = COSENT["batch_size"] // GRAD_ACCUM
    tlog(logger, f"Batch: physical={phys_batch} x accum={GRAD_ACCUM} "
                 f"= effective={COSENT['batch_size']}")

    dataset    = PairDataset(train_pairs, tokenizer, COSENT["max_seq_len"])
    dataloader = DataLoader(
        dataset, batch_size=phys_batch, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=(NUM_WORKERS > 0),
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    # ── Optimiser ───────────────────────────────────────────────────────────
    total_opt = (len(dataloader) // GRAD_ACCUM) * COSENT["epochs"]
    warmup    = int(COSENT["warmup_ratio"] * total_opt)
    try:
        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=COSENT["lr"],
            weight_decay=COSENT["weight_decay"], fused=True,
        )
        tlog(logger, "Fused AdamW ✓")
    except TypeError:
        optimizer = torch.optim.AdamW(
            encoder.parameters(), lr=COSENT["lr"],
            weight_decay=COSENT["weight_decay"],
        )
        tlog(logger, "Standard AdamW")

    scheduler  = get_linear_schedule_with_warmup(optimizer, warmup, total_opt)
    scaler     = GradScaler() if (use_bf16 and not torch.cuda.is_bf16_supported()) else None
    tlog(logger, f"Opt steps: {total_opt:,}  warmup: {warmup}  "
                 f"micro-batches/epoch: {len(dataloader):,}")

    # ── Training loop ───────────────────────────────────────────────────────
    best_f1 = 0.0
    patience_counter = 0
    history = []
    nan_reported = False
    epoch_bar = tqdm(range(COSENT["epochs"]), desc="Epochs", unit="epoch")

    for epoch in epoch_bar:
        encoder.train()
        epoch_loss = 0.0
        n_batches  = 0
        t_start    = time.time()
        optimizer.zero_grad()

        batch_bar = tqdm(dataloader, desc=f"  Epoch {epoch+1:02d}",
                         leave=False, unit="batch", dynamic_ncols=True)

        for batch_idx, batch in enumerate(batch_bar):
            ids_a  = batch["input_ids_a"].to(DEVICE, non_blocking=True)
            mask_a = batch["attention_mask_a"].to(DEVICE, non_blocking=True)
            ids_b  = batch["input_ids_b"].to(DEVICE, non_blocking=True)
            mask_b = batch["attention_mask_b"].to(DEVICE, non_blocking=True)
            labels = batch["label"].to(DEVICE, non_blocking=True)

            with amp_ctx(use_bf16):
                ea   = F.normalize(encode(encoder, ids_a, mask_a), dim=-1)
                eb   = F.normalize(encode(encoder, ids_b, mask_b), dim=-1)
                loss = cosent_loss(ea, eb, labels, COSENT["lambda_param"]) / GRAD_ACCUM

            if torch.isnan(loss) and not nan_reported:
                nan_reported = True
                tlog(logger,
                     f"  [NaN @ epoch {epoch+1} step {batch_idx+1}] "
                     f"ea_nan={torch.isnan(ea).any().item()} "
                     f"eb_nan={torch.isnan(eb).any().item()} "
                     f"labels={labels.unique().tolist()}")

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % GRAD_ACCUM == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), COSENT["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(encoder.parameters(), COSENT["max_grad_norm"])
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            lv = loss.item() * GRAD_ACCUM
            if not np.isnan(lv):
                epoch_loss += lv
                n_batches  += 1
            avg = epoch_loss / n_batches if n_batches else float("nan")
            batch_bar.set_postfix(
                loss=f"{avg:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
                nan="⚠" if np.isnan(lv) else "",
            )

        epoch_time = time.time() - t_start
        avg_loss   = epoch_loss / n_batches if n_batches else float("nan")

        if (epoch + 1) % COSENT["eval_every_n_epochs"] == 0:
            vs = evaluate_cosent(
                encoder, tokenizer, val_pairs, DEVICE,
                COSENT["max_seq_len"], use_bf16=use_bf16,
            )
            tlog(logger,
                 f"Epoch {epoch+1:2d}/{COSENT['epochs']} | "
                 f"Loss {avg_loss:.4f} | "
                 f"Val F1 {vs['f1']:.4f} (thr={vs['threshold']:.2f} dir={vs['direction']}) | "
                 f"Sep {vs['separation']:.4f} "
                 f"(pos={vs['avg_pos_sim']:.3f} neg={vs['avg_neg_sim']:.3f}) | "
                 f"{epoch_time/60:.1f} min")
            epoch_bar.set_postfix(
                loss=f"{avg_loss:.4f}",
                val_f1=f"{vs['f1']:.4f}",
                sep=f"{vs['separation']:.4f}",
            )
            history.append({
                "epoch":      epoch + 1,
                "loss":       avg_loss,
                "val_f1":     vs["f1"],
                "separation": vs["separation"],
                "direction":  vs["direction"],
            })

            # Warn if encoder is still collapsing after epoch 3
            if epoch >= 2 and vs["separation"] < 0.05:
                tlog(logger,
                     f"  ⚠ WARNING: separation={vs['separation']:.4f} after epoch {epoch+1}. "
                     f"Encoder may be collapsing. Check model_name and loss direction.")

            if vs["f1"] > best_f1:
                best_f1 = vs["f1"]
                patience_counter = 0
                os.makedirs(save_dir, exist_ok=True)
                raw = encoder._orig_mod if hasattr(encoder, "_orig_mod") else encoder
                raw.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                torch.save(
                    {
                        "epoch":     epoch + 1,
                        "best_f1":   best_f1,
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "direction": vs["direction"],
                    },
                    os.path.join(save_dir, "training_state.pt"),
                )
                tlog(logger, f"  ✓ Best saved  F1={best_f1:.4f}  sep={vs['separation']:.4f}")
            else:
                patience_counter += 1
                tlog(logger, f"  No improvement ({patience_counter}/{COSENT['patience']})")
                if patience_counter >= COSENT["patience"]:
                    tlog(logger, f"  Early stopping at epoch {epoch+1}")
                    break

    json.dump(
        history,
        open(os.path.join(LOG_DIR, f"cosent_{encoder_name}_history.json"), "w"),
        indent=2,
    )
    tlog(logger, "=" * 60)
    tlog(logger, f"DONE {encoder_name} | Best Val F1: {best_f1:.4f} | Checkpoint: {save_dir}")
    tlog(logger, "=" * 60)
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, default=None, choices=list(ENCODERS.keys()))
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    encoders_to_train = [args.encoder] if args.encoder else list(ENCODERS.keys())
    results = {}
    for enc in encoders_to_train:
        results[enc] = train_encoder(enc, resume=args.resume)

    print("\n" + "=" * 40)
    print("CoSENT Training Summary")
    for name, f1 in results.items():
        print(f"  {name:15s}: Val F1 = {f1:.4f}")
    print("=" * 40)