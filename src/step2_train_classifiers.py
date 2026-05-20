"""
PAN 2026 — Step 2: Train per-difficulty MLP classifiers.
9 models: 3 encoders × 3 difficulties. Frozen encoder — trains MLP only.

Fixes vs original:
  • extract_embeddings uses mean pooling (matches step1 encode())
  • embedding extraction uses bf16 autocast for speed
  • batch_size bumped to 512 (embeddings are tiny, no grad storage)
  • tqdm progress on extraction
  • threshold sweep starts at 0.05 to catch medium difficulty properly

Usage:
    python step2_train_classifiers.py
    python step2_train_classifiers.py --encoder deberta --difficulty hard
"""

import argparse
import logging
import os
import sys
import time
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score
from tqdm import tqdm

from config import *
from data_utils import load_dataset, create_sentence_pairs, evaluate_document_level, format_metrics
from models import StyleChangeClassifier
from losses import FocalBCELoss, rdrop_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "step2_classifiers.log")),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


# ── Mean pooling — must match step1 encode() exactly ──────────────────────
def encode_batch(encoder, input_ids, attention_mask):
    out    = encoder(input_ids=input_ids, attention_mask=attention_mask)
    hidden = out.last_hidden_state
    mask   = attention_mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class EmbeddingDataset(Dataset):
    def __init__(self, emb_a, emb_b, labels):
        self.emb_a = emb_a; self.emb_b = emb_b; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return self.emb_a[idx], self.emb_b[idx], self.labels[idx]


def extract_embeddings(encoder, tokenizer, pairs, device, max_len, batch_size=256):
    """Extract mean-pool embeddings from frozen encoder. bf16 for speed."""
    use_bf16 = torch.cuda.is_bf16_supported()
    encoder.eval()
    all_ea, all_eb, all_labels = [], [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(pairs), batch_size), desc="  Extracting", leave=False):
            batch   = pairs[i:i+batch_size]
            texts_a = [p["text_a"] for p in batch]
            texts_b = [p["text_b"] for p in batch]
            labels  = [p["label"]  for p in batch]
            ta = tokenizer(texts_a, truncation=True, max_length=max_len,
                           padding=True, return_tensors="pt").to(device)
            tb = tokenizer(texts_b, truncation=True, max_length=max_len,
                           padding=True, return_tensors="pt").to(device)
            ctx = autocast(dtype=torch.bfloat16) if use_bf16 else torch.no_grad()
            with autocast(dtype=torch.bfloat16) if use_bf16 else _NullCtx():
                ea = encode_batch(encoder, ta["input_ids"], ta["attention_mask"]).float().cpu()
                eb = encode_batch(encoder, tb["input_ids"], tb["attention_mask"]).float().cpu()
            all_ea.append(ea); all_eb.append(eb); all_labels.extend(labels)
    return torch.cat(all_ea), torch.cat(all_eb), torch.tensor(all_labels, dtype=torch.float)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def train_one_classifier(encoder_name, difficulty):
    cfg_enc  = ENCODERS[encoder_name]
    cfg_loss = LOSS_CONFIG[difficulty]
    hidden   = cfg_enc["hidden_dim"]

    logger.info(f"\n{'='*50}")
    logger.info(f"Classifier: {encoder_name} / {difficulty}  "
                f"pos_weight={cfg_loss['pos_weight']} gamma={cfg_loss['focal_gamma']}")
    logger.info(f"{'='*50}")

    enc_dir = os.path.join(CHECKPOINT_DIR, f"{encoder_name}_cosent_best")
    if not os.path.exists(enc_dir):
        logger.error(f"Encoder checkpoint not found: {enc_dir} — run step1 first.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(enc_dir)
    encoder   = AutoModel.from_pretrained(enc_dir).to(DEVICE).eval()

    train_probs = load_dataset(DATA_DIR, difficulty, "train")
    val_probs   = load_dataset(DATA_DIR, difficulty, "validation")
    train_pairs = create_sentence_pairs(train_probs)
    val_pairs   = create_sentence_pairs(val_probs)

    pos = sum(1 for p in train_pairs if p["label"] == 1)
    logger.info(f"Train pairs: {len(train_pairs):,} | Val pairs: {len(val_pairs):,} | "
                f"Pos rate: {pos/len(train_pairs)*100:.1f}%")

    logger.info("Extracting train embeddings...")
    tr_ea, tr_eb, tr_lbl = extract_embeddings(encoder, tokenizer, train_pairs,
                                               DEVICE, COSENT["max_seq_len"])
    logger.info("Extracting val embeddings...")
    va_ea, va_eb, va_lbl = extract_embeddings(encoder, tokenizer, val_pairs,
                                               DEVICE, COSENT["max_seq_len"])
    del encoder; torch.cuda.empty_cache()

    classifier = StyleChangeClassifier(hidden_dim=hidden,
                                       mlp_sizes=CLASSIFIER["hidden_sizes"],
                                       dropout_rates=CLASSIFIER["dropout_rates"]).to(DEVICE)
    base_loss  = FocalBCELoss(gamma=cfg_loss["focal_gamma"], pos_weight=cfg_loss["pos_weight"])
    optimizer  = torch.optim.AdamW(classifier.parameters(), lr=CLASSIFIER["lr"],
                                   weight_decay=CLASSIFIER["weight_decay"])

    loader = DataLoader(EmbeddingDataset(tr_ea, tr_eb, tr_lbl),
                        batch_size=CLASSIFIER["batch_size"], shuffle=True, pin_memory=True)

    best_f1   = 0.0
    save_path = os.path.join(CHECKPOINT_DIR, f"{encoder_name}_{difficulty}_classifier.pt")

    for epoch in range(CLASSIFIER["epochs"]):
        classifier.train()
        epoch_loss = 0.0; n = 0
        for ea, eb, lbl in loader:
            ea, eb, lbl = ea.to(DEVICE), eb.to(DEVICE), lbl.to(DEVICE)
            l1 = classifier(ea, eb)
            l2 = classifier(ea, eb)  # different dropout mask
            loss = rdrop_loss(l1, l2, lbl, base_loss, CLASSIFIER["rdrop_alpha"])
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            epoch_loss += loss.item(); n += 1

        classifier.eval()
        with torch.no_grad():
            logits = classifier(va_ea.to(DEVICE), va_eb.to(DEVICE))
            probs  = torch.sigmoid(logits).cpu().numpy()

        best_ep_f1, best_thr = 0.0, 0.5
        for thr in np.arange(0.05, 0.65, 0.02):
            preds = (probs > thr).astype(int)
            f1    = f1_score(va_lbl.numpy(), preds, average="macro", zero_division=0)
            if f1 > best_ep_f1: best_ep_f1, best_thr = f1, thr

        logger.info(f"  Epoch {epoch+1:2d}/{CLASSIFIER['epochs']} | "
                    f"Loss {epoch_loss/n:.4f} | Val F1 {best_ep_f1:.4f} (thr={best_thr:.2f})")

        if best_ep_f1 > best_f1:
            best_f1 = best_ep_f1
            torch.save({"model_state": classifier.state_dict(), "best_f1": best_f1,
                        "best_threshold": best_thr, "hidden_dim": hidden,
                        "encoder_name": encoder_name, "difficulty": difficulty}, save_path)
            logger.info(f"  ✓ Saved F1={best_f1:.4f} thr={best_thr:.2f}")

    logger.info(f"Done {encoder_name}/{difficulty} → F1={best_f1:.4f}")
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder",    type=str, default=None, choices=list(ENCODERS.keys()))
    parser.add_argument("--difficulty", type=str, default=None, choices=DIFFICULTIES)
    args = parser.parse_args()
    encoders = [args.encoder]    if args.encoder    else list(ENCODERS.keys())
    diffs    = [args.difficulty] if args.difficulty else DIFFICULTIES
    results  = {}
    for enc in encoders:
        for diff in diffs:
            results[f"{enc}/{diff}"] = train_one_classifier(enc, diff)
    print("\n" + "="*50)
    print("Classifier Training Summary:")
    for k, f1 in results.items():
        print(f"  {k}: F1={f1:.4f}" if f1 else f"  {k}: FAILED")
    print("="*50)