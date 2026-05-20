"""
PAN 2026 — Step 4a: Extract ONE shard (one encoder × one difficulty × one split).

Run this script once per combination, then run step4_merge.py to merge,
then step4_train_bilstm.py to train.

Usage:
    python step4_extract_shard.py --encoder deberta --difficulty easy --split train
    python step4_extract_shard.py --encoder deberta --difficulty easy --split val
    python step4_extract_shard.py --encoder roberta  --difficulty hard --split train
    ... etc.

Tip — run all combos in a shell loop:
    for enc in deberta roberta mpnet; do
      for diff in easy medium hard; do
        for split in train val; do
          python step4_extract_shard.py --encoder $enc --difficulty $diff --split $split
        done
      done
    done
"""

import argparse, logging, os, sys, time
import torch
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from config import *
from data_utils import load_dataset
from models import StyleChangeClassifier

# ── logging ──────────────────────────────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "step4_extract.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── memory helpers ────────────────────────────────────────────────────────────

def log_ram():
    try:
        with open("/proc/meminfo") as f:
            lines = {l.split(":")[0]: int(l.split()[1])
                     for l in f if ":" in l}
        total = lines.get("MemTotal", 0) / 1e6
        avail = lines.get("MemAvailable", 0) / 1e6
        return f"RAM used {total-avail:.1f}/{total:.1f} GB"
    except Exception:
        return "RAM n/a"


def log_gpu():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        res   = torch.cuda.memory_reserved()  / 1e9
        return f"GPU alloc {alloc:.2f} GB / reserved {res:.2f} GB"
    return "GPU n/a"


# ── encode helper ─────────────────────────────────────────────────────────────

def encode_mean(encoder, input_ids, attention_mask):
    out  = encoder(input_ids=input_ids, attention_mask=attention_mask)
    h    = out.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


# ── main extraction ───────────────────────────────────────────────────────────

def extract_shard(enc_name: str, difficulty: str, split: str):
    split_name = "validation" if split == "val" else "train"
    shard_path = os.path.join(
        FEATURE_CACHE_DIR, f"shard_{enc_name}_{difficulty}_{split}.pt"
    )

    if os.path.exists(shard_path):
        size_mb = os.path.getsize(shard_path) / 1e6
        logger.info(f"Shard already exists ({size_mb:.1f} MB), nothing to do: {shard_path}")
        return

    logger.info("=" * 60)
    logger.info(f"Encoder : {enc_name}")
    logger.info(f"Difficulty: {difficulty}")
    logger.info(f"Split   : {split} ({split_name})")
    logger.info(f"Output  : {shard_path}")
    logger.info(f"Device  : {DEVICE}")
    logger.info(f"{log_ram()} | {log_gpu()}")
    logger.info("=" * 60)

    # ── 1. load text data ────────────────────────────────────────────────────
    logger.info(f"[1/5] Loading dataset {difficulty}/{split_name}...")
    problems = load_dataset(DATA_DIR, difficulty, split_name)
    logger.info(f"      {len(problems)} docs loaded | {log_ram()}")

    # ── 2. load encoder ──────────────────────────────────────────────────────
    enc_dir    = os.path.join(CHECKPOINT_DIR, f"{enc_name}_cosent_best")
    hidden_dim = ENCODERS[enc_name]["hidden_dim"]
    batch_size = 128 if hidden_dim >= 1024 else 512

    logger.info(f"[2/5] Loading encoder from {enc_dir}  (hidden={hidden_dim}, batch={batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained(enc_dir)
    encoder   = AutoModel.from_pretrained(enc_dir).to(DEVICE).eval()
    logger.info(f"      Encoder loaded | {log_ram()} | {log_gpu()}")

    # ── 3. load classifier ───────────────────────────────────────────────────
    clf_path = os.path.join(CHECKPOINT_DIR, f"{enc_name}_{difficulty}_classifier.pt")
    logger.info(f"[3/5] Loading classifier from {clf_path}...")
    ckpt = torch.load(clf_path, map_location=DEVICE, weights_only=False)
    clf  = StyleChangeClassifier(hidden_dim=hidden_dim).to(DEVICE)
    clf.load_state_dict(ckpt["model_state"])
    clf.eval()
    logger.info(f"      Classifier loaded | {log_ram()} | {log_gpu()}")

    # ── 4. flatten all sentence pairs ────────────────────────────────────────
    logger.info("[4/5] Flattening sentence pairs...")
    all_a, all_b, index = [], [], []
    for doc_idx, prob in enumerate(problems):
        sents = prob["sentences"]
        for i in range(len(sents) - 1):
            all_a.append(sents[i])
            all_b.append(sents[i + 1])
            index.append((doc_idx, i))

    if not all_a:
        torch.save([None] * len(problems), shard_path)
        logger.info("No sentence pairs found — saved empty shard.")
        return

    total_pairs   = len(all_a)
    total_batches = (total_pairs + batch_size - 1) // batch_size
    logger.info(
        f"      {total_pairs} pairs across {len(problems)} docs | "
        f"{total_batches} batches | {log_ram()}"
    )

    # ── 5. forward pass ───────────────────────────────────────────────────────
    logger.info("[5/5] Running forward pass...")
    max_len    = COSENT["max_seq_len"]
    all_feats  = []
    all_probs  = []
    t_start    = time.time()

    with torch.no_grad():
        for bn, i in enumerate(
            tqdm(range(0, total_pairs, batch_size),
                 desc=f"{enc_name}/{difficulty}/{split}",
                 total=total_batches,
                 unit="batch",
                 leave=True)
        ):
            ta = tokenizer(
                all_a[i : i + batch_size],
                truncation=True, max_length=max_len,
                padding=True, return_tensors="pt",
            ).to(DEVICE)
            tb = tokenizer(
                all_b[i : i + batch_size],
                truncation=True, max_length=max_len,
                padding=True, return_tensors="pt",
            ).to(DEVICE)

            with autocast("cuda", dtype=torch.bfloat16):
                ea = encode_mean(encoder, ta["input_ids"], ta["attention_mask"]).float()
                eb = encode_mean(encoder, tb["input_ids"], tb["attention_mask"]).float()

            all_feats.append(classifier_features(clf, ea, eb))
            all_probs.append(torch.sigmoid(clf(ea, eb)).detach().cpu())

            if (bn + 1) % 50 == 0:
                elapsed    = time.time() - t_start
                done_pairs = min(i + batch_size, total_pairs)
                pct        = 100 * done_pairs / total_pairs
                eta        = (elapsed / done_pairs) * (total_pairs - done_pairs)
                logger.info(
                    f"  batch {bn+1:>5}/{total_batches} | "
                    f"{done_pairs}/{total_pairs} pairs ({pct:.1f}%) | "
                    f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m | "
                    f"{log_ram()} | {log_gpu()}"
                )
                torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    logger.info(
        f"  Forward pass complete: {elapsed/60:.1f} min "
        f"({total_pairs / elapsed:.0f} pairs/s)"
    )

    # ── free encoder + classifier before building per-doc tensors ────────────
    logger.info("  Freeing encoder and classifier...")
    del encoder, tokenizer, clf
    torch.cuda.empty_cache()
    logger.info(f"  Freed | {log_ram()} | {log_gpu()}")

    # ── assemble per-doc tensors ──────────────────────────────────────────────
    logger.info("  Assembling per-doc tensors...")
    all_feats_cat = torch.cat(all_feats, 0)   # [N_pairs, feat_dim]
    all_probs_cat = torch.cat(all_probs, 0)   # [N_pairs]
    del all_feats, all_probs

    n_bounds = {}
    for doc_idx, b_idx in index:
        n_bounds[doc_idx] = max(n_bounds.get(doc_idx, 0), b_idx + 1)

    doc_feats = [None] * len(problems)
    doc_probs = [None] * len(problems)
    for fi, (doc_idx, b_idx) in enumerate(index):
        if doc_feats[doc_idx] is None:
            doc_feats[doc_idx] = torch.zeros(n_bounds[doc_idx], all_feats_cat.shape[1])
            doc_probs[doc_idx] = torch.zeros(n_bounds[doc_idx])
        doc_feats[doc_idx][b_idx] = all_feats_cat[fi]
        doc_probs[doc_idx][b_idx] = all_probs_cat[fi]

    del all_feats_cat, all_probs_cat

    shard = [
        {"feat": doc_feats[i], "prob": doc_probs[i]}
        for i in range(len(problems))
    ]
    torch.save(shard, shard_path)
    size_mb = os.path.getsize(shard_path) / 1e6
    logger.info(f"  ✓ Shard saved: {shard_path}  ({size_mb:.1f} MB) | {log_ram()}")


def classifier_features(clf, ea, eb):
    """Wrapper so we can call clf.get_features and detach in one place."""
    return clf.get_features(ea, eb).detach().cpu()


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract one shard: one encoder × one difficulty × one split."
    )
    parser.add_argument("--encoder",    required=True, choices=list(ENCODERS.keys()),
                        help="Encoder name, e.g. deberta")
    parser.add_argument("--difficulty", required=True, choices=DIFFICULTIES,
                        help="easy | medium | hard")
    parser.add_argument("--split",      required=True, choices=["train", "val"],
                        help="train or val")
    args = parser.parse_args()

    extract_shard(args.encoder, args.difficulty, args.split)