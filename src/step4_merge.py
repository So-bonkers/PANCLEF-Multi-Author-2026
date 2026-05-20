"""
PAN 2026 — Step 4b: Finalise shards (no giant merged object).

Instead of loading all encoder shards into RAM at once, this script:
  1. Validates all shards exist
  2. Processes one encoder at a time, saving feat_{enc}_{diff}_{split}.pt
  3. Accumulates ensemble probs as a running sum (no extra RAM)
  4. Saves a tiny meta_{diff}_{split}.pt with only changes + ensemble_prob

The BiLSTM trainer loads one encoder's feat file at a time per batch.

Usage:
    python step4_merge.py --difficulty easy  --split train
    python step4_merge.py --difficulty easy  --split val
    python step4_merge.py --difficulty medium --split train
    ... etc.
"""

import argparse, json, logging, os, sys
import torch

from config import *
from data_utils import load_dataset

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "step4_merge.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


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


def merge(difficulty: str, split: str):
    split_name = "validation" if split == "val" else "train"
    meta_path  = os.path.join(FEATURE_CACHE_DIR, f"meta_{difficulty}_{split}.pt")

    if os.path.exists(meta_path):
        logger.info(f"Meta cache already exists, nothing to do: {meta_path}")
        return

    # Verify all shards present
    missing = []
    for enc_name in ENCODERS:
        sp = os.path.join(FEATURE_CACHE_DIR,
                          f"shard_{enc_name}_{difficulty}_{split}.pt")
        if not os.path.exists(sp):
            missing.append(sp)
    if missing:
        logger.error("Missing shards — run step4_extract_shard.py first:")
        for m in missing:
            logger.error(f"  {m}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Finalising: difficulty={difficulty}  split={split}")
    logger.info(f"Peak RAM = one shard at a time (~20 GB max)")
    logger.info(f"{log_ram()}")
    logger.info("=" * 60)

    # Load only the 'changes' labels (text strings, tiny)
    logger.info(f"[1/3] Loading labels for {difficulty}/{split_name}...")
    problems = load_dataset(DATA_DIR, difficulty, split_name)
    n_docs   = len(problems)
    meta     = [{"changes": p["changes"], "ensemble_prob": None} for p in problems]
    del problems
    logger.info(f"      {n_docs} docs, labels only | {log_ram()}")

    # Ensemble weights
    ens_cfg = json.load(open(os.path.join(CHECKPOINT_DIR, "ensemble_config.json")))
    weights = ens_cfg[difficulty]["weights"]

    # Running ensemble prob sum — never holds more than one shard at a time
    ensemble_probs = [None] * n_docs

    logger.info(f"[2/3] Processing shards one encoder at a time...")
    for enc_name in sorted(ENCODERS.keys()):
        shard_path = os.path.join(FEATURE_CACHE_DIR,
                                  f"shard_{enc_name}_{difficulty}_{split}.pt")
        feat_path  = os.path.join(FEATURE_CACHE_DIR,
                                  f"feat_{enc_name}_{difficulty}_{split}.pt")
        size_mb    = os.path.getsize(shard_path) / 1e6

        logger.info(f"  [{enc_name}] Loading shard ({size_mb:.1f} MB) | {log_ram()}")
        shard = torch.load(shard_path, weights_only=False)

        # Save features for this encoder as its own separate file
        if not os.path.exists(feat_path):
            feat_only = [
                (entry["feat"] if entry is not None and entry["feat"] is not None else None)
                for entry in shard
            ]
            torch.save(feat_only, feat_path)
            logger.info(f"  [{enc_name}] Saved {feat_path}  "
                        f"({os.path.getsize(feat_path)/1e6:.1f} MB)")
        else:
            logger.info(f"  [{enc_name}] feat file already exists, skipping save")

        # Accumulate weighted prob into running sum
        w = weights.get(enc_name, 0.0)
        for idx, entry in enumerate(shard):
            if entry is not None and entry["prob"] is not None:
                wp = w * entry["prob"]
                ensemble_probs[idx] = wp if ensemble_probs[idx] is None \
                                      else ensemble_probs[idx] + wp

        del shard
        logger.info(f"  [{enc_name}] Shard freed | {log_ram()}")

    # Write ensemble probs into meta
    for idx in range(n_docs):
        if ensemble_probs[idx] is not None:
            meta[idx]["ensemble_prob"] = ensemble_probs[idx]
    del ensemble_probs

    # Save meta (tiny: only changes labels + ensemble_prob tensors)
    logger.info(f"[3/3] Saving meta cache...")
    torch.save(meta, meta_path)
    logger.info(f"  ✓ {meta_path}  ({os.path.getsize(meta_path)/1e6:.1f} MB) | {log_ram()}")

    # Delete raw shards to free disk
    for enc_name in ENCODERS:
        sp = os.path.join(FEATURE_CACHE_DIR,
                          f"shard_{enc_name}_{difficulty}_{split}.pt")
        if os.path.exists(sp):
            os.remove(sp)
            logger.info(f"  Removed shard: {sp}")

    logger.info(f"\nDone. Files on disk for {difficulty}/{split}:")
    logger.info(f"  {meta_path}")
    for enc_name in sorted(ENCODERS.keys()):
        fp = os.path.join(FEATURE_CACHE_DIR,
                          f"feat_{enc_name}_{difficulty}_{split}.pt")
        if os.path.exists(fp):
            logger.info(f"  {fp}  ({os.path.getsize(fp)/1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", required=True, choices=DIFFICULTIES)
    parser.add_argument("--split",      required=True, choices=["train", "val"])
    args = parser.parse_args()
    merge(args.difficulty, args.split)