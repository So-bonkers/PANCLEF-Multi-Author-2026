#!/usr/bin/env python3
"""
ensemble_medium.py — Weighted classifier ensemble for medium (no shards needed).
Usage: python ensemble_medium.py --difficulty medium
"""

import argparse
import logging
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score

from config import *
from data_utils import load_dataset, compute_pk, compute_windowdiff

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_classifier_probs_and_labels(difficulty, split='val'):
    """
    Load per-boundary classifier probabilities from feature cache.
    Returns: probs_dict {enc_name: [total_boundaries]}, labels [total_boundaries]
    """
    # Load meta to get labels and doc structure
    meta_path = os.path.join(FEATURE_CACHE_DIR, f"meta_{difficulty}_{split}.pt")
    meta = torch.load(meta_path, weights_only=False)
    
    # Collect valid docs with labels
    docs = []
    for idx, entry in enumerate(meta):
        if entry.get("ensemble_prob") is None or not entry.get("changes"):
            continue
        docs.append({
            "idx": idx,
            "changes": entry["changes"],
            "n_bounds": len(entry["changes"]),
        })
    
    # Load probabilities from each encoder
    probs = {}
    for enc_name in ENCODERS:
        prob_path = os.path.join(FEATURE_CACHE_DIR, f"prob_{enc_name}_{difficulty}_{split}.pt")
        if not os.path.exists(prob_path):
            logger.warning(f"Missing {prob_path}, skipping {enc_name}")
            continue
            
        prob_list = torch.load(prob_path, weights_only=False)
        all_probs = []
        
        for doc in docs:
            p = prob_list[doc["idx"]]
            if p is None:
                # Fallback to ensemble prob
                ep = meta[doc["idx"]]["ensemble_prob"]
                if ep.dim() == 0:
                    p = ep.unsqueeze(0).expand(doc["n_bounds"])
                else:
                    p = ep[:doc["n_bounds"]]
            all_probs.extend(p.tolist())
        
        probs[enc_name] = np.array(all_probs)
        logger.info(f"  Loaded {enc_name}: {len(all_probs)} probs")
    
    # Build labels array
    labels = []
    for doc in docs:
        labels.extend(doc["changes"])
    labels = np.array(labels)
    
    logger.info(f"Total boundaries: {len(labels)}, positive rate: {labels.mean():.4f}")
    return probs, labels, docs


def evaluate_ensemble(probs_dict, labels, weights, threshold):
    """Evaluate weighted ensemble."""
    # Weighted average
    ensemble = sum(w * probs_dict[name] for name, w in weights.items() if name in probs_dict)
    ensemble = ensemble / sum(w for name, w in weights.items() if name in probs_dict)
    
    preds = (ensemble > threshold).astype(int)
    
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    f1b = f1_score(labels, preds, average='binary', zero_division=0)
    
    # Per-doc metrics (approximate since we flattened)
    # For proper Pk/WD we'd need doc structure, but F1 is the main metric
    
    return {
        "f1_macro": f1,
        "f1_binary": f1b,
        "precision": (preds & labels).sum() / preds.sum() if preds.sum() > 0 else 0,
        "recall": (preds & labels).sum() / labels.sum() if labels.sum() > 0 else 0,
        "change_rate_pred": preds.mean(),
        "change_rate_true": labels.mean(),
    }


def optimize_weights(probs_dict, labels, n_trials=500):
    """Random search over weights and thresholds."""
    enc_names = list(probs_dict.keys())
    logger.info(f"Optimizing over: {enc_names}")
    
    best_f1, best_w, best_thr = 0, None, 0.5
    
    for trial in range(n_trials):
        # Sample weights from Dirichlet (ensures sum to 1)
        weights = np.random.dirichlet(np.ones(len(enc_names)))
        w_dict = {name: weights[i] for i, name in enumerate(enc_names)}
        
        # Sample threshold
        # Medium needs low threshold due to class imbalance
        thr = np.random.uniform(0.02, 0.30) if labels.mean() < 0.1 else np.random.uniform(0.1, 0.6)
        
        m = evaluate_ensemble(probs_dict, labels, w_dict, thr)
        f1 = m["f1_macro"]
        
        if f1 > best_f1:
            best_f1, best_w, best_thr = f1, w_dict, thr
            logger.info(f"  Trial {trial}: F1={f1:.4f} | w={ {k: round(v,3) for k,v in w_dict.items()} } | thr={thr:.3f}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST ENSEMBLE: F1={best_f1:.4f}")
    logger.info(f"  Weights: { {k: round(v,4) for k,v in best_w.items()} }")
    logger.info(f"  Threshold: {best_thr:.3f}")
    logger.info(f"{'='*60}")
    
    return best_w, best_thr, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--difficulty', default='medium', choices=DIFFICULTIES)
    parser.add_argument('--split', default='val')
    parser.add_argument('--n_trials', type=int, default=500)
    args = parser.parse_args()

    logger.info(f"Loading data for {args.difficulty}/{args.split}...")
    probs, labels, docs = load_classifier_probs_and_labels(args.difficulty, args.split)
    
    if len(probs) < 2:
        logger.error("Need at least 2 classifiers for ensemble")
        return
    
    logger.info(f"Optimizing ensemble with {args.n_trials} trials...")
    best_w, best_thr, best_f1 = optimize_weights(probs, labels, args.n_trials)
    
    # Final evaluation with best
    final = evaluate_ensemble(probs, labels, best_w, best_thr)
    logger.info(f"\nFinal metrics: F1={final['f1_macro']:.4f} | F1b={final['f1_binary']:.4f} | "
                f"P={final['precision']:.3f} | R={final['recall']:.3f}")


if __name__ == '__main__':
    main()