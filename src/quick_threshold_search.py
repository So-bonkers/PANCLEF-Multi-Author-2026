#!/usr/bin/env python3
"""
quick_threshold_search.py — Find best threshold for existing Bi-LSTM checkpoint.
Usage: python quick_threshold_search.py --checkpoint checkpoints/bilstm_hard.pt --difficulty hard
"""

import argparse
import logging
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

from config import *
from data_utils import load_dataset, evaluate_document_level
from models import SequentialCPDRefinement

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_meta(difficulty, split):
    meta_path = os.path.join(FEATURE_CACHE_DIR, f"meta_{difficulty}_{split}.pt")
    raw = torch.load(meta_path, weights_only=False)
    meta = []
    for idx, entry in enumerate(raw):
        if entry.get("ensemble_prob") is None or not entry.get("changes"):
            continue
        meta.append({"idx": idx, "changes": entry["changes"], "ensemble_prob": entry["ensemble_prob"]})
    return meta


def build_docs_simple(meta, difficulty, split):
    """Load features only, use ensemble_prob as fallback for all probs."""
    valid_idxs = {m["idx"] for m in meta}
    docs = {m["idx"]: {"idx": m["idx"], "changes": m["changes"], 
                       "ensemble_prob": m["ensemble_prob"], "features": {}, "probs": {}} 
            for m in meta}
    
    for enc_name in ENCODERS:
        feat_path = os.path.join(FEATURE_CACHE_DIR, f"feat_{enc_name}_{difficulty}_{split}.pt")
        feat_list = torch.load(feat_path, weights_only=False)
        for idx in valid_idxs:
            feat = feat_list[idx]
            if feat is None: continue
            docs[idx]["features"][enc_name] = feat
            n_bounds = feat.shape[0]
            # Use ensemble prob as fallback (your current broken behavior, but consistent)
            ep = docs[idx]["ensemble_prob"]
            if ep.dim() == 0:
                ep = ep.unsqueeze(0).expand(n_bounds)
            elif ep.shape[0] < n_bounds:
                pad = ep[-1:].expand(n_bounds - ep.shape[0])
                ep = torch.cat([ep, pad])
            else:
                ep = ep[:n_bounds]
            docs[idx]["probs"][enc_name] = ep
        del feat_list
    
    return [d for d in docs.values() if len(d["features"]) == len(ENCODERS)]


def run_val(model, docs):
    model.eval()
    preds = []
    with torch.no_grad():
        for doc in tqdm(docs, desc="Val"):
            n_bounds = list(doc["features"].values())[0].shape[0]
            ef = {k: v.unsqueeze(0).to(DEVICE) for k, v in doc["features"].items()}
            ep = {k: v.unsqueeze(0).unsqueeze(-1).to(DEVICE) for k, v in doc["probs"].items()}
            ens = doc["ensemble_prob"]
            if ens.dim() == 0:
                ens = ens.unsqueeze(0).expand(n_bounds)
            elif ens.shape[0] < n_bounds:
                pad = ens[-1:].expand(n_bounds - ens.shape[0])
                ens = torch.cat([ens, pad])
            else:
                ens = ens[:n_bounds]
            ens = ens.unsqueeze(0).unsqueeze(-1).to(DEVICE)
            logits = model(ef, ep, ens).squeeze(0)
            preds.append(torch.sigmoid(logits).cpu().numpy().tolist())
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--difficulty', required=True, choices=DIFFICULTIES)
    parser.add_argument('--threshold_min', type=float, default=0.05)
    parser.add_argument('--threshold_max', type=float, default=0.95)
    parser.add_argument('--threshold_step', type=float, default=0.01)
    args = parser.parse_args()

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    
    # Build model with same config as checkpoint
    encoder_dims = {n: ENCODERS[n]["hidden_dim"] for n in ENCODERS}
    model = SequentialCPDRefinement(
        encoder_dims=encoder_dims,
        feature_proj_dim=BILSTM["feature_proj_dim"],
        hidden_dim=BILSTM["hidden_dim"],
        num_layers=BILSTM["num_layers"],
        dropout=BILSTM["dropout"],
        use_positional_features=BILSTM.get("use_positional_features", True),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    logger.info(f"Loading validation data for {args.difficulty}...")
    val_problems = load_dataset(DATA_DIR, args.difficulty, "validation")
    val_meta = load_meta(args.difficulty, "val")
    val_docs = build_docs_simple(val_meta, args.difficulty, "val")
    
    logger.info(f"Running validation with {len(val_docs)} docs...")
    all_preds = run_val(model, val_docs)
    
    # Grid search
    best_f1, best_thr, best_metrics = 0, 0.5, None
    thresholds = np.arange(args.threshold_min, args.threshold_max, args.threshold_step)
    
    logger.info(f"Searching {len(thresholds)} thresholds from {args.threshold_min} to {args.threshold_max}...")
    for thr in thresholds:
        m = evaluate_document_level(val_problems, all_preds, threshold=thr)
        f1 = m["f1_macro"]
        if f1 > best_f1:
            best_f1, best_thr, best_metrics = f1, thr, m
            logger.info(f"  New best: thr={thr:.3f} | F1={f1:.4f} | F1b={m['f1_binary']:.4f} | Pk={m['pk']:.4f}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"BEST: thr={best_thr:.3f} | F1={best_f1:.4f}")
    logger.info(f"  F1b={best_metrics['f1_binary']:.4f} | Pk={best_metrics['pk']:.4f} | WD={best_metrics['windowdiff']:.4f}")
    logger.info(f"  cr_pred={best_metrics['change_rate_pred']:.4f} | cr_true={best_metrics['change_rate_true']:.4f}")
    logger.info(f"{'='*60}")
    
    # Save results
    result_path = os.path.join(LOG_DIR, f"threshold_search_{args.difficulty}.json")
    import json
    with open(result_path, 'w') as f:
        json.dump({
            "best_threshold": float(best_thr),
            "best_f1": float(best_f1),
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                       for k, v in best_metrics.items()}
        }, f, indent=2)
    logger.info(f"Saved results to {result_path}")


if __name__ == '__main__':
    main()