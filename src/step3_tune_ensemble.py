"""
PAN 2026 — Step 3: Tune ensemble weights + thresholds per difficulty.

Fixes vs original:
  • get_classifier_predictions uses mean pooling (matches step1/step2)
  • bf16 inference for speed

Usage:
    python step3_tune_ensemble.py
"""

import logging, os, sys, json
import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics import f1_score
import optuna
from tqdm import tqdm

from config import *
from data_utils import load_dataset, create_sentence_pairs

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(os.path.join(LOG_DIR, "step3_ensemble.log")),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass

def encode_batch(encoder, input_ids, attention_mask):
    out  = encoder(input_ids=input_ids, attention_mask=attention_mask)
    h    = out.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def get_classifier_predictions(encoder_name, difficulty):
    from models import StyleChangeClassifier
    use_bf16  = torch.cuda.is_bf16_supported()
    enc_dir   = os.path.join(CHECKPOINT_DIR, f"{encoder_name}_cosent_best")
    clf_path  = os.path.join(CHECKPOINT_DIR, f"{encoder_name}_{difficulty}_classifier.pt")
    tokenizer = AutoTokenizer.from_pretrained(enc_dir)
    encoder   = AutoModel.from_pretrained(enc_dir).to(DEVICE).eval()
    ckpt      = torch.load(clf_path, map_location=DEVICE)
    clf       = StyleChangeClassifier(hidden_dim=ENCODERS[encoder_name]["hidden_dim"]).to(DEVICE)
    clf.load_state_dict(ckpt["model_state"]); clf.eval()

    val_probs = load_dataset(DATA_DIR, difficulty, "validation")
    val_pairs = create_sentence_pairs(val_probs)
    all_probs, all_labels = [], []

    with torch.no_grad():
        for i in tqdm(range(0, len(val_pairs), 256), desc=f"  {encoder_name}/{difficulty}", leave=False):
            batch   = val_pairs[i:i+256]
            texts_a = [p["text_a"] for p in batch]
            texts_b = [p["text_b"] for p in batch]
            labels  = [p["label"]  for p in batch]
            ta = tokenizer(texts_a, truncation=True, max_length=COSENT["max_seq_len"],
                           padding=True, return_tensors="pt").to(DEVICE)
            tb = tokenizer(texts_b, truncation=True, max_length=COSENT["max_seq_len"],
                           padding=True, return_tensors="pt").to(DEVICE)
            with autocast(dtype=torch.bfloat16) if use_bf16 else _NullCtx():
                ea = encode_batch(encoder, ta["input_ids"], ta["attention_mask"])
                eb = encode_batch(encoder, tb["input_ids"], tb["attention_mask"])
            logits = clf(ea.float(), eb.float())
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist()); all_labels.extend(labels)

    del encoder, clf; torch.cuda.empty_cache()
    return np.array(all_probs), np.array(all_labels)


def tune_one_difficulty(difficulty):
    logger.info(f"\n{'='*50}\nTuning ensemble: {difficulty}\n{'='*50}")
    preds  = {}; labels = None
    for enc in ENCODERS:
        logger.info(f"  Predictions: {enc}/{difficulty}")
        p, l = get_classifier_predictions(enc, difficulty)
        preds[enc] = p; labels = l

    enc_names = sorted(preds.keys())

    def objective(trial):
        weights = {}; remaining = 1.0
        for i, name in enumerate(enc_names[:-1]):
            w = trial.suggest_float(f"w_{name}", ENSEMBLE["weight_min"],
                                    min(ENSEMBLE["weight_max"],
                                        remaining - 0.05*(len(enc_names)-i-1)))
            weights[name] = w; remaining -= w
        weights[enc_names[-1]] = remaining
        if weights[enc_names[-1]] < ENSEMBLE["weight_min"]: return 0.0
        thr      = trial.suggest_float("threshold", ENSEMBLE["threshold_min"], ENSEMBLE["threshold_max"])
        ensemble = sum(weights[n] * preds[n] for n in enc_names)
        binary   = (ensemble > thr).astype(int)
        return f1_score(labels, binary, average="macro", zero_division=0)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=ENSEMBLE["n_trials"])
    best = study.best_params
    weights = {}; remaining = 1.0
    for name in enc_names[:-1]:
        weights[name] = best[f"w_{name}"]; remaining -= weights[name]
    weights[enc_names[-1]] = remaining
    config = {"weights": weights, "threshold": best["threshold"], "f1": study.best_value}
    logger.info(f"  Best F1: {study.best_value:.4f}  thr={best['threshold']:.3f}  weights={weights}")
    return config


if __name__ == "__main__":
    all_config = {}
    for diff in DIFFICULTIES:
        all_config[diff] = tune_one_difficulty(diff)
    save_path = os.path.join(CHECKPOINT_DIR, "ensemble_config.json")
    json.dump(all_config, open(save_path, "w"), indent=2)
    logger.info(f"\nEnsemble config saved: {save_path}")
    print("\n" + "="*50)
    print("Ensemble Tuning Summary:")
    for diff, cfg in all_config.items():
        w_str = " | ".join(f"{k}={v:.3f}" for k, v in cfg["weights"].items())
        print(f"  {diff}: F1={cfg['f1']:.4f} thr={cfg['threshold']:.3f} | {w_str}")
    print("="*50)