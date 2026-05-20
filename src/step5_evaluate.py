"""
PAN 2026 — Step 5: Full evaluation + generate solution JSONs.

Usage:
    python step5_evaluate.py --mode val
    python step5_evaluate.py --mode test
    python step5_evaluate.py --mode both

Changes vs previous version:
  • SequentialCPDRefinement instantiated with use_positional_features=True
    and updated feature_proj_dim/hidden_dim from BILSTM config
  • StyleChangeClassifier input_dim is now hidden_dim*4+1 (cosine sim feature)
  • get_features call unchanged in API — classifier handles it internally
"""

import argparse, logging, os, sys, json, time
import numpy as np
import torch
from torch.cuda.amp import autocast
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from config import *
from data_utils import load_dataset, evaluate_document_level, format_metrics
from models import StyleChangeClassifier, SequentialCPDRefinement

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "step5_evaluate.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): pass


def encode_mean(encoder, input_ids, attention_mask):
    out  = encoder(input_ids=input_ids, attention_mask=attention_mask)
    h    = out.last_hidden_state
    mask = attention_mask.unsqueeze(-1).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def load_all_models(device):
    use_bf16 = torch.cuda.is_bf16_supported()
    models = {}
    classifiers = {}

    for enc_name in sorted(ENCODERS.keys()):
        enc_dir   = os.path.join(CHECKPOINT_DIR, f"{enc_name}_cosent_best")
        logger.info(f"Loading encoder: {enc_name}")
        models[enc_name] = {
            "encoder":   AutoModel.from_pretrained(enc_dir).to(device).eval(),
            "tokenizer": AutoTokenizer.from_pretrained(enc_dir),
        }
        classifiers[enc_name] = {}
        for diff in DIFFICULTIES:
            clf_path = os.path.join(CHECKPOINT_DIR, f"{enc_name}_{diff}_classifier.pt")
            ckpt     = torch.load(clf_path, map_location=device)
            clf      = StyleChangeClassifier(
                hidden_dim=ENCODERS[enc_name]["hidden_dim"],
                mlp_sizes=CLASSIFIER["hidden_sizes"],
                dropout_rates=CLASSIFIER["dropout_rates"],
            ).to(device)
            clf.load_state_dict(ckpt["model_state"])
            clf.eval()
            classifiers[enc_name][diff] = clf

    bilstm       = {}
    encoder_dims = {n: ENCODERS[n]["hidden_dim"] for n in ENCODERS}
    use_pos      = BILSTM.get("use_positional_features", True)

    for diff in DIFFICULTIES:
        bl_path = os.path.join(CHECKPOINT_DIR, f"bilstm_{diff}.pt")
        if os.path.exists(bl_path):
            ckpt = torch.load(bl_path, map_location=device)
            bl   = SequentialCPDRefinement(
                encoder_dims=encoder_dims,
                feature_proj_dim=BILSTM["feature_proj_dim"],
                hidden_dim=BILSTM["hidden_dim"],
                num_layers=BILSTM["num_layers"],
                dropout=BILSTM["dropout"],
                use_positional_features=use_pos,
            ).to(device)
            bl.load_state_dict(ckpt["model_state"])
            bl.eval()
            bilstm[diff] = bl
            logger.info(f"Loaded Bi-LSTM: {diff}")
        else:
            bilstm[diff] = None
            logger.warning(f"No Bi-LSTM for {diff} — ensemble-only")

    ens_config = json.load(open(os.path.join(CHECKPOINT_DIR, "ensemble_config.json")))
    return models, classifiers, bilstm, ens_config


def predict_document(sentences, difficulty, models, classifiers, bilstm, ens_config, device):
    n = len(sentences) - 1
    if n <= 0:
        return []
    use_bf16 = torch.cuda.is_bf16_supported()

    enc_features = {}
    enc_probs    = {}

    for enc_name in sorted(models.keys()):
        encoder   = models[enc_name]["encoder"]
        tokenizer = models[enc_name]["tokenizer"]
        clf       = classifiers[enc_name][difficulty]
        feat_list, prob_list = [], []

        with torch.no_grad():
            for i in range(n):
                ta = tokenizer(
                    sentences[i],
                    truncation=True, max_length=COSENT["max_seq_len"],
                    padding="max_length", return_tensors="pt",
                ).to(device)
                tb = tokenizer(
                    sentences[i + 1],
                    truncation=True, max_length=COSENT["max_seq_len"],
                    padding="max_length", return_tensors="pt",
                ).to(device)
                ctx = autocast(dtype=torch.bfloat16) if use_bf16 else _NullCtx()
                with ctx:
                    ea = encode_mean(encoder, ta["input_ids"], ta["attention_mask"])
                    eb = encode_mean(encoder, tb["input_ids"], tb["attention_mask"])
                ea, eb = ea.float(), eb.float()
                feat_list.append(clf.get_features(ea, eb).squeeze(0))
                prob_list.append(torch.sigmoid(clf(ea, eb)).squeeze(0))

        enc_features[enc_name] = torch.stack(feat_list).unsqueeze(0).to(device)
        enc_probs[enc_name]    = torch.stack(prob_list).unsqueeze(0).unsqueeze(-1).to(device)

    weights  = ens_config[difficulty]["weights"]
    ens_prob = sum(weights.get(n, 0) * enc_probs[n] for n in enc_probs)

    if bilstm[difficulty] is not None:
        with torch.no_grad():
            logits = bilstm[difficulty](enc_features, enc_probs, ens_prob).squeeze(0)
            probs  = torch.sigmoid(logits).cpu().numpy().tolist()
    else:
        probs = ens_prob.squeeze().cpu().numpy().tolist()
        if isinstance(probs, float):
            probs = [probs]
    return probs


def run_evaluation(split="validation"):
    device = DEVICE
    logger.info("Loading all models...")
    models, classifiers, bilstm, ens_config = load_all_models(device)

    for difficulty in DIFFICULTIES:
        logger.info(f"\n{'='*60}\n{difficulty} / {split}\n{'='*60}")
        problems = load_dataset(DATA_DIR, difficulty, split)
        if not problems:
            logger.warning(f"No data: {difficulty}/{split}")
            continue

        threshold = ens_config[difficulty]["threshold"]
        bl_path   = os.path.join(CHECKPOINT_DIR, f"bilstm_{difficulty}.pt")
        if os.path.exists(bl_path):
            bl_ckpt   = torch.load(bl_path, map_location="cpu")
            threshold = bl_ckpt.get("threshold", threshold)

        all_probs = []
        t_start   = time.time()
        for idx, prob in enumerate(tqdm(problems, desc=f"  {difficulty}")):
            all_probs.append(
                predict_document(
                    prob["sentences"], difficulty,
                    models, classifiers, bilstm, ens_config, device,
                )
            )
        elapsed = time.time() - t_start
        logger.info(
            f"  {len(problems)} docs in {elapsed:.0f}s "
            f"({len(problems)/elapsed:.1f} doc/s)"
        )

        out_dir = os.path.join(OUTPUT_DIR, difficulty)
        os.makedirs(out_dir, exist_ok=True)
        for prob, probs in zip(problems, all_probs):
            changes  = [1 if p > threshold else 0 for p in probs]
            out_path = os.path.join(out_dir, f"solution-problem-{prob['id']}.json")
            json.dump({"changes": changes}, open(out_path, "w"))
        logger.info(f"  Solutions → {out_dir}")

        if problems[0]["changes"] is not None:
            metrics      = evaluate_document_level(problems, all_probs, threshold)
            metrics_path = os.path.join(LOG_DIR, f"metrics_{difficulty}_{split}.json")
            json.dump(metrics, open(metrics_path, "w"), indent=2, default=float)
            logger.info(format_metrics(metrics, f"{difficulty} thr={threshold:.3f}"))

    logger.info("\n✓ All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="val",
                        choices=["val", "test", "both"])
    args = parser.parse_args()
    if args.mode in ["val",  "both"]: run_evaluation("validation")
    if args.mode in ["test", "both"]: run_evaluation("test")