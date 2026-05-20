#!/usr/bin/env python3
"""
PAN @ CLEF 2026 — TIRA submission entry point.
"""

import argparse
import json
import logging
import os
import sys
import time

import torch
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer
from glob import glob
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from config import (
    ENCODERS,
    DIFFICULTIES,
    COSENT,
    CLASSIFIER,
    BILSTM,
    CHECKPOINT_DIR,
    DEVICE,
)

from models import (
    StyleChangeClassifier,
    SequentialCPDRefinement,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

BEST_THRESHOLDS = {
    "easy": 0.52,
    "medium": 0.650,
    "hard": 0.570,
}


# ── helpers ───────────────────────────────────────────────────────────────────

class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def encode_mean(encoder, input_ids, attention_mask):
    out = encoder(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    h = out.last_hidden_state

    mask = attention_mask.unsqueeze(-1).float()

    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def read_sentences(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [
        s.strip()
        for s in text.split("\n")
        if s.strip()
    ]


# ── model loading ─────────────────────────────────────────────────────────────

def load_models(device):
    models = {}
    classifiers = {}

    TOKENIZER_PATHS = {
        "deberta": "/opt/hf_cache/deberta",
        "roberta": "/opt/hf_cache/roberta",
        "sentbert": "/opt/hf_cache/sentbert",
    }

    for enc_name in sorted(ENCODERS.keys()):
        enc_dir = os.path.join(
            CHECKPOINT_DIR,
            f"{enc_name}_cosent_best"
        )

        if not os.path.exists(enc_dir):
            logger.error(
                f"Encoder checkpoint not found: {enc_dir}"
            )
            sys.exit(1)

        tok_dir = TOKENIZER_PATHS[enc_name]

        if not os.path.exists(tok_dir):
            logger.error(
                f"Tokenizer directory not found: {tok_dir}"
            )
            sys.exit(1)

        logger.info(f"  Loading encoder: {enc_name}")

        models[enc_name] = {
            "encoder": AutoModel.from_pretrained(
                enc_dir
            ).to(device).eval(),

            "tokenizer": AutoTokenizer.from_pretrained(
                tok_dir,
                local_files_only=True,
            ),
        }

        classifiers[enc_name] = {}

        for diff in DIFFICULTIES:
            clf_path = os.path.join(
                CHECKPOINT_DIR,
                f"{enc_name}_{diff}_classifier.pt"
            )

            if not os.path.exists(clf_path):
                logger.error(
                    f"Classifier checkpoint not found: {clf_path}"
                )
                sys.exit(1)

            ckpt = torch.load(
                clf_path,
                map_location=device,
                weights_only=False
            )

            clf = StyleChangeClassifier(
                hidden_dim=ENCODERS[enc_name]["hidden_dim"],
                mlp_sizes=CLASSIFIER["hidden_sizes"],
                dropout_rates=CLASSIFIER["dropout_rates"],
            ).to(device)

            clf.load_state_dict(ckpt["model_state"])

            clf.eval()

            classifiers[enc_name][diff] = clf

    bilstm = {}

    encoder_dims = {
        n: ENCODERS[n]["hidden_dim"]
        for n in ENCODERS
    }

    use_pos = BILSTM.get(
        "use_positional_features",
        True
    )

    for diff in DIFFICULTIES:
        bl_path = os.path.join(
            CHECKPOINT_DIR,
            f"bilstm_{diff}.pt"
        )

        if os.path.exists(bl_path):
            ckpt = torch.load(
                bl_path,
                map_location=device,
                weights_only=False
            )

            bl = SequentialCPDRefinement(
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

            logger.info(f"  Loaded BiLSTM: {diff}")

        else:
            bilstm[diff] = None

            logger.warning(
                f"  No BiLSTM for {diff} — ensemble only"
            )

    ens_config_path = os.path.join(
        CHECKPOINT_DIR,
        "ensemble_config.json"
    )

    if not os.path.exists(ens_config_path):
        logger.error(
            f"ensemble_config.json not found: "
            f"{ens_config_path}"
        )

        sys.exit(1)

    with open(ens_config_path, "r") as f:
        ens_config = json.load(f)

    return models, classifiers, bilstm, ens_config

# ── inference ─────────────────────────────────────────────────────────────────

def predict_document(
    sentences,
    difficulty,
    models,
    classifiers,
    bilstm,
    ens_config,
    device
):
    n = len(sentences) - 1

    if n <= 0:
        return []

    use_bf16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )

    BATCH_SIZE = 32 if device == "cuda" else 8

    enc_features = {}
    enc_probs = {}

    sent_a = sentences[:-1]
    sent_b = sentences[1:]

    for enc_name in sorted(models.keys()):
        encoder = models[enc_name]["encoder"]

        tokenizer = models[enc_name]["tokenizer"]

        clf = classifiers[enc_name][difficulty]

        feat_batches = []
        prob_batches = []

        with torch.no_grad():

            for start in range(0, n, BATCH_SIZE):
                end = min(start + BATCH_SIZE, n)

                batch_a = sent_a[start:end]
                batch_b = sent_b[start:end]

                ta = tokenizer(
                    batch_a,
                    truncation=True,
                    max_length=COSENT["max_seq_len"],
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                tb = tokenizer(
                    batch_b,
                    truncation=True,
                    max_length=COSENT["max_seq_len"],
                    padding=True,
                    return_tensors="pt",
                ).to(device)

                ctx = (
                    autocast(
                        "cuda",
                        dtype=torch.bfloat16
                    )
                    if use_bf16
                    else _NullCtx()
                )

                with ctx:
                    ea = encode_mean(
                        encoder,
                        ta["input_ids"],
                        ta["attention_mask"]
                    )

                    eb = encode_mean(
                        encoder,
                        tb["input_ids"],
                        tb["attention_mask"]
                    )

                ea = ea.float()
                eb = eb.float()

                feats = clf.get_features(ea, eb)

                probs = torch.sigmoid(
                    clf(ea, eb)
                )

                feat_batches.append(feats)

                prob_batches.append(probs)

        feats = torch.cat(feat_batches, dim=0)

        probs = torch.cat(prob_batches, dim=0)

        enc_features[enc_name] = (
            feats.unsqueeze(0).to(device)
        )

        enc_probs[enc_name] = (
            probs.unsqueeze(0).unsqueeze(-1).to(device)
        )

    weights = ens_config[difficulty]["weights"]

    ens_prob = sum(
        weights.get(name, 0.0) * enc_probs[name]
        for name in enc_probs
    )

    if bilstm[difficulty] is not None:
        with torch.no_grad():
            logits = bilstm[difficulty](
                enc_features,
                enc_probs,
                ens_prob
            ).squeeze(0)

            probs = torch.sigmoid(
                logits
            ).cpu().numpy().tolist()

    else:
        probs = ens_prob.squeeze().cpu().numpy().tolist()

        if isinstance(probs, float):
            probs = [probs]

    return probs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PAN 2026 Style Change Detection"
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input directory"
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output directory"
    )

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input)

    output_dir = os.path.abspath(args.output)

    if not os.path.exists(input_dir):
        logger.error(
            f"Input directory not found: {input_dir}"
        )
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    device = DEVICE if torch.cuda.is_available() else "cpu"

    logger.info(f"Device: {device}")

    logger.info(
        f"Loading all models from: {CHECKPOINT_DIR}"
    )

    t_load = time.time()

    models, classifiers, bilstm, ens_config = load_models(device)

    logger.info(
        f"Models loaded in "
        f"{time.time() - t_load:.1f}s"
    )

    total_docs = 0

    t_start = time.time()

    for difficulty in DIFFICULTIES:

        candidate_dirs = [
            os.path.join(input_dir, difficulty, "train"),
            os.path.join(
                input_dir,
                "input-data",
                difficulty,
                "train"
            ),
            os.path.join(input_dir, difficulty),
        ]

        diff_input_dir = None

        for cand in candidate_dirs:
            if os.path.exists(cand):
                diff_input_dir = cand
                break

        if diff_input_dir is None:
            logger.warning(
                f"No valid input directory found "
                f"for {difficulty}"
            )
            continue

        diff_output_dir = os.path.join(
            output_dir,
            difficulty
        )

        os.makedirs(diff_output_dir, exist_ok=True)

        txt_files = sorted(
            glob(
                os.path.join(
                    diff_input_dir,
                    "problem-*.txt"
                )
            )
        )

        if not txt_files:
            logger.warning(
                f"No problem files found in "
                f"{diff_input_dir}"
            )
            continue

        threshold = BEST_THRESHOLDS.get(
            difficulty,
            0.5
        )

        logger.info(
            f"\n{difficulty}: {len(txt_files)} problems | "
            f"threshold={threshold}"
        )

        for txt_path in tqdm(
            txt_files,
            desc=f"  {difficulty}",
            unit="doc"
        ):
            fname = os.path.basename(txt_path)

            problem_id = (
                fname.replace("problem-", "")
                     .replace(".txt", "")
            )

            sentences = read_sentences(txt_path)

            probs = predict_document(
                sentences,
                difficulty,
                models,
                classifiers,
                bilstm,
                ens_config,
                device,
            )

            changes = [
                1 if p > threshold else 0
                for p in probs
            ]

            out_path = os.path.join(
                diff_output_dir,
                f"solution-problem-{problem_id}.json"
            )

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"changes": changes}, f)

        total_docs += len(txt_files)

        logger.info(
            f"  Done {difficulty} → solutions in "
            f"{diff_output_dir}"
        )

    elapsed = time.time() - t_start

    logger.info(
        f"\nFinished: {total_docs} docs in "
        f"{elapsed:.1f}s "
        f"({total_docs / elapsed:.1f} doc/s)"
    )


if __name__ == "__main__":
    main()

