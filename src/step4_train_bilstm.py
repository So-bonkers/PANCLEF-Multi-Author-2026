"""
PAN 2026 — Step 4c: Train Bi-LSTM for ONE difficulty.

Reads feat_{enc}_{diff}_{split}.pt and meta_{diff}_{split}.pt
produced by step4_merge.py. One encoder's feat file is loaded
at a time to keep RAM under control.

File layout expected in FEATURE_CACHE_DIR:
    meta_{difficulty}_{split}.pt          — list of {changes, ensemble_prob}
    feat_{enc}_{difficulty}_{split}.pt    — list of CPU tensors [n_bounds, feat_dim]

Usage:
    python step4_train_bilstm.py --difficulty easy
    python step4_train_bilstm.py --difficulty medium
    python step4_train_bilstm.py --difficulty hard
"""

import argparse, json, logging, os, sys, random, time
import numpy as np
import torch
from tqdm import tqdm

from config import *
from data_utils import load_dataset, evaluate_document_level
from models import SequentialCPDRefinement
from losses import FocalBCELoss

os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "step4_bilstm.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

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


# ── file checks ───────────────────────────────────────────────────────────────

def check_merged_files(difficulty: str, split: str):
    """Verify that step4_merge.py outputs exist for this difficulty/split."""
    missing = []
    meta_path = os.path.join(FEATURE_CACHE_DIR, f"meta_{difficulty}_{split}.pt")
    if not os.path.exists(meta_path):
        missing.append(meta_path)
    for enc_name in ENCODERS:
        fp = os.path.join(FEATURE_CACHE_DIR, f"feat_{enc_name}_{difficulty}_{split}.pt")
        if not os.path.exists(fp):
            missing.append(fp)
    if missing:
        logger.error("Missing merged files — run step4_merge.py first:")
        for m in missing:
            logger.error(f"  {m}")
        sys.exit(1)


# ── data loading ──────────────────────────────────────────────────────────────

def load_meta(difficulty: str, split: str) -> list:
    """
    Load meta_{difficulty}_{split}.pt.
    Returns list of {idx, changes, ensemble_prob} for valid docs only.
    """
    meta_path = os.path.join(FEATURE_CACHE_DIR, f"meta_{difficulty}_{split}.pt")
    logger.info(f"  Loading meta: {meta_path}")
    raw  = torch.load(meta_path, weights_only=False)
    meta = []
    for idx, entry in enumerate(raw):
        if entry["ensemble_prob"] is None:
            continue
        if not entry["changes"]:
            continue
        meta.append({
            "idx":           idx,
            "changes":       entry["changes"],
            "ensemble_prob": entry["ensemble_prob"],
        })
    logger.info(f"  {len(meta)}/{len(raw)} docs valid | {log_ram()}")
    return meta


def _make_prob_fallback(ensemble_prob: torch.Tensor, n_bounds: int) -> torch.Tensor:
    """
    When no per-encoder prob file exists, broadcast ensemble_prob to [n_bounds].
    ensemble_prob may be a scalar tensor or a 1-D tensor of any length.
    """
    if ensemble_prob.dim() == 0:
        return ensemble_prob.unsqueeze(0).expand(n_bounds)
    L = ensemble_prob.shape[0]
    if L == n_bounds:
        return ensemble_prob
    if L > n_bounds:
        return ensemble_prob[:n_bounds]
    # L < n_bounds: pad by repeating last element
    pad = ensemble_prob[-1:].expand(n_bounds - L)
    return torch.cat([ensemble_prob, pad])


# ── REPLACE your entire build_docs function with this ─────────────────────────

def build_docs_with_classifier_probs(meta: list, difficulty: str, split: str, 
                                      classifier_checkpoints: dict = None) -> list:
    """
    Load features + generate CORRECT per-boundary classifier probabilities.
    Since saved prob files are corrupted (scalar instead of per-boundary),
    we load classifier checkpoints and run forward pass to get real per-boundary probs.
    
    classifier_checkpoints: dict of {enc_name: path_to_classifier.pt}
    """
    valid_idxs = {m["idx"] for m in meta}

    docs = {m["idx"]: {
        "idx":           m["idx"],
        "changes":       m["changes"],
        "ensemble_prob": m["ensemble_prob"],
        "features":      {},
        "probs":         {},
    } for m in meta}

    # Step 1: Load all feature files (same as before)
    for enc_name in sorted(ENCODERS.keys()):
        feat_path = os.path.join(
            FEATURE_CACHE_DIR, f"feat_{enc_name}_{difficulty}_{split}.pt"
        )
        size_mb = os.path.getsize(feat_path) / 1e6
        logger.info(f"    Loading feat: {enc_name} ({size_mb:.1f} MB) | {log_ram()}")
        feat_list = torch.load(feat_path, weights_only=False)

        for idx in valid_idxs:
            feat = feat_list[idx]
            if feat is None:
                continue
            docs[idx]["features"][enc_name] = feat  # [n_bounds, feat_dim]

        del feat_list
        logger.info(f"    {enc_name} features loaded | {log_ram()}")

    # Step 2: Generate per-boundary probs from classifiers (FIXED)
    if classifier_checkpoints is None:
        # Default paths based on your naming convention
        classifier_checkpoints = {
            "deberta": os.path.join(CHECKPOINT_DIR, f"classifier_deberta_{difficulty}.pt"),
            "roberta": os.path.join(CHECKPOINT_DIR, f"classifier_roberta_{difficulty}.pt"),
            "sentbert": os.path.join(CHECKPOINT_DIR, f"classifier_sentbert_{difficulty}.pt"),
        }

    from models import StyleChangeClassifier  # Your classifier class
    
    for enc_name, ckpt_path in classifier_checkpoints.items():
        if not os.path.exists(ckpt_path):
            logger.warning(f"  Classifier checkpoint not found: {ckpt_path}")
            continue
            
        logger.info(f"  Loading classifier: {enc_name} | {log_ram()}")
        
        # Load classifier
        hidden_dim = ENCODERS[enc_name]["hidden_dim"]
        classifier = StyleChangeClassifier(
            hidden_dim=hidden_dim,
            mlp_sizes=CLASSIFIER["hidden_sizes"],
            dropout_rates=CLASSIFIER["dropout_rates"],
        ).to(DEVICE)
        
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        classifier.load_state_dict(ckpt.get("model_state", ckpt))
        classifier.eval()
        
        # Process each document
        with torch.no_grad():
            for idx in valid_idxs:
                if idx not in docs or enc_name not in docs[idx]["features"]:
                    continue
                    
                feat = docs[idx]["features"][enc_name].to(DEVICE)  # [n_bounds, hidden*4+1]
                
                # Forward pass through classifier to get per-boundary logits
                logits = classifier.mlp(feat).squeeze(-1)  # [n_bounds]
                probs = torch.sigmoid(logits).cpu()  # [n_bounds]
                
                docs[idx]["probs"][enc_name] = probs
                
                # Clean up
                del logits
                
        del classifier
        torch.cuda.empty_cache()
        logger.info(f"    {enc_name} classifier probs generated | {log_ram()}")

    # Step 3: Filter incomplete docs
    result = [
        d for d in docs.values()
        if len(d["features"]) == len(ENCODERS) and len(d["probs"]) == len(ENCODERS)
    ]
    logger.info(f"    Built {len(result)} complete docs | {log_ram()}")
    return result

# ── training / validation ─────────────────────────────────────────────────────

def _prep_ens(ensemble_prob: torch.Tensor, n_bounds: int) -> torch.Tensor:
    """Shape ensemble_prob to [1, n_bounds, 1] for BiLSTM input."""
    ep = _make_prob_fallback(ensemble_prob, n_bounds)   # [n_bounds]
    return ep.unsqueeze(0).unsqueeze(-1).to(DEVICE)     # [1, n_bounds, 1]


def run_train(bilstm, docs, loss_fn, optimizer, batch_size, epoch_num, difficulty):
    bilstm.train()
    random.shuffle(docs)
    epoch_loss = 0.0
    n          = 0
    optimizer.zero_grad()
    t_start    = time.time()

    consecutive_penalty_weight = (
        BILSTM.get("medium_consecutive_penalty", 0.0)
        if difficulty == "medium" else 0.0
    )

    for i, doc in enumerate(tqdm(docs, desc=f"  Epoch {epoch_num:02d} train", leave=False)):
        changes  = torch.tensor(doc["changes"], dtype=torch.float).to(DEVICE)
        n_bounds = changes.shape[0]

        ef  = {k: v.unsqueeze(0).to(DEVICE) for k, v in doc["features"].items()}
        ep  = {k: v.unsqueeze(0).unsqueeze(-1).to(DEVICE) for k, v in doc["probs"].items()}
        ens = _prep_ens(doc["ensemble_prob"], n_bounds)

        logits  = bilstm(ef, ep, ens).squeeze(0)
        min_len = min(len(logits), len(changes))
        loss    = loss_fn(logits[:min_len], changes[:min_len]) / batch_size

        if consecutive_penalty_weight > 0.0 and min_len > 1:
            probs_seq           = torch.sigmoid(logits[:min_len])
            consecutive_penalty = (probs_seq[:-1] * probs_seq[1:]).mean()
            loss = loss + (consecutive_penalty_weight * consecutive_penalty / batch_size)

        loss.backward()
        epoch_loss += loss.item() * batch_size
        n += 1

        if n % batch_size == 0:
            torch.nn.utils.clip_grad_norm_(bilstm.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t_start
            pct     = 100 * (i + 1) / len(docs)
            eta     = (elapsed / (i + 1)) * (len(docs) - i - 1)
            logger.info(
                f"    doc {i+1}/{len(docs)} ({pct:.1f}%) | "
                f"loss {epoch_loss/n:.4f} | "
                f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m | "
                f"{log_ram()}"
            )

    # Final gradient step for any remainder
    torch.nn.utils.clip_grad_norm_(bilstm.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    return epoch_loss / max(n, 1)


def run_val(bilstm, docs):
    bilstm.eval()
    preds = []
    with torch.no_grad():
        for doc in tqdm(docs, desc="  Val", leave=False):
            n_bounds = list(doc["features"].values())[0].shape[0]
            ef  = {k: v.unsqueeze(0).to(DEVICE) for k, v in doc["features"].items()}
            ep  = {k: v.unsqueeze(0).unsqueeze(-1).to(DEVICE) for k, v in doc["probs"].items()}
            ens = _prep_ens(doc["ensemble_prob"], n_bounds)
            logits = bilstm(ef, ep, ens).squeeze(0)
            preds.append(torch.sigmoid(logits).cpu().numpy().tolist())
    return preds


# ── main ──────────────────────────────────────────────────────────────────────

def train(difficulty: str):
    logger.info("=" * 60)
    logger.info(f"Bi-LSTM training: difficulty={difficulty}")
    logger.info(f"{log_ram()} | {log_gpu()}")
    logger.info("=" * 60)

    check_merged_files(difficulty, "train")
    check_merged_files(difficulty, "val")

    logger.info("Loading val ground-truth labels...")
    val_problems = load_dataset(DATA_DIR, difficulty, "validation")
    logger.info(f"  {len(val_problems)} val docs | {log_ram()}")

    logger.info("Loading train meta...")
    train_meta = load_meta(difficulty, "train")
    logger.info("Loading val meta...")
    val_meta   = load_meta(difficulty, "val")
    logger.info(f"Train: {len(train_meta)} | Val: {len(val_meta)} | {log_ram()}")

    encoder_dims = {n: ENCODERS[n]["hidden_dim"] for n in ENCODERS}
    use_pos      = BILSTM.get("use_positional_features", True)

    bilstm = SequentialCPDRefinement(
        encoder_dims=encoder_dims,
        feature_proj_dim=BILSTM["feature_proj_dim"],
        hidden_dim=BILSTM["hidden_dim"],
        num_layers=BILSTM["num_layers"],
        dropout=BILSTM["dropout"],
        use_positional_features=use_pos,
    ).to(DEVICE)
    logger.info(f"Model built | {log_gpu()}")

    cfg_loss  = LOSS_CONFIG[difficulty]
    loss_fn   = FocalBCELoss(
        gamma=cfg_loss["focal_gamma"],
        pos_weight=cfg_loss["pos_weight"],
    )
    optimizer = torch.optim.Adam(
        bilstm.parameters(),
        lr=BILSTM["lr"],
        weight_decay=BILSTM["weight_decay"],
    )

    best_f1   = 0.0
    patience  = 0
    save_path = os.path.join(CHECKPOINT_DIR, f"bilstm_{difficulty}.pt")

    thr_min, thr_max, thr_step = BILSTM["threshold_search"][difficulty]

    for epoch in range(BILSTM["epochs"]):
        logger.info(f"\n{'='*40}")
        logger.info(f"Epoch {epoch+1}/{BILSTM['epochs']} | {log_ram()}")

        logger.info("  Loading train feat files (one encoder at a time)...")
        train_docs = build_docs_with_classifier_probs(train_meta, difficulty, "train")

        t0         = time.time()
        epoch_loss = run_train(
            bilstm, train_docs, loss_fn, optimizer,
            BILSTM["batch_size"], epoch + 1, difficulty,
        )
        train_time = time.time() - t0
        logger.info(f"  Train: {train_time/60:.1f}m | loss {epoch_loss:.4f} | {log_ram()}")

        del train_docs
        logger.info(f"  Train docs freed | {log_ram()}")

        logger.info("  Loading val feat files (one encoder at a time)...")
        val_docs  = build_docs_with_classifier_probs(val_meta, difficulty, "val")
        all_preds = run_val(bilstm, val_docs)
        del val_docs
        logger.info(f"  Val docs freed | {log_ram()}")

        # Per-difficulty threshold search
        best_f, best_t = 0.0, (thr_min + thr_max) / 2
        for thr in np.arange(thr_min, thr_max, thr_step):
            m = evaluate_document_level(val_problems, all_preds, threshold=thr)
            if m["f1_macro"] > best_f:
                best_f, best_t = m["f1_macro"], thr

        metrics = evaluate_document_level(val_problems, all_preds, threshold=best_t)
        logger.info(
            f"  Epoch {epoch+1:2d}/{BILSTM['epochs']} | "
            f"Loss {epoch_loss:.4f} | "
            f"F1 {metrics['f1_macro']:.4f} | "
            f"F1b {metrics['f1_binary']:.4f} | "
            f"Pk {metrics['pk']:.4f} | "
            f"WD {metrics['windowdiff']:.4f} | "
            f"thr {best_t:.3f} | "
            f"cr_pred {metrics['change_rate_pred']:.4f} | "
            f"{log_ram()} | {log_gpu()}"
        )

        if metrics["f1_macro"] > best_f1:
            best_f1  = metrics["f1_macro"]
            patience = 0
            torch.save(
                {
                    "model_state": bilstm.state_dict(),
                    "best_f1":     best_f1,
                    "threshold":   best_t,
                    "metrics":     metrics,
                    "difficulty":  difficulty,
                },
                save_path,
            )
            logger.info(f"  ✓ Saved  F1={best_f1:.4f}  →  {save_path}")
        else:
            patience += 1
            logger.info(f"  No improvement ({patience}/{BILSTM['patience']})")
            if patience >= BILSTM["patience"]:
                logger.info(f"  Early stopping at epoch {epoch+1}")
                break

    logger.info(f"\nFinished {difficulty} → best F1={best_f1:.4f}")
    return best_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", required=True, choices=DIFFICULTIES)
    args = parser.parse_args()

    f1 = train(args.difficulty)
    print(f"\nFinal best F1 ({args.difficulty}): {f1:.4f}")