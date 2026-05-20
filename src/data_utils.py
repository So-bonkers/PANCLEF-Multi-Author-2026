"""
PAN 2026 — Data loading, pair generation, and evaluation metrics.
Handles sentence-level data (one sentence per line).
"""

import json
import os
import random
import numpy as np
from glob import glob
from collections import Counter
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_dataset(data_dir, difficulty, split="train"):
    """
    Load all problems from a difficulty level.
    
    Returns list of dicts:
        id, sentences, n_sentences, authors, changes, n_boundaries
    """
    problems = []
    path = os.path.join(data_dir, difficulty, split)
    
    if not os.path.exists(path):
        logger.warning(f"Path not found: {path}")
        return problems
    
    txt_files = sorted(glob(os.path.join(path, "problem-*.txt")))
    
    for txt_file in txt_files:
        fname = os.path.basename(txt_file)
        problem_id = fname.replace("problem-", "").replace(".txt", "")
        
        # SENTENCE-LEVEL: one sentence per line
        with open(txt_file, "r", encoding="utf-8") as f:
            text = f.read()
        sentences = [s.strip() for s in text.split("\n") if s.strip()]
        
        # Ground truth
        truth_name = fname.replace("problem-", "truth-problem-").replace(".txt", ".json")
        truth_path = os.path.join(path, truth_name)
        
        changes, authors = None, None
        if os.path.exists(truth_path):
            with open(truth_path, "r") as f:
                truth = json.load(f)
            changes = truth.get("changes", [])
            authors = truth.get("authors", None)
        
        n_boundaries = len(sentences) - 1
        
        problems.append({
            "id": problem_id,
            "sentences": sentences,
            "n_sentences": len(sentences),
            "authors": authors,
            "changes": changes,
            "n_boundaries": n_boundaries,
        })
    
    return problems


def print_dataset_stats(problems, name=""):
    """Print summary statistics for a dataset."""
    n = len(problems)
    if n == 0:
        logger.info(f"[{name}] Empty dataset")
        return
    
    total_sents = sum(p["n_sentences"] for p in problems)
    avg_sents = total_sents / n
    
    if problems[0]["changes"] is not None:
        total_changes = sum(sum(p["changes"]) for p in problems)
        total_boundaries = sum(p["n_boundaries"] for p in problems)
        change_rate = total_changes / total_boundaries if total_boundaries > 0 else 0
        avg_changes = total_changes / n
        
        author_dist = Counter(p["authors"] for p in problems if p["authors"])
        
        logger.info(
            f"[{name}] {n} docs | {total_sents:,} sents | "
            f"avg {avg_sents:.1f} sents/doc | avg {avg_changes:.1f} changes/doc | "
            f"change rate: {change_rate:.4f} ({change_rate*100:.1f}%) | "
            f"authors: {dict(sorted(author_dist.items()))}"
        )
    else:
        logger.info(f"[{name}] {n} docs | {total_sents:,} sents | avg {avg_sents:.1f} sents/doc")


# ============================================================
# PAIR GENERATION (for CoSENT + MLP)
# ============================================================

def generate_pairs_grouped(sentences, changes, max_per_group=6, max_cross=3):
    """
    Group-based pair generation (Chen et al. 2023).
    
    Positive: pairs within same-author groups.
    Negative: pairs across adjacent groups (at style change boundaries).
    """
    if not changes or len(sentences) < 2:
        return []
    
    # Build author groups
    groups = [[0]]
    for i, c in enumerate(changes):
        if c == 1:
            groups.append([i + 1])
        else:
            groups[-1].append(i + 1)
    
    positives, negatives = [], []
    
    # Positives: sample within each group
    for group in groups:
        if len(group) < 2:
            continue
        indices = group if len(group) <= max_per_group else random.sample(group, max_per_group)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                positives.append((sentences[indices[i]], sentences[indices[j]], 1))
    
    # Negatives: sample across adjacent groups
    for g in range(len(groups) - 1):
        g1 = groups[g][-max_cross:] if len(groups[g]) > max_cross else groups[g]
        g2 = groups[g + 1][:max_cross] if len(groups[g + 1]) > max_cross else groups[g + 1]
        for i_idx in g1:
            for j_idx in g2:
                negatives.append((sentences[i_idx], sentences[j_idx], 0))
    
    return positives + negatives


def generate_all_pairs(problems, max_per_group=6, max_cross=3, neg_pos_ratio=2.0):
    """Generate CoSENT pairs from all problems."""
    all_pairs = []
    for prob in problems:
        if prob["changes"] is None:
            continue
        pairs = generate_pairs_grouped(
            prob["sentences"], prob["changes"], max_per_group, max_cross
        )
        all_pairs.extend(pairs)
    
    # Balance
    pos = [p for p in all_pairs if p[2] == 1]
    neg = [p for p in all_pairs if p[2] == 0]
    max_neg = int(len(pos) * neg_pos_ratio)
    if len(neg) > max_neg:
        neg = random.sample(neg, max_neg)
    
    all_pairs = pos + neg
    random.shuffle(all_pairs)
    
    logger.info(f"Generated {len(all_pairs)} pairs ({len(pos)} pos, {len(neg)} neg)")
    return all_pairs


def create_sentence_pairs(problems):
    """Create consecutive (sent_i, sent_{i+1}, label) pairs for MLP training."""
    pairs = []
    for prob in problems:
        if prob["changes"] is None:
            continue
        for i in range(min(len(prob["sentences"]) - 1, len(prob["changes"]))):
            pairs.append({
                "text_a": prob["sentences"][i],
                "text_b": prob["sentences"][i + 1],
                "label": prob["changes"][i],
                "doc_id": prob["id"],
                "position": i,
            })
    return pairs


# ============================================================
# EVALUATION METRICS
# ============================================================

def compute_f1_metrics(y_true, y_pred):
    """Compute F1, precision, recall, accuracy."""
    return {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_binary": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def compute_pk(reference, hypothesis, k=None):
    """
    Pk metric (Beeferman et al. 1999).
    Measures probability that two randomly chosen points k apart
    are incorrectly classified as same/different segment.
    
    Lower is better. Range [0, 1]. Perfect = 0.
    """
    n = len(reference)
    if n < 2:
        return 0.0
    
    # Default k = half of average segment length
    if k is None:
        changes = sum(reference)
        n_segments = changes + 1
        k = max(1, int(n / (2 * n_segments)))
    
    # Convert changes to segment IDs
    ref_segs = changes_to_segments(reference)
    hyp_segs = changes_to_segments(hypothesis)
    
    errors = 0
    total = 0
    for i in range(n - k):
        ref_same = (ref_segs[i] == ref_segs[i + k])
        hyp_same = (hyp_segs[i] == hyp_segs[i + k])
        if ref_same != hyp_same:
            errors += 1
        total += 1
    
    return errors / total if total > 0 else 0.0


def compute_windowdiff(reference, hypothesis, k=None):
    """
    WindowDiff metric (Pevzner & Hearst 2002).
    Stricter than Pk — penalizes near-misses and false alarms.
    
    Lower is better. Range [0, 1]. Perfect = 0.
    """
    n = len(reference)
    if n < 2:
        return 0.0
    
    if k is None:
        changes = sum(reference)
        n_segments = changes + 1
        k = max(1, int(n / (2 * n_segments)))
    
    errors = 0
    total = 0
    for i in range(n - k):
        ref_changes_in_window = sum(reference[i:i + k])
        hyp_changes_in_window = sum(hypothesis[i:i + k])
        if ref_changes_in_window != hyp_changes_in_window:
            errors += 1
        total += 1
    
    return errors / total if total > 0 else 0.0


def changes_to_segments(changes):
    """Convert binary change array to segment IDs. [0,1,0,0,1] → [0,0,1,1,1,2]"""
    segments = [0]
    seg_id = 0
    for c in changes:
        if c == 1:
            seg_id += 1
        segments.append(seg_id)
    return segments


def evaluate_document_level(problems, all_preds, threshold=0.5):
    """
    Evaluate at document level — computes per-doc metrics then averages.
    
    Args:
        problems: list of problem dicts with 'changes'
        all_preds: list of probability arrays (one per document)
        threshold: decision threshold
    
    Returns:
        dict with averaged metrics
    """
    all_f1, all_pk, all_wd = [], [], []
    all_true, all_pred_binary = [], []
    
    for prob, pred_probs in zip(problems, all_preds):
        if prob["changes"] is None:
            continue
        
        true = prob["changes"]
        pred_binary = [1 if p > threshold else 0 for p in pred_probs]
        
        # Ensure same length
        min_len = min(len(true), len(pred_binary))
        true = true[:min_len]
        pred_binary = pred_binary[:min_len]
        
        if len(true) == 0:
            continue
        
        # Per-doc F1
        doc_f1 = f1_score(true, pred_binary, average="macro", zero_division=0)
        all_f1.append(doc_f1)
        
        # Pk and WindowDiff
        all_pk.append(compute_pk(true, pred_binary))
        all_wd.append(compute_windowdiff(true, pred_binary))
        
        all_true.extend(true)
        all_pred_binary.extend(pred_binary)
    
    # Global metrics
    global_metrics = compute_f1_metrics(all_true, all_pred_binary)
    
    return {
        "f1_macro": global_metrics["f1_macro"],
        "f1_binary": global_metrics["f1_binary"],
        "precision": global_metrics["precision"],
        "recall": global_metrics["recall"],
        "accuracy": global_metrics["accuracy"],
        "avg_doc_f1": np.mean(all_f1) if all_f1 else 0.0,
        "pk": np.mean(all_pk) if all_pk else 0.0,
        "windowdiff": np.mean(all_wd) if all_wd else 0.0,
        "n_docs": len(all_f1),
        "n_boundaries": len(all_true),
        "change_rate_true": np.mean(all_true) if all_true else 0.0,
        "change_rate_pred": np.mean(all_pred_binary) if all_pred_binary else 0.0,
    }


def format_metrics(metrics, prefix=""):
    """Format metrics dict into a readable string."""
    lines = []
    if prefix:
        lines.append(f"--- {prefix} ---")
    lines.append(f"  F1 (macro):    {metrics['f1_macro']:.4f}")
    lines.append(f"  F1 (binary):   {metrics['f1_binary']:.4f}")
    lines.append(f"  Precision:     {metrics['precision']:.4f}")
    lines.append(f"  Recall:        {metrics['recall']:.4f}")
    lines.append(f"  Accuracy:      {metrics['accuracy']:.4f}")
    lines.append(f"  Avg Doc F1:    {metrics['avg_doc_f1']:.4f}")
    lines.append(f"  Pk:            {metrics['pk']:.4f}  (lower=better)")
    lines.append(f"  WindowDiff:    {metrics['windowdiff']:.4f}  (lower=better)")
    lines.append(f"  Docs: {metrics['n_docs']} | Boundaries: {metrics['n_boundaries']}")
    lines.append(f"  Change rate: true={metrics['change_rate_true']:.4f} pred={metrics['change_rate_pred']:.4f}")
    return "\n".join(lines)
