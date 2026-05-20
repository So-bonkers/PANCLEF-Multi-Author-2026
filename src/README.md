# PAN @ CLEF 2026 — Execution Guide
## Multi-Author Writing Style Analysis — Step-by-Step

**Instance:** `shubhankar_kam@instance-20260428-162052`  
**GPU:** NVIDIA L4 24GB  
**Session:** `tmux attach -t Guneesh`

---

## 0. Initial Setup (30 min)

```bash
# SSH in and attach to your session
tmux attach -t Guneesh

# Navigate to working directory
cd ~/PAN-CLEF

# Create project structure
mkdir -p src checkpoints logs outputs feature_cache

# Copy all scripts into src/
# (Upload the .py files or git clone)
cp *.py src/
cd src

# Install dependencies
pip install torch transformers scikit-learn optuna numpy

# Verify GPU
nvidia-smi  # Should show L4 24GB

# Verify data structure
ls ~/PAN-CLEF/data/
# → easy  medium  hard

ls ~/PAN-CLEF/data/easy/train/ | head -5
# → problem-1.txt  problem-2.txt  ...

ls ~/PAN-CLEF/data/easy/train/ | wc -l
# → Should be ~10500 (txt) + ~10500 (json) = ~21000 files

# Quick data sanity check
head -5 ~/PAN-CLEF/data/easy/train/problem-1.txt
cat ~/PAN-CLEF/data/easy/train/truth-problem-1.json
```

---

## 1. Train CoSENT Encoders — HARD DATA ONLY (~15 hrs)

This is the longest step. Train 3 encoders sequentially on the L4.

```bash
cd ~/PAN-CLEF/src

# Option A: Train all 3 sequentially (safest, ~15 hrs total)
python step1_train_encoders.py --encoder deberta   # ~5 hrs
python step1_train_encoders.py --encoder roberta   # ~5 hrs
python step1_train_encoders.py --encoder electra   # ~5 hrs

# Option B: Train one at a time, check results, then continue
python step1_train_encoders.py --encoder deberta
# Check log: tail -f ~/PAN-CLEF/logs/step1_cosent_deberta.log
# When done, start next:
python step1_train_encoders.py --encoder roberta
python step1_train_encoders.py --encoder electra
```

**What to look for:**
- Val F1 should climb each epoch (aim for >0.65 by epoch 5)
- Separation (avg_pos_sim - avg_neg_sim) should increase
- If F1 plateaus early → training is done, early stopping will kick in

**Checkpoints saved to:**
```
~/PAN-CLEF/checkpoints/deberta_cosent_best/
~/PAN-CLEF/checkpoints/roberta_cosent_best/
~/PAN-CLEF/checkpoints/electra_cosent_best/
```

**If ELECTRA is too slow**, use this fallback:
```bash
# Edit config.py and replace electra entry:
# "electra": {"model_name": "microsoft/deberta-v3-large", "hidden_dim": 1024}
```

**Monitor training (in another tmux window):**
```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# Watch logs
tail -f ~/PAN-CLEF/logs/step1_cosent_deberta.log
```

---

## 2. Train MLP Classifiers (~6 hrs)

9 classifiers: 3 encoders × 3 difficulties. Each ~40 min.

```bash
cd ~/PAN-CLEF/src

# Train all 9
python step2_train_classifiers.py

# Or train specific ones:
python step2_train_classifiers.py --encoder deberta --difficulty hard
python step2_train_classifiers.py --encoder deberta --difficulty medium
python step2_train_classifiers.py --encoder deberta --difficulty easy
# ... repeat for roberta, electra
```

**What to look for:**
- Easy: Val F1 > 0.95
- Medium: Val F1 > 0.60 (hard due to imbalance — check that threshold is LOW ~0.10-0.20)
- Hard: Val F1 > 0.80

**Checkpoints saved to:**
```
~/PAN-CLEF/checkpoints/deberta_easy_classifier.pt
~/PAN-CLEF/checkpoints/deberta_medium_classifier.pt
~/PAN-CLEF/checkpoints/deberta_hard_classifier.pt
# ... 6 more for roberta + electra
```

---

## 3. Tune Ensemble (~1 hr)

Finds optimal weights and thresholds per difficulty.

```bash
cd ~/PAN-CLEF/src
python step3_tune_ensemble.py
```

**What to look for:**
- Easy threshold: ~0.40-0.55
- Medium threshold: ~0.08-0.20 (VERY LOW — catches rare changes)
- Hard threshold: ~0.40-0.55
- Ensemble F1 should be higher than any single-model F1

**Config saved to:**
```
~/PAN-CLEF/checkpoints/ensemble_config.json
```

**⚠️ SAFETY NET: At this point, you have a working ensemble submission.**
If you're running low on time, skip step 4 and go straight to step 5.

---

## 4. Train Bi-LSTM Sequential Refinement (~6 hrs)

This is the NOVEL component. Processes boundaries sequentially.

```bash
cd ~/PAN-CLEF/src

# Pre-extract features first (cached to disk — saves GPU time later)
# This happens automatically in step4, but takes ~2 hrs first run.

# Train all 3
python step4_train_bilstm.py

# Or one at a time:
python step4_train_bilstm.py --difficulty easy    # ~1.5 hrs
python step4_train_bilstm.py --difficulty hard    # ~1.5 hrs
python step4_train_bilstm.py --difficulty medium  # ~2 hrs (longer docs)
```

**What to look for:**
- F1 should be HIGHER than ensemble-only from step 3
- Pk and WindowDiff should be LOWER
- If Bi-LSTM hurts F1 → skip it. The ensemble alone is competitive.

**Checkpoints saved to:**
```
~/PAN-CLEF/checkpoints/bilstm_easy.pt
~/PAN-CLEF/checkpoints/bilstm_medium.pt
~/PAN-CLEF/checkpoints/bilstm_hard.pt
```

---

## 5. Evaluate + Generate Solutions (~1 hr)

Full evaluation on validation set with all metrics.

```bash
cd ~/PAN-CLEF/src

# Evaluate on validation (with F1, Pk, WindowDiff)
python step5_evaluate.py --mode val

# Generate test solutions (no metrics — just JSONs)
python step5_evaluate.py --mode test

# Both
python step5_evaluate.py --mode both
```

**Outputs:**
```
~/PAN-CLEF/outputs/easy/solution-problem-1.json
~/PAN-CLEF/outputs/easy/solution-problem-2.json
~/PAN-CLEF/outputs/medium/solution-problem-1.json
...
~/PAN-CLEF/logs/metrics_easy_validation.json
~/PAN-CLEF/logs/metrics_medium_validation.json
~/PAN-CLEF/logs/metrics_hard_validation.json
```

**Expected validation metrics (targets):**

| Metric | Easy | Medium | Hard |
|--------|------|--------|------|
| F1 (macro) | >0.96 | >0.75 | >0.85 |
| Pk | <0.10 | <0.15 | <0.12 |
| WindowDiff | <0.12 | <0.18 | <0.15 |

---

## 6. Docker + TIRA Submission (~2 hrs)

```bash
cd ~/PAN-CLEF

# Create Dockerfile
cat > Dockerfile <<'EOF'
FROM python:3.10-slim
RUN pip install --no-cache-dir torch==2.1.0+cpu \
    -f https://download.pytorch.org/whl/cpu \
    transformers==4.36.0 scikit-learn numpy
COPY src/*.py /app/
COPY checkpoints/ /app/checkpoints/
WORKDIR /app
ENTRYPOINT ["python", "step5_evaluate.py", "--mode", "test"]
EOF

# Build
docker build -t pan2026-mawsa .

# Test locally
mkdir -p test_output
docker run -v ~/PAN-CLEF/data:/input -v ~/PAN-CLEF/test_output:/output \
    pan2026-mawsa --input /input --output /output

# Verify outputs
ls test_output/easy/
cat test_output/easy/solution-problem-1.json

# Push to TIRA (follow their instructions)
# https://www.tira.io
```

---

## File Reference

| File | Purpose | Run Step |
|------|---------|----------|
| `config.py` | All hyperparameters, paths, settings | — (imported) |
| `data_utils.py` | Data loading, pairs, metrics (F1, Pk, WD) | — (imported) |
| `models.py` | MLP classifier + Bi-LSTM definitions | — (imported) |
| `losses.py` | CoSENT, Focal, R-Drop losses | — (imported) |
| `step1_train_encoders.py` | Train 3 CoSENT encoders on hard data | Step 1 |
| `step2_train_classifiers.py` | Train 9 MLP classifiers per-difficulty | Step 2 |
| `step3_tune_ensemble.py` | Optuna weight + threshold tuning | Step 3 |
| `step4_train_bilstm.py` | Train 3 Bi-LSTM models | Step 4 |
| `step5_evaluate.py` | Full eval + generate solution JSONs | Step 5 |

---

## Checkpoint Reference

After all steps complete:

```
~/PAN-CLEF/checkpoints/
├── deberta_cosent_best/          # Step 1
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
├── roberta_cosent_best/          # Step 1
├── electra_cosent_best/          # Step 1
├── deberta_easy_classifier.pt    # Step 2
├── deberta_medium_classifier.pt
├── deberta_hard_classifier.pt
├── roberta_easy_classifier.pt
├── roberta_medium_classifier.pt
├── roberta_hard_classifier.pt
├── electra_easy_classifier.pt
├── electra_medium_classifier.pt
├── electra_hard_classifier.pt
├── ensemble_config.json          # Step 3
├── bilstm_easy.pt               # Step 4
├── bilstm_medium.pt
└── bilstm_hard.pt
```

---

## Troubleshooting

**GPU OOM during CoSENT training:**
→ Reduce batch_size in config.py: `COSENT["batch_size"] = 32`

**GPU OOM during feature extraction:**
→ Process fewer docs at once in step4, or reduce max_seq_len to 96

**Medium F1 is very low (<0.5):**
→ Check threshold is LOW (~0.10-0.15). Increase pos_weight to 25-30.

**Bi-LSTM makes things worse:**
→ Skip it. Ensemble-only is still competitive. Delete bilstm_{diff}.pt files.

**Running out of time:**
→ Priority order: Step 1 (DeBERTa only) → Step 2 (DeBERTa only) → Step 5
→ Single DeBERTa + CoSENT + MLP is still top 5-8.

---

## Quick Summary

```
Step 1: python step1_train_encoders.py          # ~15 hrs (3 encoders on hard)
Step 2: python step2_train_classifiers.py        # ~6 hrs (9 MLPs)
Step 3: python step3_tune_ensemble.py            # ~1 hr (Optuna)
Step 4: python step4_train_bilstm.py             # ~6 hrs (3 Bi-LSTMs)
Step 5: python step5_evaluate.py --mode both     # ~1 hr (eval + solutions)
Step 6: Docker + TIRA submit                     # ~2 hrs

Total: ~31 hrs GPU + ~4 hrs CPU
```
