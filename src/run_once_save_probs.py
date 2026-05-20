# run_once_save_probs.py  — drop this in your src/ and run it once
import torch, os
from config import FEATURE_CACHE_DIR, ENCODERS, DIFFICULTIES

for enc_name in ENCODERS:
    for difficulty in DIFFICULTIES:
        for split in ["train", "val"]:
            shard_path = os.path.join(
                FEATURE_CACHE_DIR, f"shard_{enc_name}_{difficulty}_{split}.pt"
            )
            prob_path = os.path.join(
                FEATURE_CACHE_DIR, f"prob_{enc_name}_{difficulty}_{split}.pt"
            )
            if not os.path.exists(shard_path):
                print(f"SKIP (no shard): {shard_path}")
                continue
            if os.path.exists(prob_path):
                print(f"SKIP (exists):   {prob_path}")
                continue
            print(f"Extracting probs: {enc_name}/{difficulty}/{split} ...", end=" ", flush=True)
            shard    = torch.load(shard_path, weights_only=False)
            prob_only = [
                (entry["prob"] if entry is not None and entry["prob"] is not None else None)
                for entry in shard
            ]
            del shard
            torch.save(prob_only, prob_path)
            mb = os.path.getsize(prob_path) / 1e6
            print(f"saved {mb:.1f} MB → {prob_path}")