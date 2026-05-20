FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    transformers==4.40.1 \
    sentence-transformers==2.7.0 \
    scikit-learn \
    tqdm \
    huggingface_hub \
    sentencepiece

# Copy source code only
COPY src /app/src

# Download checkpoints from HF DURING BUILD
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download( \
    repo_id='Shubhankar1708/pan26-style-change-models', \
    local_dir='/app', \
    local_dir_use_symlinks=False \
)"

RUN python -c "\
from transformers import AutoTokenizer; \
AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base', use_fast=False); \
AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1', use_fast=False); \
AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2', use_fast=False); \
"

WORKDIR /app/src

ENTRYPOINT ["python", "predict.py"]
