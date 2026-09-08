#!/usr/bin/env bash
# Source this file after setting PROJECT_ROOT for the active server.
: "${PROJECT_ROOT:?set PROJECT_ROOT to the experiment root}"
export PATH="$PROJECT_ROOT/envs/neuroadapter/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT/repo/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export MPLCONFIGDIR="$PROJECT_ROOT/cache/matplotlib"
