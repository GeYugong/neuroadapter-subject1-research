#!/usr/bin/env bash
set -euo pipefail

if [[ $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) -ne 2 ]]; then
  echo "expected exactly two GPUs" >&2
  exit 1
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "GPU gate failed: active compute processes are present" >&2
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader
  exit 1
fi

echo "GPU gate passed: both GPUs have no active compute process"

