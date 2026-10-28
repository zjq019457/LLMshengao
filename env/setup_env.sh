#!/bin/bash
# Recreate the minimal environment on a new Linux GPU machine. Usage: bash env/setup_env.sh /path/to/project_root
# Original machine: Python 3.12.3, 3x NVIDIA A100 80GB PCIe, driver 580.82.07, torch CUDA 13.0.
set -e
ROOT=${1:?project root}
python3.12 -m venv $ROOT/.venv
$ROOT/.venv/bin/pip install -U pip
$ROOT/.venv/bin/pip install -r "$(dirname "$0")/requirements.txt"
mkdir -p $ROOT/{scripts,data,out,out_bench,out_dev,out_bench24,logs,models}
echo "Next: copy scripts/ and data/ into $ROOT, download models into $ROOT/models,"
echo "then set ROOT / MODELS in scripts/common.py (or export K0_ROOT) and R= / PY= at the top of run_k0.sh and run_paper.sh."
