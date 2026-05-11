#!/bin/bash
#SBATCH --job-name=prism6
#SBATCH --output=prism6_%j.out
#SBATCH --error=prism6_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=100gb

set -euo pipefail
source ~/spp_ahq/bin/activate

# ── Preprocess (skip if already done) ──

python preprocess.py --config config.yaml

echo "DONE"