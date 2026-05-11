#!/bin/bash
#SBATCH --job-name=nmr_13c
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=50gb
#SBATCH --output=nmr_13c_%j.log
#SBATCH --error=nmr_13c_%j.err

set -euo pipefail
source ~/spp_ahq/bin/activate

echo "═══════════════════════════════════════════"
echo "  13C NMR — Auto-resumes from latest.pt"
echo "═══════════════════════════════════════════"

# Preprocess (uncomment on first run only)
python nmr_preprocess.py --config config_13c.yaml

# Train
python nmr_train.py --config config_13c.yaml --precision fp32 --best_metric f1

# Test set inference
python nmr_inference.py --checkpoint checkpoints/nmr_13c/best.pt \
    --output_csv predictions_13c.csv --output_json predictions_13c.json

echo "DONE — 13C"