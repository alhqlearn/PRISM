#!/bin/bash
#SBATCH --job-name=nmr_1h
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=50gb
#SBATCH --output=nmr_1h_%j.log
#SBATCH --error=nmr_1h_%j.err

set -euo pipefail
source ~/spp_ahq/bin/activate

echo "═══════════════════════════════════════════"
echo "  1H NMR — Auto-resumes from latest.pt"
echo "═══════════════════════════════════════════"

# Preprocess (uncomment on first run only)
python nmr_preprocess.py --config config_1h.yaml

# Train (auto-resumes if checkpoints/nmr_1h/latest.pt exists)
python nmr_train.py --config config_1h.yaml --precision fp32 --best_metric f1

# Test set inference
python nmr_inference.py --checkpoint checkpoints/nmr_1h/best.pt \
    --output_csv predictions_1h.csv --output_json predictions_1h.json

echo "DONE — 1H"