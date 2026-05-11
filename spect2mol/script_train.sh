#!/bin/bash
#SBATCH --job-name=prism6
#SBATCH --output=prism6_%j.out
#SBATCH --error=prism6_%j.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=100gb


set -euo pipefail
source ~/spp_ahq/bin/activate

echo "═══════════════════════════════════════════"


python train.py \
    --config config.yaml \
    --best_metric exact_match \
    --top_k_checkpoints 5 \
    --use_molecular_formula_encoder --use_fcspe --use_gsfu \
    --use_sarab --use_bicma --use_peak_atom_head \
    > output_train.txt

echo "DONE "




