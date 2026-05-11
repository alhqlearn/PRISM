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

python test.py \
    --checkpoint_dir checkpoints/prism_full_all_true \
    --ensemble_mode weight_avg \
    --weight_avg_ks 5 \
    --top_k_values 1 5 10 \
    --tanimoto_stereo both \
    --input_csv test.csv --id_col ID \
    --beam_size 10 \
    --encoder_chunk 256 \
    --rerank_by_formula \
    --per_beam_csv summary_only \
    --output_csv results_wavg_k5.csv \
    > output_wavg_k5.txt

echo "DONE "
