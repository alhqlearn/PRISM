#!/bin/bash
#SBATCH --job-name=nmr_1h
#SBATCH --partition=GPU
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=50gb
#SBATCH --output=nmr_1h_%j.log
#SBATCH --error=nmr_1h_%j.err

set -euo pipefail
source ~/spp_ahq/bin/activate

# .txt with one SMILES per line:
python nmr_predict_smiles_list.py \
    --checkpoint_1h  checkpoints/nmr_1h/best.pt \
    --checkpoint_13c checkpoints/nmr_13c/best.pt \
    --input_file smiles.txt \
    --output_csv predictions.csv

# .csv with a SMILES column:
python nmr_predict_smiles_list.py \
    --checkpoint_1h  checkpoints/nmr_1h/best.pt \
    --checkpoint_13c checkpoints/nmr_13c/best.pt \
    --input_file mols.csv --smiles_col SMILES \
    --output_csv predictions.csv

# 1H only (skip 13C):
python nmr_predict_smiles_list.py \
    --checkpoint_1h checkpoints/nmr_1h/best.pt \
    --input_file smiles.txt --output_csv predictions_1h.csv
