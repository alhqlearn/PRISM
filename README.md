# PRISM: Bidirectional NMR–SMILES Translation

**P**eak **R**esolved **I**ntegrative **S**pectral **M**odel.

PRISM is a paired pair of transformer models for the two ends of small-molecule NMR analysis:

| Direction          | Module      | Task                                                                    |
| ------------------ | ----------- | ----------------------------------------------------------------------- |
| **Forward**        | `mol2spec`  | Given a SMILES string, predict its `¹H` and/or `¹³C` NMR peak list.     |
| **Inverse**        | `spect2mol` | Given `¹H` and/or `¹³C` NMR peak lists and/or `Molecular Formula`, decode the SMILES string.       |

Both models are trained on the same in-house curated NMR corpus and share a tokenizer / vocabulary convention, so the predicted spectra from `mol2spec` can be fed directly into `spect2mol` for round-trip validation.

> 📄 *Paper:* **From spectra to structure and back: PRISM, a bidirectional NMR foundation model**
> 🧪 *Pre-trained checkpoints & training data:* **\<Zenodo DOI — to be added on release\>**
> 💻 *Code:* this repository

---

## Table of Contents

1. [Overview](#overview)
2. [Repository layout](#repository-layout)
3. [Installation](#installation)
4. [Data & checkpoints](#data--checkpoints)
5. [`mol2spec` — SMILES → NMR](#mol2spec--smiles--nmr)
6. [`spect2mol` — NMR → SMILES](#spect2mol--nmr--smiles)
7. [Reproducing the paper](#reproducing-the-paper)
8. [Citing PRISM](#citing-prism)
9. [License](#license)
10. [Contact](#contact)

---

## Overview

Both networks are encoder–decoder transformers, but the encoder/decoder roles are swapped between the two directions.

**`mol2spec` (forward)** uses a SMILES character-level encoder feeding a *set-prediction* decoder with learnable peak queries (DETR-style). Each query emits a peak with `(δ, splitting, J, integration, line-width, existence)`. Loss is matched to ground-truth peaks via a Hungarian assignment on `δ` and existence probability. The `1H` and `13C` variants are trained independently from the same code base via separate YAML configs.

**`spect2mol` (inverse)** uses a peak-token NMR encoder (per-peak embeddings of `δ`, splitting, `J`, integration, with optional Fourier chemical-shift positional encodings, shift-region tags, molecular-formula and solvent conditioning) and an autoregressive SMILES decoder with greedy or beam-search decoding.

A complete schematic and ablations are reported in the paper.

---

## Repository layout

```
PRISM/
├── README.md
├── requirements.txt
├── mol2spec/                       # SMILES → NMR (forward)
│   ├── nmr_model.py                # encoder + DETR-style peak decoder
│   ├── nmr_preprocess.py           # build tokenizer / vocab / tensors
│   ├── nmr_train.py                # training loop (auto-resume, AMP)
│   ├── nmr_inference.py            # test-set evaluation
│   ├── nmr_predict_smiles_list.py  # batch inference on a SMILES list / CSV
│   ├── config_1h.yaml              # 1H training configuration
│   ├── config_13c.yaml             # 13C training configuration
│   ├── run_1h.sh                   # SLURM script: preprocess + train + test (1H)
│   ├── run_13c.sh                  # SLURM script: preprocess + train + test (13C)
│   └── oob_1h_13c.sh               # SLURM script: batch SMILES → NMR inference
│
└── spect2mol/                      # NMR → SMILES (inverse)
    ├── model.py                    # NMR encoder + SMILES decoder (PRISM)
    ├── preprocess.py               # build vocab / tensors from CSV
    ├── train.py                    # training loop
    ├── inference.py                # single-model + top-K ensemble inference
    ├── test.py                     # held-out evaluation
    ├── script_preprocess.sh        # SLURM script: preprocess 
    ├── script_train.sh             # SLURM script: train 
    └── script_test.sh              # SLURM script: test 
```

---

## Installation

PRISM is developed against **Python 3.10+** and **PyTorch ≥ 2.1** with CUDA 11.8 / 12.x. We recommend [`uv`](https://github.com/astral-sh/uv) for environment management; `venv` + `pip` also works.

### Using `uv` (recommended)

```bash
[git clone https://github.com/<user>/PRISM.git](https://github.com/alhqlearn/PRISM.git)
cd PRISM
uv venv ~/spp_ahq                       # named to match the SLURM scripts
source ~/spp_ahq/bin/activate
uv pip install -r requirements.txt
```

### Using `pip`

```bash
python -m venv ~/spp_ahq
source ~/spp_ahq/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Minimum dependencies

```
torch >= 2.1
numpy
pandas
pyyaml
tqdm
rdkit                # canonical SMILES + Tanimoto evaluation
scipy                # Hungarian assignment in spect2mol
```

GPU is strongly recommended: training a single mode (`1H` *or* `13C`) takes roughly **6–8 h on one NVIDIA A40** for 50 epochs at `batch_size=128`. Inference runs comfortably on CPU.

---

## Data & checkpoints

The curated NMR dataset (`df_C_H_merged_NMR_unique.csv`) and all pre-trained checkpoints (`mol2spec/checkpoints/nmr_1h/best.pt`, `mol2spec/checkpoints/nmr_13c/best.pt`, and the `spect2mol/checkpoints/` directory including the top-K ensemble manifest) are released through **Zenodo** under the same DOI as the paper.

After downloading, place files as follows:

```
mol2spec/
    df_C_H_merged_NMR_unique.csv             # training CSV
    checkpoints/nmr_1h/best.pt
    checkpoints/nmr_13c/best.pt
spect2mol/
    checkpoints/best.pt
    checkpoints/top_k_manifest.json
    checkpoints/top_k_ep*.pt
```

Each CSV row contains a canonical SMILES, a `¹H` peak list, and a `¹³C` peak list, both in the bracketed-tuple format described below.

---

## `mol2spec` — SMILES → NMR

### Input

A single SMILES string (or a `.txt` / `.csv` of them). The tokenizer is character-level and shared across both spectra.

### Output

For each molecule, a sorted list of tuples.

**`¹H` format:** `(splitting, [J_list], "<n>H", δ_start, δ_end)`

```python
[('d', ['7.2Hz'], '2H', 7.85, 7.83),
 ('dd', ['8.1Hz'], '1H', 7.42, 7.39),
 ('s', [], '3H', 3.91, 3.91),
 ...]
```

**`¹³C` format:** `(δ, splitting | None, J | None)`

```python
[(168.2, None, None),
 (152.7, None, None),
 (114.3, 'd', 21.4),
 ...]
```

### Configuration

The two configs (`config_1h.yaml`, `config_13c.yaml`) differ only in `mode`, `nmr_col`, `max_peaks`, `processed_dir`, and a couple of loss weights specific to `¹H` (integration, line width). Loss weights are exposed at the top level for easy tuning.

### Reproducing training

The end-to-end pipeline is wrapped in a single shell script per channel:

```bash
# 1H
sbatch mol2spec/run_1h.sh

# 13C
sbatch mol2spec/run_13c.sh
```

Each script runs `nmr_preprocess.py` → `nmr_train.py` → `nmr_inference.py`. Training auto-resumes from `checkpoints/nmr_<mode>/latest.pt` if present; the best model (by `F1`) is also kept as `best.pt`.

Manual invocation:

```bash
python mol2spec/nmr_preprocess.py --config mol2spec/config_1h.yaml
python mol2spec/nmr_train.py      --config mol2spec/config_1h.yaml \
                                  --precision bf16 --best_metric f1
python mol2spec/nmr_inference.py  --checkpoint mol2spec/checkpoints/nmr_1h/best.pt \
                                  --output_json mol2spec/predictions_1h.json \
                                  --precision bf16
```

### Batch inference on new SMILES

```bash
# Both 1H and 13C from a plain text file
python mol2spec/nmr_predict_smiles_list.py \
    --checkpoint_1h  mol2spec/checkpoints/nmr_1h/best.pt \
    --checkpoint_13c mol2spec/checkpoints/nmr_13c/best.pt \
    --input_file     my_smiles.txt \
    --output_csv     my_predictions.csv

# CSV input with a named SMILES column
python mol2spec/nmr_predict_smiles_list.py \
    --checkpoint_1h  mol2spec/checkpoints/nmr_1h/best.pt \
    --checkpoint_13c mol2spec/checkpoints/nmr_13c/best.pt \
    --input_file     mols.csv --smiles_col SMILES \
    --output_csv     my_predictions.csv

# 1H only
python mol2spec/nmr_predict_smiles_list.py \
    --checkpoint_1h mol2spec/checkpoints/nmr_1h/best.pt \
    --input_file    my_smiles.txt \
    --output_csv    my_predictions_1h.csv
```

### Evaluation metrics

`nmr_inference.py` reports peak-level **precision / recall / F1** (existence), **ppm MAE / R²**, **J MAE / R²**, **splitting-pattern accuracy**, and per-molecule **count MAE**, on a held-out test split fixed by `seed=42` in the config.

---

## `spect2mol` — NMR → SMILES

### Input

For each molecule, the same `¹H` and `¹³C` tuple-list format as above (either or both — see *Out-of-bag* below). At inference time, peaks predicted by `mol2spec` can be fed in directly.

### Output

A canonical SMILES string per molecule.

### Reproducing training

```bash
# Preprocess (builds vocabulary + tensorized train/val/test splits)
bash spect2mol/script_preprocess.sh

# Train (saves both best.pt and top-K checkpoints for ensembling)
bash spect2mol/script_train.sh

# Test
bash spect2mol/script_test.sh
```

### Inference

Three modes are supported by `spect2mol/inference.py`:

```bash
# Single best model
python spect2mol/inference.py \
    --checkpoint spect2mol/checkpoints/best.pt \
    --output_csv predictions.csv \
    --beam_size 5

# Top-K weight-averaged ensemble (uses top_k_manifest.json)
python spect2mol/inference.py \
    --checkpoint_dir spect2mol/checkpoints \
    --ensemble_top_k 5 \
    --output_csv predictions_ensemble.csv

# Both, with side-by-side comparison report
python spect2mol/inference.py \
    --checkpoint     spect2mol/checkpoints/best.pt \
    --checkpoint_dir spect2mol/checkpoints \
    --ensemble_top_k 5 \
    --compare \
    --output_csv predictions.csv
```


### Evaluation metrics

`evaluate_predictions` in `model.py` reports **exact-match SMILES accuracy** (after RDKit canonicalisation), **mean Tanimoto similarity** on Morgan-2 fingerprints, and **chemical validity** of the decoded strings.

---

## Reproducing the paper

A full reproduction (single A40 GPU) involves four jobs:

```bash
# Forward direction
sbatch mol2spec/run_1h.sh        # ≈ 7 h
sbatch mol2spec/run_13c.sh       # ≈ 8 h

# Inverse direction
sbatch spect2mol/script_train.sh # ≈ 12 h
sbatch spect2mol/script_test.sh
```

All randomness is controlled by `seed: 42` in each config. The reference numbers in the paper were obtained with `--precision bf16` (or `fp32` on A40 for `mol2spec`), `batch_size=128`, and 50 epochs.

---

## Citing PRISM

If PRISM is useful in your work, please cite:

```bibtex
@article{hoque2026prism,
  title   = {PRISM: Bidirectional NMR\textendash SMILES Translation with Set-Prediction
             and Peak-Conditioned Transformers},
  author  = {Hoque, A. and <co-authors>},
  journal = {<journal>},
  year    = {2026},
  doi     = {<DOI>}
}
```

and the Zenodo deposit for the data + checkpoints:

```bibtex
@dataset{hoque2026prism_data,
  title     = {PRISM dataset and pre-trained checkpoints},
  author    = {Hoque, A. and <co-authors>},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {<Zenodo DOI>}
}
```

---

## License

Source code is released under the **MIT License** (see `LICENSE`). The dataset and pre-trained checkpoints on Zenodo are released under **CC BY 4.0**.

---

## Contact

Questions, bug reports and pull requests are very welcome. Please open a GitHub issue, or contact the corresponding author:

**Ajnabiul Hoque** — `ajnabiul.hoque@uni-muenster.de` — `Postdoctoral Researcher
Glorius Group, Organic Chemistry Institute
University of Münster`
