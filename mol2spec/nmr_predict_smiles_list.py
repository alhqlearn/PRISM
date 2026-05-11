
import argparse, ast, csv, json, logging, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nmr_model as nm
from nmr_model import NMRPredictor, SMILESTokenizer, load_vocab, SMILES_REGEX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("NMR-predict-list")


# ────────────────────────── SMILES dataset ──────────────────────────

class SMILESListDataset(Dataset):
    def __init__(self, smiles_list, tokenizer, max_len=600):
        self.smiles = smiles_list
        self.tok = tokenizer
        self.ml = max_len
    def __len__(self): return len(self.smiles)
    def __getitem__(self, idx):
        smi = self.smiles[idx]
        ids = self.tok.encode(smi, max_len=self.ml)
        padded = ids + [0] * (self.ml - len(ids))
        return {"smiles_ids": torch.tensor(padded[:self.ml], dtype=torch.long),
                "smiles_len": len(ids), "smiles_str": smi}


def collate_oob(batch):
    return {"smiles_ids": torch.stack([b["smiles_ids"] for b in batch]),
            "smiles_len": torch.tensor([b["smiles_len"] for b in batch]),
            "smiles_str": [b["smiles_str"] for b in batch]}


# ────────────────────────── Read SMILES from file ──────────────────────────

def read_smiles(path, smiles_col=None):
    ext = os.path.splitext(path)[1].lower()
    smiles = []
    if ext in (".csv", ".tsv"):
        delim = "\t" if ext == ".tsv" else ","
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delim)
            cols = reader.fieldnames or []
            if smiles_col and smiles_col in cols:
                col = smiles_col
            else:
                cands = ["smiles", "SMILES", "canon_smiles", "smi", "canonical_smiles"]
                col = next((c for c in cands if c in cols), cols[0] if cols else None)
                if col is None:
                    raise SystemExit(f"No SMILES column found in {path}")
            for row in reader:
                s = (row.get(col) or "").strip()
                if s: smiles.append(s)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"): continue
                smiles.append(line.split()[0])
    return smiles


# ────────────────────────── Denorm + format (same as nmr_eval_and_plot.py) ──────────────────────────

def _splitname(idx):
    p = nm.SPLITTING_PATTERNS
    if not p or idx < 0 or idx >= len(p):
        return "m"
    n = p[idx]
    return None if n == "<pad>" else ("m" if n == "other" else n)


def _stat(ns, mode_key, fallback_key):
    src = ns[mode_key] if mode_key in ns else ns[fallback_key]
    return src["mean"], src["std"]


def fmt_1h(peaks_dicts, ns):
    pm, ps = _stat(ns, "ppm_1h", "ppm")
    jm, js = _stat(ns, "j_1h", "j")
    im = ns.get("integ", {}).get("mean", 3.0); ist = ns.get("integ", {}).get("std", 2.0)
    wm = ns.get("width", {}).get("mean", 0.02); wst = ns.get("width", {}).get("std", 0.05)
    out = []
    for pk in peaks_dicts:
        ppm_real = pk["ppm"] * ps + pm
        j_real = pk["j_coupling"] * js + jm
        sp = _splitname(pk["splitting"]) or "s"
        integ = max(1, min(20, int(round(pk.get("integration", 1.0) * ist + im))))
        if "half_width" in pk:
            hw = max(0.0, pk["half_width"] * wst + wm)
            ppm_s = round(ppm_real + hw, 2); ppm_e = round(ppm_real - hw, 2)
        else:
            ppm_s = round(ppm_real, 2); ppm_e = round(ppm_real, 2)
        jl = [f"{abs(j_real):.1f}Hz"] if sp not in ("s", "br_s") and abs(j_real) > 0.1 else []
        out.append((sp, jl, f"{integ}H", ppm_s, ppm_e))
    out.sort(key=lambda t: -t[3])
    return repr(out)


def fmt_13c(peaks_dicts, ns):
    pm, ps = _stat(ns, "ppm_13c", "ppm")
    jm, js = _stat(ns, "j_13c", "j")
    out = []
    for pk in peaks_dicts:
        ppm_real = round(pk["ppm"] * ps + pm, 1)
        j_real = pk["j_coupling"] * js + jm
        sp = _splitname(pk["splitting"])
        if sp is None or sp == "s":
            out.append((ppm_real, None, None))
        else:
            out.append((ppm_real, sp, round(abs(j_real), 1)))
    out.sort(key=lambda t: -t[0])
    return repr(out)


# ────────────────────────── Inference for one checkpoint ──────────────────────────

def run_one_model(checkpoint_path, mode, smiles_list, batch_size, precision, threshold,
                  num_workers=0):
    """Run one checkpoint on the FULL smiles_list. Returns list of strings (parallel)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[{mode.upper()}] Loading checkpoint: {checkpoint_path}")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    ck_mode = ck.get("mode", cfg.get("data", {}).get("mode", "1h"))
    if ck_mode != mode and ck_mode != "joint":
        raise SystemExit(f"[{mode}] Checkpoint mode '{ck_mode}' != requested.")

    pd_dir = ck.get("processed_dir", f"processed/nmr_{mode}")
    load_vocab(pd_dir)
    tok = SMILESTokenizer()
    tok.token2id = {k: int(v) for k, v in ck["tokenizer"].items()}
    tok.id2token = {v: k for k, v in tok.token2id.items()}
    model = NMRPredictor(tok, cfg).to(device)
    model.load_state_dict(
        {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()},
        strict=False,
    )
    model.eval()
    ns = ck.get("norm_stats") or json.load(open(os.path.join(pd_dir, "vocab", "norm_stats.json")))
    is_joint = (ck_mode == "joint")
    max_smi = cfg.get("data", {}).get("max_smiles_len", 600)
    logger.info(f"[{mode.upper()}] Params: {model.count_params():,}, joint={is_joint}")

    valid_pairs = [(i, s) for i, s in enumerate(smiles_list)
                   if isinstance(s, str) and s.strip() and len(SMILES_REGEX.findall(s)) > 0]
    logger.info(f"[{mode.upper()}] {len(valid_pairs)}/{len(smiles_list)} SMILES tokenizable")

    results = ["[]"] * len(smiles_list)
    if not valid_pairs:
        del model
        if device.type == "cuda": torch.cuda.empty_cache()
        return results

    valid_smiles = [s for _, s in valid_pairs]
    valid_idx    = [i for i, _ in valid_pairs]
    ds = SMILESListDataset(valid_smiles, tok, max_smi)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_oob, num_workers=num_workers, pin_memory=True)
    use_amp = precision != "fp32" and device.type == "cuda"

    t0 = time.time(); pos = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Predicting {mode.upper()}"):
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            preds = model.predict(bd, threshold=threshold, use_amp=use_amp)
            if is_joint: preds = preds[mode]
            for r in preds:
                s_out = fmt_1h(r["peaks"], ns) if mode == "1h" else fmt_13c(r["peaks"], ns)
                results[valid_idx[pos]] = s_out
                pos += 1
    logger.info(f"[{mode.upper()}] Done in {time.time()-t0:.1f}s")
    del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return results


# ────────────────────────── Main ──────────────────────────

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                 description=__doc__)
    pa.add_argument("--input_file", required=True,
                    help=".txt (one SMILES per line) or .csv (with --smiles_col).")
    pa.add_argument("--smiles_col", default=None)
    pa.add_argument("--output_csv", required=True)
    pa.add_argument("--checkpoint_1h",  default=None,
                    help="Path to 1H checkpoint. Omit to skip 1H prediction.")
    pa.add_argument("--checkpoint_13c", default=None,
                    help="Path to 13C checkpoint. Omit to skip 13C prediction.")
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--threshold", type=float, default=0.5)
    pa.add_argument("--precision", default="bf16", choices=["fp16", "fp32", "bf16"])
    pa.add_argument("--num_workers", type=int, default=0)
    args = pa.parse_args()

    if not args.checkpoint_1h and not args.checkpoint_13c:
        raise SystemExit("Provide at least one of --checkpoint_1h / --checkpoint_13c.")

    logger.info(f"Reading SMILES from: {args.input_file}")
    smiles = read_smiles(args.input_file, args.smiles_col)
    logger.info(f"  {len(smiles)} SMILES read")
    if not smiles:
        raise SystemExit("No SMILES found.")

    out = {"smiles": smiles}

    if args.checkpoint_1h:
        logger.info(f"\n{'='*60}\n  1H predictions ({len(smiles)} molecules)\n{'='*60}")
        out["NMR_processed_1H"] = run_one_model(
            args.checkpoint_1h, "1h", smiles,
            args.batch_size, args.precision, args.threshold, args.num_workers,
        )

    if args.checkpoint_13c:
        logger.info(f"\n{'='*60}\n  13C predictions ({len(smiles)} molecules)\n{'='*60}")
        out["NMR_processed_13C"] = run_one_model(
            args.checkpoint_13c, "13c", smiles,
            args.batch_size, args.precision, args.threshold, args.num_workers,
        )

    df = pd.DataFrame(out)

    def _safe_n(s):
        try: return len(ast.literal_eval(s))
        except Exception: return 0

    if "NMR_processed_1H"  in df.columns:
        df["n_peaks_1h"]  = df["NMR_processed_1H"].apply(_safe_n)
    if "NMR_processed_13C" in df.columns:
        df["n_peaks_13c"] = df["NMR_processed_13C"].apply(_safe_n)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"\nSaved: {out_path}")
    if "n_peaks_1h" in df.columns:
        logger.info(f"  Mean 1H  peaks/mol: {df['n_peaks_1h'].mean():.2f} "
                    f"(median {df['n_peaks_1h'].median():.0f})")
    if "n_peaks_13c" in df.columns:
        logger.info(f"  Mean 13C peaks/mol: {df['n_peaks_13c'].mean():.2f} "
                    f"(median {df['n_peaks_13c'].median():.0f})")


if __name__ == "__main__":
    main()
