"""
NMR Preprocessor — builds tensors for 1H, 13C, or joint mode.
==============================================================
Key fixes:
  - Tighter percentile clipping (2.5/97.5) for ALL features
  - NORM_CLIP=4.0 to stay safely in fp16/bf16 range
  - Logs actual tensor max values to verify clipping works
  - Reduced default max_peaks to match real data distributions

Usage:
  python nmr_preprocess.py --config config_1h.yaml
  python nmr_preprocess.py --config config_13c.yaml
  python nmr_preprocess.py --config config_joint.yaml
"""

import argparse, ast, csv, json, logging, math, os, random, re, sys
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("NMR-preprocess")

SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|[BCNOPSFIcnos]|[=#/\\@+\-\.\(\)]|[0-9]|%[0-9]{2})")

NORM_CLIP = 4.0  # Maximum absolute value for any normalized tensor

# ═══════════════════════ NMR PARSING ═══════════════════════

def normalize_splitting(raw):
    if raw is None: return None
    ns = raw.strip().lower().replace("-","_").replace(".","").replace(" ","")
    if not ns: return None
    syn = {"singlet":"s","doublet":"d","triplet":"t","quartet":"q",
           "quintet":"quint","pentet":"p","multiplet":"m",
           "brs":"br_s","br":"br_s","brd":"br_d","brt":"br_d",
           "brdd":"br_d","brq":"br_s","septet":"hept","heptet":"hept",
           "sext":"hex","dm":"m","quint":"quint"}
    if ns in syn: return syn[ns]
    if ns.startswith("app"):
        base = ns[3:].strip("_")
        return syn.get(base, base) if base else None
    orig = raw.strip()
    if orig in ("AB","ABq"): return orig
    if ns == "ab": return "AB"
    if ns == "abq": return "ABq"
    return ns

def _parse_j(j_list):
    if not j_list: return []
    if isinstance(j_list, str):
        try: j_list = ast.literal_eval(j_list)
        except: j_list = [j_list]
    if not isinstance(j_list, (list, tuple)): return []
    out = []
    for j in j_list:
        try:
            if isinstance(j, (int, float)):
                v = float(j)
                if not math.isnan(v): out.append(v)
            elif isinstance(j, str):
                m = re.search(r"(\d+\.?\d*)", j)
                if m: out.append(float(m.group(1)))
        except: pass
    return out

def parse_h1(raw_str):
    if not raw_str or raw_str.strip() in ("[]","None","nan",""): return []
    try: data = ast.literal_eval(raw_str.strip())
    except: return []
    if not isinstance(data, (list, tuple)): return []
    peaks = []
    for e in data:
        try:
            if not isinstance(e, (list,tuple)) or len(e) < 5: continue
            sp, jr, ig, ps, pe = e[:5]
            ps_f, pe_f = float(ps), float(pe)
            if math.isnan(ps_f) or math.isnan(pe_f): continue
            ppm = (ps_f + pe_f) / 2.0
            jv = _parse_j(jr)
            sp_n = normalize_splitting(sp) if isinstance(sp, str) else None
            m = re.search(r"(\d+)", str(ig))
            integ = min(int(m.group(1)), 20) if m else 1
            peaks.append((ppm, sp_n, integ, sum(jv)/len(jv) if jv else 0.0, ps_f, pe_f))
        except: continue
    peaks.sort(key=lambda p: -p[0])
    return peaks

def parse_c13(raw_str):
    if not raw_str or raw_str.strip() in ("[]","None","nan",""): return []
    try: data = ast.literal_eval(raw_str.strip())
    except: return []
    if not isinstance(data, (list, tuple)): return []
    peaks = []
    for e in data:
        try:
            if not isinstance(e, (list,tuple)) or len(e) < 3: continue
            ppm_raw, sp_raw, j_raw = e[:3]
            ppm = sum(float(x) for x in ppm_raw)/len(ppm_raw) if isinstance(ppm_raw,(list,tuple)) else float(ppm_raw)
            if math.isnan(ppm): continue
            jv = 0.0
            if j_raw is not None:
                try:
                    jv = float(j_raw[0]) if isinstance(j_raw,(list,tuple)) else float(j_raw)
                    if math.isnan(jv): jv = 0.0
                except: jv = 0.0
            sp_n = normalize_splitting(sp_raw) if sp_raw is not None else None
            peaks.append((ppm, sp_n, jv))
        except: continue
    peaks.sort(key=lambda p: -p[0])
    return peaks


# ═══════════════════════ MAIN ═══════════════════════

def preprocess(cfg, output_dir):
    dc = cfg.get("data", {})
    mode = dc.get("mode", "1h")
    sc = dc.get("smiles_col", "canon_smiles")
    max_peaks_1h = dc.get("max_peaks_1h", dc.get("max_peaks", 30))
    max_peaks_13c = dc.get("max_peaks_13c", dc.get("max_peaks", 40))
    max_smi = dc.get("max_smiles_len", 600)
    min_sf = dc.get("min_split_freq", 50)  # Higher threshold = fewer classes

    is_joint = (mode == "joint")
    nmr_col_1h = dc.get("nmr_col_1h", dc.get("nmr_col", "NMR_processed_1H"))
    nmr_col_13c = dc.get("nmr_col_13c", dc.get("nmr_col", "NMR_processed_13C"))

    for d in ["vocab", "splits", "tensors"]:
        os.makedirs(os.path.join(output_dir, d), exist_ok=True)

    # Read CSV
    logger.info(f"Reading {dc.get('csv_path','data.csv')} (mode={mode})...")
    rows = []
    with open(dc.get("csv_path","data.csv"), "r", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            smi = r.get(sc, "").strip()
            if not smi: continue
            if is_joint:
                if len(parse_h1(r.get(nmr_col_1h, ""))) > 0 and len(parse_c13(r.get(nmr_col_13c, ""))) > 0:
                    rows.append(r)
            elif mode == "1h":
                if len(parse_h1(r.get(nmr_col_1h, ""))) > 0: rows.append(r)
            else:
                if len(parse_c13(r.get(nmr_col_13c, ""))) > 0: rows.append(r)
    logger.info(f"  {len(rows)} valid rows")

    # Vocabularies
    logger.info("Extracting vocabularies...")
    spc = Counter()
    all_smi = []
    for r in rows:
        all_smi.append(r.get(sc, ""))
        if mode in ("1h", "joint"):
            for pk in parse_h1(r.get(nmr_col_1h, "")):
                if pk[1]: spc[pk[1]] += 1
        if mode in ("13c", "joint"):
            for pk in parse_c13(r.get(nmr_col_13c, "")):
                if pk[1]: spc[pk[1]] += 1

    # Keep only frequent splitting patterns to reduce class count
    data_patterns = [s for s, c in spc.most_common() if c >= min_sf and s != "s"]
    sp = ["<pad>", "s"] + data_patterns + ["other"]
    s2id = {s: i for i, s in enumerate(sp)}
    with open(os.path.join(output_dir, "vocab", "splitting_patterns.json"), "w") as f:
        json.dump({"patterns": sp, "pattern2id": s2id, "n_patterns": len(sp)}, f, indent=2)
    logger.info(f"  {len(sp)} splitting patterns (min_freq={min_sf})")

    # SMILES tokenizer
    tc = Counter()
    for smi in all_smi: tc.update(SMILES_REGEX.findall(smi))
    t2id = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
    mtf = dc.get("min_token_freq", 2)
    for t, c in tc.most_common():
        if c >= mtf and t not in t2id: t2id[t] = len(t2id)
    with open(os.path.join(output_dir, "vocab", "smiles_tokenizer.json"), "w") as f:
        json.dump({"token2id": t2id, "vocab_size": len(t2id)}, f, indent=2)
    logger.info(f"  SMILES vocab: {len(t2id)}")

    # Split
    logger.info("Splitting...")
    seed = cfg.get("training", {}).get("seed", 42)
    rng = random.Random(seed)
    rng.shuffle(rows)
    val_f = dc.get("val_frac", 0.10)
    test_f = dc.get("test_frac", 0.10)
    n = len(rows)
    n_val, n_test = int(n * val_f), int(n * test_f)
    te_rows = rows[:n_test]
    vl_rows = rows[n_test:n_test+n_val]
    tr_rows = rows[n_test+n_val:]
    logger.info(f"  train={len(tr_rows)} val={len(vl_rows)} test={len(te_rows)}")

    fn = list(rows[0].keys())
    for name, sub in [("train", tr_rows), ("val", vl_rows), ("test", te_rows)]:
        with open(os.path.join(output_dir, "splits", f"{name}.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fn); w.writeheader(); w.writerows(sub)

    # ═══ Norm stats (training only) ═══
    logger.info("Computing normalization stats (training set only)...")

    def collect_1h(subset):
        ppms, js, integs, widths = [], [], [], []
        for r in subset:
            for pk in parse_h1(r.get(nmr_col_1h, "")):
                ppms.append(pk[0]); js.append(pk[3])
                integs.append(pk[2]); widths.append((pk[4] - pk[5]) / 2.0)
        return ppms, js, integs, widths

    def collect_13c(subset):
        ppms, js = [], []
        for r in subset:
            for pk in parse_c13(r.get(nmr_col_13c, "")):
                ppms.append(pk[0]); js.append(pk[2])
        return ppms, js

    def compute_stats(vals, name, lo_pct=2.5, hi_pct=97.5):
        """Clip to percentiles, compute stats. After normalization, values ∈ ~[-3, 3]."""
        if not vals:
            return {"mean": 0.0, "std": 1.0, "clip_lo": -1e6, "clip_hi": 1e6, "n": 0}
        a = np.array(vals, dtype=np.float64)
        lo = float(np.percentile(a, lo_pct))
        hi = float(np.percentile(a, hi_pct))
        if abs(hi - lo) < 1e-8:
            hi = lo + 1.0
        ac = np.clip(a, lo, hi)
        m, s = float(np.mean(ac)), float(np.std(ac))
        if s < 1e-6: s = 1.0
        max_norm = (hi - m) / s
        min_norm = (lo - m) / s
        logger.info(f"  {name}: mean={m:.4f} std={s:.4f} clip=[{lo:.2f}, {hi:.2f}] "
                     f"norm_range=[{min_norm:.2f}, {max_norm:.2f}] n={len(vals)}")
        return {"mean": m, "std": s, "clip_lo": lo, "clip_hi": hi, "n": len(vals)}

    ns = {}

    if mode in ("1h", "joint"):
        ppms_1h, js_1h, integs_1h, widths_1h = collect_1h(tr_rows)
        ns["ppm_1h"] = compute_stats(ppms_1h, "ppm_1h", 0.5, 99.5)
        ns["j_1h"] = compute_stats(js_1h, "j_1h", 0, 97.5)
        ns["integ"] = compute_stats(integs_1h, "integ", 0, 97.5)
        ns["width"] = compute_stats(widths_1h, "width", 0, 97.5)

    if mode in ("13c", "joint"):
        ppms_13c, js_13c = collect_13c(tr_rows)
        ns["ppm_13c"] = compute_stats(ppms_13c, "ppm_13c", 0.5, 99.5)
        ns["j_13c"] = compute_stats(js_13c, "j_13c", 0, 97.5)

    if mode == "1h":
        ns["ppm"] = ns["ppm_1h"]; ns["j"] = ns["j_1h"]
    elif mode == "13c":
        ns["ppm"] = ns["ppm_13c"]; ns["j"] = ns["j_13c"]

    with open(os.path.join(output_dir, "vocab", "norm_stats.json"), "w") as f:
        json.dump(ns, f, indent=2)

    # ═══ Build tensors ═══
    logger.info("Building tensors...")
    import torch

    other_id = s2id.get("other", len(s2id) - 1)
    s_default = s2id.get("s", 1)

    def norm_clip(val, stat_key):
        st = ns[stat_key]
        clamped = max(min(val, st["clip_hi"]), st["clip_lo"])
        normed = (clamped - st["mean"]) / st["std"]
        return max(min(normed, NORM_CLIP), -NORM_CLIP)

    def build(subset, name):
        N = len(subset)
        has_1h = mode in ("1h", "joint")
        has_13c = mode in ("13c", "joint")

        if has_1h:
            ppm_1h_t = np.zeros((N, max_peaks_1h), np.float32)
            ppm_1h_raw_t = np.zeros((N, max_peaks_1h), np.float32)
            split_1h_t = np.zeros((N, max_peaks_1h), np.int64)
            j_1h_t = np.zeros((N, max_peaks_1h), np.float32)
            mask_1h_t = np.zeros((N, max_peaks_1h), np.bool_)
            integ_t = np.zeros((N, max_peaks_1h), np.float32)
            width_t = np.zeros((N, max_peaks_1h), np.float32)

        if has_13c:
            ppm_13c_t = np.zeros((N, max_peaks_13c), np.float32)
            ppm_13c_raw_t = np.zeros((N, max_peaks_13c), np.float32)
            split_13c_t = np.zeros((N, max_peaks_13c), np.int64)
            j_13c_t = np.zeros((N, max_peaks_13c), np.float32)
            mask_13c_t = np.zeros((N, max_peaks_13c), np.bool_)

        smi_ids = np.zeros((N, max_smi), np.int64)
        smi_len = np.zeros((N,), np.int64)
        smi_str = []

        for idx, row in enumerate(subset):
            smi = row.get(sc, "").strip()
            smi_str.append(smi)

            if has_1h:
                peaks = parse_h1(row.get(nmr_col_1h, ""))[:max_peaks_1h]
                for i, pk in enumerate(peaks):
                    ppm, sp, integ, jv, ppm_s, ppm_e = pk
                    half_w = (ppm_s - ppm_e) / 2.0
                    ppm_1h_raw_t[idx, i] = ppm
                    ppm_1h_t[idx, i] = norm_clip(ppm, "ppm_1h")
                    split_1h_t[idx, i] = s2id.get(sp, other_id) if sp else s_default
                    j_1h_t[idx, i] = norm_clip(jv, "j_1h")
                    integ_t[idx, i] = norm_clip(integ, "integ")
                    width_t[idx, i] = norm_clip(half_w, "width")
                    mask_1h_t[idx, i] = True

            if has_13c:
                peaks = parse_c13(row.get(nmr_col_13c, ""))[:max_peaks_13c]
                for i, pk in enumerate(peaks):
                    ppm, sp, jv = pk
                    ppm_13c_raw_t[idx, i] = ppm
                    ppm_13c_t[idx, i] = norm_clip(ppm, "ppm_13c")
                    split_13c_t[idx, i] = s2id.get(sp, other_id) if sp else s_default
                    j_13c_t[idx, i] = norm_clip(jv, "j_13c")
                    mask_13c_t[idx, i] = True

            toks = SMILES_REGEX.findall(smi)[:max_smi - 2]
            ids = [1] + [t2id.get(t, 3) for t in toks] + [2]
            smi_ids[idx, :len(ids)] = ids
            smi_len[idx] = len(ids)

        T = {"smiles_ids": torch.from_numpy(smi_ids),
             "smiles_len": torch.from_numpy(smi_len), "smiles_str": smi_str}

        if has_1h:
            T.update({"ppm_1h": torch.from_numpy(ppm_1h_t),
                       "ppm_1h_raw": torch.from_numpy(ppm_1h_raw_t),
                       "split_1h": torch.from_numpy(split_1h_t),
                       "j_1h": torch.from_numpy(j_1h_t),
                       "mask_1h": torch.from_numpy(mask_1h_t),
                       "integ": torch.from_numpy(integ_t),
                       "width": torch.from_numpy(width_t)})
            if mode == "1h":
                T["ppm"]=T["ppm_1h"]; T["ppm_raw"]=T["ppm_1h_raw"]
                T["split"]=T["split_1h"]; T["j"]=T["j_1h"]; T["mask"]=T["mask_1h"]

        if has_13c:
            T.update({"ppm_13c": torch.from_numpy(ppm_13c_t),
                       "ppm_13c_raw": torch.from_numpy(ppm_13c_raw_t),
                       "split_13c": torch.from_numpy(split_13c_t),
                       "j_13c": torch.from_numpy(j_13c_t),
                       "mask_13c": torch.from_numpy(mask_13c_t)})
            if mode == "13c":
                T["ppm"]=T["ppm_13c"]; T["ppm_raw"]=T["ppm_13c_raw"]
                T["split"]=T["split_13c"]; T["j"]=T["j_13c"]; T["mask"]=T["mask_13c"]

        torch.save(T, os.path.join(output_dir, "tensors", f"{name}.pt"))

        # Verify actual tensor ranges
        for key in ["ppm_1h", "ppm_13c", "j_1h", "j_13c", "integ", "width", "ppm", "j"]:
            if key in T and isinstance(T[key], torch.Tensor):
                mx = T[key].abs().max().item()
                status = "OK" if mx <= NORM_CLIP + 0.01 else "EXCEEDS!"
                if mx > 0.01:
                    logger.info(f"    {key}: max_abs={mx:.3f} [{status}]")

        logger.info(f"  Saved {name}.pt ({N} samples)")

    for name, sub in [("train", tr_rows), ("val", vl_rows), ("test", te_rows)]:
        build(sub, name)

    # Peak count stats
    if mode in ("1h", "joint"):
        counts = [len(parse_h1(r.get(nmr_col_1h, ""))) for r in rows]
        c = np.array(counts)
        logger.info(f"  1H peaks/mol: mean={c.mean():.1f} p95={np.percentile(c,95):.0f} max={c.max()} max_peaks={max_peaks_1h}")
    if mode in ("13c", "joint"):
        counts = [len(parse_c13(r.get(nmr_col_13c, ""))) for r in rows]
        c = np.array(counts)
        logger.info(f"  13C peaks/mol: mean={c.mean():.1f} p95={np.percentile(c,95):.0f} max={c.max()} max_peaks={max_peaks_13c}")

    logger.info(f"\n{'='*60}\n  DONE: {output_dir} (mode={mode})\n{'='*60}")


if __name__ == "__main__":
    import yaml
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", required=True)
    args = pa.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)
    od = cfg.get("data", {}).get("processed_dir", "processed")
    preprocess(cfg, od)
