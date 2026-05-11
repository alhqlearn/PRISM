"""
NMR Inference — test-set evaluation
Usage:
  python nmr_inference.py --checkpoint checkpoints/nmr_1h/best.pt --output_json pred.json --precision bf16
"""
import argparse, csv, json, logging, os, time
import numpy as np, torch
from tqdm import tqdm
import nmr_model as nm
from nmr_model import NMRPredictor, NMRDataset, SMILESTokenizer, collate_fn, load_vocab, evaluate
from torch.utils.data import DataLoader

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("NMR-infer")

def _sn(idx):
    p = nm.SPLITTING_PATTERNS
    if not p or idx < 0 or idx >= len(p): return "m"
    n = p[idx]
    return None if n == "<pad>" else ("m" if n == "other" else n)

def denorm(results, ns, mode):
    pk_ = f"ppm_{mode}" if f"ppm_{mode}" in ns else "ppm"
    jk_ = f"j_{mode}" if f"j_{mode}" in ns else "j"
    pm, ps = ns[pk_]["mean"], ns[pk_]["std"]; jm, js = ns[jk_]["mean"], ns[jk_]["std"]
    im = ns.get("integ",{}).get("mean",3.0); ist = ns.get("integ",{}).get("std",2.0)
    wm = ns.get("width",{}).get("mean",0.02); wst = ns.get("width",{}).get("std",0.05)
    for r in results:
        for pk in r["peaks"]:
            pk["ppm"] = pk["ppm"]*ps+pm; pk["j_coupling"] = pk["j_coupling"]*js+jm
            pk["splitting_name"] = _sn(pk["splitting"])
            if mode=="1h" and "integration" in pk: pk["integration"] = max(1,min(20,round(pk["integration"]*ist+im)))
            if mode=="1h" and "half_width" in pk:
                hw = max(0.0, pk["half_width"]*wst+wm)
                pk["ppm_start"]=round(pk["ppm"]+hw,2); pk["ppm_end"]=round(pk["ppm"]-hw,2)
            else: pk["ppm_start"]=round(pk["ppm"],2); pk["ppm_end"]=round(pk["ppm"],2)
    return results

def fmt_1h(peaks):
    out = []
    for pk in peaks:
        sp = pk.get("splitting_name") or "s"; j = pk.get("j_coupling",0.0)
        jl = [f"{abs(j):.1f}Hz"] if sp not in ("s","br_s") and abs(j) > 0.1 else []
        out.append((sp, jl, f"{pk.get('integration',1)}H", pk.get("ppm_start",round(pk["ppm"],2)), pk.get("ppm_end",round(pk["ppm"],2))))
    return out

def fmt_13c(peaks):
    out = []
    for pk in peaks:
        ppm = round(pk["ppm"],1); sp = pk.get("splitting_name"); j = pk.get("j_coupling",0.0)
        out.append((ppm, None, None) if sp is None or sp == "s" else (ppm, sp, round(abs(j),1)))
    return out

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", required=True); pa.add_argument("--processed_dir", default=None)
    pa.add_argument("--output_csv", default=None); pa.add_argument("--output_json", default=None)
    pa.add_argument("--batch_size", type=int, default=64); pa.add_argument("--threshold", type=float, default=0.5)
    pa.add_argument("--precision", default="bf16", choices=["fp16","fp32","bf16"])
    args = pa.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["cfg"]; mode = ck.get("mode", cfg.get("data",{}).get("mode","1h"))
    pd = args.processed_dir or ck.get("processed_dir","processed"); load_vocab(pd)
    tok = SMILESTokenizer(); tok.token2id = {k:int(v) for k,v in ck["tokenizer"].items()}
    tok.id2token = {v:k for k,v in tok.token2id.items()}
    model = NMRPredictor(tok, cfg).to(device)
    model.load_state_dict({k.replace("_orig_mod.",""):v for k,v in ck["model"].items()}, strict=False); model.eval()
    ns = ck.get("norm_stats") or json.load(open(os.path.join(pd,"vocab","norm_stats.json")))
    is_joint = (mode == "joint")
    test_ds = NMRDataset(os.path.join(pd,"tensors","test.pt"), tok, cfg, augment=False)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    use_amp = args.precision != "fp32" and device.type == "cuda"

    all_r1h, all_r13c, all_r = [], [], []; all_m = {}; nb = 0; t0 = time.time()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            bd = {k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            preds = model.predict(bd, threshold=args.threshold, use_amp=use_amp)
            out = model.forward(bd)
            if is_joint:
                all_r1h.extend(preds["1h"]); all_r13c.extend(preds["13c"])
                m1 = evaluate(out, bd, ns, "1h", pred_key="pred_1h", ppm_raw_key="ppm_1h_raw",
                    mask_key="mask_1h", split_key="split_1h", j_key="j_1h", ppm_stat="ppm_1h", j_stat="j_1h")
                m2 = evaluate(out, bd, ns, "13c", pred_key="pred_13c", ppm_raw_key="ppm_13c_raw",
                    mask_key="mask_13c", split_key="split_13c", j_key="j_13c", ppm_stat="ppm_13c", j_stat="j_13c")
                met = {f"1h_{k}":v for k,v in m1.items()}; met.update({f"13c_{k}":v for k,v in m2.items()})
            else:
                all_r.extend(preds)
                ps = f"ppm_{mode}" if f"ppm_{mode}" in ns else "ppm"
                jst = f"j_{mode}" if f"j_{mode}" in ns else "j"
                met = evaluate(out, bd, ns, mode=mode, threshold=args.threshold, ppm_stat=ps, j_stat=jst)
            for k,v in met.items(): all_m[k] = all_m.get(k,0)+v
            nb += 1
    for k in all_m: all_m[k] /= max(nb,1)
    elapsed = time.time()-t0

    if is_joint:
        all_r1h = denorm(all_r1h, ns, "1h"); all_r13c = denorm(all_r13c, ns, "13c")
        for r in all_r1h: r["NMR_processed_1H"] = repr(fmt_1h(r["peaks"]))
        for r in all_r13c: r["NMR_processed_13C"] = repr(fmt_13c(r["peaks"]))
    else:
        all_r = denorm(all_r, ns, mode)
        for r in all_r:
            r["NMR_processed_1H" if mode=="1h" else "NMR_processed_13C"] = repr(
                fmt_1h(r["peaks"]) if mode=="1h" else fmt_13c(r["peaks"]))

    logger.info(f"\n{'='*60}\n  RESULTS ({elapsed:.1f}s)\n{'='*60}")
    for k,v in sorted(all_m.items()):
        logger.info(f"  {k:20s}: {v*100:.2f}%" if any(x in k for x in ["acc","f1","prec","rec"]) else f"  {k:20s}: {v:.4f}")

    if args.output_json:
        sd = {"mode": mode, "metrics": all_m}
        if is_joint: sd["predictions_1h"] = all_r1h; sd["predictions_13c"] = all_r13c
        else: sd["predictions"] = all_r
        with open(args.output_json, "w") as f: json.dump(sd, f, indent=2, default=str)
        logger.info(f"Saved: {args.output_json}")
    if args.output_csv and not is_joint:
        col = "NMR_processed_1H" if mode=="1h" else "NMR_processed_13C"
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["smiles", col, "n_peaks"])
            for r in all_r: w.writerow([r["smiles"], r[col], len(r["peaks"])])
        logger.info(f"Saved: {args.output_csv}")

if __name__ == "__main__": main()
