import argparse, collections, copy, csv, json, logging, os, re, sys, time
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import (PRISM, NMRDataset, SMILESTokenizer, collate_fn,
                          evaluate_predictions, load_vocab)

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("PRISM-infer")


# ═══════════════════════ MODEL LOADING ═══════════════════════

def load_model(ckpt_path, device="cpu", processed_dir=None):
    """Load a single PRISM model from checkpoint."""
    logger.info(f"Loading {ckpt_path}...")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    pd = processed_dir or ck.get("processed_dir", "processed")
    load_vocab(pd)
    tok = SMILESTokenizer()
    tok.token2id = {k: int(v) for k, v in ck["tokenizer"].items()}
    tok.id2token = {v: k for k, v in tok.token2id.items()}
    tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id = 0, 1, 2, 3
    model = PRISM(tok, cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Loaded. vocab={tok.vocab_size}, params={n_params:,}")
    return model, tok, cfg


def load_ensemble_weights(checkpoint_dir, top_k=5, device="cpu",
                           processed_dir=None):
    """
    Load top-K checkpoints and average their weights.
    Uses the top_k_manifest.json produced by train_v2.py.
    """
    manifest_path = os.path.join(checkpoint_dir, "top_k_manifest.json")

    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        paths = [e["path"] for e in manifest[:top_k]]
        logger.info(f"Ensemble from manifest: {len(paths)} checkpoints")
        for e in manifest[:top_k]:
            logger.info(f"  Epoch {e['epoch']}: {e['metric']:.4f} ({e['path']})")
    else:
        # Fallback: find top_k_*.pt files
        import glob
        paths = sorted(glob.glob(os.path.join(checkpoint_dir, "top_k_ep*.pt")))
        if not paths:
            raise FileNotFoundError(
                f"No top-K checkpoints found in {checkpoint_dir}. "
                f"Expected top_k_manifest.json or top_k_ep*.pt files.")
        paths = paths[:top_k]
        logger.info(f"Ensemble from glob: {len(paths)} checkpoints")

    if len(paths) < 2:
        logger.warning(f"Only {len(paths)} checkpoint(s) found, "
                       f"ensemble may not be meaningful.")

    # Load first checkpoint to get config, tokenizer, and model structure
    ck0 = torch.load(paths[0], map_location=device, weights_only=False)
    cfg = ck0["cfg"]
    pd = processed_dir or ck0.get("processed_dir", "processed")
    load_vocab(pd)

    tok = SMILESTokenizer()
    tok.token2id = {k: int(v) for k, v in ck0["tokenizer"].items()}
    tok.id2token = {v: k for k, v in tok.token2id.items()}
    tok.pad_id, tok.bos_id, tok.eos_id, tok.unk_id = 0, 1, 2, 3

    # Average weights
    logger.info(f"Averaging weights from {len(paths)} checkpoints...")
    avg_state = None
    for i, path in enumerate(paths):
        ck = torch.load(path, map_location=device, weights_only=False)
        sd = ck["model"]
        if avg_state is None:
            avg_state = collections.OrderedDict()
            for k, v in sd.items():
                avg_state[k] = v.float().clone()
        else:
            for k, v in sd.items():
                avg_state[k] += v.float()

    # Divide by number of checkpoints
    n = len(paths)
    for k in avg_state:
        avg_state[k] /= n
        # Cast back to original dtype
        orig_dtype = ck0["model"][k].dtype
        avg_state[k] = avg_state[k].to(orig_dtype)

    # Build model with averaged weights
    model = PRISM(tok, cfg).to(device)
    model.load_state_dict(avg_state)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Ensemble model built. vocab={tok.vocab_size}, "
                f"params={n_params:,}, n_models={n}")

    return model, tok, cfg


# ═══════════════════════ INFERENCE ═══════════════════════

def predict_csv(model, tok, cfg, output_csv, beam=5, max_len=550,
                device="cpu", eval_spectra="both"):
    """Run prediction on preprocessed test tensors."""
    from torch.utils.data import DataLoader

    pd = cfg.get("data", {}).get("processed_dir", "processed")
    td = os.path.join(pd, "tensors", "test.pt")
    if not os.path.exists(td):
        logger.error(f"Test tensors not found: {td}")
        return None, None

    ds = NMRDataset(td, tok, cfg, augment=False)

    def _collate(batch):
        out = collate_fn(batch)
        if eval_spectra == "h_only":
            out["c_mask"] = torch.zeros_like(out["c_mask"])
            out["c_ppm"] = torch.zeros_like(out["c_ppm"])
            out["c_split"] = torch.zeros_like(out["c_split"])
            out["c_j"] = torch.zeros_like(out["c_j"])
            if "c_region" in out:
                out["c_region"] = torch.zeros_like(out["c_region"])
        elif eval_spectra == "c_only":
            out["h_mask"] = torch.zeros_like(out["h_mask"])
            out["h_ppm"] = torch.zeros_like(out["h_ppm"])
            out["h_split"] = torch.zeros_like(out["h_split"])
            out["h_integ"] = torch.zeros_like(out["h_integ"])
            out["h_j"] = torch.zeros_like(out["h_j"])
            if "h_region" in out:
                out["h_region"] = torch.zeros_like(out["h_region"])
        return out

    loader = DataLoader(ds, batch_size=512, shuffle=False,
                        collate_fn=_collate, num_workers=2)
    preds, gts = [], []
    t0 = time.time()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", unit="batch"):
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
            preds.extend(model.generate(bd, max_len=max_len, beam_size=beam))
            gts.extend(batch["smiles_str"])

    elapsed = time.time() - t0
    logger.info(f"Predicted {len(preds)} in {elapsed:.1f}s "
                f"({len(preds) / elapsed:.1f}/s)")

    # Save predictions
    with open(output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["index", "ground_truth", "prediction"])
        w.writeheader()
        for i, (p, g) in enumerate(zip(preds, gts)):
            w.writerow({"index": i, "ground_truth": g, "prediction": p})
    logger.info(f"Saved to {output_csv}")

    # Evaluate
    if gts and gts[0]:
        met = evaluate_predictions(preds, gts)
        logger.info(
            f"  Exact: {met['exact_match'] * 100:.2f}% | "
            f"Tanimoto: {met['avg_tanimoto']:.4f} | "
            f"Valid: {met['validity'] * 100:.2f}%")
        return preds, met

    return preds, None


# ═══════════════════════ COMPARISON ═══════════════════════

def compare_models(single_preds, single_met, ensemble_preds, ensemble_met,
                   gts, output_path="comparison_results.json"):
    """Compare single best model vs ensemble performance."""
    results = {
        "single_model": single_met,
        "ensemble_model": ensemble_met,
    }

    # Per-sample comparison
    n = min(len(single_preds), len(ensemble_preds), len(gts))
    single_correct = 0
    ensemble_correct = 0
    both_correct = 0
    neither_correct = 0

    try:
        from rdkit import Chem
        def canon(smi):
            try:
                m = Chem.MolFromSmiles(smi)
                return Chem.MolToSmiles(m, canonical=True) if m else smi
            except:
                return smi
    except ImportError:
        def canon(smi):
            return smi.strip()

    for i in range(n):
        sc = (canon(single_preds[i]) == canon(gts[i]))
        ec = (canon(ensemble_preds[i]) == canon(gts[i]))
        if sc:
            single_correct += 1
        if ec:
            ensemble_correct += 1
        if sc and ec:
            both_correct += 1
        if not sc and not ec:
            neither_correct += 1

    only_single = single_correct - both_correct
    only_ensemble = ensemble_correct - both_correct

    results["comparison"] = {
        "n_samples": n,
        "single_correct": single_correct,
        "ensemble_correct": ensemble_correct,
        "both_correct": both_correct,
        "only_single": only_single,
        "only_ensemble": only_ensemble,
        "neither_correct": neither_correct,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"  Model Comparison ({n} samples)")
    logger.info(f"{'=' * 50}")
    logger.info(f"  Single  exact match: {single_correct / n * 100:.2f}%")
    logger.info(f"  Ensemble exact match: {ensemble_correct / n * 100:.2f}%")
    logger.info(f"  Both correct: {both_correct}")
    logger.info(f"  Only single correct: {only_single}")
    logger.info(f"  Only ensemble correct: {only_ensemble}")
    logger.info(f"  Neither correct: {neither_correct}")
    logger.info(f"  Saved to {output_path}")

    return results


# ═══════════════════════ MAIN ═══════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="PRISM v3.1 Inference with ensemble support")

    # Model selection
    pa.add_argument("--checkpoint", default=None,
                    help="Path to single best checkpoint")
    pa.add_argument("--checkpoint_dir", default=None,
                    help="Directory containing top-K checkpoints for ensemble")
    pa.add_argument("--ensemble_top_k", type=int, default=5,
                    help="Number of top models to ensemble")
    pa.add_argument("--processed_dir", default=None,
                    help="Override processed data directory")

    # Input/Output
    pa.add_argument("--input_csv", default=None,
                    help="Input CSV (uses preprocessed test tensors)")
    pa.add_argument("--output_csv", default="predictions.csv")

    # Generation settings
    pa.add_argument("--beam_size", type=int, default=5)
    pa.add_argument("--max_len", type=int, default=550)

    # Device
    pa.add_argument("--device", default="auto")

    # Enhanced options
    pa.add_argument("--eval_spectra", type=str, default="both",
                    choices=["both", "h_only", "c_only"],
                    help="Which NMR spectra to use at inference. "
                         "'both'=H1+C13, 'h_only'=H1 only, 'c_only'=C13 only. "
                         "Use for out-of-bag testing when model was trained on both.")
    pa.add_argument("--compare", action="store_true",
                    help="Compare single vs ensemble model")

    args = pa.parse_args()

    if not args.checkpoint and not args.checkpoint_dir:
        pa.error("Provide --checkpoint and/or --checkpoint_dir")

    # Device
    if args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)
    logger.info(f"Device: {dev}")

    # ── Single model ──
    single_preds, single_met = None, None
    if args.checkpoint:
        logger.info("=" * 50)
        logger.info("  SINGLE MODEL INFERENCE")
        logger.info("=" * 50)
        model, tok, cfg = load_model(args.checkpoint, dev, args.processed_dir)
        out_csv = args.output_csv
        if args.compare and args.checkpoint_dir:
            out_csv = args.output_csv.replace(".csv", "_single.csv")
        single_preds, single_met = predict_csv(
            model, tok, cfg, out_csv,
            args.beam_size, args.max_len, dev, args.eval_spectra)
        del model
        torch.cuda.empty_cache()

    # ── Ensemble model ──
    ensemble_preds, ensemble_met = None, None
    if args.checkpoint_dir:
        logger.info("=" * 50)
        logger.info("  ENSEMBLE MODEL INFERENCE")
        logger.info("=" * 50)
        model, tok, cfg = load_ensemble_weights(
            args.checkpoint_dir, args.ensemble_top_k, dev, args.processed_dir)
        out_csv = args.output_csv
        if args.compare and args.checkpoint:
            out_csv = args.output_csv.replace(".csv", "_ensemble.csv")
        ensemble_preds, ensemble_met = predict_csv(
            model, tok, cfg, out_csv,
            args.beam_size, args.max_len, dev, args.eval_spectra)
        del model
        torch.cuda.empty_cache()

    # ── Comparison ──
    if args.compare and single_preds and ensemble_preds:
        # Reload ground truths
        pd = cfg.get("data", {}).get("processed_dir", "processed")
        td = os.path.join(pd, "tensors", "test.pt")
        data = torch.load(td, map_location="cpu", weights_only=False)
        gts = data["smiles_str"]
        compare_models(
            single_preds, single_met,
            ensemble_preds, ensemble_met,
            gts,
            args.output_csv.replace(".csv", "_comparison.json"))


if __name__ == "__main__":
    main()
