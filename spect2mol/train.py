

import argparse, csv, json, logging, math, os, random, time, warnings
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml
from torch.utils.data import DataLoader, DistributedSampler, Dataset

try:
    from torch.amp import GradScaler
    def _autocast(enabled=True): return torch.amp.autocast("cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import GradScaler
    def _autocast(enabled=True): return torch.cuda.amp.autocast(enabled=enabled)

try:
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
except Exception:
    DDP = None

try:
    import mlflow
except Exception:
    mlflow = None

from model import (PRISM, NMRDataset, SMILESTokenizer, collate_fn,
                          evaluate_predictions, compute_tf_metrics, load_vocab)
from rdkit import RDLogger, rdBase
import warnings

# 1. Disable RDKit internal C++ logging (Kekulize and SMILES Parse errors)
RDLogger.DisableLog('rdApp.*') 

# 2. Disable Python Deprecation Warnings (MorganGenerator messages)
warnings.filterwarnings("ignore", category=DeprecationWarning)                          
                          

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("PRISM")


# ═══════════════════════ DDP HELPERS ═══════════════════════

def is_main():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        r = dist.get_rank()
        torch.cuda.set_device(r)
        return r
    return 0

def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def banner(msg):
    if is_main():
        logger.info(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ═══════════════════════ SCHEDULER ═══════════════════════

class CosineWarmup:
    def __init__(self, opt, warmup, total):
        self.opt, self.warmup, self.total = opt, warmup, total
        self.step_num = 0
        self.base_lrs = [g["lr"] for g in opt.param_groups]

    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup:
            s = self.step_num / max(self.warmup, 1)
        else:
            p = (self.step_num - self.warmup) / max(self.total - self.warmup, 1)
            s = 0.5 * (1 + math.cos(math.pi * p))
        s = max(s, 1e-2)
        for g, b in zip(self.opt.param_groups, self.base_lrs):
            g["lr"] = b * s

    def get_lr(self):
        return self.opt.param_groups[0]["lr"]


# ═══════════════════════ MODALITY-AWARE DATASET WRAPPER ═══════════════════════

class ModalityAwareDataset(Dataset):
    """
    Wraps NMRDataset to support:
      1. Samples that naturally have only H1 or C13 NMR
      2. Random modality dropout during training (data augmentation)
      3. Explicit-H SMILES target conversion
    """
    def __init__(self, base_ds, allow_single_modality=False,
                 modality_dropout_prob=0.0, use_explicit_h=False,
                 tokenizer=None, max_smiles_len=550):
        self.base = base_ds
        self.allow_single = allow_single_modality
        self.mod_drop = modality_dropout_prob  # prob of dropping one modality
        self.use_explicit_h = use_explicit_h
        self.tokenizer = tokenizer
        self.max_smiles_len = max_smiles_len
        # Pre-compute explicit-H SMILES if needed
        self._explicit_h_cache = {}
        if use_explicit_h and tokenizer is not None:
            self._build_explicit_h_cache()

    def _build_explicit_h_cache(self):
        """Convert all SMILES to explicit-H version (cached)."""
        try:
            from rdkit import Chem
            has_rdkit = True
        except ImportError:
            has_rdkit = False
            logger.warning("RDKit not available; explicit-H disabled.")
            self.use_explicit_h = False
            return

        logger.info("Building explicit-H SMILES cache...")
        data = self.base.data
        n = len(data["smiles_str"])
        for i in range(n):
            smi = data["smiles_str"][i]
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    mol_h = Chem.AddHs(mol)
                    smi_h = Chem.MolToSmiles(mol_h, allHsExplicit=True)
                    self._explicit_h_cache[i] = smi_h
            except Exception:
                pass  # Keep original if conversion fails
        logger.info(f"  Cached {len(self._explicit_h_cache)}/{n} explicit-H SMILES")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]

        # ── Modality dropout (training augmentation) ──
        if self.mod_drop > 0 and self.base.augment:
            if random.random() < self.mod_drop:
                h_has = item["h_mask"].any().item()
                c_has = item["c_mask"].any().item()
                if h_has and c_has:
                    # Randomly drop one modality
                    if random.random() < 0.5:
                        item["h_mask"] = torch.zeros_like(item["h_mask"])
                        item["h_ppm"] = torch.zeros_like(item["h_ppm"])
                        item["h_split"] = torch.zeros_like(item["h_split"])
                        item["h_integ"] = torch.zeros_like(item["h_integ"])
                        item["h_j"] = torch.zeros_like(item["h_j"])
                        item["h_region"] = torch.zeros_like(item["h_region"])
                    else:
                        item["c_mask"] = torch.zeros_like(item["c_mask"])
                        item["c_ppm"] = torch.zeros_like(item["c_ppm"])
                        item["c_split"] = torch.zeros_like(item["c_split"])
                        item["c_j"] = torch.zeros_like(item["c_j"])
                        item["c_region"] = torch.zeros_like(item["c_region"])

        # ── Filter samples with no NMR at all (unless single modality allowed) ──
        if not self.allow_single:
            h_has = item["h_mask"].any().item()
            c_has = item["c_mask"].any().item()
            if not h_has and not c_has:
                # Return with masks zeroed — collate will handle
                pass

        # ── Explicit-H SMILES target ──
        if self.use_explicit_h and idx in self._explicit_h_cache:
            smi_h = self._explicit_h_cache[idx]
            ids = self.tokenizer.encode(smi_h, max_len=self.max_smiles_len)
            padded = ids + [0] * (self.max_smiles_len - len(ids))
            item["smiles_ids"] = torch.tensor(padded[:self.max_smiles_len], dtype=torch.long)
            item["smiles_len"] = len(ids)
            item["smiles_str"] = smi_h

        return item


# ═══════════════════════ FIXED AUXILIARY LOSSES ═══════════════════════

def compute_aux_losses_v2(out, cfg):
    """
    Fixed auxiliary losses:
      • fp32 casting to prevent AMP overflow (fixes spec_recon oscillation)
      • SmoothL1 (Huber) loss instead of MSE for robustness
      • Loss clamping as safety net
    """
    mc = cfg.get("modules", {})
    device = out["logits"].device
    aux_loss = torch.tensor(0.0, device=device)
    info = {}

    if mc.get("use_auxiliary_losses", False):
        if "h_ppm_pred" in out:
            # ── CRITICAL FIX: Cast to fp32 before loss computation ──
            h_pred = out["h_ppm_pred"].float()
            c_pred = out["c_ppm_pred"].float()
            h_gt = out["h_ppm_gt"].float()
            c_gt = out["c_ppm_gt"].float()
            hm = out["h_mask"].float()
            cm = out["c_mask"].float()

            # Use SmoothL1 (Huber) loss — more robust than MSE
            h_diff = (h_pred - h_gt) * hm
            c_diff = (c_pred - c_gt) * cm
            h_r = F.smooth_l1_loss(h_diff, torch.zeros_like(h_diff),
                                    reduction="sum") / hm.sum().clamp(min=1)
            c_r = F.smooth_l1_loss(c_diff, torch.zeros_like(c_diff),
                                    reduction="sum") / cm.sum().clamp(min=1)

            sw = mc.get("aux_spectral_weight", 0.1)
            sl = sw * (h_r + c_r)

            # ── Safety clamp ──
            sl = sl.clamp(max=2.0)

            aux_loss = aux_loss + sl
            info["spec_recon"] = sl.item()

        if "formula_pred" in out and "formula_gt" in out:
            fp = out["formula_pred"].float()
            fg = out["formula_gt"].float()
            fw = mc.get("aux_formula_weight", 0.05)
            fl = fw * F.mse_loss(fp, fg)
            fl = fl.clamp(max=1.0)
            aux_loss = aux_loss + fl
            info["formula"] = fl.item()

    if "peak_attn_entropy" in out:
        ew = mc.get("aux_peak_entropy_weight", 0.01)
        el = ew * out["peak_attn_entropy"].float()
        el = el.clamp(max=0.5)
        aux_loss = aux_loss + el
        info["peak_ent"] = el.item()

    return aux_loss, info


# ═══════════════════════ TOP-K CHECKPOINT MANAGER ═══════════════════════

class TopKCheckpointManager:
    """Keeps top-K checkpoints ranked by a metric (for ensemble)."""
    def __init__(self, ckpt_dir, k=5, metric_name="exact_match"):
        self.ckpt_dir = ckpt_dir
        self.k = k
        self.metric_name = metric_name
        self.entries = []  # list of (metric_value, epoch, path)
        self.manifest_path = os.path.join(ckpt_dir, "top_k_manifest.json")

    def update(self, metric_val, epoch, state_dict_fn):
        """
        Check if this epoch qualifies for top-K.
        state_dict_fn: callable that returns the checkpoint dict (lazy).
        """
        path = os.path.join(self.ckpt_dir, f"top_k_ep{epoch:03d}.pt")

        if len(self.entries) < self.k:
            ck = state_dict_fn()
            torch.save(ck, path)
            self.entries.append((metric_val, epoch, path))
            self.entries.sort(key=lambda x: -x[0])
            self._save_manifest()
            return True

        # Check if better than worst in top-K
        worst_val = self.entries[-1][0]
        if metric_val > worst_val:
            # Remove worst checkpoint file
            worst_path = self.entries[-1][2]
            if os.path.exists(worst_path):
                os.remove(worst_path)
            self.entries.pop()

            ck = state_dict_fn()
            torch.save(ck, path)
            self.entries.append((metric_val, epoch, path))
            self.entries.sort(key=lambda x: -x[0])
            self._save_manifest()
            return True
        return False

    def _save_manifest(self):
        manifest = [{"metric": v, "epoch": e, "path": p}
                    for v, e, p in self.entries]
        with open(self.manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def get_paths(self):
        return [p for _, _, p in self.entries]


# ═══════════════════════ EPOCH METRICS CSV LOGGER ═══════════════════════

class EpochMetricsLogger:
    """Logs per-epoch metrics to CSV for easy plotting."""
    def __init__(self, path):
        self.path = path
        self.rows = []
        self.fieldnames = None

    def log(self, epoch, metrics_dict):
        row = {"epoch": epoch}
        row.update(metrics_dict)
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        self.rows.append(row)
        # Write incrementally
        with open(self.path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames)
            w.writeheader()
            w.writerows(self.rows)


# ═══════════════════════ VALIDATE ═══════════════════════

@torch.no_grad()
def validate(model, loader, device, tokenizer, epoch, mode="fast",
             n_gen=64, beam=5, max_len=128, cfg=None):
    raw = model.module if isinstance(model, DDP) else model
    raw.eval()
    ce_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    ls, nb = 0.0, 0
    tf = {"token_acc": 0, "top5_acc": 0, "seq_acc": 0, "ppl": 0}
    t0 = time.time()

    for batch in loader:
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
        out = raw(bd)
        B, T, V = out["logits"].shape
        ce = ce_fn(out["logits"].reshape(B * T, V), out["labels"].reshape(B * T))
        if not torch.isnan(ce):
            ls += ce.item()
            nb += 1
        m = compute_tf_metrics(out["logits"], out["labels"], tokenizer.pad_id)
        for k in tf:
            tf[k] += m[k]
    for k in tf:
        tf[k] /= max(nb, 1)
    avg_loss = ls / max(nb, 1)
    p1 = time.time() - t0

    t1 = time.time()
    preds, gts, gc = [], [], 0
    mx = n_gen if mode == "fast" else 99999
    bs = 1 if mode == "fast" else beam
    for batch in loader:
        if gc >= mx:
            break
        bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}
        preds.extend(raw.generate(bd, max_len=max_len, beam_size=bs))
        gts.extend(batch["smiles_str"])
        gc += len(batch["smiles_str"])
    preds, gts = preds[:mx], gts[:mx]
    gm = evaluate_predictions(preds, gts)
    p2 = time.time() - t1

    if is_main():
        logger.info(f"  ┌──────────────────────────────────────────────┐")
        logger.info(f"  │  {'VAL' if mode == 'fast' else 'TEST':4s}  Epoch {str(epoch):<28s}│")
        logger.info(f"  ├──────────────────────────────────────────────┤")
        logger.info(f"  │  CE Loss      : {avg_loss:>10.4f}                 │")
        logger.info(f"  │  Token Acc    : {tf['token_acc'] * 100:>9.2f}%                 │")
        logger.info(f"  │  Top-5 Acc    : {tf['top5_acc'] * 100:>9.2f}%                 │")
        logger.info(f"  │  Perplexity   : {tf['ppl']:>10.2f}                 │")
        logger.info(f"  │  Exact Match  : {gm['exact_match'] * 100:>9.2f}%  ({len(preds):>5d} dec)  │")
        logger.info(f"  │  Tanimoto     : {gm['avg_tanimoto']:>10.4f}                 │")
        logger.info(f"  │  Validity     : {gm['validity'] * 100:>9.2f}%                 │")
        logger.info(f"  │  Time (fwd/dec): {p1:.1f}s / {p2:.1f}s              │")
        logger.info(f"  └──────────────────────────────────────────────┘")
        for i in range(min(5, len(preds))):
            s = "✓" if preds[i] == gts[i] else "✗"
            logger.info(f"    {s} GT:   {gts[i][:70]}")
            logger.info(f"      Pred: {preds[i][:70]}")

    raw.train()
    return {
        "loss": avg_loss,
        "token_acc": tf["token_acc"],
        "top5_acc": tf["top5_acc"],
        "seq_acc": tf["seq_acc"],
        "ppl": tf["ppl"],
        "exact_match": gm["exact_match"],
        "avg_tanimoto": gm["avg_tanimoto"],
        "validity": gm["validity"],
    }


# ═══════════════════════ CLI FLAGS ═══════════════════════

MODULE_FLAGS = [
    "use_molecular_formula_encoder",
    "use_fcspe",
    "use_gsfu",
    "use_sarab",
    "use_bicma",
    "use_peak_atom_head",
    "use_auxiliary_losses",
    "use_chemical_augmentation",
    "use_chemical_shift_regions",
]


def apply_flags(cfg, args):
    if "modules" not in cfg:
        cfg["modules"] = {}
    mc = cfg["modules"]
    for f in MODULE_FLAGS:
        if getattr(args, f, False):
            mc[f] = True
    return cfg


# ═══════════════════════ MAIN ═══════════════════════

def main():
    pa = argparse.ArgumentParser(
        description="PRISM v3.1 Training with enhanced features")

    # ── Core arguments ──
    pa.add_argument("--config", default="config.yaml")
    pa.add_argument("--eval", action="store_true")
    pa.add_argument("--resume", type=str, default=None)
    pa.add_argument("--best_metric", type=str, default="token_acc",
                    choices=["token_acc", "exact_match", "avg_tanimoto", "validity"],
                    help="Metric for best checkpoint selection")
    pa.add_argument("--port", type=int, default=None,
                    help="Override rendezvous port (kills any process holding it)")

    # ── 9 module flags (same as v3.0) ──
    for f in MODULE_FLAGS:
        pa.add_argument(f"--{f}", action="store_true")

    # ── NEW: Enhanced features ──
    pa.add_argument("--allow_single_modality", action="store_true",
                    help="Allow samples with only H1 or only C13 NMR in the data")
    pa.add_argument("--modality_dropout_prob", type=float, default=0.0,
                    help="Prob of dropping one modality during TRAINING (0.0-0.3). "
                         "Teaches model to handle missing spectra at test time.")
    pa.add_argument("--use_explicit_h", action="store_true",
                    help="Use explicit-hydrogen SMILES as prediction target")
    pa.add_argument("--top_k_checkpoints", type=int, default=5,
                    help="Keep top-K checkpoints for ensemble (default: 5)")
    pa.add_argument("--train_spectra", type=str, default="both",
                    choices=["both", "h_only", "c_only"],
                    help="Which NMR spectra to use during TRAINING. "
                         "'both'=H1+C13 (default), 'h_only'=H1 only, 'c_only'=C13 only")
    pa.add_argument("--eval_spectra", type=str, default=None,
                    choices=["both", "h_only", "c_only"],
                    help="Which NMR spectra to use during VAL/TEST. "
                         "Default: follows --train_spectra. "
                         "Set differently for out-of-bag testing, e.g. "
                         "--train_spectra both --eval_spectra h_only")

    args = pa.parse_args()

    # ── Resolve eval_spectra: default follows train_spectra ──
    if args.eval_spectra is None:
        args.eval_spectra = args.train_spectra

    # ── Kill zombie processes on the port ──
    if args.port and os.environ.get("LOCAL_RANK", "0") == "0":
        import subprocess, signal
        try:
            result = subprocess.run(["lsof", "-ti", f":{args.port}"],
                                     capture_output=True, text=True)
            pids = result.stdout.strip().split()
            my_pid = str(os.getpid())
            for pid in pids:
                if pid and pid != my_pid:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                        logger.info(f"  Killed zombie PID {pid} on port {args.port}")
                    except Exception:
                        pass
            time.sleep(1)
        except Exception:
            pass

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_flags(cfg, args)
    best_metric_name = args.best_metric

    rank = setup_ddp()
    seed_all(cfg.get("training", {}).get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dc, tc = cfg.get("data", {}), cfg.get("training", {})

    # ═══════════════════════ DATA ═══════════════════════
    banner("LOADING DATA")
    mc = cfg.get("modules", {})
    active = [k for k, v in mc.items() if isinstance(v, bool) and v]
    if is_main():
        logger.info(f"  Active modules: {', '.join(active) if active else 'none (base mode)'}")
        logger.info(f"  Train spectra : {args.train_spectra}")
        logger.info(f"  Eval spectra  : {args.eval_spectra}")
        if args.train_spectra != args.eval_spectra:
            logger.info(f"  ⚠ OOB MODE: training on '{args.train_spectra}', "
                         f"evaluating on '{args.eval_spectra}'")
        logger.info(f"  Single modality: {args.allow_single_modality}")
        logger.info(f"  Modality dropout: {args.modality_dropout_prob}")
        logger.info(f"  Explicit-H target: {args.use_explicit_h}")
        logger.info(f"  Top-K checkpoints: {args.top_k_checkpoints}")

    pd = dc.get("processed_dir", "processed")
    if not os.path.isdir(os.path.join(pd, "vocab")):
        logger.error(f"Run: python preprocess.py --config {args.config}")
        return
    load_vocab(pd)
    tokenizer = SMILESTokenizer.from_processed(pd)

    td = os.path.join(pd, "tensors")
    for sp in ["train.pt", "val.pt", "test.pt"]:
        if not os.path.isfile(os.path.join(td, sp)):
            logger.error(f"Missing {sp}. Run: python preprocess.py --config {args.config}")
            return

    max_smi_len = dc.get("max_smiles_len", 550)

    # Base datasets
    train_ds_base = NMRDataset(os.path.join(td, "train.pt"), tokenizer, cfg, augment=True)
    val_ds_base = NMRDataset(os.path.join(td, "val.pt"), tokenizer, cfg, augment=False)
    test_ds_base = NMRDataset(os.path.join(td, "test.pt"), tokenizer, cfg, augment=False)

    # Wrap with modality-aware dataset
    train_ds = ModalityAwareDataset(
        train_ds_base,
        allow_single_modality=args.allow_single_modality,
        modality_dropout_prob=args.modality_dropout_prob,
        use_explicit_h=args.use_explicit_h,
        tokenizer=tokenizer,
        max_smiles_len=max_smi_len,
    )
    val_ds = ModalityAwareDataset(
        val_ds_base,
        allow_single_modality=args.allow_single_modality,
        modality_dropout_prob=0.0,  # No dropout during validation
        use_explicit_h=args.use_explicit_h,
        tokenizer=tokenizer,
        max_smiles_len=max_smi_len,
    )
    test_ds = ModalityAwareDataset(
        test_ds_base,
        allow_single_modality=args.allow_single_modality,
        modality_dropout_prob=0.0,
        use_explicit_h=args.use_explicit_h,
        tokenizer=tokenizer,
        max_smiles_len=max_smi_len,
    )

    # ── Spectra masking collate functions (separate for train vs eval) ──
    def _make_spectra_collate(spectra_mode):
        """Returns a collate_fn that masks spectra according to the mode."""
        def _collate(batch):
            out = collate_fn(batch)
            if spectra_mode == "h_only":
                # Zero out all C13 data
                out["c_mask"] = torch.zeros_like(out["c_mask"])
                out["c_ppm"] = torch.zeros_like(out["c_ppm"])
                out["c_split"] = torch.zeros_like(out["c_split"])
                out["c_j"] = torch.zeros_like(out["c_j"])
                if "c_region" in out:
                    out["c_region"] = torch.zeros_like(out["c_region"])
            elif spectra_mode == "c_only":
                # Zero out all H1 data
                out["h_mask"] = torch.zeros_like(out["h_mask"])
                out["h_ppm"] = torch.zeros_like(out["h_ppm"])
                out["h_split"] = torch.zeros_like(out["h_split"])
                out["h_integ"] = torch.zeros_like(out["h_integ"])
                out["h_j"] = torch.zeros_like(out["h_j"])
                if "h_region" in out:
                    out["h_region"] = torch.zeros_like(out["h_region"])
            return out
        return _collate

    train_collate = _make_spectra_collate(args.train_spectra)
    eval_collate = _make_spectra_collate(args.eval_spectra)

    bs = tc.get("batch_size", 64)
    nw = tc.get("num_workers", 4)
    use_ddp = dist.is_available() and dist.is_initialized()
    tr_samp = DistributedSampler(train_ds) if use_ddp else None
    tr_ld = DataLoader(train_ds, batch_size=bs, shuffle=(tr_samp is None),
                       sampler=tr_samp, collate_fn=train_collate,
                       num_workers=nw, pin_memory=True, drop_last=True)
    vl_ld = DataLoader(val_ds, batch_size=bs * 2, shuffle=False,
                       collate_fn=eval_collate,
                       num_workers=nw, pin_memory=True)
    te_ld = DataLoader(test_ds, batch_size=bs * 2, shuffle=False,
                       collate_fn=eval_collate,
                       num_workers=nw, pin_memory=True)
    logger.info(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} Batch={bs}")

    # ═══════════════════════ MODEL ═══════════════════════
    banner("MODEL")
    model = PRISM(tokenizer, cfg).to(device)
    p = model.count_params()
    logger.info(f"Params: {p['total']:,} (enc={p['encoder']:,} dec={p['decoder']:,} aux={p['auxiliary']:,})")

    if use_ddp:
        has_opt = any(isinstance(v, bool) and v for v in mc.values())
        model = DDP(model, device_ids=[rank], find_unused_parameters=has_opt)

    # ═══════════════════════ OPTIMIZER ═══════════════════════
    lr = tc.get("learning_rate", 3e-4)
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr,
        weight_decay=tc.get("weight_decay", 0.01),
        betas=(0.9, 0.98), eps=1e-9)

    epochs = tc.get("epochs", 200)
    warmup = tc.get("warmup_steps", 1000)
    sched = CosineWarmup(opt, warmup, len(tr_ld) * epochs)
    use_amp = torch.cuda.is_available() and tc.get("use_amp", True)
    scaler = GradScaler() if use_amp else None
    ce_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_id,
        label_smoothing=tc.get("label_smoothing", 0.1))
    ga = tc.get("gradient_accumulation_steps", 1)
    mg = tc.get("max_grad_norm", 1.0)
    ve = tc.get("val_every_n_epochs", 5)
    ng = tc.get("val_n_generate", 256)
    ckd = tc.get("checkpoint_dir", "checkpoints/prism")
    os.makedirs(ckd, exist_ok=True)

    # ── Top-K checkpoint manager ──
    topk_mgr = TopKCheckpointManager(ckd, k=args.top_k_checkpoints,
                                      metric_name=best_metric_name)

    # ── Epoch metrics logger ──
    metrics_csv = os.path.join(ckd, "epoch_metrics.csv")
    metrics_logger = EpochMetricsLogger(metrics_csv)

    start_ep, best, gs = 0, 0.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        (model.module if isinstance(model, DDP) else model).load_state_dict(ck["model"])
        if "optimizer" in ck and not args.eval:
            opt.load_state_dict(ck["optimizer"])
        start_ep = ck.get("epoch", 0) + 1
        best = ck.get("best_metric", 0)
        gs = ck.get("global_step", 0)
        logger.info(f"Resumed epoch {start_ep}, best={best:.4f}")

    if args.eval:
        banner("EVAL")
        validate(model, te_ld, device, tokenizer, "FINAL", "full",
                 len(test_ds), tc.get("val_beam_size", 5), cfg=cfg)
        cleanup_ddp()
        return

    # ── MLflow (rank 0 only) ──
    _use_mlflow = False
    if mlflow and is_main():
        try:
            mlc = cfg.get("mlflow", {})
            mlflow.set_tracking_uri(mlc.get("tracking_uri", "mlruns"))
            mlflow.set_experiment(mlc.get("experiment_name", "PRISM"))
            mlflow.start_run(run_name=mlc.get("run_name", "train"))
            mlflow.log_params({
                "params": p["total"], "train": len(train_ds),
                "vocab": tokenizer.vocab_size, "epochs": epochs,
                "d_model": cfg["model"]["d_model"], "lr": lr, "bs": bs,
                "best_metric": best_metric_name,
                "train_spectra": args.train_spectra,
                "eval_spectra": args.eval_spectra,
                "explicit_h": args.use_explicit_h,
                "single_modality": args.allow_single_modality,
                "modality_dropout": args.modality_dropout_prob,
            })
            for mk, mv in mc.items():
                if isinstance(mv, bool):
                    mlflow.log_param(f"mod/{mk}", mv)
            _use_mlflow = True
        except Exception as e:
            logger.warning(f"  MLflow init failed: {e}. Continuing without MLflow.")

    # ═══════════════════════ TRAIN ═══════════════════════
    banner("TRAINING")
    logger.info(f"  Epochs={epochs} LR={lr} Warmup={warmup} GA={ga} AMP={use_amp}")
    nan_count = 0

    for ep in range(start_ep, epochs):
        if tr_samp:
            tr_samp.set_epoch(ep)
        model.train()
        ls, ns = 0.0, 0
        aux_tot = {}
        opt.zero_grad()
        t0 = time.time()

        for st, batch in enumerate(tr_ld):
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}

            with _autocast(scaler is not None):
                out = model(bd)
                B, T, V = out["logits"].shape
                ce_loss = ce_fn(out["logits"].reshape(B * T, V),
                                out["labels"].reshape(B * T))

            # ── FIXED: Compute aux losses OUTSIDE autocast for fp32 stability ──
            aux_loss, aux_info = compute_aux_losses_v2(out, cfg)
            loss = (ce_loss + aux_loss) / ga

            # NaN detection — skip batch
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count <= 5 and is_main():
                    logger.warning(
                        f"  NaN/Inf loss at epoch {ep + 1} step {st}, "
                        f"skipping batch (count={nan_count})")
                opt.zero_grad()
                continue

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (st + 1) % ga == 0:
                if scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), mg)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
                sched.step()
                gs += 1

            ls += ce_loss.item()
            ns += 1
            for ak, av in aux_info.items():
                aux_tot[ak] = aux_tot.get(ak, 0) + av

        el = ls / max(ns, 1)
        et = time.time() - t0
        aux_avg = {k: v / max(ns, 1) for k, v in aux_tot.items()}
        aux_str = ""
        if aux_avg:
            aux_str = " | " + " ".join(f"{k}={v:.4f}" for k, v in aux_avg.items())

        if is_main():
            logger.info(
                f"Epoch {ep + 1:>3d}/{epochs} | loss={el:.4f} | "
                f"lr={sched.get_lr():.2e} | time={et:.1f}s{aux_str}")

            # Log training metrics
            train_metrics = {"train_loss": el, "lr": sched.get_lr()}
            train_metrics.update({f"train_{k}": v for k, v in aux_avg.items()})

            if _use_mlflow:
                try:
                    mlflow.log_metric("train/loss", el, step=ep)
                    for ak, av in aux_avg.items():
                        mlflow.log_metric(f"train/{ak}", av, step=ep)
                except Exception:
                    pass

        # ── Validation ──
        if (ep + 1) % ve == 0 or ep == epochs - 1:
            met = validate(model, vl_ld, device, tokenizer, ep + 1,
                           "fast", ng, cfg=cfg)

            if is_main():
                cur = met[best_metric_name]

                # Log all metrics to CSV
                epoch_row = {"train_loss": el, "lr": sched.get_lr()}
                epoch_row.update({f"train_{k}": v for k, v in aux_avg.items()})
                epoch_row.update({f"val_{k}": v for k, v in met.items()})
                metrics_logger.log(ep + 1, epoch_row)

                if _use_mlflow:
                    try:
                        for k, v in met.items():
                            mlflow.log_metric(f"val/{k}", v, step=ep)
                    except Exception:
                        pass

                rm = (model.module if isinstance(model, DDP) else model)

                def _make_ckpt():
                    return {
                        "model": rm.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": ep, "global_step": gs,
                        "best_metric": best,
                        "best_metric_name": best_metric_name,
                        "tokenizer": tokenizer.token2id,
                        "cfg": cfg,
                        "processed_dir": pd,
                        "metrics": met,
                        "train_spectra": args.train_spectra,
                        "eval_spectra": args.eval_spectra,
                    }

                # Save latest
                torch.save(_make_ckpt(), os.path.join(ckd, "latest.pt"))

                # Save best
                if cur > best:
                    best = cur
                    torch.save(_make_ckpt(), os.path.join(ckd, "best.pt"))
                    logger.info(f"  ★ New best! {best_metric_name}={best:.4f}")

                # Top-K checkpoint for ensemble
                topk_mgr.update(cur, ep + 1, _make_ckpt)

        barrier()

    # ═══════════════════════ FINAL TEST ═══════════════════════
    if is_main():
        banner("FINAL TEST")
        bp = os.path.join(ckd, "best.pt")
        if os.path.exists(bp):
            (model.module if isinstance(model, DDP) else model).load_state_dict(
                torch.load(bp, map_location=device, weights_only=False)["model"])

        met = validate(model, te_ld, device, tokenizer, "FINAL", "full",
                       len(test_ds), tc.get("val_beam_size", 5), cfg=cfg)

        if _use_mlflow:
            try:
                for k, v in met.items():
                    mlflow.log_metric(f"test/{k}", v)
                mlflow.end_run()
            except Exception:
                pass

        banner("DONE")
        logger.info(f"  Best val {best_metric_name}: {best:.4f}")
        logger.info(f"  Test exact_match: {met['exact_match'] * 100:.2f}%")
        logger.info(f"  Top-K checkpoints: {topk_mgr.get_paths()}")
        if nan_count > 0:
            logger.info(f"  NaN batches skipped: {nan_count}")

    cleanup_ddp()


if __name__ == "__main__":
    main()