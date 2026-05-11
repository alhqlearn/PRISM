
import argparse, json, logging, math, os, random, time, warnings, contextlib
import numpy as np, torch, torch.nn as nn, yaml
from torch.utils.data import DataLoader

from nmr_model import (
    NMRPredictor, NMRDataset, AugmentedNMRDataset, SMILESTokenizer,
    collate_fn, load_vocab, hungarian_match,
    compute_loss, compute_loss_joint, evaluate,
)

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("NMR")

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

class CosineWarmup:
    def __init__(self, opt, warmup, total):
        self.opt, self.warmup, self.total = opt, warmup, total
        self.step_num = 0; self.base_lrs = [g["lr"] for g in opt.param_groups]
    def step(self):
        self.step_num += 1
        if self.step_num <= self.warmup: s = self.step_num / max(self.warmup, 1)
        else:
            p = (self.step_num - self.warmup) / max(self.total - self.warmup, 1)
            s = 0.5 * (1 + math.cos(math.pi * p))
        s = max(s, 1e-2)
        for g, b in zip(self.opt.param_groups, self.base_lrs): g["lr"] = b * s
    def get_lr(self): return self.opt.param_groups[0]["lr"]
    def state_dict(self): return {"step_num": self.step_num, "base_lrs": self.base_lrs}
    def load_state_dict(self, sd):
        self.step_num = sd["step_num"]; self.base_lrs = sd["base_lrs"]
        if self.step_num <= self.warmup: s = self.step_num / max(self.warmup, 1)
        else:
            p = (self.step_num - self.warmup) / max(self.total - self.warmup, 1)
            s = 0.5 * (1 + math.cos(math.pi * p))
        s = max(s, 1e-2)
        for g, b in zip(self.opt.param_groups, self.base_lrs): g["lr"] = b * s

def validate(model, loader, device, norm_stats, epoch, cfg):
    raw = model.module if hasattr(model, 'module') else model; raw.eval()
    mode = cfg.get("data", {}).get("mode", "1h"); is_joint = (mode == "joint")
    all_metrics, nb = {}, 0
    with torch.no_grad():
        for batch in loader:
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = raw.forward(bd)
            if is_joint:
                m1 = evaluate(out, bd, norm_stats, "1h", pred_key="pred_1h",
                    ppm_raw_key="ppm_1h_raw", mask_key="mask_1h", split_key="split_1h",
                    j_key="j_1h", ppm_stat="ppm_1h", j_stat="j_1h")
                m2 = evaluate(out, bd, norm_stats, "13c", pred_key="pred_13c",
                    ppm_raw_key="ppm_13c_raw", mask_key="mask_13c", split_key="split_13c",
                    j_key="j_13c", ppm_stat="ppm_13c", j_stat="j_13c")
                met = {f"1h_{k}": v for k, v in m1.items()}
                met.update({f"13c_{k}": v for k, v in m2.items()})
                met["f1"] = (m1["f1"] + m2["f1"]) / 2.0
            else:
                ps = f"ppm_{mode}" if f"ppm_{mode}" in norm_stats else "ppm"
                jst = f"j_{mode}" if f"j_{mode}" in norm_stats else "j"
                met = evaluate(out, bd, norm_stats, mode=mode, ppm_stat=ps, j_stat=jst)
            for k, v in met.items(): all_metrics[k] = all_metrics.get(k, 0) + v
            nb += 1
    for k in all_metrics: all_metrics[k] /= max(nb, 1)

    label = mode.upper()
    logger.info(f"  ┌──────────────────────────────────────────────────────┐")
    logger.info(f"  │  VAL Epoch {str(epoch):<5s} ({label})                          │")
    logger.info(f"  ├──────────────────────────────────────────────────────┤")
    for k in ["ppm_mae","ppm_r2","j_mae","j_r2","count_mae","split_acc","precision","recall","f1",
              "1h_f1","13c_f1","1h_ppm_mae","13c_ppm_mae"]:
        if k in all_metrics:
            v = all_metrics[k]
            vs = f"{v*100:.2f}%" if any(x in k for x in ["acc","f1","prec","rec"]) else f"{v:.4f}"
            logger.info(f"  │  {k:14s}: {vs:>12s}                          │")
    logger.info(f"  └──────────────────────────────────────────────────────┘")
    raw.train(); return all_metrics

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", required=True)
    pa.add_argument("--precision", default="bf16", choices=["fp16","fp32","bf16"])
    pa.add_argument("--best_metric", default="f1")
    pa.add_argument("--smiles_augment_n", type=int, default=1)
    pa.add_argument("--resume", type=str, default=None)
    args = pa.parse_args()

    with open(args.config) as f: cfg = yaml.safe_load(f)
    dc, tc = cfg.get("data", {}), cfg.get("training", {})
    mode = dc.get("mode", "1h"); is_joint = (mode == "joint")
    seed_all(tc.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda": logger.info(f"  Device: {torch.cuda.get_device_name(device)}")

    pd = dc.get("processed_dir", "processed"); load_vocab(pd)
    tokenizer = SMILESTokenizer.from_processed(pd)
    with open(os.path.join(pd, "vocab", "norm_stats.json")) as f: norm_stats = json.load(f)

    td = os.path.join(pd, "tensors")
    train_ds_base = NMRDataset(os.path.join(td, "train.pt"), tokenizer, cfg, augment=True)
    val_ds = NMRDataset(os.path.join(td, "val.pt"), tokenizer, cfg, augment=False)
    test_ds = NMRDataset(os.path.join(td, "test.pt"), tokenizer, cfg, augment=False)
    train_ds = AugmentedNMRDataset(train_ds_base, tokenizer, dc.get("max_smiles_len", 600), args.smiles_augment_n)

    bs = tc.get("batch_size", 64); nw = tc.get("num_workers", 4)
    dl_kw = {"persistent_workers": True, "prefetch_factor": 4} if nw > 0 else {}
    tr_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, collate_fn=collate_fn,
                       num_workers=nw, pin_memory=True, drop_last=True, **dl_kw)
    vl_ld = DataLoader(val_ds, batch_size=bs*2, shuffle=False, collate_fn=collate_fn,
                       num_workers=nw, pin_memory=True, **dl_kw)
    te_ld = DataLoader(test_ds, batch_size=bs*2, shuffle=False, collate_fn=collate_fn,
                       num_workers=nw, pin_memory=True, **dl_kw)

    logger.info(f"Mode={mode}  Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} Batch={bs}")

    model = NMRPredictor(tokenizer, cfg).to(device)
    logger.info(f"Params: {model.count_params():,}")
    if hasattr(torch, "compile") and device.type == "cuda":
        try: model = torch.compile(model, mode="default"); logger.info("  torch.compile() enabled")
        except Exception as e: logger.info(f"  torch.compile() skipped: {e}")

    lr = tc.get("learning_rate", 3e-4)
    opt = torch.optim.AdamW(model.parameters(), lr=lr,
        weight_decay=tc.get("weight_decay", 0.01), betas=(0.9, 0.98), eps=1e-8)
    epochs = tc.get("epochs", 50)
    sched = CosineWarmup(opt, tc.get("warmup_steps", 2000), len(tr_ld) * epochs)

    precision = args.precision if device.type == "cuda" else "fp32"
    use_amp = precision != "fp32"
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision, torch.float16)
    # bf16 doesn't need GradScaler; fp16 does
    scaler = None
    if precision == "fp16":
        from torch.amp import GradScaler; scaler = GradScaler()
    logger.info(f"  Precision: {precision}")

    def autocast():
        if use_amp: return torch.amp.autocast("cuda", enabled=True, dtype=amp_dtype)
        return contextlib.nullcontext()

    ga = tc.get("gradient_accumulation_steps", 1)
    mg = tc.get("max_grad_norm", 1.0)
    ve = tc.get("val_every_n_epochs", 5)
    ckd = tc.get("checkpoint_dir", f"checkpoints/nmr_{mode}")
    os.makedirs(ckd, exist_ok=True)

    lower_better = {"ppm_mae","count_mae","j_mae","1h_ppm_mae","13c_ppm_mae"}
    best_metric = args.best_metric
    maximize = best_metric not in lower_better
    best = -float("inf") if maximize else float("inf")

    # Auto-resume
    start_epoch = 0
    rp = args.resume or (os.path.join(ckd, "latest.pt") if os.path.exists(os.path.join(ckd, "latest.pt")) else None)
    if rp and os.path.exists(rp):
        logger.info(f"Resuming from {rp}...")
        ck = torch.load(rp, map_location=device, weights_only=False)
        rm = model.module if hasattr(model, 'module') else model
        rm.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}, strict=False)
        if "optimizer" in ck:
            try: opt.load_state_dict(ck["optimizer"])
            except: pass
        if "scheduler" in ck:
            try: sched.load_state_dict(ck["scheduler"])
            except: pass
        if "scaler" in ck and scaler: 
            try: scaler.load_state_dict(ck["scaler"])
            except: pass
        start_epoch = ck.get("epoch", -1) + 1
        best = ck.get("best_value", best)
        logger.info(f"  Epoch {start_epoch+1}/{epochs}, best {best_metric}={best:.4f}")
    else:
        logger.info("Starting fresh training")

    if start_epoch >= epochs: logger.info("Already done."); return

    logger.info(f"\n{'='*60}\n  TRAINING ({mode.upper()}) — epochs {start_epoch+1}→{epochs}\n{'='*60}")

    nan_total = 0
    for ep in range(start_epoch, epochs):
        model.train(); t0 = time.time()
        el, es, lc, nan_ep = 0.0, 0, {}, 0
        opt.zero_grad(set_to_none=True)

        for st, batch in enumerate(tr_ld):
            bd = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            with autocast():
                out = model(bd)
                if is_joint:
                    p1, p2 = out["pred_1h"], out["pred_13c"]
                    m1 = hungarian_match(p1["ppm"].detach(), p1["exist"].detach(), bd["ppm_1h"], bd["mask_1h"])
                    m2 = hungarian_match(p2["ppm"].detach(), p2["exist"].detach(), bd["ppm_13c"], bd["mask_13c"])
                    loss, ld, valid = compute_loss_joint(out, bd, m1, m2, tc)
                else:
                    p = out["pred"]
                    m = hungarian_match(p["ppm"].detach(), p["exist"].detach(), bd["ppm"], bd["mask"])
                    loss, ld, valid = compute_loss(p, bd, m, tc, mode=mode)
                loss = loss / ga

            if not valid.item():
                nan_ep += 1; nan_total += 1; opt.zero_grad(set_to_none=True)
                if nan_ep <= 3: logger.warning(f"  NaN loss ep {ep+1} step {st}")
                continue

            if scaler: scaler.scale(loss).backward()
            else: loss.backward()

            if (st + 1) % ga == 0:
                if scaler: scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), mg)
                if scaler: scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad(set_to_none=True); sched.step()

            el += ld.get("total", 0); es += 1
            for k, v in ld.items(): lc[k] = lc.get(k, 0) + v

        if es > 0:
            avg = el / es
            cs = " | ".join(f"{k}={v/es:.4f}" for k, v in lc.items() if k != "total")
        else: avg = float('nan'); cs = "ALL NaN"
        ns = f" | nan={nan_ep}" if nan_ep > 0 else ""
        logger.info(f"Epoch {ep+1:>3d}/{epochs} | loss={avg:.4f} | lr={sched.get_lr():.2e} | "
                     f"time={time.time()-t0:.1f}s | {cs}{ns}")
        if es == 0: logger.error("ALL NaN — stopping."); break

        # Save every epoch
        rm = model.module if hasattr(model, 'module') else model
        def _ck(met=None):
            d = {"model": rm.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "epoch": ep, "cfg": cfg,
                 "processed_dir": pd, "tokenizer": tokenizer.token2id,
                 "norm_stats": norm_stats, "mode": mode,
                 "best_value": best, "best_metric_name": best_metric}
            if scaler: d["scaler"] = scaler.state_dict()
            if met: d["metrics"] = met
            return d
        lp = os.path.join(ckd, "latest.pt"); tmp = lp + ".tmp"
        torch.save(_ck(), tmp); os.replace(tmp, lp)

        if (ep + 1) % ve == 0 or ep == epochs - 1:
            met = validate(model, vl_ld, device, norm_stats, ep + 1, cfg)
            cur = met.get(best_metric, 0)
            torch.save(_ck(met), lp)
            improved = (cur > best) if maximize else (cur < best)
            if improved:
                best = cur; bc = _ck(met); bc["best_value"] = best
                torch.save(bc, os.path.join(ckd, "best.pt"))
                logger.info(f"  ★ New best! {best_metric}={best:.4f}")

    logger.info(f"\n{'='*60}\n  FINAL TEST\n{'='*60}")
    bp = os.path.join(ckd, "best.pt")
    if os.path.exists(bp):
        rm = model.module if hasattr(model, 'module') else model
        rm.load_state_dict({k.replace("_orig_mod.", ""): v
            for k, v in torch.load(bp, map_location=device, weights_only=False)["model"].items()}, strict=False)
    validate(model, te_ld, device, norm_stats, "FINAL", cfg)

if __name__ == "__main__": main()
