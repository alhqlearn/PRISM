
import json, logging, math, os, re
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

logger = logging.getLogger("NMR")

SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|[BCNOPSFIcnos]|[=#/\\@+\-\.\(\)]|[0-9]|%[0-9]{2})")

SPLITTING_PATTERNS, SPLIT2ID, N_SPLITS = [], {}, 0
_VOCAB_LOADED = False

def load_vocab(processed_dir):
    global SPLITTING_PATTERNS, SPLIT2ID, N_SPLITS, _VOCAB_LOADED
    vd = os.path.join(processed_dir, "vocab")
    with open(os.path.join(vd, "splitting_patterns.json")) as f:
        d = json.load(f)
        SPLITTING_PATTERNS = d["patterns"]
        SPLIT2ID = {s: int(i) for s, i in d["pattern2id"].items()}
        N_SPLITS = len(SPLITTING_PATTERNS)
    logger.info(f"  Vocab: {N_SPLITS} splits")
    _VOCAB_LOADED = True

def ensure_vocab():
    if not _VOCAB_LOADED:
        raise RuntimeError("Call load_vocab(processed_dir) first")


# ═══════════════════════ TOKENIZER ═══════════════════════

class SMILESTokenizer:
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
    def __init__(self):
        self.token2id = {self.PAD:0, self.BOS:1, self.EOS:2, self.UNK:3}
        self.id2token = {0:self.PAD, 1:self.BOS, 2:self.EOS, 3:self.UNK}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
    @staticmethod
    def tokenize(smi): return SMILES_REGEX.findall(smi)
    @property
    def vocab_size(self): return len(self.token2id)
    def encode(self, smi, max_len=128):
        toks = self.tokenize(smi)[:max_len - 2]
        return [self.bos_id] + [self.token2id.get(t, self.unk_id) for t in toks] + [self.eos_id]
    def save(self, path):
        with open(path, "w") as f: json.dump({"token2id": self.token2id}, f)
    def load(self, path):
        with open(path) as f: d = json.load(f)
        self.token2id = {k: int(v) for k, v in d["token2id"].items()}
        self.id2token = {v: k for k, v in self.token2id.items()}
    @classmethod
    def from_processed(cls, pd):
        t = cls(); t.load(os.path.join(pd, "vocab", "smiles_tokenizer.json"))
        logger.info(f"Loaded tokenizer (vocab_size={t.vocab_size})"); return t


# ═══════════════════════ DATASET ═══════════════════════

class NMRDataset(Dataset):
    def __init__(self, tensor_path, tokenizer, cfg, augment=False):
        ensure_vocab()
        self.tokenizer = tokenizer; self.augment = augment
        self.data = torch.load(tensor_path, map_location="cpu", weights_only=False)
        self.n = self.data["smiles_ids"].size(0)
        self.mode = cfg.get("data", {}).get("mode", "1h")
        logger.info(f"Loaded {self.n} samples (mode={self.mode})")
    def __len__(self): return self.n
    def __getitem__(self, idx):
        d = self.data
        result = {"smiles_ids": d["smiles_ids"][idx],
                  "smiles_len": d["smiles_len"][idx].item(),
                  "smiles_str": d["smiles_str"][idx]}
        for k in ["ppm","ppm_raw","split","j","mask",
                   "ppm_1h","ppm_1h_raw","split_1h","j_1h","mask_1h",
                   "ppm_13c","ppm_13c_raw","split_13c","j_13c","mask_13c",
                   "integ","width"]:
            if k in d: result[k] = d[k][idx]
        return result

def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == "smiles_str": out[k] = [b[k] for b in batch]
        elif k == "smiles_len": out[k] = torch.tensor([b[k] for b in batch])
        else: out[k] = torch.stack([b[k] for b in batch])
    return out

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*'); HAS_RDKIT = True
except ImportError: HAS_RDKIT = False

def randomize_smiles(smi):
    if not HAS_RDKIT: return smi
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, doRandom=True, canonical=False) or smi if mol else smi
    except: return smi

class AugmentedNMRDataset(Dataset):
    def __init__(self, base_ds, tokenizer, max_smiles_len=600, n_aug=1):
        self.base = base_ds; self.tok = tokenizer
        self.max_smi = max_smiles_len; self.n_aug = max(1, n_aug)
    def __len__(self): return len(self.base) * self.n_aug
    def __getitem__(self, idx):
        real = idx % len(self.base); copy = idx // len(self.base)
        item = self.base[real]
        if copy > 0 and self.base.augment and HAS_RDKIT:
            rand_smi = randomize_smiles(item["smiles_str"])
            ids = self.tok.encode(rand_smi, max_len=self.max_smi)
            padded = ids + [0] * (self.max_smi - len(ids))
            item["smiles_ids"] = torch.tensor(padded[:self.max_smi], dtype=torch.long)
            item["smiles_len"] = len(ids)
        return item


# ═══════════════════════ MODEL ═══════════════════════

class SinPE(nn.Module):
    def __init__(self, d, max_len=1024, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])

class SMILESEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, n_layers, d_ff, dropout, max_len=600):
        super().__init__()
        self.d = d_model
        self.emb = nn.Embedding(vocab_size, d_model)
        self.pe = SinPE(d_model, max_len, dropout)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=d_ff, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
    def forward(self, smiles_ids, smiles_len):
        B, S = smiles_ids.shape; dev = smiles_ids.device
        x = self.pe(self.emb(smiles_ids) * math.sqrt(self.d))
        pad_mask = torch.arange(S, device=dev).unsqueeze(0) >= smiles_len.unsqueeze(1)
        noop = torch.zeros(S, S, device=dev, dtype=x.dtype)
        return self.enc(x, mask=noop, src_key_padding_mask=pad_mask), pad_mask

class PeakDecoder(nn.Module):
    def __init__(self, d_model, nhead, n_layers, d_ff, dropout, max_peaks):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(max_peaks, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead,
            dim_feedforward=d_ff, dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, n_layers)
    def forward(self, mem, mem_pad):
        B = mem.size(0); q = self.queries.unsqueeze(0).expand(B, -1, -1)
        Q = q.size(1); noop = torch.zeros(Q, Q, device=mem.device, dtype=q.dtype)
        return self.dec(q, mem, tgt_mask=noop, memory_key_padding_mask=mem_pad)

class PropertyHead(nn.Module):
    def __init__(self, d_model, n_splits, has_integration=True):
        super().__init__()
        self.has_integration = has_integration
        self.refine = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))
        self.ppm_head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.GELU(), nn.Linear(d_model//2, 1))
        self.split_head = nn.Linear(d_model, n_splits)
        self.j_head = nn.Sequential(nn.Linear(d_model, d_model//4), nn.GELU(), nn.Linear(d_model//4, 1))
        self.exist_head = nn.Sequential(nn.Linear(d_model, d_model//4), nn.GELU(), nn.Linear(d_model//4, 1))
        if has_integration:
            self.integ_head = nn.Sequential(nn.Linear(d_model, d_model//4), nn.GELU(), nn.Linear(d_model//4, 1))
            self.width_head = nn.Sequential(nn.Linear(d_model, d_model//4), nn.GELU(), nn.Linear(d_model//4, 1))
    def forward(self, x):
        h = self.refine(x)
        out = {"ppm": self.ppm_head(h).squeeze(-1), "split": self.split_head(h),
               "j": self.j_head(h).squeeze(-1), "exist": self.exist_head(h).squeeze(-1)}
        if self.has_integration:
            out["integ"] = self.integ_head(h).squeeze(-1)
            out["width"] = F.softplus(self.width_head(h).squeeze(-1))
        return out


class NMRPredictor(nn.Module):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        ensure_vocab()
        self.tokenizer = tokenizer; self.cfg = cfg
        mc = cfg.get("model", {}); dc = cfg.get("data", {})
        d = mc.get("d_model", 512); nh = mc.get("nhead", 8)
        enc_l = mc.get("encoder_layers", 8); dec_l = mc.get("decoder_layers", 8)
        ff = mc.get("d_ff", 1024); dr = mc.get("dropout", 0.1)
        max_smi = dc.get("max_smiles_len", 600)
        self.mode = dc.get("mode", "1h")

        self.encoder = SMILESEncoder(tokenizer.vocab_size, d, nh, enc_l, ff, dr, max_smi)

        if self.mode == "joint":
            mp1h = dc.get("max_peaks_1h", 30); mp13c = dc.get("max_peaks_13c", 40)
            self.decoder_1h = PeakDecoder(d, nh, dec_l, ff, dr, mp1h)
            self.decoder_13c = PeakDecoder(d, nh, dec_l, ff, dr, mp13c)
            self.head_1h = PropertyHead(d, N_SPLITS, has_integration=True)
            self.head_13c = PropertyHead(d, N_SPLITS, has_integration=False)
        else:
            mp = dc.get("max_peaks", 30 if self.mode == "1h" else 40)
            self.decoder = PeakDecoder(d, nh, dec_l, ff, dr, mp)
            self.head = PropertyHead(d, N_SPLITS, has_integration=(self.mode == "1h"))
        self._init()

    def _init(self):
        for n, p in self.named_parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)
        # Bias existence head strongly toward "no peak"
        if self.mode == "joint":
            nn.init.constant_(self.head_1h.exist_head[-1].bias, -4.0)
            nn.init.constant_(self.head_13c.exist_head[-1].bias, -4.0)
        else:
            nn.init.constant_(self.head.exist_head[-1].bias, -4.0)

    def forward(self, batch):
        mem, mem_pad = self.encoder(batch["smiles_ids"], batch["smiles_len"])
        if self.mode == "joint":
            return {"pred_1h": self.head_1h(self.decoder_1h(mem, mem_pad)),
                    "pred_13c": self.head_13c(self.decoder_13c(mem, mem_pad)), "mem": mem}
        else:
            return {"pred": self.head(self.decoder(mem, mem_pad)), "mem": mem}

    @torch.inference_mode()
    def predict(self, batch, threshold=0.5, use_amp=True):
        self.eval(); dev = next(self.parameters()).device
        amp_ctx = torch.amp.autocast("cuda", enabled=use_amp and dev.type=="cuda")
        with amp_ctx: out = self.forward(batch)
        B = batch["smiles_ids"].size(0)
        def _extract(pred, ml):
            results = []
            for b in range(B):
                sample = {"smiles": batch["smiles_str"][b]}
                ep = torch.sigmoid(pred["exist"][b].float()).cpu().numpy()
                valid = ep > threshold; peaks = []
                if valid.any():
                    for i in np.where(valid)[0]:
                        pk = {"ppm": float(pred["ppm"][b,i].float().cpu()),
                              "splitting": int(pred["split"][b,i].float().argmax().cpu()),
                              "j_coupling": float(pred["j"][b,i].float().cpu()),
                              "confidence": float(ep[i])}
                        if ml == "1h" and "integ" in pred:
                            pk["integration"] = float(pred["integ"][b,i].float().cpu())
                        if ml == "1h" and "width" in pred:
                            pk["half_width"] = float(pred["width"][b,i].float().cpu())
                        peaks.append(pk)
                peaks.sort(key=lambda p: -p["ppm"]); sample["peaks"] = peaks
                results.append(sample)
            return results
        if self.mode == "joint":
            return {"1h": _extract(out["pred_1h"],"1h"), "13c": _extract(out["pred_13c"],"13c")}
        return _extract(out["pred"], self.mode)

    def count_params(self): return sum(p.numel() for p in self.parameters())


# ═══════════════════════ HUNGARIAN MATCHING ═══════════════════════

@torch.no_grad()
def hungarian_match(pred_ppm, pred_exist, gt_ppm, gt_mask):
    from scipy.optimize import linear_sum_assignment
    B = pred_ppm.size(0); matches = []
    pp = pred_ppm.detach().float().cpu().numpy()
    pe = torch.sigmoid(pred_exist.detach().float()).cpu().numpy()
    gp = gt_ppm.detach().float().cpu().numpy()
    gm = gt_mask.detach().cpu().numpy()
    for b in range(B):
        n_gt = int(gm[b].sum())
        if n_gt == 0:
            matches.append((np.array([], dtype=np.int64), np.array([], dtype=np.int64))); continue
        N = pp.shape[1]; gt_idx = np.where(gm[b])[0]
        cost_ppm = np.abs(pp[b, :, None] - gp[b][gt_idx][None, :])
        cost_exist = -np.log(np.clip(pe[b, :, None], 1e-6, 1.0))
        cost = cost_ppm + 0.5 * np.broadcast_to(cost_exist, (N, n_gt))
        if not np.isfinite(cost).all():
            cost = np.nan_to_num(cost, nan=1e4, posinf=1e4, neginf=1e4)
        pi, gi = linear_sum_assignment(cost)
        matches.append((pi.astype(np.int64), gt_idx[gi].astype(np.int64)))
    return matches


# ═══════════════════════ DETR-STYLE LOSS ═══════════════════════

def compute_loss_single(pred, batch, matches, cfg, mode="1h",
                         ppm_key="ppm", split_key="split", j_key="j", mask_key="mask"):
    """
    DETR-style loss with no-object weighting.

    Key design: existence loss uses per-element weighting:
      - Matched queries (target=1): weight = 1.0
      - Unmatched queries (target=0): weight = no_object_weight (default 0.1)

    This ensures the decoder gets 10× stronger gradient from matched queries
    (which carry ppm/split/j information) than from unmatched queries
    (which only carry "I don't exist" information).
    """
    device = pred["ppm"].device
    B = pred["ppm"].size(0)
    Q = pred["ppm"].size(1)  # number of query slots
    lw = cfg.get("loss_weights", {})
    w_ppm = lw.get("ppm", 5.0)
    w_split = lw.get("split", 1.5)
    w_j = lw.get("j", 1.0)
    w_integ = lw.get("integ", 0.5)
    w_exist = lw.get("exist", 2.0)
    w_width = lw.get("width", 0.5)
    no_obj_w = lw.get("no_object_weight", 0.1)
    label_smooth = lw.get("label_smoothing", 0.1)

    gt_ppm = batch[ppm_key]; gt_split = batch[split_key]
    gt_j = batch[j_key]; gt_mask = batch[mask_key]
    gt_integ = batch.get("integ")
    losses = {}

    # ═══ EXISTENCE LOSS (DETR-style with no-object weighting) ═══
    exist_target = torch.zeros(B, Q, device=device)
    exist_weight = torch.full((B, Q), no_obj_w, device=device)  # unmatched: 0.1
    for b, (pi, gi) in enumerate(matches):
        if len(pi) > 0:
            pi_t = torch.from_numpy(pi).long().to(device)
            exist_target[b, pi_t] = 1.0
            exist_weight[b, pi_t] = 1.0  # matched: 1.0

    # Per-element BCE, then weighted mean
    exist_bce = F.binary_cross_entropy_with_logits(
        pred["exist"].float(), exist_target, reduction="none")
    exist_loss = (exist_bce * exist_weight).sum() / exist_weight.sum()
    losses["exist"] = exist_loss.item()
    total = w_exist * exist_loss

    # ═══ MATCHED PROPERTY LOSSES ═══
    all_p_ppm, all_g_ppm = [], []
    all_p_split, all_g_split = [], []
    all_p_j, all_g_j = [], []
    all_p_integ, all_g_integ = [], []
    all_p_width, all_g_width = [], []

    for b, (pi, gi) in enumerate(matches):
        if len(pi) == 0: continue
        pi_t = torch.from_numpy(pi).long().to(device)
        gi_t = torch.from_numpy(gi).long().to(device)
        all_p_ppm.append(pred["ppm"][b, pi_t])
        all_g_ppm.append(gt_ppm[b, gi_t])
        all_p_split.append(pred["split"][b, pi_t])
        all_g_split.append(gt_split[b, gi_t])
        all_p_j.append(pred["j"][b, pi_t])
        all_g_j.append(gt_j[b, gi_t])
        if mode == "1h" and gt_integ is not None and "integ" in pred:
            all_p_integ.append(pred["integ"][b, pi_t])
            all_g_integ.append(gt_integ[b, gi_t].float())
        if mode == "1h" and "width" in pred and "width" in batch:
            all_p_width.append(pred["width"][b, pi_t])
            all_g_width.append(batch["width"][b, gi_t])

    if all_p_ppm:
        cat_p_ppm = torch.cat(all_p_ppm).float()
        cat_g_ppm = torch.cat(all_g_ppm).float()
        cat_p_j = torch.cat(all_p_j).float()
        cat_g_j = torch.cat(all_g_j).float()

        m_ppm = F.mse_loss(cat_p_ppm, cat_g_ppm)
        m_split = F.cross_entropy(torch.cat(all_p_split).float(),
                                   torch.cat(all_g_split),
                                   label_smoothing=label_smooth)
        m_j = F.mse_loss(cat_p_j, cat_g_j)

        losses["ppm"] = m_ppm.item()
        losses["split"] = m_split.item()
        losses["j"] = m_j.item()
        total = total + w_ppm * m_ppm + w_split * m_split + w_j * m_j

        if all_p_integ:
            m_integ = F.mse_loss(torch.cat(all_p_integ).float(),
                                  torch.cat(all_g_integ).float())
            losses["integ"] = m_integ.item()
            total = total + w_integ * m_integ

        if all_p_width:
            m_width = F.mse_loss(torch.cat(all_p_width).float(),
                                  torch.cat(all_g_width).float())
            losses["width"] = m_width.item()
            total = total + w_width * m_width
    else:
        losses["ppm"] = 0.0; losses["split"] = 0.0; losses["j"] = 0.0

    losses["total"] = total.item()
    valid = torch.isfinite(total)
    return total, losses, valid


def compute_loss(pred, batch, matches, cfg, mode="1h"):
    return compute_loss_single(pred, batch, matches, cfg, mode=mode)


def compute_loss_joint(out, batch, matches_1h, matches_13c, cfg):
    lw = cfg.get("loss_weights", {})
    w_1h = lw.get("weight_1h", 1.0); w_13c = lw.get("weight_13c", 1.0)
    l1, ld1, v1 = compute_loss_single(out["pred_1h"], batch, matches_1h, cfg, "1h",
        "ppm_1h","split_1h","j_1h","mask_1h")
    l2, ld2, v2 = compute_loss_single(out["pred_13c"], batch, matches_13c, cfg, "13c",
        "ppm_13c","split_13c","j_13c","mask_13c")
    total = w_1h * l1 + w_13c * l2
    losses = {f"1h_{k}": v for k, v in ld1.items()}
    losses.update({f"13c_{k}": v for k, v in ld2.items()})
    losses["total"] = total.item()
    return total, losses, torch.isfinite(total)


# ═══════════════════════ EVALUATION ═══════════════════════

def evaluate(predictions, batch, norm_stats, mode="1h", threshold=0.5,
             pred_key="pred", ppm_raw_key="ppm_raw", mask_key="mask",
             split_key="split", j_key="j", ppm_stat="ppm", j_stat="j"):
    pm = norm_stats[ppm_stat]["mean"]; ps = norm_stats[ppm_stat]["std"]
    jm = norm_stats[j_stat]["mean"]; js = norm_stats[j_stat]["std"]
    pred = predictions[pred_key]
    gt_ppm_raw = batch[ppm_raw_key]; gt_mask = batch[mask_key]
    gt_split = batch[split_key]; gt_j = batch[j_key]
    B = gt_ppm_raw.size(0)

    ppm_errors, count_errors = [], []
    split_correct, split_total = 0, 0
    all_prec, all_rec = [], []
    m_pred_ppms, m_gt_ppms, m_pred_js, m_gt_js = [], [], [], []

    ep = torch.sigmoid(pred["exist"].float()).detach().cpu()
    ppn = pred["ppm"].float().detach().cpu()
    psc = pred["split"].float().detach().cpu().argmax(-1)
    pjn = pred["j"].float().detach().cpu()
    gm = gt_mask.cpu(); gpr = gt_ppm_raw.cpu()
    gsc = gt_split.cpu(); gjc = gt_j.cpu()
    tol = 0.08 if mode == "1h" else 0.8

    for b in range(B):
        vp = ep[b] > threshold; np_ = vp.sum().item(); ng = gm[b].sum().item()
        count_errors.append(abs(np_ - ng))
        if ng == 0 and np_ == 0: all_prec.append(1.0); all_rec.append(1.0); continue
        if ng == 0: all_prec.append(0.0); all_rec.append(1.0); continue
        if np_ == 0: all_prec.append(1.0); all_rec.append(0.0); continue

        p_ppm = ppn[b][vp].numpy() * ps + pm
        g_ppm = gpr[b][gm[b]].numpy()
        p_j = pjn[b][vp].numpy() * js + jm
        g_j = gjc[b][gm[b]].numpy() * js + jm
        p_sp = psc[b][vp].numpy(); g_sp = gsc[b][gm[b]].numpy()

        from scipy.optimize import linear_sum_assignment
        cost = np.abs(p_ppm[:, None] - g_ppm[None, :])
        pi_arr, gi_arr = linear_sum_assignment(cost)
        mp, mg = set(), set()
        for pi, gi in zip(pi_arr, gi_arr):
            if cost[pi, gi] < tol:
                mp.add(pi); mg.add(gi); ppm_errors.append(cost[pi, gi])
                m_pred_ppms.append(p_ppm[pi]); m_gt_ppms.append(g_ppm[gi])
                m_pred_js.append(p_j[pi]); m_gt_js.append(g_j[gi])
                if p_sp[pi] == g_sp[gi]: split_correct += 1
                split_total += 1
        all_prec.append(len(mp)/max(np_,1)); all_rec.append(len(mg)/max(ng,1))

    avg_p = np.mean(all_prec) if all_prec else 0
    avg_r = np.mean(all_rec) if all_rec else 0
    f1 = 2*avg_p*avg_r/max(avg_p+avg_r, 1e-8)
    metrics = {"ppm_mae": np.mean(ppm_errors) if ppm_errors else 0,
               "count_mae": np.mean(count_errors) if count_errors else 0,
               "split_acc": split_correct/max(split_total,1),
               "precision": avg_p, "recall": avg_r, "f1": f1}
    if len(m_gt_ppms) >= 2:
        ga = np.array(m_gt_ppms); pa = np.array(m_pred_ppms)
        metrics["ppm_r2"] = float(1 - np.sum((ga-pa)**2)/max(np.sum((ga-ga.mean())**2), 1e-8))
    else: metrics["ppm_r2"] = 0.0
    if len(m_gt_js) >= 2:
        gj = np.array(m_gt_js); pj = np.array(m_pred_js)
        metrics["j_mae"] = float(np.mean(np.abs(gj-pj)))
        metrics["j_r2"] = float(1 - np.sum((gj-pj)**2)/max(np.sum((gj-gj.mean())**2), 1e-8))
    else: metrics["j_mae"] = 0.0; metrics["j_r2"] = 0.0
    return metrics
