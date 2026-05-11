

import ast, csv, json, logging, math, os, random, re, warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

logger = logging.getLogger("PRISM")

SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|[BCNOPSFIcnos]|[=#/\\@+\-\.\(\)]|[0-9]|%[0-9]{2})")

# ═══════════ Module-level vocab (populated by load_vocab) ═══════════
SPLITTING_PATTERNS, SPLIT2ID, N_SPLITS = [], {}, 0
H_SHIFT_REGIONS, N_H_REGIONS = [], 0
C_SHIFT_REGIONS, N_C_REGIONS = [], 0
SOLVENTS, SOLV2ID, N_SOLVENTS = [], {}, 0
MF_ATOMS, N_MF_ATOMS = [], 0
_VOCAB_LOADED = False

def load_vocab(processed_dir):
    global SPLITTING_PATTERNS, SPLIT2ID, N_SPLITS
    global H_SHIFT_REGIONS, N_H_REGIONS, C_SHIFT_REGIONS, N_C_REGIONS
    global SOLVENTS, SOLV2ID, N_SOLVENTS, MF_ATOMS, N_MF_ATOMS, _VOCAB_LOADED
    vd = os.path.join(processed_dir, "vocab")
    if not os.path.isdir(vd):
        raise FileNotFoundError(f"Vocab dir not found: {vd}\nRun: python preprocess.py --config config.yaml")
    with open(os.path.join(vd,"splitting_patterns.json")) as f:
        d=json.load(f); SPLITTING_PATTERNS=d["patterns"]; SPLIT2ID={s:int(i) for s,i in d["pattern2id"].items()}; N_SPLITS=len(SPLITTING_PATTERNS)
    logger.info(f"  Loaded {N_SPLITS} splitting patterns")
    with open(os.path.join(vd,"h_shift_regions.json")) as f:
        d=json.load(f); H_SHIFT_REGIONS=[(r["lower"],r["upper"],r["label"]) for r in d["regions"]]; N_H_REGIONS=len(H_SHIFT_REGIONS)
    with open(os.path.join(vd,"c_shift_regions.json")) as f:
        d=json.load(f); C_SHIFT_REGIONS=[(r["lower"],r["upper"],r["label"]) for r in d["regions"]]; N_C_REGIONS=len(C_SHIFT_REGIONS)
    logger.info(f"  Loaded {N_H_REGIONS} 1H + {N_C_REGIONS} 13C shift regions")
    with open(os.path.join(vd,"solvents.json")) as f:
        d=json.load(f); SOLVENTS=d["solvents"]; SOLV2ID={s:int(i) for s,i in d["solvent2id"].items()}; N_SOLVENTS=len(SOLVENTS)
    with open(os.path.join(vd,"atom_types.json")) as f:
        d=json.load(f); MF_ATOMS=d["atom_types"]; N_MF_ATOMS=len(MF_ATOMS)
    logger.info(f"  Loaded {N_SOLVENTS} solvents, {N_MF_ATOMS} atom types")
    _VOCAB_LOADED = True

def ensure_vocab():
    if not _VOCAB_LOADED:
        raise RuntimeError("Call load_vocab(processed_dir) first")

# ═══════════════════════ SMILES TOKENIZER ═══════════════════════

class SMILESTokenizer:
    PAD,BOS,EOS,UNK = "<pad>","<bos>","<eos>","<unk>"
    def __init__(self):
        self.token2id = {self.PAD:0, self.BOS:1, self.EOS:2, self.UNK:3}
        self.id2token = {0:self.PAD, 1:self.BOS, 2:self.EOS, 3:self.UNK}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0,1,2,3
    @staticmethod
    def tokenize(smi): return SMILES_REGEX.findall(smi)
    def build_vocab(self, smiles_list, min_freq=2):
        c = Counter()
        for s in smiles_list: c.update(self.tokenize(s))
        for t,n in c.most_common():
            if n>=min_freq and t not in self.token2id:
                idx=len(self.token2id); self.token2id[t]=idx; self.id2token[idx]=t
    @property
    def vocab_size(self): return len(self.token2id)
    def encode(self, smi, max_len=128):
        toks = self.tokenize(smi)[:max_len-2]
        return [self.bos_id]+[self.token2id.get(t,self.unk_id) for t in toks]+[self.eos_id]
    def decode(self, ids):
        out = []
        for i in ids:
            if i==self.eos_id: break
            if i in (self.pad_id,self.bos_id): continue
            out.append(self.id2token.get(i,"?"))
        return "".join(out)
    def save(self, path):
        with open(path,"w") as f: json.dump({"token2id":self.token2id},f)
    def load(self, path):
        with open(path) as f: d=json.load(f)
        self.token2id={k:int(v) for k,v in d["token2id"].items()}
        self.id2token={v:k for k,v in self.token2id.items()}
        self.pad_id,self.bos_id,self.eos_id,self.unk_id=0,1,2,3
    @classmethod
    def from_processed(cls, processed_dir):
        t=cls()
        t.load(os.path.join(processed_dir,"vocab","smiles_tokenizer.json"))
        logger.info(f"Loaded tokenizer (vocab_size={t.vocab_size})")
        return t

class SELFIESTokenizer:
    """
    SELFIES tokenizer — encodes molecules as SELFIES strings.
    SELFIES are 100% valid by construction: every SELFIES string
    decodes to a valid molecule, eliminating the "invalid SMILES" problem.

    Requires: pip install selfies
    """
    PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"

    def __init__(self):
        self.token2id = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK: 3}
        self.id2token = {0: self.PAD, 1: self.BOS, 2: self.EOS, 3: self.UNK}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
        self._sf = None  # lazy import

    def _ensure_selfies(self):
        if self._sf is None:
            try:
                import selfies as sf
                self._sf = sf
            except ImportError:
                raise ImportError(
                    "SELFIES not installed. Run: pip install selfies")
        return self._sf

    @staticmethod
    def tokenize(selfies_str):
        """Split a SELFIES string into tokens: [C][=C][Branch1]... → ['[C]','[=C]','[Branch1]',...]"""
        return SELFIES_REGEX.findall(selfies_str)

    def smiles_to_selfies(self, smi):
        """Convert SMILES to SELFIES string. Returns None on failure."""
        sf = self._ensure_selfies()
        try:
            result = sf.encoder(smi)
            return result if result else None
        except Exception:
            return None

    def selfies_to_smiles(self, sel):
        """Convert SELFIES string back to SMILES. Returns None on failure."""
        sf = self._ensure_selfies()
        try:
            result = sf.decoder(sel)
            return result if result else None
        except Exception:
            return None

    def build_vocab(self, smiles_list, min_freq=2):
        """Build vocabulary from SMILES list (converts to SELFIES internally)."""
        sf = self._ensure_selfies()
        c = Counter()
        n_fail = 0
        for smi in smiles_list:
            sel = self.smiles_to_selfies(smi)
            if sel:
                c.update(self.tokenize(sel))
            else:
                n_fail += 1
        for t, n in c.most_common():
            if n >= min_freq and t not in self.token2id:
                idx = len(self.token2id)
                self.token2id[t] = idx
                self.id2token[idx] = t
        if n_fail > 0:
            logger.warning(f"  SELFIES: {n_fail} SMILES failed to encode")

    @property
    def vocab_size(self):
        return len(self.token2id)

    def encode(self, smi_or_selfies, max_len=128):
        """
        Encode a molecule to token IDs.
        Accepts either SMILES (auto-converts to SELFIES) or SELFIES.
        """
        # Auto-detect: SELFIES always start with '[', SMILES usually don't
        if smi_or_selfies and smi_or_selfies[0] == '[' and '][' in smi_or_selfies:
            sel = smi_or_selfies  # Already SELFIES
        else:
            sel = self.smiles_to_selfies(smi_or_selfies)
            if sel is None:
                return [self.bos_id, self.unk_id, self.eos_id]
        toks = self.tokenize(sel)[:max_len - 2]
        return [self.bos_id] + [self.token2id.get(t, self.unk_id) for t in toks] + [self.eos_id]

    def decode(self, ids):
        """Decode token IDs to SELFIES string, then convert back to SMILES."""
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.id2token.get(i, "?"))
        selfies_str = "".join(out)
        # Convert SELFIES back to SMILES for evaluation
        smi = self.selfies_to_smiles(selfies_str)
        return smi if smi else selfies_str  # Fallback to raw SELFIES if decode fails

    def decode_selfies(self, ids):
        """Decode token IDs to raw SELFIES string (no SMILES conversion)."""
        out = []
        for i in ids:
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.id2token.get(i, "?"))
        return "".join(out)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"token2id": self.token2id, "type": "selfies"}, f)

    def load(self, path):
        with open(path) as f:
            d = json.load(f)
        self.token2id = {k: int(v) for k, v in d["token2id"].items()}
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3

    @classmethod
    def from_processed(cls, processed_dir):
        t = cls()
        path = os.path.join(processed_dir, "vocab", "selfies_tokenizer.json")
        if os.path.exists(path):
            t.load(path)
            logger.info(f"Loaded SELFIES tokenizer (vocab_size={t.vocab_size})")
        else:
            # Fall back to building from SMILES tokenizer vocab
            logger.warning("No SELFIES tokenizer found — will build during training")
        return t


# ═══════════════════════ DATASET (tensor-based) ═══════════════════════

class NMRDataset(Dataset):
    """Loads pre-processed & normalized tensors from .pt file. Very fast."""
    def __init__(self, tensor_path, tokenizer, cfg, augment=False, **kwargs):
        ensure_vocab()
        self.tokenizer = tokenizer
        self.augment = augment
        ac = cfg.get("training",{}).get("augmentation",{})
        self.ppm_noise = ac.get("ppm_noise_std", 0.03) if augment else 0
        self.peak_drop = ac.get("peak_dropout_prob", 0.08) if augment else 0
        self.j_noise_std = ac.get("j_coupling_noise_std", 0.02) if augment else 0  # In normalized space
        self.integ_noise = ac.get("integration_noise_prob", 0.05) if augment else 0
        # Load norm stats for noise scaling
        pd = cfg.get("data",{}).get("processed_dir","processed")
        ns_path = os.path.join(pd, "vocab", "norm_stats.json")
        if os.path.exists(ns_path):
            with open(ns_path) as f: ns = json.load(f)
            self.h_ppm_std = ns["h_ppm"]["std"]
            self.c_ppm_std = ns["c_ppm"]["std"]
        else:
            self.h_ppm_std, self.c_ppm_std = 3.0, 50.0
        # Load tensors
        logger.info(f"Loading tensors from {tensor_path}...")
        self.data = torch.load(tensor_path, map_location="cpu", weights_only=False)
        self.n = self.data["h_ppm"].size(0)
        logger.info(f"Loaded {self.n} samples")

    def __len__(self): return self.n

    def __getitem__(self, idx):
        d = self.data
        h_ppm = d["h_ppm"][idx].clone()
        h_split = d["h_split"][idx].clone()
        h_integ = d["h_integ"][idx].clone()
        h_j = d["h_j"][idx].clone()
        h_mask = d["h_mask"][idx].clone()
        h_region = d["h_region"][idx].clone()
        c_ppm = d["c_ppm"][idx].clone()
        c_split = d["c_split"][idx].clone()
        c_j = d["c_j"][idx].clone()
        c_mask = d["c_mask"][idx].clone()
        c_region = d["c_region"][idx].clone()

        # Augmentation (in normalized space)
        if self.augment:
            nh = h_mask.sum().item()
            nc = c_mask.sum().item()
            # PPM noise (normalized: noise_ppm / std)
            if self.ppm_noise > 0 and nh > 0:
                h_ppm[:nh] += torch.randn(nh) * (self.ppm_noise / self.h_ppm_std)
            if self.ppm_noise > 0 and nc > 0:
                c_ppm[:nc] += torch.randn(nc) * (self.ppm_noise * 10 / self.c_ppm_std)
            # J noise
            if self.j_noise_std > 0:
                if nh > 0: h_j[:nh] += torch.randn(nh) * self.j_noise_std
                if nc > 0: c_j[:nc] += torch.randn(nc) * self.j_noise_std
            # Peak dropout
            if self.peak_drop > 0:
                if nh > 1:
                    keep = torch.rand(nh) > self.peak_drop
                    if not keep.any(): keep[0] = True
                    drop_mask = torch.ones_like(h_mask)
                    drop_mask[:nh] = keep
                    h_mask = h_mask & drop_mask
                if nc > 1:
                    keep = torch.rand(nc) > self.peak_drop
                    if not keep.any(): keep[0] = True
                    drop_mask = torch.ones_like(c_mask)
                    drop_mask[:nc] = keep
                    c_mask = c_mask & drop_mask
            # Integration noise
            if self.integ_noise > 0 and nh > 0:
                flip = torch.rand(nh) < self.integ_noise
                delta = torch.where(torch.rand(nh)>0.5, torch.ones(nh,dtype=torch.long), -torch.ones(nh,dtype=torch.long))
                h_integ[:nh] = (h_integ[:nh] + flip.long()*delta).clamp(1, 20)

        result = {
            "h_ppm": h_ppm, "h_split": h_split, "h_integ": h_integ,
            "h_j": h_j, "h_mask": h_mask,
            "c_ppm": c_ppm, "c_split": c_split, "c_j": c_j, "c_mask": c_mask,
            "smiles_ids": d["smiles_ids"][idx],
            "smiles_len": d["smiles_len"][idx].item(),
            "smiles_str": d["smiles_str"][idx],
            "h_region": h_region, "c_region": c_region,
            "mol_formula": d["mol_formula"][idx],
            "freq_h": d["freq_h"][idx], "freq_c": d["freq_c"][idx],
            "solvent_h": d["solvent_h"][idx], "solvent_c": d["solvent_c"][idx],
            "h_ppm_raw": d["h_ppm_raw"][idx], "c_ppm_raw": d["c_ppm_raw"][idx],
        }
        return result

def collate_fn(batch):
    out = {}
    for k in batch[0]:
        if k == "smiles_str": out[k] = [b[k] for b in batch]
        elif k == "smiles_len": out[k] = torch.tensor([b[k] for b in batch])
        else: out[k] = torch.stack([b[k] for b in batch])
    return out

# ═══════════════════════ MODEL COMPONENTS ═══════════════════════

class SinPE(nn.Module):
    def __init__(self, d, max_len=512, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:,:x.size(1)])

class FourierCSPE(nn.Module):
    def __init__(self, d_model, n_freqs=None, dropout=0.1):
        super().__init__()
        n_freqs = n_freqs or d_model//4
        self.freqs = nn.Parameter(torch.logspace(-2,2,n_freqs))
        self.proj = nn.Linear(2*n_freqs, d_model)
        self.drop = nn.Dropout(dropout)
    def forward(self, ppm):
        angles = 2*math.pi*ppm.unsqueeze(-1)*self.freqs
        return self.drop(self.proj(torch.cat([torch.sin(angles),torch.cos(angles)],-1)))

class ChemicalShiftRegionEncoder(nn.Module):
    def __init__(self, d_model, n_h=None, n_c=None):
        super().__init__()
        self.h_emb = nn.Embedding(n_h or N_H_REGIONS, d_model)
        self.c_emb = nn.Embedding(n_c or N_C_REGIONS, d_model)
    def forward_h(self, ids): return self.h_emb(ids)
    def forward_c(self, ids): return self.c_emb(ids)

class GSFU(nn.Module):
    def __init__(self, d, n_extra):
        super().__init__()
        self.proj = nn.Linear(d*n_extra, d)
        self.gate = nn.Sequential(nn.Linear(d*2, d), nn.Sigmoid())
        self.norm = nn.LayerNorm(d)
    def forward(self, backbone, extras):
        ep = self.proj(torch.cat(extras, -1))
        g = self.gate(torch.cat([backbone, ep], -1))
        return self.norm(backbone + g * ep)

class SARAB(nn.Module):
    def __init__(self, nhead, max_peaks, n_splits=None):
        super().__init__()
        ns = n_splits or N_SPLITS
        self.nhead, self.n_splits = nhead, ns
        self.split_bias = nn.Embedding(ns*ns, nhead)
        self.ppm_scale = nn.Parameter(torch.ones(nhead)*0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))
    def compute_bias(self, ppm, split_ids, mask):
        B,S = ppm.shape
        pd = (ppm.unsqueeze(2)-ppm.unsqueeze(1)).pow(2)
        sc = self.ppm_scale.abs().view(1,self.nhead,1,1)
        pb = torch.exp(-sc * pd.unsqueeze(1))
        pi = split_ids.unsqueeze(2)*self.n_splits + split_ids.unsqueeze(1)
        sb = self.split_bias(pi).permute(0,3,1,2)
        bias = self.alpha*(pb+sb)
        valid = mask.unsqueeze(2)&mask.unsqueeze(1)
        return bias.masked_fill(~valid.unsqueeze(1), 0.0)

class BiCMA(nn.Module):
    def __init__(self, d, nhead, drop=0.1):
        super().__init__()
        self.h2c = nn.MultiheadAttention(d,nhead,dropout=drop,batch_first=True)
        self.c2h = nn.MultiheadAttention(d,nhead,dropout=drop,batch_first=True)
        self.hg = nn.Sequential(nn.Linear(d*2,d),nn.Sigmoid())
        self.cg = nn.Sequential(nn.Linear(d*2,d),nn.Sigmoid())
        self.hn = nn.LayerNorm(d); self.cn = nn.LayerNorm(d)
    def forward(self, h, c, hm, cm):
        cc,_ = self.h2c(c,h,h,key_padding_mask=~hm)
        c = self.cn(c + self.cg(torch.cat([c,cc],-1))*cc)
        hc,_ = self.c2h(h,c,c,key_padding_mask=~cm)
        h = self.hn(h + self.hg(torch.cat([h,hc],-1))*hc)
        return h, c

class MolecularFormulaEncoder(nn.Module):
    def __init__(self, d, n_atoms=None, hidden=128):
        super().__init__()
        na = n_atoms or N_MF_ATOMS
        self.mlp = nn.Sequential(nn.Linear(na,hidden),nn.GELU(),nn.LayerNorm(hidden),
                                  nn.Linear(hidden,hidden),nn.GELU(),nn.LayerNorm(hidden),
                                  nn.Linear(hidden,d))
        self.gate = nn.Sequential(nn.Linear(d*2,d),nn.Sigmoid())
    def forward(self, fv): return self.mlp(fv).unsqueeze(1)
    def fuse(self, mem, fe):
        fe2 = fe.expand_as(mem)
        return mem + self.gate(torch.cat([mem,fe2],-1))*fe2

class PeakAtomHead(nn.Module):
    def __init__(self, d, nhead, top_k=5):
        super().__init__()
        self.top_k = top_k
        self.attn = nn.MultiheadAttention(d,nhead,batch_first=True)
    def forward(self, dec_h, enc_m, mask):
        _, aw = self.attn(dec_h, enc_m, enc_m, key_padding_mask=mask)
        k = min(self.top_k, aw.size(-1))
        _,ti = aw.topk(k,-1)
        entropy = -(aw * torch.log(aw+1e-8)).sum(-1).mean()
        return aw, ti, entropy

class SpectralReconHead(nn.Module):
    """Predicts NORMALIZED ppm from encoder memory."""
    def __init__(self, d, max_h, max_c):
        super().__init__()
        self.hp = nn.Linear(d,1); self.cp = nn.Linear(d,1); self.mh = max_h
    def forward(self, mem, hm, cm):
        return self.hp(mem[:,:self.mh]).squeeze(-1), self.cp(mem[:,self.mh:]).squeeze(-1)

class FormulaPredHead(nn.Module):
    def __init__(self, d, n_atoms=None):
        super().__init__()
        na = n_atoms or N_MF_ATOMS
        self.head = nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,na))
    def forward(self, mem, mask):
        vm = (~mask).unsqueeze(-1).float()
        pooled = (mem*vm).sum(1)/vm.sum(1).clamp(min=1)
        return self.head(pooled)

class SARABEncoderLayer(nn.Module):
    def __init__(self, d, nhead, d_ff, drop):
        super().__init__()
        self.sa = nn.MultiheadAttention(d,nhead,dropout=drop,batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d,d_ff),nn.GELU(),nn.Dropout(drop),nn.Linear(d_ff,d),nn.Dropout(drop))
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop); self.nhead = nhead
    def forward(self, src, pad=None, bias=None):
        x = self.n1(src)
        if bias is not None:
            B,S,d = x.shape; hd = d//self.nhead
            qkv = F.linear(x, self.sa.in_proj_weight, self.sa.in_proj_bias)
            q,k,v = qkv.chunk(3,-1)
            q=q.view(B,S,self.nhead,hd).transpose(1,2)
            k=k.view(B,S,self.nhead,hd).transpose(1,2)
            v=v.view(B,S,self.nhead,hd).transpose(1,2)
            sc = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(hd) + bias
            if pad is not None: sc=sc.masked_fill(pad.unsqueeze(1).unsqueeze(2),float("-inf"))
            aw = F.softmax(sc,-1); aw = F.dropout(aw,p=self.sa.dropout,training=self.training)
            ao = torch.matmul(aw,v).transpose(1,2).contiguous().view(B,S,d)
            ao = self.sa.out_proj(ao)
        else:
            ao,_ = self.sa(x,x,x,key_padding_mask=pad)
        src = src + self.drop(ao)
        return src + self.ffn(self.n2(src))

# ═══════════════════════ NMR ENCODER ═══════════════════════

class NMREncoder(nn.Module):
    def __init__(self, d, nhead, n_layers, d_ff, drop, max_pk, cfg=None):
        super().__init__()
        mc = (cfg or {}).get("modules",{})
        self.use_fcspe = mc.get("use_fcspe",False)
        self.use_gsfu = mc.get("use_gsfu",False)
        self.use_sarab = mc.get("use_sarab",False)
        self.use_bicma = mc.get("use_bicma",False)
        self.use_csr = mc.get("use_chemical_shift_regions",False)
        self.use_mfe = mc.get("use_molecular_formula_encoder",False)
        dc = (cfg or {}).get("data",{})
        self.max_h, self.max_c = dc.get("max_h_peaks",32), dc.get("max_c_peaks",40)
        # Core embeddings
        self.h_cont = nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,d))
        self.h_sp = nn.Embedding(N_SPLITS,d); self.h_ig = nn.Embedding(24,d)
        self.c_cont = nn.Sequential(nn.Linear(2,d),nn.GELU(),nn.Linear(d,d))
        self.c_sp = nn.Embedding(N_SPLITS,d); self.mod = nn.Embedding(2,d)
        self.norm = nn.LayerNorm(d); self.pe = SinPE(d,max_pk,drop)
        if self.use_fcspe: self.h_fcspe=FourierCSPE(d,dropout=drop); self.c_fcspe=FourierCSPE(d,dropout=drop)
        if self.use_csr: self.csr = ChemicalShiftRegionEncoder(d)
        if self.use_gsfu:
            self.solv_emb = nn.Embedding(N_SOLVENTS,d)
            self.freq_proj = nn.Sequential(nn.Linear(1,d),nn.GELU())
            nh_ex = 5 if self.use_csr else 4; nc_ex = 4 if self.use_csr else 3
            self.h_gsfu = GSFU(d,nh_ex); self.c_gsfu = GSFU(d,nc_ex)
        if self.use_sarab: self.sarab = SARAB(nhead,max_pk)
        if self.use_bicma:
            nl = mc.get("bicma_layers",2)
            self.bicma_layers = nn.ModuleList([BiCMA(d,nhead,drop) for _ in range(nl)])
        if self.use_mfe: self.mfe = MolecularFormulaEncoder(d)
        if self.use_sarab:
            self.enc_layers = nn.ModuleList([SARABEncoderLayer(d,nhead,d_ff,drop) for _ in range(n_layers)])
        else:
            el = nn.TransformerEncoderLayer(d_model=d,nhead=nhead,dim_feedforward=d_ff,dropout=drop,activation="gelu",batch_first=True,norm_first=True)
            self.enc = nn.TransformerEncoder(el, n_layers)

    def forward(self, h_ppm, h_split, h_integ, h_j, h_mask,
                c_ppm, c_split, c_j, c_mask,
                h_region=None, c_region=None, mol_formula=None,
                freq_h=None, freq_c=None, solvent_h=None, solvent_c=None):
        # Base embeddings (ppm and j are ALREADY normalized)
        h_e = self.h_cont(torch.stack([h_ppm, h_j],-1))
        h_se, h_ie = self.h_sp(h_split), self.h_ig(h_integ.clamp(0,23))
        c_e = self.c_cont(torch.stack([c_ppm, c_j],-1))
        c_se = self.c_sp(c_split)
        if self.use_fcspe:
            h_e = h_e + self.h_fcspe(h_ppm); c_e = c_e + self.c_fcspe(c_ppm)
        # Always add modality
        h_e = h_e + self.mod(torch.zeros_like(h_split))
        c_e = c_e + self.mod(torch.ones_like(c_split))
        # GSFU or original
        if self.use_gsfu and freq_h is not None:
            B = h_ppm.shape[0]
            se_h = self.solv_emb(solvent_h if solvent_h is not None else torch.zeros(B,dtype=torch.long,device=h_ppm.device))
            se_c = self.solv_emb(solvent_c if solvent_c is not None else torch.zeros(B,dtype=torch.long,device=h_ppm.device))
            fe_h = self.freq_proj(freq_h.view(B,1)); fe_c = self.freq_proj(freq_c.view(B,1))
            mh = (se_h+fe_h).unsqueeze(1).expand(-1,self.max_h,-1)
            mc = (se_c+fe_c).unsqueeze(1).expand(-1,self.max_c,-1)
            hj_e = self.h_cont(torch.stack([torch.zeros_like(h_j),h_j],-1))
            cj_e = self.c_cont(torch.stack([torch.zeros_like(c_j),c_j],-1))
            h_ex = [h_se, h_ie, hj_e, mh]; c_ex = [c_se, cj_e, mc]
            if self.use_csr and h_region is not None:
                h_ex.append(self.csr.forward_h(h_region)); c_ex.append(self.csr.forward_c(c_region))
            h_e = self.h_gsfu(h_e, h_ex); c_e = self.c_gsfu(c_e, c_ex)
        else:
            h_e = h_e + h_se + h_ie; c_e = c_e + c_se
            if self.use_csr and h_region is not None:
                h_e = h_e + self.csr.forward_h(h_region); c_e = c_e + self.csr.forward_c(c_region)
        if self.use_bicma:
            for bl in self.bicma_layers: h_e, c_e = bl(h_e, c_e, h_mask, c_mask)
        emb = self.pe(self.norm(torch.cat([h_e,c_e],1)))
        pad = ~torch.cat([h_mask,c_mask],1)
        if self.use_sarab:
            fp = torch.cat([h_ppm,c_ppm],1); fs = torch.cat([h_split,c_split],1); fm = torch.cat([h_mask,c_mask],1)
            ab = self.sarab.compute_bias(fp,fs,fm)
            x = emb
            for l in self.enc_layers: x = l(x,pad,ab)
            mem = x
        else:
            mem = self.enc(emb, src_key_padding_mask=pad)
        if self.use_mfe and mol_formula is not None:
            fe = self.mfe(mol_formula); mem = self.mfe.fuse(mem, fe)
        return mem, pad

# ═══════════════════════ DECODER ═══════════════════════

class SMILESDecoder(nn.Module):
    def __init__(self, vocab, d, nhead, n_layers, d_ff, drop, max_len):
        super().__init__()
        self.d = d; self.emb = nn.Embedding(vocab,d); self.pe = SinPE(d,max_len,drop)
        dl = nn.TransformerDecoderLayer(d_model=d,nhead=nhead,dim_feedforward=d_ff,dropout=drop,activation="gelu",batch_first=True,norm_first=True)
        self.dec = nn.TransformerDecoder(dl,n_layers); self.proj = nn.Linear(d,vocab)
    def forward(self, tgt_ids, mem, mem_pad, tgt_pad=None):
        T,dev = tgt_ids.size(1), tgt_ids.device
        causal = torch.zeros(T,T,device=dev)
        causal.masked_fill_(torch.triu(torch.ones(T,T,device=dev,dtype=torch.bool),1),float("-inf"))
        x = self.pe(self.emb(tgt_ids)*math.sqrt(self.d))
        out = self.dec(x,mem,tgt_mask=causal,tgt_key_padding_mask=tgt_pad,memory_key_padding_mask=mem_pad)
        return self.proj(out), out
    def forward_logits(self, tgt_ids, mem, mem_pad, tgt_pad=None):
        logits, _ = self.forward(tgt_ids, mem, mem_pad, tgt_pad)
        return logits

# ═══════════════════════ PRISM ═══════════════════════

class PRISM(nn.Module):
    def __init__(self, tokenizer, cfg):
        super().__init__()
        ensure_vocab()
        self.tokenizer = tokenizer; self.cfg = cfg
        mc = cfg.get("model",{}); d=mc.get("d_model",256); nh=mc.get("nhead",8)
        el=mc.get("encoder_layers",6); dl=mc.get("decoder_layers",6)
        ff=mc.get("d_ff",1024); dr=mc.get("dropout",0.1)
        dc = cfg.get("data",{}); mh=dc.get("max_h_peaks",32); mc_=dc.get("max_c_peaks",40); ms=dc.get("max_smiles_len",128)
        modules = cfg.get("modules",{})
        self.use_pah = modules.get("use_peak_atom_head",False)
        self.use_aux = modules.get("use_auxiliary_losses",False)
        self.use_mfe = modules.get("use_molecular_formula_encoder",False)
        self.use_csr = modules.get("use_chemical_shift_regions",False)
        self.use_gsfu = modules.get("use_gsfu",False)
        self.encoder = NMREncoder(d,nh,el,ff,dr,mh+mc_,cfg)
        self.decoder = SMILESDecoder(tokenizer.vocab_size,d,nh,dl,ff,dr,ms)
        if self.use_pah: self.pah = PeakAtomHead(d,nh,modules.get("peak_atom_top_k",5))
        if self.use_aux:
            self.srh = SpectralReconHead(d,mh,mc_)
            self.fph = FormulaPredHead(d)
        self._init()

    def _init(self):
        for p in self.parameters():
            if p.dim()>1: nn.init.xavier_uniform_(p)

    def _enc_kwargs(self, batch):
        kw = {}
        if self.use_csr: kw["h_region"]=batch.get("h_region"); kw["c_region"]=batch.get("c_region")
        if self.use_mfe: kw["mol_formula"]=batch.get("mol_formula")
        if self.use_gsfu:
            for k in ["freq_h","freq_c","solvent_h","solvent_c"]: kw[k]=batch.get(k)
        return kw

    def forward(self, batch):
        mem,mp = self.encoder(batch["h_ppm"],batch["h_split"],batch["h_integ"],
                              batch["h_j"],batch["h_mask"],
                              batch["c_ppm"],batch["c_split"],batch["c_j"],batch["c_mask"],
                              **self._enc_kwargs(batch))
        smi = batch["smiles_ids"]; ti,to = smi[:,:-1],smi[:,1:]
        tp = (ti==self.tokenizer.pad_id)
        logits, dh = self.decoder(ti,mem,mp,tp)
        result = {"logits":logits,"labels":to}
        if self.use_pah:
            aw,tk,ent = self.pah(dh,mem,mp)
            result["peak_attn_weights"]=aw; result["peak_top_k"]=tk; result["peak_attn_entropy"]=ent
        if self.use_aux:
            hp,cp = self.srh(mem,batch["h_mask"],batch["c_mask"])
            result["h_ppm_pred"]=hp; result["c_ppm_pred"]=cp
            result["h_ppm_gt"]=batch["h_ppm"]  # NORMALIZED ppm
            result["c_ppm_gt"]=batch["c_ppm"]   # NORMALIZED ppm
            result["h_mask"]=batch["h_mask"]; result["c_mask"]=batch["c_mask"]
            result["formula_pred"] = self.fph(mem,mp)
            if "mol_formula" in batch: result["formula_gt"]=batch["mol_formula"]
        return result

    @torch.no_grad()
    def generate(self, batch, max_len=128, beam_size=1):
        self.eval()
        mem,mp = self.encoder(batch["h_ppm"],batch["h_split"],batch["h_integ"],
                              batch["h_j"],batch["h_mask"],
                              batch["c_ppm"],batch["c_split"],batch["c_j"],batch["c_mask"],
                              **self._enc_kwargs(batch))
        return self._greedy(mem,mp,max_len) if beam_size<=1 else self._beam(mem,mp,max_len,beam_size)

    def _greedy(self, mem, mp, ml):
        B,dev = mem.size(0),mem.device
        seqs = torch.full((B,1),self.tokenizer.bos_id,dtype=torch.long,device=dev)
        done = torch.zeros(B,dtype=torch.bool,device=dev)
        for _ in range(ml):
            logits = self.decoder.forward_logits(seqs,mem,mp)
            nxt = logits[:,-1].argmax(-1); nxt[done]=self.tokenizer.pad_id
            seqs = torch.cat([seqs,nxt.unsqueeze(1)],1)
            done |= (nxt==self.tokenizer.eos_id)
            if done.all(): break
        return [self.tokenizer.decode(seqs[i].tolist()) for i in range(B)]

    def _beam(self, mem, mp, ml, bsz):
        B,dev = mem.size(0),mem.device; results = []
        for b in range(B):
            mb=mem[b:b+1].expand(bsz,-1,-1); pb=mp[b:b+1].expand(bsz,-1)
            seqs=torch.full((bsz,1),self.tokenizer.bos_id,dtype=torch.long,device=dev)
            sc=torch.zeros(bsz,device=dev); sc[1:]=-1e9
            best_sc,best_sq = -1e9, seqs[0]
            for step in range(ml):
                lp = F.log_softmax(self.decoder.forward_logits(seqs,mb,pb)[:,-1],-1)
                V=lp.size(-1); cand=(sc.unsqueeze(1)+lp).view(-1)
                ts,ti=cand.topk(bsz); bi,toki=ti//V,ti%V
                seqs=torch.cat([seqs[bi],toki.unsqueeze(1)],1); sc=ts
                for k in range(bsz):
                    if toki[k].item()==self.tokenizer.eos_id:
                        ns=sc[k].item()/((5+step+1)/6)**0.6
                        if ns>best_sc: best_sc,best_sq=ns,seqs[k]
                        sc[k]=-1e9
                if (sc<-1e8).all(): break
            results.append(self.tokenizer.decode(best_sq.tolist()))
        return results

    def count_params(self):
        t=sum(p.numel() for p in self.parameters())
        e=sum(p.numel() for p in self.encoder.parameters())
        d=sum(p.numel() for p in self.decoder.parameters())
        return {"total":t,"encoder":e,"decoder":d,"auxiliary":t-e-d}

# ═══════════════════════ EVALUATION ═══════════════════════

def _canon(smi):
    try:
        from rdkit import Chem
        m=Chem.MolFromSmiles(smi)
        if m: return Chem.MolToSmiles(m,canonical=True)
    except: pass
    return smi.strip()

def _tanimoto(s1,s2):
    try:
        from rdkit import Chem; from rdkit.Chem import AllChem,DataStructs
        m1,m2=Chem.MolFromSmiles(s1),Chem.MolFromSmiles(s2)
        if m1 and m2:
            f1=AllChem.GetMorganFingerprintAsBitVect(m1,2,2048)
            f2=AllChem.GetMorganFingerprintAsBitVect(m2,2,2048)
            return DataStructs.TanimotoSimilarity(f1,f2)
    except: pass
    a,b=set(s1),set(s2); return len(a&b)/max(len(a|b),1)

def evaluate_predictions(preds, targets):
    if not preds: return {"exact_match":0,"avg_tanimoto":0,"validity":0,"n_samples":0}
    n=len(preds); ex=ta=va=0
    for p,t in zip(preds,targets):
        if _canon(p)==_canon(t): ex+=1
        ta+=_tanimoto(p,t)
        try:
            from rdkit import Chem
            if Chem.MolFromSmiles(p) is not None: va+=1
        except: va += 1 if p else 0
    return {"exact_match":ex/n,"avg_tanimoto":ta/n,"validity":va/n,"n_samples":n}

def compute_tf_metrics(logits, labels, pad_id=0):
    B,T,V = logits.shape; mask=(labels!=pad_id); n=mask.sum().item()
    if n==0: return {"token_acc":0,"top5_acc":0,"seq_acc":0,"ppl":999}
    preds=logits.argmax(-1); correct=(preds==labels)&mask
    tok_acc=correct.sum().item()/n
    t5=logits.topk(min(5,V),-1).indices; t5h=(t5==labels.unsqueeze(-1)).any(-1)&mask
    seq_acc=(correct==mask).all(1).float().mean().item()
    ce=F.cross_entropy(logits.reshape(B*T,V),labels.reshape(B*T),ignore_index=pad_id)
    return {"token_acc":tok_acc,"top5_acc":t5h.sum().item()/n,"seq_acc":seq_acc,"ppl":min(float(torch.exp(ce)),10000)}

def split_dataset(ds, val_frac=0.10, test_frac=0.10, seed=42):
    """Legacy split for CSV-based datasets."""
    rng=random.Random(seed); n=len(ds); groups={}
    for i in range(n):
        k=_canon(ds.data["smiles_str"][i]); groups.setdefault(k,[]).append(i)
    keys=list(groups.keys()); rng.shuffle(keys)
    vt,tt=int(n*val_frac),int(n*test_frac); vi,ti,tri=[],[],[]
    for k in keys:
        ix=groups[k]
        if len(vi)<vt: vi.extend(ix)
        elif len(ti)<tt: ti.extend(ix)
        else: tri.extend(ix)
    return Subset(ds,tri),Subset(ds,vi),Subset(ds,ti)