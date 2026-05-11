
import argparse, csv, json, logging, os, re, sys, math

# ── Pre-parse GPU/CPU selection BEFORE importing torch ─────────────────
# This is essential: if torch is imported while another process has the
# default device pinned, we need CUDA_VISIBLE_DEVICES set before *any*
# torch import. Doing it inside main() is too late.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--gpu", type=str, default=None,
                  help="Which physical GPU to use, e.g. '2'. Overrides any "
                       "existing CUDA_VISIBLE_DEVICES. Use '' or 'cpu' for CPU.")
_pre.add_argument("--cpu", action="store_true",
                  help="Force CPU inference regardless of CUDA availability")
_pre_args, _ = _pre.parse_known_args()
if _pre_args.cpu or (_pre_args.gpu is not None and
                     _pre_args.gpu.lower() in ("", "cpu", "none")):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif _pre_args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_pre_args.gpu)
# ────────────────────────────────────────────────────────────────────────

import numpy as np
import torch
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k): return x   # no-op fallback

# Make sure prism_model / inference_v2 are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import (
    PRISM, SMILESTokenizer, SELFIESTokenizer, load_vocab, evaluate_predictions,
)
from inference import load_model, load_ensemble_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("PRISM-OOB")

# Suppress RDKit Morgan deprecation + parse-error noise
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

# ═══════════════════════════════════════════════════════════════════════
#  RAW NMR TEXT → STRUCTURED PEAKS
# ═══════════════════════════════════════════════════════════════════════
# Your preprocessing (preprocess.py) expects tuples that already went through
# parse_h1 / parse_c13. Those parsers consume lists-of-lists of the form:
#   H : [splitting, J_list, integ, ppm_start, ppm_end]
#   C : [ppm, splitting, J_list]
# The raw text from a paper looks like:
#   "δ 8.23 (dd, J = 5.2, 1.6 Hz, 1H), 7.39 (dd, ...), 1.94 – 1.81 (m, 2H), ..."
# so we just need to tokenize each peak and emit those tuples.

# A single peak entry inside the δ-list:
#   <ppm or ppm_range> <optional "(" details ")">
_PEAK_RE = re.compile(
    r"""
    (?P<ppm>
        \d+\.\d+ \s*[–\-−] \s*\d+\.\d+    # range like "1.94 – 1.81"
      | \d+\.\d+                           # single like "8.23"
    )
    \s*
    (?:\(\s*(?P<body>[^)]*)\))?            # optional "(dd, J=..., 1H)"
    """,
    re.VERBOSE,
)

_H_NMR_HEADER   = re.compile(r"1H\s*NMR\b",  re.IGNORECASE)
_C_NMR_HEADER   = re.compile(r"13C\s*NMR\b", re.IGNORECASE)
_DELTA_MARK     = re.compile(r"δ")

_SOLVENT_RE = re.compile(
    r"\(\s*\d+\s*MHz\s*,\s*([^)]+?)\s*\)", re.IGNORECASE)
_FREQ_RE    = re.compile(r"\(\s*(\d+)\s*MHz", re.IGNORECASE)


def _split_sections(text):
    """Split a full NMR block into 1H-section-text and 13C-section-text."""
    h_match = _H_NMR_HEADER.search(text)
    c_match = _C_NMR_HEADER.search(text)
    h_text = c_text = ""
    if h_match and c_match:
        if h_match.start() < c_match.start():
            h_text = text[h_match.start():c_match.start()]
            c_text = text[c_match.start():]
        else:
            c_text = text[c_match.start():h_match.start()]
            h_text = text[h_match.start():]
    elif h_match:
        h_text = text[h_match.start():]
    elif c_match:
        c_text = text[c_match.start():]
    # Cut off anything after a likely end-of-section marker (HRMS, etc.)
    end_mark = re.compile(r"\b(HRMS|MS\s*\(|IR|UV|HPLC|mp\s*\()", re.IGNORECASE)
    em = end_mark.search(c_text)
    if em:
        c_text = c_text[:em.start()]
    em = end_mark.search(h_text)
    if em:
        # don't cut if the HRMS falls inside the 13C block we're trimming
        pass
    return h_text, c_text


def _parse_J_text(body):
    """Pull J values out of a body like 'dd, J = 5.2, 1.6 Hz, 1H'."""
    js = []
    m = re.search(r"J\s*=\s*([0-9.,\s]+?)\s*Hz", body, re.IGNORECASE)
    if m:
        for tok in re.split(r"[,\s]+", m.group(1).strip()):
            if tok:
                try:
                    js.append(float(tok))
                except ValueError:
                    pass
    return js


def _parse_integ_text(body):
    """Pull the integration (e.g. '2H') out of a peak body. Default 1."""
    m = re.search(r"(\d+)\s*H\b", body)
    return int(m.group(1)) if m else 1


def _parse_splitting_text(body):
    """First token before ',' or 'J' is usually the multiplicity."""
    if not body:
        return "s"
    # take up to the first comma or 'J'
    head = re.split(r",|J\s*=", body, maxsplit=1)[0].strip()
    # strip any leading number (rare)
    head = re.sub(r"^\d+\s*", "", head).strip()
    return head or "s"


def parse_h_nmr_text(text):
    """
    Convert a raw 1H-NMR paragraph into the list-of-lists format that
    preprocess.parse_h1 expects, i.e.
        [[splitting, [J,J,...], integ, ppm_start, ppm_end], ...]
    """
    peaks = []
    # Start from δ if present (skip the "(400 MHz, CDCl3)" header)
    m = _DELTA_MARK.search(text)
    body_text = text[m.end():] if m else text
    for pm in _PEAK_RE.finditer(body_text):
        ppm_raw = pm.group("ppm")
        body    = (pm.group("body") or "").strip()
        # ppm range → start/end; single → identical
        if re.search(r"[–\-−]", ppm_raw):
            parts = re.split(r"[–\-−]", ppm_raw)
            ps, pe = float(parts[0]), float(parts[1])
        else:
            ps = pe = float(ppm_raw)
        split = _parse_splitting_text(body)
        js    = _parse_J_text(body)
        integ = _parse_integ_text(body)
        peaks.append([split, js, integ, ps, pe])
    return peaks


def parse_c_nmr_text(text):
    """
    Convert a raw 13C-NMR paragraph into the format parse_c13 expects:
        [[ppm, splitting_or_None, J_or_None], ...]
    Most reported 13C peaks are bare ppm numbers — splitting/J are usually None.
    """
    peaks = []
    m = _DELTA_MARK.search(text)
    body_text = text[m.end():] if m else text
    # bare numeric list: "162.7, 144.7, 138.18, ..."
    # also allow "162.7 (d, J=…)" etc.
    for pm in _PEAK_RE.finditer(body_text):
        ppm_raw = pm.group("ppm")
        body    = (pm.group("body") or "").strip()
        if re.search(r"[–\-−]", ppm_raw):
            parts = re.split(r"[–\-−]", ppm_raw)
            ppm = (float(parts[0]) + float(parts[1])) / 2
        else:
            ppm = float(ppm_raw)
        if body:
            split = _parse_splitting_text(body)
            js    = _parse_J_text(body)
            j_val = js if js else None
        else:
            split = None
            j_val = None
        peaks.append([ppm, split, j_val])
    return peaks


def parse_solvent(text):
    m = _SOLVENT_RE.search(text)
    return m.group(1).strip() if m else None


def parse_frequency(text):
    m = _FREQ_RE.search(text)
    return float(m.group(1)) if m else 0.0


# ═══════════════════════════════════════════════════════════════════════
#  BUILD INFERENCE TENSORS (mirrors preprocess.build_tensors)
# ═══════════════════════════════════════════════════════════════════════
# We re-use the *exact* parse functions and regex from preprocess.py so the
# splitting / J / integration normalization is bit-for-bit identical to what
# the model was trained on.

from preprocess import (
    parse_h1 as _preprocess_parse_h1,
    parse_c13 as _preprocess_parse_c13,
    normalize_solvent, normalize_splitting, smiles_to_formula,
    SMILES_REGEX as _SMILES_REGEX_PRE,
)


def build_oob_tensors(molecules, processed_dir, cfg):
    """
    molecules : list of dicts with keys
        h_nmr : str (raw text) OR pre-parsed list
        c_nmr : str (raw text) OR pre-parsed list
        solvent_h, solvent_c : str (optional)
        freq_h,    freq_c    : float (optional)

    Returns a dict of tensors in the same layout as preprocess.build_tensors.
    """
    # Load vocabs and norm stats
    vd = os.path.join(processed_dir, "vocab")
    with open(os.path.join(vd, "splitting_patterns.json")) as f:
        sp_json = json.load(f)
    split2id = {s: int(i) for s, i in sp_json["pattern2id"].items()}
    with open(os.path.join(vd, "solvents.json")) as f:
        sv_json = json.load(f)
    solv2id = {s: int(i) for s, i in sv_json["solvent2id"].items()}
    with open(os.path.join(vd, "atom_types.json")) as f:
        at_json = json.load(f)
    atom_types = at_json["atom_types"]
    with open(os.path.join(vd, "h_shift_regions.json")) as f:
        hr_json = json.load(f)
    h_regions = hr_json["regions"]
    with open(os.path.join(vd, "c_shift_regions.json")) as f:
        cr_json = json.load(f)
    c_regions = cr_json["regions"]
    with open(os.path.join(vd, "norm_stats.json")) as f:
        ns = json.load(f)
    with open(os.path.join(vd, "smiles_tokenizer.json")) as f:
        tk_json = json.load(f)
    tok2id = {k: int(v) for k, v in tk_json["token2id"].items()}

    dc = cfg.get("data", {})
    mxh = dc.get("max_h_peaks", 100)
    mxc = dc.get("max_c_peaks", 200)
    mxs = dc.get("max_smiles_len", 1000)

    hp_m, hp_s = ns["h_ppm"]["mean"], ns["h_ppm"]["std"]
    cp_m, cp_s = ns["c_ppm"]["mean"], ns["c_ppm"]["std"]
    hj_m, hj_s = ns["h_j"]["mean"],   ns["h_j"]["std"]
    cj_m, cj_s = ns["c_j"]["mean"],   ns["c_j"]["std"]
    fr_m, fr_s = ns["freq"]["mean"],  ns["freq"]["std"]
    fm_m, fm_s = ns["formula_mean"],  ns["formula_std"]
    other_id = split2id.get("other", len(split2id) - 1)
    s_default = split2id.get("s", 1)

    def h_reg(ppm):
        for i, r in enumerate(h_regions):
            if r["lower"] <= ppm < r["upper"]:
                return i
        return len(h_regions) - 1

    def c_reg(ppm):
        for i, r in enumerate(c_regions):
            if r["lower"] <= ppm < r["upper"]:
                return i
        return len(c_regions) - 1

    N = len(molecules)
    D = {
        "h_ppm":       np.zeros((N, mxh), np.float32),
        "h_ppm_raw":   np.zeros((N, mxh), np.float32),
        "h_split":     np.zeros((N, mxh), np.int64),
        "h_integ":     np.zeros((N, mxh), np.int64),
        "h_j":         np.zeros((N, mxh), np.float32),
        "h_mask":      np.zeros((N, mxh), np.bool_),
        "h_region":    np.zeros((N, mxh), np.int64),
        "c_ppm":       np.zeros((N, mxc), np.float32),
        "c_ppm_raw":   np.zeros((N, mxc), np.float32),
        "c_split":     np.zeros((N, mxc), np.int64),
        "c_j":         np.zeros((N, mxc), np.float32),
        "c_mask":      np.zeros((N, mxc), np.bool_),
        "c_region":    np.zeros((N, mxc), np.int64),
        "smiles_ids":  np.zeros((N, mxs), np.int64),
        "smiles_len":  np.zeros((N,),      np.int64),
        "mol_formula": np.zeros((N, len(atom_types)), np.float32),
        "freq_h":      np.zeros((N,), np.float32),
        "freq_c":      np.zeros((N,), np.float32),
        "solvent_h":   np.zeros((N,), np.int64),
        "solvent_c":   np.zeros((N,), np.int64),
    }
    smiles_str_list = []

    for idx, mol in enumerate(molecules):
        # ── 1H ──
        h_input = mol.get("h_nmr", "")
        if isinstance(h_input, str):
            # Two sub-cases for string cells:
            #   (a) already a repr'd list-of-tuples from a training-style CSV
            #       (starts with '[(' or '[[')  → pass straight to the
            #       preprocessing parser, don't re-parse as publication text.
            #   (b) publication-style NMR text with 'δ' markers
            #       → run our raw-text parser first.
            s = h_input.lstrip()
            if s.startswith("[(") or s.startswith("[["):
                h_peaks = _preprocess_parse_h1(h_input)
            else:
                h_list = parse_h_nmr_text(h_input)
                h_peaks = _preprocess_parse_h1(repr(h_list))
        else:
            # Already a Python list object
            h_peaks = _preprocess_parse_h1(repr(h_input))
        for i, (ppm, sp, integ, jm) in enumerate(h_peaks[:mxh]):
            D["h_ppm_raw"][idx, i] = ppm
            D["h_ppm"][idx, i]     = (ppm - hp_m) / hp_s
            D["h_split"][idx, i]   = split2id.get(sp, other_id) if sp else s_default
            D["h_integ"][idx, i]   = integ
            D["h_j"][idx, i]       = (jm - hj_m) / hj_s
            D["h_mask"][idx, i]    = True
            D["h_region"][idx, i]  = h_reg(ppm)
        if idx < 5:   # only log for the first few rows to avoid spam on big CSVs
            log.info(f"  mol[{idx}] parsed {len(h_peaks)} 1H peaks")

        # ── 13C ──
        c_input = mol.get("c_nmr", "")
        if isinstance(c_input, str):
            s = c_input.lstrip()
            if s.startswith("[(") or s.startswith("[["):
                c_peaks = _preprocess_parse_c13(c_input)
            else:
                c_list = parse_c_nmr_text(c_input)
                c_peaks = _preprocess_parse_c13(repr(c_list))
        else:
            c_peaks = _preprocess_parse_c13(repr(c_input))
        for i, (ppm, sp, jv) in enumerate(c_peaks[:mxc]):
            D["c_ppm_raw"][idx, i] = ppm
            D["c_ppm"][idx, i]     = (ppm - cp_m) / cp_s
            D["c_split"][idx, i]   = split2id.get(sp, other_id) if sp else s_default
            D["c_j"][idx, i]       = (jv - cj_m) / cj_s
            D["c_mask"][idx, i]    = True
            D["c_region"][idx, i]  = c_reg(ppm)
        if idx < 5:
            log.info(f"  mol[{idx}] parsed {len(c_peaks)} 13C peaks")

        # ── Placeholder SMILES (not used at inference, but shape required) ──
        smi = mol.get("smiles", "")
        smiles_str_list.append(smi)
        toks = _SMILES_REGEX_PRE.findall(smi)[:mxs - 2]
        ids  = [1] + [tok2id.get(t, 3) for t in toks] + [2]
        D["smiles_ids"][idx, :len(ids)] = ids
        D["smiles_len"][idx] = len(ids)

        # ── Molecular formula ──
        # If user passed a SMILES (e.g. ground truth) we use its formula;
        # otherwise we zero it out (model was trained w/ formula encoder off
        # in your config, so this is safe). The formula encoder is disabled
        # anyway: modules.use_molecular_formula_encoder = false in config.yaml.
        if smi:
            fc = smiles_to_formula(smi)
            for j, a in enumerate(atom_types):
                D["mol_formula"][idx, j] = (fc.get(a, 0) - fm_m[j]) / fm_s[j]
        # else leave as zeros → normalized zero-vector

        # ── Frequency / solvent ──
        fh  = float(mol.get("freq_h",  0.0) or 0.0)
        fcv = float(mol.get("freq_c",  0.0) or 0.0)
        D["freq_h"][idx] = (fh  - fr_m) / fr_s if fh  > 0 else 0.0
        D["freq_c"][idx] = (fcv - fr_m) / fr_s if fcv > 0 else 0.0
        sh  = normalize_solvent(mol.get("solvent_h", "") or "")
        scv = normalize_solvent(mol.get("solvent_c", "") or "")
        D["solvent_h"][idx] = solv2id.get(sh,  solv2id.get("other", 0)) if sh  else 0
        D["solvent_c"][idx] = solv2id.get(scv, solv2id.get("other", 0)) if scv else 0

    T = {k: torch.from_numpy(v) for k, v in D.items()}
    T["smiles_str"] = smiles_str_list
    return T


# ═══════════════════════════════════════════════════════════════════════
#  BEAM GENERATION — return ALL beam candidates, not just the best
# ═══════════════════════════════════════════════════════════════════════
# prism_model.generate() returns only the top-1 beam. We want the full
# beam so we can keep any candidate that matches ground truth. Easiest
# way: call generate() in beam mode multiple times with different beam
# widths is *not* correct (same top-1). Instead, we piggyback on the fact
# that v2 inference just calls `model.generate`, and your model's
# _beam_cached returns one SMILES per example. To get N candidates per
# molecule we duplicate the batch N times (batch expansion).
#
# Simpler and cleaner: run greedy + beam=K, but also run multiple
# smaller beams. In practice, you want the *top-K* per-example from one
# beam search. So we monkey-patch `_beam_cached` to also stash all final
# beams into a sidecar attribute.

import types

def generate_all_beams(model, batch, max_len, beam_size, use_amp=True,
                       length_penalty=0.0, verbose=True,
                       encoder_chunk=128):
    """
    Run the encoder in chunks of `encoder_chunk` molecules, then do beam
    search ONE molecule at a time. Chunking the encoder keeps peak
    memory bounded on 10k-row CSVs; beam-searching per-molecule keeps
    the decoder footprint at O(K * T * d).

    Returns: list[ list[ (score, smiles_str) ] ] of length B, where the
    inner lists are ordered best→worst.
    """
    model.eval()
    dev = next(model.parameters()).device
    amp_enabled = use_amp and dev.type == "cuda"

    def _amp():
        return torch.amp.autocast("cuda", enabled=amp_enabled)

    B = batch["h_ppm"].size(0)

    # ── Encode in chunks to bound memory on very large inputs ──
    mems, mps = [], []
    n_chunks = (B + encoder_chunk - 1) // encoder_chunk
    enc_iter = range(n_chunks)
    if verbose and n_chunks > 1:
        enc_iter = tqdm(enc_iter, desc="Encoding", unit="chunk")
    for c in enc_iter:
        s, e = c * encoder_chunk, min((c + 1) * encoder_chunk, B)
        sub = {k: (v[s:e] if isinstance(v, torch.Tensor) else v[s:e])
               for k, v in batch.items()}
        with torch.no_grad(), _amp():
            mem_c, mp_c = model.encoder(
                sub["h_ppm"],   sub["h_split"], sub["h_integ"],
                sub["h_j"],     sub["h_mask"],
                sub["c_ppm"],   sub["c_split"], sub["c_j"],
                sub["c_mask"],
                **model._enc_kwargs(sub),
            )
        mems.append(mem_c); mps.append(mp_c)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    mem = torch.cat(mems, dim=0)
    mp  = torch.cat(mps,  dim=0)
    del mems, mps

    # ── Beam search, one molecule at a time ──
    out = []
    beam_iter = range(B)
    if verbose:
        beam_iter = tqdm(beam_iter, desc="Beam search", unit="mol")
    for i in beam_iter:
        mem_i = mem[i:i + 1]        # [1, S, D]
        mp_i  = mp[i:i + 1]         # [1, S]
        with torch.no_grad(), _amp():
            cand = _beam_with_all_candidates(
                model, mem_i, mp_i, max_len, beam_size, length_penalty)
        out.append(cand[0])
        if dev.type == "cuda" and (i % 50 == 49):
            torch.cuda.empty_cache()
    return out


def _beam_with_all_candidates(model, mem, mp, max_len, beam_size,
                              length_penalty=0.0):
    """
    Plain-python beam search that records every final candidate per
    example. Called with a batch of ONE example (B=1) to minimize peak
    memory; for a full batch just loop in the caller.

    Memory note: we do NOT cache keys/values across steps (that would
    cut decoder time by ~10x on long sequences), because wiring up
    KV-cache into the upstream TransformerDecoder requires reaching
    into private PyTorch state. For 2 molecules × 10 beams × ≤550 tokens
    this simple version finishes in seconds.

    Returns: list of length B=1, each inner list is [(score, ids), ...]
    sorted best first (up to `beam_size` entries).
    """
    import torch.nn.functional as F
    tok = model.tokenizer
    B, S, D = mem.shape
    assert B == 1, "call one example at a time"
    dev = mem.device
    dec = model.decoder
    K = beam_size

    # Expand memory so it's shared across K beams
    mem_e = mem.expand(K, S, D).contiguous()   # [K, S, D]
    mp_e  = mp.expand(K, S).contiguous()       # [K, S]

    seqs   = torch.full((K, 1), tok.bos_id, dtype=torch.long, device=dev)
    scores = torch.zeros(K, device=dev)
    # Step 0: only beam 0 is active so we don't get K duplicate top-Ks
    scores[1:] = float("-inf")
    done = torch.zeros(K, dtype=torch.bool, device=dev)

    finished = []           # list of (score, ids) for this example

    for step in range(max_len):
        logits = dec.forward_logits(seqs, mem_e, mp_e)     # [K, T, V]
        log_probs = F.log_softmax(logits[:, -1].float(), dim=-1)   # [K, V]
        V = log_probs.size(-1)

        # Finished beams emit <pad> with 0 extra log-prob
        if done.any():
            log_probs[done] = float("-inf")
            log_probs[done, tok.pad_id] = 0.0

        cand = scores.unsqueeze(1) + log_probs             # [K, V]
        flat = cand.view(-1)                               # [K*V]
        top_scores, top_idx = flat.topk(K)                 # [K]
        beam_idx = torch.div(top_idx, V, rounding_mode="floor")
        tok_idx  = top_idx - beam_idx * V

        # Reorder state by chosen beam origins
        seqs  = seqs[beam_idx]
        done  = done[beam_idx]
        seqs  = torch.cat([seqs, tok_idx.unsqueeze(1)], dim=1)
        scores = top_scores

        # Newly finished
        just_eos = (tok_idx == tok.eos_id) & (~done)
        for k in just_eos.nonzero(as_tuple=False).view(-1).tolist():
            ids = seqs[k].tolist()
            raw = scores[k].item()
            finished.append((raw, ids))
        done = done | (tok_idx == tok.eos_id)

        if done.all():
            break

    # Flush un-terminated beams
    for k in (~done).nonzero(as_tuple=False).view(-1).tolist():
        finished.append((scores[k].item(), seqs[k].tolist()))

    # Rank (optionally with length penalty) and deduplicate SMILES
    def _rank_score(score, ids):
        if length_penalty <= 0:
            return score
        L = max(1, len(ids) - 1)     # minus BOS
        return score / ((5 + L) / 6.0) ** length_penalty
    finished.sort(key=lambda x: -_rank_score(*x))

    seen, dedup = set(), []
    for score, ids in finished:
        s = tok.decode(ids)
        if s in seen:
            continue
        seen.add(s)
        dedup.append((score, s))
        if len(dedup) >= K:
            break
    return [dedup]


# ═══════════════════════════════════════════════════════════════════════
#  SMILES / SELFIES POST-PROCESSING + CORRECTNESS CHECK
# ═══════════════════════════════════════════════════════════════════════

def _canon(smi, strip_stereo=None):
    """Canonicalize a SMILES. If strip_stereo=True, @/@@/slashes are removed
    before canonicalization — use this when the model was trained on
    stereo-free SMILES (WO_Stereo / WO_Isomeric datasets).

    When strip_stereo is None (default), falls back to the module-level
    _STRIP_STEREO toggle, set from CLI via --strip_stereo / --stereo_strict."""
    if strip_stereo is None:
        strip_stereo = _STRIP_STEREO
    try:
        from rdkit import Chem
        if not smi:
            return ""
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return smi.strip()
        return Chem.MolToSmiles(m, canonical=True,
                                 isomericSmiles=not strip_stereo)
    except Exception:
        return (smi or "").strip()


# Module-level default: keep stereo (matches pre-existing behavior).
# main() flips this to True when --strip_stereo is passed (the norm for
# models trained on WO_Stereo datasets).
_STRIP_STEREO = False


def _selfies_to_smiles(s):
    try:
        import selfies as sf
        return sf.decoder(s)
    except Exception:
        return s


def _tanimoto(s1, s2, use_chirality=False):
    """Morgan (ECFP) Tanimoto similarity, radius 2, 2048 bits.

    use_chirality=False (default): 2D similarity. Stereo information is
        discarded during fingerprint hashing. Matches the training-time
        implementation in prism_model._tanimoto so inference numbers
        are directly comparable to training/validation metrics.
    use_chirality=True: stereo-aware similarity. Tetrahedral chirality
        and double-bond geometry are folded into the bit hashing, so
        two molecules differing only by a stereocenter flip will yield
        Tanimoto < 1.0.

    Falls back to a character-set Jaccard if either SMILES fails to parse.
    """
    if not s1 or not s2:
        return 0.0
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 and m2:
            f1 = AllChem.GetMorganFingerprintAsBitVect(
                m1, 2, 2048, useChirality=use_chirality)
            f2 = AllChem.GetMorganFingerprintAsBitVect(
                m2, 2, 2048, useChirality=use_chirality)
            return float(DataStructs.TanimotoSimilarity(f1, f2))
    except Exception:
        pass
    # Character-set fallback if RDKit isn't happy with one of the SMILES
    a, b = set(s1), set(s2)
    return len(a & b) / max(len(a | b), 1)


def _tanimoto_both(s1, s2):
    """Compute both 2D and stereo-aware Morgan Tanimoto in one go.
    Returns (tani_2d, tani_iso). Only one RDKit parse per SMILES — the
    fingerprint generation is the cheap part. Used internally so the
    rest of the pipeline always has both values available regardless
    of which the user asked us to display.
    """
    if not s1 or not s2:
        return 0.0, 0.0
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
        if m1 and m2:
            f1_2d  = AllChem.GetMorganFingerprintAsBitVect(m1, 2, 2048)
            f2_2d  = AllChem.GetMorganFingerprintAsBitVect(m2, 2, 2048)
            f1_iso = AllChem.GetMorganFingerprintAsBitVect(
                m1, 2, 2048, useChirality=True)
            f2_iso = AllChem.GetMorganFingerprintAsBitVect(
                m2, 2, 2048, useChirality=True)
            return (
                float(DataStructs.TanimotoSimilarity(f1_2d, f2_2d)),
                float(DataStructs.TanimotoSimilarity(f1_iso, f2_iso)),
            )
    except Exception:
        pass
    a, b = set(s1), set(s2)
    j = len(a & b) / max(len(a | b), 1)
    return j, j


def is_correct(pred, truth, is_selfies=False, strip_stereo=None):
    if not truth:
        return None  # unknown
    p = _selfies_to_smiles(pred) if is_selfies else pred
    return _canon(p, strip_stereo) == _canon(truth, strip_stereo)


# ═══════════════════════════════════════════════════════════════════════
#  PREDICTION-LEVEL ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════
# Strategy: load each top-K checkpoint one at a time (so K models don't
# sit in GPU memory simultaneously), run beam search with each, then
# merge the per-molecule candidate sets. Merging rules:
#   - Dedupe by canonical isomeric SMILES.
#   - For each unique candidate, track (a) max log-prob across models,
#     (b) count of models whose beam produced it ("vote" count).
#   - Final rank = primary by vote count (descending), secondary by
#     max-score. This matches the intuition "if 4 of 5 models agree,
#     that candidate is more trustworthy than one that appeared once
#     with a slightly higher score."
# The resulting merged beam acts as a drop-in replacement for the
# single-model beam; formula reranking applies on top exactly as before.

def list_ensemble_checkpoints(checkpoint_dir, top_k):
    """Resolve the K checkpoint paths for prediction-level ensembling.
    Accepts either top_k_manifest.json or top_k_ep*.pt files."""
    import glob
    manifest_path = os.path.join(checkpoint_dir, "top_k_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        return [e["path"] for e in manifest[:top_k]]
    paths = sorted(glob.glob(os.path.join(checkpoint_dir, "top_k_ep*.pt")))
    if not paths:
        raise FileNotFoundError(
            f"No ensemble checkpoints in {checkpoint_dir}")
    return paths[:top_k]


def run_prediction_ensemble(checkpoint_paths, batch, processed_dir,
                             max_len, beam_size, use_amp, length_penalty,
                             encoder_chunk, device):
    """
    Run beam search independently with each checkpoint and return
    merged beams. Returns the SAME shape as generate_all_beams():
    list[ list[ (score, pred_str) ] ] with length = B molecules.

    The score for the merged candidate is `max_logprob - 0.01 * (K - votes)`
    where K is the number of models in the ensemble — this preserves
    log-prob as the dominant score but gently downranks one-off beams.
    The merged list is truncated to `beam_size` per molecule.
    """
    from collections import defaultdict
    # Per-molecule, per-canon_iso: (max_score, vote_count, best_raw_pred)
    merged = defaultdict(lambda: defaultdict(lambda: [-1e9, 0, None]))

    K = len(checkpoint_paths)
    for k_idx, ckpt_path in enumerate(checkpoint_paths):
        log.info(f"Ensemble model {k_idx + 1}/{K}: {ckpt_path}")
        model, tok, cfg = load_model(ckpt_path, device, processed_dir)
        is_self = isinstance(tok, SELFIESTokenizer)
        try:
            per_model_beams = generate_all_beams(
                model, batch, max_len, beam_size,
                use_amp=use_amp, length_penalty=length_penalty,
                encoder_chunk=encoder_chunk,
            )
        finally:
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Accumulate into merged bag of candidates per molecule
        for i, beams in enumerate(per_model_beams):
            for score, pred in beams:
                smi = _selfies_to_smiles(pred) if is_self else pred
                key = _canon(smi, strip_stereo=False) or smi   # dedupe on iso
                entry = merged[i][key]
                if score > entry[0]:
                    entry[0] = score
                    entry[2] = pred       # store the raw pred with highest score
                entry[1] += 1             # vote count (each model contributes ≤1)
            log.info(f"  mol {i + 1}/{len(per_model_beams)} aggregated "
                     f"({len(merged[i])} unique so far)") if (i % max(1, len(per_model_beams)//10) == 0) else None

    # Final rank per molecule: by votes desc, then max-score desc
    out = [None] * len(merged)
    # Note: merged keys are 0..B-1 in insertion order but dict ordering
    # is insertion-ordered in py3.7+, so iterate by index
    B = max(merged.keys()) + 1 if merged else 0
    for i in range(B):
        bag = merged.get(i, {})
        items = [(score, pred, votes) for key, (score, votes, pred) in bag.items()]
        items.sort(key=lambda t: (-t[2], -t[0]))    # votes desc, score desc
        # Package as (score, pred) with an ensemble-adjusted score
        scored = []
        for score, pred, votes in items[:beam_size]:
            adj_score = score - 0.01 * (K - votes)
            scored.append((adj_score, pred))
        out[i] = scored
    # Empty molecules (shouldn't happen but guard anyway)
    out = [b if b is not None else [] for b in out]
    log.info(f"Prediction-level ensemble: merged {K} models into "
             f"{sum(len(b) for b in out)} total candidates across "
             f"{len(out)} molecules.")
    return out


# ═══════════════════════════════════════════════════════════════════════
#  DUAL-STEREO EVALUATION + FORMULA-CONSTRAINED RERANKING
# ═══════════════════════════════════════════════════════════════════════
# Instead of picking ONE canonicalization mode we always compute both —
# isomeric (keeps @/@@ and /\) and non-isomeric (2D only). The output CSV
# + HTML show both correctness flags so the user can see how much of the
# gap is stereo vs. connectivity.

def _canon_both(smi):
    """Return (canon_with_stereo, canon_2d_only) for a SMILES.
    Returns ("","") if the SMILES is empty or invalid."""
    iso   = _canon(smi, strip_stereo=False)
    flat  = _canon(smi, strip_stereo=True)
    return iso, flat


def _formula_key(smi_or_canon):
    """Hill-order formula string used as a comparison key for reranking.
    Returns '' if RDKit can't parse the SMILES (those candidates never
    match a ground-truth formula, so they sink in the reranked list)."""
    if not smi_or_canon:
        return ""
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMolDescriptors
        m = Chem.MolFromSmiles(smi_or_canon)
        if m is None:
            return ""
        return rdMolDescriptors.CalcMolFormula(m)
    except Exception:
        return ""


def rerank_by_formula(beams, target_formula, is_selfies=False,
                      heavy_only=False):
    """
    Stable re-sort of beams so candidates whose molecular formula matches
    `target_formula` come first. Within each group, original beam-score
    order is preserved.

    Parameters
    ----------
    beams : list of (score, pred_str)
    target_formula : str      Hill-order formula, e.g. 'C15H19N'. If '',
                              return beams unchanged.
    is_selfies : bool         decode SELFIES → SMILES before formula check
    heavy_only : bool         compare formulas excluding hydrogen counts.
                              Useful when the model predicts explicit-H
                              SMILES differently from the canonical target.

    Returns
    -------
    new_beams : list of (score, pred_str), reordered
    matches   : list of bool, same length, aligned to new_beams
    """
    if not target_formula or not beams:
        return beams, [False] * len(beams)

    def _strip_H(formula):
        # "C15H19N" → "C15N" ; "CH4O" → "CO"
        return re.sub(r"H\d*", "", formula)

    target = _strip_H(target_formula) if heavy_only else target_formula

    matches = []
    keyed = []
    for idx, (score, pred) in enumerate(beams):
        smi = _selfies_to_smiles(pred) if is_selfies else pred
        f = _formula_key(smi)
        f_cmp = _strip_H(f) if heavy_only else f
        is_match = bool(f) and f_cmp == target
        matches.append(is_match)
        # Sort key: (no-match first → True=1, match=0) then orig idx (stable)
        keyed.append(((0 if is_match else 1), idx, (score, pred)))

    keyed.sort(key=lambda t: (t[0], t[1]))
    new_beams  = [t[2] for t in keyed]
    new_matches = [t[0] == 0 for t in keyed]
    return new_beams, new_matches


# ═══════════════════════════════════════════════════════════════════════
#  RDKIT VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════
# For each input molecule we emit:
#   oob_report.html            — one scrollable HTML report (self-contained,
#                                SVG embedded inline, easy to share)
#   oob_mol{i}_top{K}.png      — optional grid image per molecule
#
# The HTML report is the main deliverable because:
#   - It embeds SVG (crisp at any zoom, no font/rasterization issues)
#   - It's one file, no external assets
#   - Correct hits are highlighted green, invalid SMILES marked, ground
#     truth rendered next to the beam candidates for direct comparison

def _mol_to_svg(smi, size=300):
    """Return an inline <svg> string for a SMILES, or an 'invalid' placeholder."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
    except ImportError:
        return (f'<div style="width:{size}px;height:{size}px;'
                f'display:flex;align-items:center;justify-content:center;'
                f'border:1px dashed #999;color:#666;font-size:12px;">'
                f'RDKit not installed</div>')
    if not smi or not smi.strip():
        return _invalid_svg(size, "empty")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return _invalid_svg(size, "invalid SMILES")
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(size, size)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.bondLineWidth = 2
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Strip the <?xml ...?> declaration so the SVG can be inlined in HTML
        svg = re.sub(r"<\?xml[^?]+\?>\s*", "", svg)
        # Make the SVG responsive
        svg = svg.replace(
            '<svg ',
            '<svg style="max-width:100%;height:auto;" ',
            1,
        )
        return svg
    except Exception as e:
        return _invalid_svg(size, f"draw error: {e}")


def _invalid_svg(size, reason):
    return (f'<div style="width:{size}px;height:{size}px;'
            f'display:flex;align-items:center;justify-content:center;'
            f'border:1px dashed #c00;color:#c00;font-size:12px;'
            f'background:#fff5f5;">{reason}</div>')


def _card(rank, score, smi_raw, canon_pred, gt_canon, size=280):
    """Render one prediction card (SVG + metadata) for the HTML report."""
    valid = False
    try:
        from rdkit import Chem
        valid = Chem.MolFromSmiles(smi_raw) is not None
    except ImportError:
        pass
    is_hit = bool(gt_canon) and canon_pred == gt_canon
    border = ("3px solid #16a34a" if is_hit
              else ("1px solid #d1d5db" if valid else "2px solid #dc2626"))
    bg = "#f0fdf4" if is_hit else ("#fff" if valid else "#fef2f2")
    tag = ""
    if is_hit:
        tag = ('<span style="background:#16a34a;color:#fff;padding:2px 8px;'
               'border-radius:4px;font-size:11px;font-weight:600;">'
               '✓ CORRECT</span>')
    elif not valid:
        tag = ('<span style="background:#dc2626;color:#fff;padding:2px 8px;'
               'border-radius:4px;font-size:11px;font-weight:600;">'
               'INVALID</span>')
    svg = _mol_to_svg(smi_raw, size=size)
    return f'''
    <div style="border:{border};background:{bg};border-radius:8px;padding:12px;
                margin:6px;width:{size + 30}px;display:inline-block;vertical-align:top;">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:6px;">
        <span style="font-weight:700;font-size:14px;color:#111;">
          Rank #{rank}
        </span>
        <span style="font-size:11px;color:#555;font-variant-numeric:tabular-nums;">
          score: {score:.3f}
        </span>
      </div>
      <div style="margin-bottom:6px;">{tag}</div>
      <div style="text-align:center;margin-bottom:8px;">{svg}</div>
      <div style="font-family:'SFMono-Regular',Menlo,Consolas,monospace;
                  font-size:11px;word-break:break-all;color:#333;
                  background:#f9fafb;padding:4px 6px;border-radius:4px;">
        {smi_raw if smi_raw else '<em>(empty)</em>'}
      </div>
    </div>
    '''


def _ground_truth_panel(gt, size=320):
    if not gt:
        return ""
    svg = _mol_to_svg(gt, size=size)
    return f'''
    <div style="border:2px solid #2563eb;background:#eff6ff;border-radius:8px;
                padding:14px;margin:6px 0 16px 0;display:inline-block;">
      <div style="font-weight:700;color:#1e40af;margin-bottom:6px;font-size:14px;">
        Ground truth
      </div>
      <div style="text-align:center;">{svg}</div>
      <div style="font-family:monospace;font-size:11px;margin-top:6px;
                  color:#1e3a8a;word-break:break-all;max-width:{size}px;">
        {gt}<br>
        <span style="color:#6b7280;">(canonical: {_canon(gt)})</span>
      </div>
    </div>
    '''


def render_html_report(all_beams, gts, is_selfies, args, output_path):
    """Write a single self-contained HTML file with all molecules + beams."""
    parts = []
    parts.append('''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>PRISM OOB Predictions</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #f3f4f6; margin: 0; padding: 24px; color: #111; }
  h1 { margin: 0 0 8px 0; font-size: 22px; }
  h2 { margin: 24px 0 4px 0; font-size: 18px; color: #111;
       padding-bottom: 6px; border-bottom: 2px solid #e5e7eb; }
  .meta { color: #6b7280; font-size: 13px; margin-bottom: 8px; }
  .mol-section { background: #fff; padding: 18px; border-radius: 10px;
                 box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 24px; }
  .summary { background: #fffbeb; border: 1px solid #fcd34d;
             padding: 10px 14px; border-radius: 8px; margin-bottom: 20px;
             font-size: 13px; color: #78350f; }
  details { margin-top: 10px; }
  summary { cursor: pointer; font-weight: 600; color: #374151; }
</style>
</head>
<body>''')
    parts.append(f'<h1>PRISM OOB Predictions</h1>')
    stereo_note = ("stereo stripped" if _STRIP_STEREO
                   else "stereo retained")
    parts.append(f'<div class="meta">'
                 f'beam_size={args.beam_size} · molecules={len(all_beams)} · '
                 f'{"ensemble" if args.checkpoint_dir else "single"} model · '
                 f'{stereo_note}'
                 f'</div>')

    # Top-level summary (only if we have any ground truth)
    if any(gts):
        n_gt = sum(1 for g in gts if g)
        top1_hit = 0
        inbeam_hit = 0
        for i, beams in enumerate(all_beams):
            if not gts[i] or not beams:
                continue
            gt_c = _canon(gts[i])
            top_smi = _selfies_to_smiles(beams[0][1]) if is_selfies else beams[0][1]
            if _canon(top_smi) == gt_c:
                top1_hit += 1
            for _, p in beams:
                smi = _selfies_to_smiles(p) if is_selfies else p
                if _canon(smi) == gt_c:
                    inbeam_hit += 1
                    break
        parts.append(f'<div class="summary">'
                     f'<strong>Top-1 exact match:</strong> {top1_hit}/{n_gt} &nbsp;·&nbsp; '
                     f'<strong>In-beam (top-{args.beam_size}):</strong> {inbeam_hit}/{n_gt}'
                     f'</div>')

    for i, beams in enumerate(all_beams):
        gt = gts[i]
        gt_c = _canon(gt) if gt else ""
        parts.append(f'<div class="mol-section">')
        parts.append(f'<h2>Molecule {i}</h2>')

        if gt:
            parts.append(_ground_truth_panel(gt))

        # Decode every beam candidate
        cards = []
        for rank, (score, pred) in enumerate(beams, start=1):
            smi = _selfies_to_smiles(pred) if is_selfies else pred
            canon_pred = _canon(smi)
            cards.append(_card(rank, score, smi, canon_pred, gt_c))
        parts.append('<div>' + ''.join(cards) + '</div>')
        parts.append('</div>')

    parts.append('</body></html>')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    log.info(f"HTML report saved to {output_path}")


def render_png_grids(all_beams, gts, is_selfies, args, output_dir):
    """Save one PNG grid per molecule using RDKit's MolsToGridImage."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        log.warning("RDKit not available — skipping PNG grid rendering")
        return

    os.makedirs(output_dir, exist_ok=True)
    for i, beams in enumerate(all_beams):
        mols = []
        legends = []
        gt_c = _canon(gts[i]) if gts[i] else ""

        # prepend ground-truth cell when available
        if gts[i]:
            gt_mol = Chem.MolFromSmiles(gts[i])
            mols.append(gt_mol if gt_mol is not None else Chem.MolFromSmiles("*"))
            legends.append(f"GT: {gts[i][:28]}")

        for rank, (score, pred) in enumerate(beams, start=1):
            smi = _selfies_to_smiles(pred) if is_selfies else pred
            m = Chem.MolFromSmiles(smi)
            if m is None:
                # placeholder atom so the grid cell stays aligned
                m = Chem.MolFromSmiles("*")
                tag = "INVALID"
            else:
                tag = "✓" if (gt_c and _canon(smi) == gt_c) else ""
            legends.append(f"#{rank} s={score:.2f} {tag}\n{smi[:28]}")
            mols.append(m)

        n_per_row = 4
        img = Draw.MolsToGridImage(
            mols, molsPerRow=n_per_row, subImgSize=(260, 260),
            legends=legends, useSVG=False,
        )
        out_path = os.path.join(output_dir, f"oob_mol{i}_top{len(beams)}.png")
        try:
            img.save(out_path)
            log.info(f"  PNG saved: {out_path}")
        except Exception as e:
            # Some environments return a PIL image, others a bytes-like object
            with open(out_path, "wb") as f:
                f.write(img.data if hasattr(img, "data") else img)
            log.info(f"  PNG saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════
#  INPUT PARSING (CLI strings → molecule dicts)
# ═══════════════════════════════════════════════════════════════════════

def split_raw_nmr_block(block):
    """Given one combined string of '1H NMR (...) δ ... 13C NMR (...) δ ...',
    return (h_text, c_text, solvent_h, solvent_c, freq_h, freq_c)."""
    h_text, c_text = _split_sections(block)
    sh = parse_solvent(h_text) if h_text else None
    sc = parse_solvent(c_text) if c_text else None
    fh = parse_frequency(h_text) if h_text else 0.0
    fc = parse_frequency(c_text) if c_text else 0.0
    return {
        "h_nmr": h_text,
        "c_nmr": c_text,
        "solvent_h": sh,
        "solvent_c": sc,
        "freq_h": fh,
        "freq_c": fc,
    }


def load_nmr_file(path):
    with open(path) as f:
        txt = f.read()
    blocks = [b.strip() for b in re.split(r"^\s*-{3,}\s*$", txt, flags=re.M) if b.strip()]
    return [split_raw_nmr_block(b) for b in blocks]


# ═══════════════════════════════════════════════════════════════════════
#  CSV INPUT MODE
# ═══════════════════════════════════════════════════════════════════════
# Two supported CSV layouts, auto-detected per-row:
#
# A) TRAINING-STYLE (same columns as the preprocessing pipeline used):
#    Required: NMR_processed_1H, NMR_processed_13C  — each cell is a
#    repr()'d nested list, e.g. "[['dd',[5.2,1.6],1,8.23,8.23], ...]"
#    Optional: NMR_frequency_1H, NMR_frequency_13C, NMR_solvent_1H,
#              NMR_solvent_13C, canon_smiles (or --smiles_col)
#
# B) RAW-TEXT:
#    At minimum one of: h_nmr_text, c_nmr_text, nmr_text (combined block)
#    Optional: solvent_h, solvent_c, freq_h, freq_c, smiles
#
# Column names are configurable via CLI. The loader figures out which
# format a row is in by checking whether the 1H cell starts with "[[".

def _looks_like_parsed_list(cell):
    """Is this cell the repr() of a nested Python list (training format)?"""
    if not cell:
        return False
    s = str(cell).strip()
    return s.startswith("[[") or s.startswith("[(")


def load_nmr_csv(path, smiles_col="canon_smiles",
                  h_parsed_col="NMR_processed_1H",
                  c_parsed_col="NMR_processed_13C",
                  h_text_col="h_nmr_text",
                  c_text_col="c_nmr_text",
                  combined_text_col="nmr_text",
                  freq_h_col="NMR_frequency_1H",
                  freq_c_col="NMR_frequency_13C",
                  solv_h_col="NMR_solvent_1H",
                  solv_c_col="NMR_solvent_13C",
                  id_col=None,
                  formula_col=None):
    """
    Parse a CSV of molecules, one per row. Returns (mols, gts, row_ids, formulas_csv).

    - mols:          list of dicts in the same shape `build_oob_tensors` expects.
    - gts:           list of ground-truth SMILES (possibly empty strings).
    - row_ids:       list of IDs (from id_col if given, else "row_{idx}").
    - formulas_csv:  list of formula strings from formula_col, if provided
                     (else empty strings — caller will derive from SMILES).
    """
    mols, gts, ids, formulas_csv = [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV {path} has no header row")
        cols = set(reader.fieldnames)
        log.info(f"CSV columns: {list(reader.fieldnames)}")
        for idx, row in enumerate(reader):
            # ── SMILES / GT ──
            smi = (row.get(smiles_col) or "").strip()
            # ── NMR content: detect format ──
            h_parsed = row.get(h_parsed_col, "") if h_parsed_col in cols else ""
            c_parsed = row.get(c_parsed_col, "") if c_parsed_col in cols else ""
            h_text   = row.get(h_text_col,   "") if h_text_col   in cols else ""
            c_text   = row.get(c_text_col,   "") if c_text_col   in cols else ""
            combined = row.get(combined_text_col, "") if combined_text_col in cols else ""

            h_input, c_input = None, None

            if _looks_like_parsed_list(h_parsed) or _looks_like_parsed_list(c_parsed):
                # Training-style pre-parsed lists → pass the strings through
                # to build_oob_tensors, which will run preprocess.parse_h1 /
                # parse_c13 on them (that's what those functions are FOR).
                h_input = h_parsed
                c_input = c_parsed
                fmt = "parsed"
            elif combined:
                # One merged text block ("1H NMR ... 13C NMR ...")
                d = split_raw_nmr_block(combined)
                h_input, c_input = d["h_nmr"], d["c_nmr"]
                # Fill in solvent/freq from the combined text if not overridden
                row.setdefault(solv_h_col, d.get("solvent_h") or "")
                row.setdefault(solv_c_col, d.get("solvent_c") or "")
                if not row.get(freq_h_col): row[freq_h_col] = d.get("freq_h") or 0.0
                if not row.get(freq_c_col): row[freq_c_col] = d.get("freq_c") or 0.0
                fmt = "combined_text"
            else:
                # Separate text cells
                h_input, c_input = h_text, c_text
                fmt = "text_pair"

            # ── Frequency & solvent ──
            def _f(x):
                try: return float(x)
                except (TypeError, ValueError):
                    m = re.search(r"(\d+\.?\d*)", str(x or ""))
                    return float(m.group(1)) if m else 0.0
            freq_h = _f(row.get(freq_h_col, 0))
            freq_c = _f(row.get(freq_c_col, 0))
            solv_h = (row.get(solv_h_col) or "").strip() or None
            solv_c = (row.get(solv_c_col) or "").strip() or None

            mol = {
                "h_nmr":      h_input,
                "c_nmr":      c_input,
                "solvent_h":  solv_h,
                "solvent_c":  solv_c,
                "freq_h":     freq_h,
                "freq_c":     freq_c,
                "smiles":     smi,          # used for molecular-formula encoder
                "_format":    fmt,          # debug/bookkeeping
            }
            mols.append(mol)
            gts.append(smi)
            row_id = (row.get(id_col) if id_col and row.get(id_col) else f"row_{idx}")
            ids.append(str(row_id))
            formulas_csv.append(
                (row.get(formula_col) or "").strip() if formula_col else "")

    log.info(f"Loaded {len(mols)} molecules from {path}")
    # Per-format summary for confidence
    from collections import Counter
    fmt_counter = Counter(m["_format"] for m in mols)
    for fmt, n in fmt_counter.items():
        log.info(f"  format={fmt}: {n}")
    return mols, gts, ids, formulas_csv


def compute_formula_from_smiles(smi):
    """Use preprocess.smiles_to_formula for consistency with the model's
    molecular-formula encoder. Returns a Counter-like dict {'C':15,'H':19,...}
    plus a pretty-formatted Hill-order string."""
    if not smi:
        return {}, ""
    try:
        fc = smiles_to_formula(smi) or {}
    except Exception:
        fc = {}
    # Pretty format (Hill system: C, H, then alphabetical)
    parts = []
    if fc.get("C", 0): parts.append(f"C{fc['C'] if fc['C']>1 else ''}")
    if fc.get("H", 0): parts.append(f"H{fc['H'] if fc['H']>1 else ''}")
    for elem in sorted(k for k in fc if k not in ("C", "H") and fc[k] > 0):
        parts.append(f"{elem}{fc[elem] if fc[elem]>1 else ''}")
    return fc, "".join(parts)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--checkpoint", default=None)
    pa.add_argument("--checkpoint_dir", default=None)
    pa.add_argument("--ensemble_top_k", type=int, default=5)
    pa.add_argument("--ensemble_mode", default="weight_avg",
                    choices=["none", "weight_avg", "prediction", "both"],
                    help="How to use --checkpoint_dir: "
                         "'none' (single model from --checkpoint only), "
                         "'weight_avg' (average top-K weights into one model — "
                         "fast, default), "
                         "'prediction' (run each checkpoint separately, merge "
                         "beams by vote count + max-score — slower, stronger), "
                         "'both' (run weight_avg AND prediction, print a side-"
                         "by-side comparison of all three modes if --checkpoint "
                         "is also given).")
    pa.add_argument("--weight_avg_ks", type=int, nargs="+", default=None,
                    help="Space-separated list of K values for weight-averaged "
                         "ensembles, e.g. '--weight_avg_ks 2 3 4 5' runs four "
                         "separate weight-avg runs (top-2, top-3, top-4, top-5) "
                         "and shows each as a row in the cross-run comparison. "
                         "Defaults to [--ensemble_top_k] (a single K). Only "
                         "takes effect when --ensemble_mode includes weight_avg.")
    pa.add_argument("--top_k_values", type=int, nargs="+",
                    default=[1, 5, 10, 20],
                    help="List of K values at which to report accuracy and "
                         "mean Tanimoto@K. Any K > --beam_size is silently "
                         "skipped. Default: 1 5 10 20.")
    pa.add_argument("--tanimoto_stereo", default="2d",
                    choices=["iso", "2d", "both"],
                    help="Which Morgan-Tanimoto variant to report. "
                         "'2d' (default): useChirality=False; stereo info "
                         "is discarded during fingerprint hashing. Matches "
                         "training-time _tanimoto. "
                         "'iso': useChirality=True; stereo flips reduce "
                         "Tanimoto below 1.0. "
                         "'both': report 2D and stereo Tanimoto side by side "
                         "in summary tables and per-row CSVs (extra columns).")
    pa.add_argument("--processed_dir", default=None,
                    help="Override processed/ path (defaults to the "
                         "path stored in the checkpoint)")
    pa.add_argument("--nmr_file", default=None,
                    help="Text file with NMR blocks separated by '---'")
    pa.add_argument("--input_csv", default=None,
                    help="CSV with one molecule per row. Auto-detects "
                         "training-style (NMR_processed_1H) vs. raw-text "
                         "(h_nmr_text) layouts. Ground-truth SMILES are "
                         "taken from --smiles_col and molecular formulae are "
                         "derived from them automatically.")
    pa.add_argument("--smiles_col",   default="canon_smiles",
                    help="CSV column name for ground-truth SMILES (default: canon_smiles)")
    pa.add_argument("--h_parsed_col", default="NMR_processed_1H",
                    help="CSV column for pre-parsed 1H peaks (training format)")
    pa.add_argument("--c_parsed_col", default="NMR_processed_13C",
                    help="CSV column for pre-parsed 13C peaks (training format)")
    pa.add_argument("--h_text_col",   default="h_nmr_text",
                    help="CSV column for raw 1H NMR text")
    pa.add_argument("--c_text_col",   default="c_nmr_text",
                    help="CSV column for raw 13C NMR text")
    pa.add_argument("--combined_text_col", default="nmr_text",
                    help="CSV column for a combined 1H+13C NMR block")
    pa.add_argument("--freq_h_col",   default="NMR_frequency_1H")
    pa.add_argument("--freq_c_col",   default="NMR_frequency_13C")
    pa.add_argument("--solv_h_col",   default="NMR_solvent_1H")
    pa.add_argument("--solv_c_col",   default="NMR_solvent_13C")
    pa.add_argument("--id_col",       default=None,
                    help="Optional CSV column to use as row identifier "
                         "in the output (default: row_0, row_1, ...)")
    pa.add_argument("--limit", type=int, default=None,
                    help="Only process the first N rows (for quick tests)")
    pa.add_argument("--ground_truth", nargs="*", default=None,
                    help="Optional SMILES (one per molecule, same order) "
                         "— overrides any SMILES found in --input_csv")
    pa.add_argument("--beam_size", type=int, default=10)
    pa.add_argument("--max_len",  type=int, default=550)
    pa.add_argument("--output_csv", default="oob_results.csv")
    pa.add_argument("--output_correct_csv", default="oob_correct_only.csv",
                    help="Written only when --ground_truth is provided")
    pa.add_argument("--render", default="auto",
                    choices=["auto", "html", "png", "both", "none"],
                    help="'auto' (default): HTML for text/hard-coded input, "
                         "NONE for large --input_csv runs to avoid giant "
                         "HTML files. Override with 'html' / 'png' / 'both' "
                         "/ 'none' to force a specific mode.")
    pa.add_argument("--report_html", default="oob_report.html",
                    help="Output path for the HTML report")
    pa.add_argument("--image_dir", default="oob_images",
                    help="Directory for per-molecule PNG grids")
    pa.add_argument("--img_size", type=int, default=280,
                    help="Structure image size in pixels (HTML cards)")
    pa.add_argument("--max_html_rows", type=int, default=50,
                    help="If --render html is used with --input_csv, cap "
                         "the number of molecules rendered to the HTML to "
                         "this many (default 50). Full results still land "
                         "in the CSVs.")
    pa.add_argument("--device", default="auto",
                    help="'auto' (GPU if available) / 'cuda' / 'cpu'")
    pa.add_argument("--gpu", type=str, default=None,
                    help="Physical GPU index, e.g. '2'. Sets "
                         "CUDA_VISIBLE_DEVICES before torch loads — the only "
                         "reliable way to pin a GPU when another job owns "
                         "the default device. Handled at import time.")
    pa.add_argument("--cpu", action="store_true",
                    help="Force CPU inference (shortcut for --gpu cpu)")
    pa.add_argument("--no_amp", action="store_true",
                    help="Disable fp16 AMP during inference (slower, uses "
                         "more memory, but avoids any AMP edge cases)")
    pa.add_argument("--encoder_chunk", type=int, default=128,
                    help="Chunk size for the encoder forward pass on large "
                         "CSV inputs (default 128). Lower if you OOM during "
                         "encoding; irrelevant for small inputs.")
    pa.add_argument("--length_penalty", type=float, default=0.0,
                    help="GNMT-style length penalty α for beam ranking. "
                         "0 = pure log-prob (default), 0.6–1.0 favors "
                         "longer outputs. Only affects ranking, not scores.")
    pa.add_argument("--primary_stereo_mode", default="isomeric",
                    choices=["isomeric", "flat"],
                    help="Which correctness mode is used as the 'primary' "
                         "pass/fail (HTML highlighting + top-line summary). "
                         "Both modes are ALWAYS computed and reported in "
                         "the CSVs regardless.")
    # Legacy aliases — kept so old command lines still work.
    pa.add_argument("--strip_stereo", action="store_true", default=False,
                    help="Legacy alias: sets primary mode to 'flat'.")
    pa.add_argument("--keep_stereo", dest="strip_stereo", action="store_false",
                    help="Legacy alias: sets primary mode to 'isomeric'.")
    pa.add_argument("--rerank_by_formula", action="store_true", default=False,
                    help="Re-rank beam candidates so those matching the "
                         "ground-truth molecular formula come first. The "
                         "original top-K is preserved in the CSV alongside "
                         "the reranked top-K so you can compare both.")
    pa.add_argument("--rerank_heavy_only", action="store_true", default=False,
                    help="When reranking, compare formulas ignoring H counts "
                         "(useful for robustness against explicit/implicit H "
                         "mismatches). Requires --rerank_by_formula.")
    pa.add_argument("--formula_col", default=None,
                    help="Optional CSV column containing a ground-truth "
                         "molecular formula (e.g. from HRMS). If omitted, "
                         "formulas are derived from the SMILES column.")
    args = pa.parse_args()

    # Legacy --strip_stereo flag still sets the PRIMARY mode for display;
    # both modes are always computed in the output CSVs.
    if args.strip_stereo:
        args.primary_stereo_mode = "flat"
    global _STRIP_STEREO
    _STRIP_STEREO = (args.primary_stereo_mode == "flat")
    log.info(f"Stereo evaluation: BOTH modes computed "
             f"(primary for display = {args.primary_stereo_mode.upper()})")
    if args.rerank_heavy_only and not args.rerank_by_formula:
        log.warning("--rerank_heavy_only has no effect without "
                    "--rerank_by_formula; ignoring")

    # Resolve --render auto: HTML for text/hard-coded inputs, NONE for CSV.
    if args.render == "auto":
        if args.input_csv:
            args.render = "none"
            log.info("CSV input detected → skipping HTML rendering "
                     "(override with --render html to force; use "
                     "--max_html_rows to cap output size).")
        else:
            args.render = "html"

    if not args.checkpoint and not args.checkpoint_dir:
        pa.error("Provide --checkpoint or --checkpoint_dir")

    # Device resolution. Remember: --gpu and --cpu were already consumed
    # at the top of this file to set CUDA_VISIBLE_DEVICES pre-import, so
    # by now torch sees only the intended device(s).
    if args.cpu or (args.gpu is not None and
                    args.gpu.lower() in ("", "cpu", "none")):
        dev = torch.device("cpu")
    elif args.device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(args.device)

    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        log.info(f"Device: {dev} ({torch.cuda.get_device_name()}), "
                 f"free={free / 1e9:.2f} / {total / 1e9:.2f} GB")
        if free < 1.5e9:
            log.warning("Less than 1.5 GB free on target GPU — consider "
                        "--gpu <other_idx> or --cpu")
    else:
        log.info(f"Device: {dev}")

    # ── Build list of molecule dicts ──
    # DEFAULT_GTS maps index → stereo-stripped ground-truth SMILES for the
    # built-in molecules. Set to None for no GT.
    DEFAULT_GTS = None
    row_ids = None
    formulas_csv = None    # GT formulas from the CSV (optional; else derived from SMILES)
    if args.input_csv:
        mols, gts_csv, row_ids, formulas_csv = load_nmr_csv(
            args.input_csv,
            smiles_col=args.smiles_col,
            h_parsed_col=args.h_parsed_col,
            c_parsed_col=args.c_parsed_col,
            h_text_col=args.h_text_col,
            c_text_col=args.c_text_col,
            combined_text_col=args.combined_text_col,
            freq_h_col=args.freq_h_col,
            freq_c_col=args.freq_c_col,
            solv_h_col=args.solv_h_col,
            solv_c_col=args.solv_c_col,
            id_col=args.id_col,
            formula_col=args.formula_col,
        )
        if args.limit:
            mols         = mols[:args.limit]
            gts_csv      = gts_csv[:args.limit]
            row_ids      = row_ids[:args.limit]
            formulas_csv = formulas_csv[:args.limit]
            log.info(f"Limited to first {len(mols)} rows")
        # GTs come from the CSV unless explicitly overridden on the CLI
        if args.ground_truth:
            log.info("CLI --ground_truth overrides SMILES column from CSV")
            DEFAULT_GTS = None    # let the resolver below pick CLI
        else:
            DEFAULT_GTS = gts_csv
    elif args.nmr_file:
        mols = load_nmr_file(args.nmr_file)
        log.info(f"Loaded {len(mols)} molecules from {args.nmr_file}")
    else:
        # Fall back to the two hard-coded OOB molecules from the user's query
        RAW_BLOCKS = [
            # ── Molecule 3a (C15H20N+ [M+H]+ = 214.1590, i.e. neutral C15H19N) ──
            """1H NMR (400 MHz, Chloroform-d) δ 8.23 (dd, J = 5.2, 1.6 Hz, 1H), 7.39 (dd, J = 7.4, 1.6 Hz, 1H),
            7.02 (dd, J = 7.3, 5.2 Hz, 1H), 6.60 (ddd, J = 7.8, 6.2, 1.5 Hz, 1H), 6.45 (ddd, J = 7.5, 5.9, 1.4 Hz, 1H),
            3.96 (dt, J = 6.3, 1.8 Hz, 1H), 3.88 (ddd, J = 6.0, 3.0, 1.7 Hz, 1H), 1.94 - 1.81 (m, 2H), 1.39 - 1.31 (m, 1H),
            1.25 - 1.15 (m, 3H), 1.07 - 0.98 (m, 1H), 0.90 (ddd, J = 11.1, 3.9, 2.4 Hz, 1H), 0.81 (t, J = 7.1 Hz, 3H),
            0.73 - 0.64 (m, 1H).
            13C NMR (101 MHz, Chloroform-d) δ 162.7, 144.7, 138.18, 135.8, 134.4, 129.4, 120.6, 47.7, 39.7, 37.9, 36.2,
            33.7, 30.2, 22.8, 14.2.""",
            # ── Molecule 3b (diastereomer — same 2D structure as 3a) ──
            """1H NMR (400 MHz, Chloroform-d) δ 8.22 (dd, J = 5.1, 1.6 Hz, 1H), 7.38 (dd, J = 7.3, 1.6 Hz, 1H),
            7.01 (dd, J = 7.3, 5.2 Hz, 1H), 6.57 - 6.49 (m, 2H), 4.03 (dq, J = 5.1, 2.4 Hz, 1H), 3.73 (dt, J = 6.0, 2.0 Hz, 1H),
            1.90 (ddd, J = 11.8, 9.8, 3.1 Hz, 1H), 1.85 - 1.77 (m, 1H), 1.28 - 1.16 (m, 4H), 0.96 (ddd, J = 11.7, 4.3, 2.3 Hz, 1H),
            0.92 - 0.87 (m, 1H), 0.82 (t, J = 7.0 Hz, 3H), 0.78 - 0.71 (m, 1H).
            13C NMR (101 MHz, Chloroform-d) δ 164.7, 144.9, 136.1, 135.3, 134.3, 131.7, 120.3, 44.5, 43.08, 37.7, 36.0,
            32.8, 30.1, 22.8, 14.2.""",
        ]
        mols = [split_raw_nmr_block(b) for b in RAW_BLOCKS]
        # Stereo-aware ground truth for a model trained WITH stereochemistry.
        # 3a and 3b are distinct diastereomers (and here encoded so that their
        # canonical 2D SMILES also differ — pyridine-N position flipped).
        # Confirmed: both are C15H19N, 19H, matches HRMS and NMR integration.
        DEFAULT_GTS = [
            "CCCC[C@H]1[C@@H]2C=C[C@@H](C3=CC=CN=C32)C1",   # 3a
            "CCCC[C@@H]1[C@H]2C=C[C@H](C3=NC=CC=C23)C1",    # 3b
        ]
        log.info(f"Using {len(mols)} hard-coded OOB molecule(s) from the prompt")

    # Resolve ground truths: CLI > DEFAULT_GTS > empty
    if args.ground_truth:
        gts = list(args.ground_truth)
    elif DEFAULT_GTS is not None:
        gts = list(DEFAULT_GTS)
        if not args.input_csv:
            log.info("Using built-in stereo-aware ground truth:")
            for i, g in enumerate(DEFAULT_GTS):
                log.info(f"  [{i}] {g}")
    else:
        gts = [""] * len(mols)
    if len(gts) < len(mols):
        gts = gts + [""] * (len(mols) - len(gts))

    # Default row_ids if not supplied by CSV
    if row_ids is None:
        row_ids = [f"row_{i}" for i in range(len(mols))]

    # Compute molecular formulae from ground-truth SMILES (for reporting;
    # also propagates into build_oob_tensors via mol["smiles"] so the
    # formula-encoder branch is filled in correctly if enabled).
    formulas = []
    for i, g in enumerate(gts):
        mols[i]["smiles"] = g
        # Preferred source: --formula_col if supplied and non-empty for this
        # row. Otherwise derive from the ground-truth SMILES.
        pretty = ""
        if formulas_csv and i < len(formulas_csv) and formulas_csv[i]:
            pretty = formulas_csv[i].strip()
        else:
            _, pretty = compute_formula_from_smiles(g)
        formulas.append(pretty)

    # Log a preview (first few rows) instead of spamming 10k lines
    _preview = min(len(gts), 5)
    for i in range(_preview):
        if formulas[i]:
            log.info(f"  [{row_ids[i]}] true SMILES: {gts[i] or '—':<60s}"
                     f"  →  formula: {formulas[i]}")
    if len(gts) > _preview:
        log.info(f"  ... and {len(gts) - _preview} more rows")

    # ─────────────────────────────────────────────────────────────────────
    # Each "run" is a full inference+post-processing cycle producing its
    # own CSVs and summary. `run_name` is used as a filename suffix and in
    # the printed headline. The function captures args/gts/formulas/row_ids
    # from the enclosing main() scope.
    # ─────────────────────────────────────────────────────────────────────
    run_summaries = []    # collected for final side-by-side comparison

    def process_one_run(run_name, all_beams, is_selfies, suffix=""):
        nonlocal run_summaries

        # Always compute BOTH 2D and stereo-aware Tanimoto so downstream
        # code has both available regardless of --tanimoto_stereo choice.
        # The "tanimoto" key is an alias for whichever the user selected
        # (or the 2D value when args.tanimoto_stereo == "both", arbitrary
        # but the per-beam CSV exposes both columns explicitly).
        def _norm_beam(pred, score, gt_smi):
            smi = _selfies_to_smiles(pred) if is_selfies else pred
            iso, flat = _canon_both(smi)
            _, pf = compute_formula_from_smiles(iso or flat or smi)
            if gt_smi:
                t_2d, t_iso = _tanimoto_both(smi, gt_smi)
            else:
                t_2d = t_iso = None
            primary = (t_iso if args.tanimoto_stereo == "iso" else t_2d)
            return {
                "raw_pred":     pred,
                "pred_smiles":  smi,
                "canon_iso":    iso,
                "canon_flat":   flat,
                "formula":      pf,
                "score":        score,
                "tanimoto":     primary,    # what summary stats look at
                "tanimoto_2d":  t_2d,
                "tanimoto_iso": t_iso,
            }

        norm_beams = []
        for i, beams in enumerate(all_beams):
            gt_i = gts[i] if i < len(gts) else ""
            norm_beams.append([_norm_beam(p, s, gt_i) for s, p in beams])

        # ── Optional: rerank each molecule's beams by formula match ──
        reranked_beams = None
        if args.rerank_by_formula:
            reranked_beams = []
            for i, nb in enumerate(norm_beams):
                tgt = formulas[i] if i < len(formulas) else ""
                if not tgt:
                    reranked_beams.append(list(nb))
                    continue
                def _strip_H(f): return re.sub(r"H\d*", "", f)
                tgt_cmp = _strip_H(tgt) if args.rerank_heavy_only else tgt
                def _match(b):
                    f_cmp = _strip_H(b["formula"]) if args.rerank_heavy_only else b["formula"]
                    return bool(b["formula"]) and f_cmp == tgt_cmp

                def _key(t):
                    idx, b = t
                    return (0 if _match(b) else 1, idx)

                reranked_beams.append(
                    sorted(enumerate(nb), key=_key)
                )
                reranked_beams[-1] = [t[1] for t in reranked_beams[-1]]
            log.info(f"[{run_name}] re-ranked beams by formula match "
                     f"(heavy_only={args.rerank_heavy_only})")

        primary_mode  = args.primary_stereo_mode
        primary_key   = "canon_iso" if primary_mode == "isomeric" else "canon_flat"

        # ── Per-beam output CSV ──
        rows_all, rows_correct = [], []
        for i, beams in enumerate(norm_beams):
            gt        = gts[i]
            gt_iso    = _canon(gt, strip_stereo=False) if gt else ""
            gt_flat   = _canon(gt, strip_stereo=True)  if gt else ""
            gt_formula = formulas[i] if i < len(formulas) else ""
            rid       = row_ids[i]
            rerank_rank = {id(b): r + 1 for r, b in enumerate(
                reranked_beams[i] if reranked_beams is not None else beams)}
            for orig_rank, b in enumerate(beams, start=1):
                correct_iso  = bool(gt_iso)  and b["canon_iso"]  == gt_iso
                correct_flat = bool(gt_flat) and b["canon_flat"] == gt_flat
                formula_hit  = bool(gt_formula) and b["formula"] == gt_formula
                row = {
                    "row_id":               rid,
                    "molecule_idx":         i,
                    "beam_rank":            orig_rank,
                    "rerank_rank":          rerank_rank.get(id(b), orig_rank),
                    "beam_score":           f"{b['score']:.4f}",
                    "prediction_raw":       b["raw_pred"],
                    "prediction_smiles":    b["pred_smiles"],
                    "prediction_canon_iso":  b["canon_iso"],
                    "prediction_canon_flat": b["canon_flat"],
                    "prediction_formula":   b["formula"],
                    "ground_truth":         gt,
                    "ground_truth_iso":     gt_iso,
                    "ground_truth_flat":    gt_flat,
                    "ground_truth_formula": gt_formula,
                    "correct_isomeric":     int(correct_iso)  if gt else "",
                    "correct_2d":           int(correct_flat) if gt else "",
                    "formula_match":        int(formula_hit)  if gt_formula else "",
                    "tanimoto_to_gt":       (f"{b['tanimoto']:.4f}"
                                             if b["tanimoto"] is not None else ""),
                }
                # When the user wants both flavors of Tanimoto, expose both
                # explicitly so the CSV is self-documenting (otherwise the
                # single tanimoto_to_gt column is whichever they selected).
                if args.tanimoto_stereo == "both":
                    row["tanimoto_2d"]  = (f"{b['tanimoto_2d']:.4f}"
                                            if b["tanimoto_2d"]  is not None else "")
                    row["tanimoto_iso"] = (f"{b['tanimoto_iso']:.4f}"
                                            if b["tanimoto_iso"] is not None else "")
                rows_all.append(row)
                if (primary_mode == "isomeric" and correct_iso) or \
                   (primary_mode == "flat"     and correct_flat):
                    rows_correct.append(row)

        fieldnames = list(rows_all[0].keys()) if rows_all else []
        output_csv_p = args.output_csv.replace(".csv", f"{suffix}.csv") if suffix else args.output_csv
        with open(output_csv_p, "w", newline="") as f:
            if fieldnames:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows_all)
        log.info(f"[{run_name}] saved {len(rows_all)} per-beam rows to {output_csv_p}")

        correct_csv_p = args.output_correct_csv.replace(".csv", f"{suffix}.csv") if suffix else args.output_correct_csv
        if any(gts):
            with open(correct_csv_p, "w", newline="") as f:
                if fieldnames:
                    w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows_correct)
            log.info(f"[{run_name}] saved {len(rows_correct)} correct-under-primary rows to {correct_csv_p}")

        # ── Per-molecule summary CSV ──
        def _first_hit_rank(beams, canon_key, target):
            if not target: return ""
            for r, b in enumerate(beams, start=1):
                if b[canon_key] == target:
                    return r
            return ""

        summary_rows = []
        for i, beams in enumerate(norm_beams):
            gt      = gts[i]
            gt_iso  = _canon(gt, strip_stereo=False) if gt else ""
            gt_flat = _canon(gt, strip_stereo=True)  if gt else ""
            gt_f    = formulas[i] if i < len(formulas) else ""

            top1       = beams[0] if beams else None
            iso_hit_1  = bool(gt_iso)  and top1 and top1["canon_iso"]  == gt_iso
            flat_hit_1 = bool(gt_flat) and top1 and top1["canon_flat"] == gt_flat
            iso_rank   = _first_hit_rank(beams, "canon_iso",  gt_iso)
            flat_rank  = _first_hit_rank(beams, "canon_flat", gt_flat)
            n_fmla_match = sum(1 for b in beams if gt_f and b["formula"] == gt_f)

            rerank_status = ""
            if gt and gt_f:
                if iso_rank == "" or not iso_rank:
                    rerank_status = "not_in_beam"
                elif iso_rank == 1:
                    rerank_status = "already_top1"
                else:
                    blockers = [r for r, b in enumerate(beams, start=1)
                                 if b["formula"] == gt_f and r < iso_rank]
                    rerank_status = ("rerank_moves_to_top1" if not blockers
                                     else "blocked_by_isomer")

            if gt and beams:
                top1_tani = top1["tanimoto"] if top1 else None
                valid_tans = [b["tanimoto"] for b in beams if b["tanimoto"] is not None]
                best_tani  = max(valid_tans) if valid_tans else None
                # Also compute the OTHER flavor's top-1 / best for "both" mode
                top1_tani_2d  = top1["tanimoto_2d"]  if top1 else None
                top1_tani_iso = top1["tanimoto_iso"] if top1 else None
                vt_2d  = [b["tanimoto_2d"]  for b in beams if b["tanimoto_2d"]  is not None]
                vt_iso = [b["tanimoto_iso"] for b in beams if b["tanimoto_iso"] is not None]
                best_tani_2d  = max(vt_2d)  if vt_2d  else None
                best_tani_iso = max(vt_iso) if vt_iso else None
            else:
                top1_tani = best_tani = None
                top1_tani_2d = top1_tani_iso = None
                best_tani_2d = best_tani_iso = None

            rb = reranked_beams[i] if reranked_beams is not None else beams
            rtop1 = rb[0] if rb else None
            r_iso_hit_1  = bool(gt_iso)  and rtop1 and rtop1["canon_iso"]  == gt_iso
            r_flat_hit_1 = bool(gt_flat) and rtop1 and rtop1["canon_flat"] == gt_flat
            r_iso_rank   = _first_hit_rank(rb, "canon_iso",  gt_iso)
            r_flat_rank  = _first_hit_rank(rb, "canon_flat", gt_flat)
            r_top1_tani  = rtop1["tanimoto"]     if (gt and rtop1) else None
            r_top1_tani_2d  = rtop1["tanimoto_2d"]  if (gt and rtop1) else None
            r_top1_tani_iso = rtop1["tanimoto_iso"] if (gt and rtop1) else None

            def _fmt(x): return f"{x:.4f}" if isinstance(x, float) else ""

            summary_row = {
                "row_id":                  row_ids[i],
                "molecule_idx":            i,
                "n_beams":                 len(beams),
                "ground_truth_smiles":     gt,
                "ground_truth_iso":        gt_iso,
                "ground_truth_flat":       gt_flat,
                "ground_truth_formula":    gt_f,
                "top1_iso":                top1["canon_iso"]  if top1 else "",
                "top1_flat":               top1["canon_flat"] if top1 else "",
                "top1_formula":            top1["formula"]    if top1 else "",
                "top1_correct_iso":        int(iso_hit_1)  if gt else "",
                "top1_correct_2d":         int(flat_hit_1) if gt else "",
                "top1_tanimoto":           _fmt(top1_tani),
                "best_beam_tanimoto":      _fmt(best_tani),
                "rank_first_hit_iso":      iso_rank,
                "rank_first_hit_2d":       flat_rank,
                "in_beam_correct_iso":     int(bool(iso_rank))  if gt else "",
                "in_beam_correct_2d":      int(bool(flat_rank)) if gt else "",
                "n_formula_match":         n_fmla_match if gt_f else "",
                "rerank_status":           rerank_status,
                "rerank_top1_iso":         rtop1["canon_iso"]  if rtop1 else "",
                "rerank_top1_flat":        rtop1["canon_flat"] if rtop1 else "",
                "rerank_top1_formula":     rtop1["formula"]    if rtop1 else "",
                "rerank_top1_correct_iso": int(r_iso_hit_1)  if gt else "",
                "rerank_top1_correct_2d":  int(r_flat_hit_1) if gt else "",
                "rerank_top1_tanimoto":    _fmt(r_top1_tani),
                "rerank_rank_first_hit_iso": r_iso_rank,
                "rerank_rank_first_hit_2d":  r_flat_rank,
            }
            if args.tanimoto_stereo == "both":
                summary_row.update({
                    "top1_tanimoto_2d":         _fmt(top1_tani_2d),
                    "top1_tanimoto_iso":        _fmt(top1_tani_iso),
                    "best_beam_tanimoto_2d":    _fmt(best_tani_2d),
                    "best_beam_tanimoto_iso":   _fmt(best_tani_iso),
                    "rerank_top1_tanimoto_2d":  _fmt(r_top1_tani_2d),
                    "rerank_top1_tanimoto_iso": _fmt(r_top1_tani_iso),
                })
            summary_rows.append(summary_row)

        summary_path = output_csv_p.replace(".csv", "_summary.csv")
        if summary_path == output_csv_p:
            summary_path = output_csv_p + ".summary.csv"
        with open(summary_path, "w", newline="") as f:
            if summary_rows:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader(); w.writerows(summary_rows)
        log.info(f"[{run_name}] saved per-molecule summary ({len(summary_rows)} rows) to {summary_path}")

        # ── Compute summary stats for this run ──
        # Includes top-K accuracy and Tanimoto@K for each K in TOP_K_LIST.
        # Any K > beam_size is silently clamped to beam_size to avoid
        # reporting nonsense ("top-20 of 10 beams").
        n_gt = sum(1 for g in gts if g)
        TOP_K_LIST = sorted(set(k for k in args.top_k_values
                                if k <= args.beam_size))
        if not TOP_K_LIST:
            TOP_K_LIST = [args.beam_size]

        def _stats(beams_src, canon_key):
            """Returns dict keyed by top-K value, plus aggregate fields.
               Per-K: n_correct (int) and per-K best-so-far mean Tanimoto.
               When args.tanimoto_stereo == 'both', mean Tanimoto is
               tracked for BOTH 2D and stereo-aware fingerprints
               (per_k_mean_tani_2d / per_k_mean_tani_iso); the legacy
               per_k_mean_tani field aliases the user-selected primary.
               Aggregates: n_tani (how many rows contributed), in_beam
               (correct anywhere in full beam), mean_best_tani."""
            per_k_correct = {k: 0 for k in TOP_K_LIST}
            per_k_tani_sum    = {k: 0.0 for k in TOP_K_LIST}
            per_k_tani_sum_2d  = {k: 0.0 for k in TOP_K_LIST}
            per_k_tani_sum_iso = {k: 0.0 for k in TOP_K_LIST}
            ib = 0
            sum_best_t = sum_best_t_2d = sum_best_t_iso = 0.0
            n_tani = 0
            for i, beams in enumerate(beams_src):
                if not gts[i] or not beams:
                    continue
                tgt = _canon(gts[i], strip_stereo=(canon_key == "canon_flat"))
                if any(c[canon_key] == tgt for c in beams): ib += 1

                # Helper to compute best-so-far running max for one tanimoto key
                def _running_max(beams_, key):
                    rm = 0.0; out = []
                    for b in beams_:
                        v = b.get(key)
                        if v is not None and v > rm:
                            rm = v
                        out.append(rm)
                    return out

                rl_primary = _running_max(beams, "tanimoto")
                rl_2d      = _running_max(beams, "tanimoto_2d")
                rl_iso     = _running_max(beams, "tanimoto_iso")
                # Use primary as the "did this row contribute" gate so the
                # denominator (n_tani) is consistent across both flavors
                if rl_primary[-1] > 0 or any(b["tanimoto"] is not None for b in beams):
                    for k in TOP_K_LIST:
                        ku = min(k, len(rl_primary))
                        per_k_tani_sum[k]    += rl_primary[ku - 1]
                        per_k_tani_sum_2d[k]  += rl_2d[ku - 1]
                        per_k_tani_sum_iso[k] += rl_iso[ku - 1]
                    sum_best_t     += rl_primary[-1]
                    sum_best_t_2d  += rl_2d[-1]
                    sum_best_t_iso += rl_iso[-1]
                    n_tani += 1

                matched_by_rank = [c[canon_key] == tgt for c in beams]
                for k in TOP_K_LIST:
                    if any(matched_by_rank[:k]):
                        per_k_correct[k] += 1

            return {
                "per_k_correct":  per_k_correct,
                "per_k_mean_tani": {k: (per_k_tani_sum[k] / n_tani if n_tani else 0.0)
                                     for k in TOP_K_LIST},
                "per_k_mean_tani_2d":  {k: (per_k_tani_sum_2d[k]  / n_tani if n_tani else 0.0)
                                         for k in TOP_K_LIST},
                "per_k_mean_tani_iso": {k: (per_k_tani_sum_iso[k] / n_tani if n_tani else 0.0)
                                         for k in TOP_K_LIST},
                "in_beam":        ib,
                "mean_best_tani":     sum_best_t     / n_tani if n_tani else 0.0,
                "mean_best_tani_2d":  sum_best_t_2d  / n_tani if n_tani else 0.0,
                "mean_best_tani_iso": sum_best_t_iso / n_tani if n_tani else 0.0,
                "n_tani":         n_tani,
                "top_k_list":     TOP_K_LIST,
            }

        stats = {
            "run_name": run_name, "n_gt": n_gt, "n_total": len(gts),
            "orig":     {"iso": _stats(norm_beams, "canon_iso"),
                         "flat": _stats(norm_beams, "canon_flat")},
            "rerank":   ({"iso": _stats(reranked_beams, "canon_iso"),
                          "flat": _stats(reranked_beams, "canon_flat")}
                         if reranked_beams is not None else None),
            "summary_rows": summary_rows,
        }
        run_summaries.append(stats)

        # ── Per-run console summary with top-K accuracy + Tanimoto@K ──
        if n_gt > 0:
            orig_iso   = stats["orig"]["iso"]
            orig_flat  = stats["orig"]["flat"]
            def _pct(n): return f"{n:>5d}/{n_gt} ({100*n/n_gt:5.1f}%)"
            Ks = orig_iso["top_k_list"]

            print()
            print("=" * 90)
            print(f"  RESULTS SUMMARY — {run_name}")
            print(f"    Total molecules : {len(gts)}   With GT : {n_gt}   "
                  f"Beam size : {args.beam_size}")
            print(f"    Reranked        : "
                  f"{'YES' if reranked_beams is not None else 'no'}"
                  f"{' (heavy-atom only)' if args.rerank_heavy_only and reranked_beams is not None else ''}")
            print("=" * 90)

            def _print_block(label, iso_d, flat_d):
                # Accuracy table: rows = iso/2D, cols = top-K
                k_hdr = "  ".join(f"{'top-'+str(k):>14s}" for k in Ks)
                print(f"\n  ── {label} ──")
                print(f"  {'Accuracy':<20s}    {k_hdr}")
                print("  " + "-" * (20 + 4 + len(Ks)*16))
                iso_row  = "  ".join(f"{_pct(iso_d['per_k_correct'][k]):>14s}" for k in Ks)
                flat_row = "  ".join(f"{_pct(flat_d['per_k_correct'][k]):>14s}" for k in Ks)
                print(f"  {'Exact (isomeric)':<20s}    {iso_row}")
                print(f"  {'Exact (2D)':<20s}    {flat_row}")
                # In-beam = correct anywhere in the full beam (== top-beam_size)
                print(f"  {'(in-beam, full)':<20s}    iso={_pct(iso_d['in_beam']):>14s}  "
                      f"2D={_pct(flat_d['in_beam']):>14s}")
                # Tanimoto table — one or two rows depending on --tanimoto_stereo
                print(f"  {'Tanimoto (best-so-far, Morgan r=2, 2048b), N=' + str(flat_d['n_tani']):<20s}")
                tani_modes = []
                if args.tanimoto_stereo in ("2d", "both"):
                    tani_modes.append(("2D", "per_k_mean_tani_2d", "mean_best_tani_2d"))
                if args.tanimoto_stereo in ("iso", "both"):
                    tani_modes.append(("iso", "per_k_mean_tani_iso", "mean_best_tani_iso"))
                for label_t, k_key, b_key in tani_modes:
                    tani_row = "  ".join(f"{flat_d[k_key][k]:>14.4f}" for k in Ks)
                    print(f"  {('mean ('+label_t+')'):<20s}    {tani_row}")
                    print(f"  {('  best-beam ('+label_t+')'):<20s}    {flat_d[b_key]:>14.4f}")

            _print_block("ORIGINAL beam ranking", orig_iso, orig_flat)

            if reranked_beams is not None:
                rr_iso  = stats["rerank"]["iso"]
                rr_flat = stats["rerank"]["flat"]
                _print_block("RERANKED (formula-constrained)", rr_iso, rr_flat)
                # Δ top-1 is what we care about most
                d_iso_t1  = rr_iso["per_k_correct"][Ks[0]]  - orig_iso["per_k_correct"][Ks[0]]
                d_flat_t1 = rr_flat["per_k_correct"][Ks[0]] - orig_flat["per_k_correct"][Ks[0]]
                print(f"\n  Δ from rerank at top-{Ks[0]}: iso {d_iso_t1:+d}  /  2D {d_flat_t1:+d}")

                # Status breakdown
                from collections import Counter
                status_counts = Counter(r["rerank_status"] for r in summary_rows
                                        if r["rerank_status"])
                n_s = sum(status_counts.values())
                print(f"\n  Rerank status (iso match, N={n_s}):")
                for label, key in [
                    ("already_top1", "already_top1"),
                    ("rerank_moves_to_top1", "rerank_moves_to_top1"),
                    ("blocked_by_isomer", "blocked_by_isomer"),
                    ("not_in_beam", "not_in_beam"),
                ]:
                    n = status_counts.get(key, 0)
                    print(f"    {label:<25s} : {n:5d} ({100*n/max(n_s,1):5.1f}%)")
            print("=" * 90)

        # ── Render visualizations (only for this run if requested) ──
        report_html_p = args.report_html.replace(".html", f"{suffix}.html") if suffix else args.report_html
        image_dir_p   = args.image_dir + suffix if suffix else args.image_dir
        if args.render in ("html", "both"):
            to_render = all_beams
            to_gts    = gts
            if args.input_csv and len(all_beams) > args.max_html_rows:
                log.info(f"[{run_name}] capping HTML to first "
                         f"{args.max_html_rows} molecules (of {len(all_beams)})")
                to_render = all_beams[:args.max_html_rows]
                to_gts    = gts[:args.max_html_rows]
            render_html_report(to_render, to_gts, is_selfies, args, report_html_p)
        if args.render in ("png", "both"):
            render_png_grids(all_beams, gts, is_selfies, args, image_dir_p)

    # ─────────────────────────────────────────────────────────────────────
    # Decide which inference runs to do. Any of these can be turned on
    # independently, and all share the same input tensors + post-processing.
    # ─────────────────────────────────────────────────────────────────────
    do_single    = bool(args.checkpoint)
    do_weight_avg = bool(args.checkpoint_dir) and args.ensemble_mode in ("weight_avg", "both")
    do_pred_ens   = bool(args.checkpoint_dir) and args.ensemble_mode in ("prediction", "both")

    # Load one model just to get cfg for tensor building (and to run single-mode)
    primary_ckpt = args.checkpoint or (
        list_ensemble_checkpoints(args.checkpoint_dir, 1)[0]
        if args.checkpoint_dir else None)
    if primary_ckpt is None:
        pa.error("Must provide --checkpoint or --checkpoint_dir")
    _init_model, _init_tok, init_cfg = load_model(
        primary_ckpt, dev, args.processed_dir)
    init_is_selfies = isinstance(_init_tok, SELFIESTokenizer)
    pd_arg = args.processed_dir or init_cfg.get("data", {}).get("processed_dir", "processed")

    # Build tensors once — they're the same for every run.
    log.info("Building OOB input tensors...")
    T = build_oob_tensors(mols, pd_arg, init_cfg)
    batch = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in T.items()}

    # ── Single-model run ──
    if do_single:
        log.info("=" * 60)
        log.info("  RUN 1/?: single-model inference")
        log.info("=" * 60)
        try:
            single_beams = generate_all_beams(
                _init_model, batch, args.max_len, args.beam_size,
                use_amp=(not args.no_amp),
                length_penalty=args.length_penalty,
                encoder_chunk=args.encoder_chunk,
            )
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"CUDA OOM during beam search: {e}")
            log.error("Try: --beam_size 5 --encoder_chunk 32 --cpu")
            sys.exit(2)
        process_one_run("single", single_beams, init_is_selfies,
                        suffix="_single" if (do_weight_avg or do_pred_ens) else "")

    # Free the single-model GPU memory before loading ensemble
    del _init_model
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # ── Weight-average ensemble(s) ──
    # If the user passes --weight_avg_ks 2 3 4 5, run one weight-avg per K.
    # Each gets its own run name/suffix and appears as a row in the cross-run
    # comparison. Defaults to a single K = --ensemble_top_k.
    if do_weight_avg:
        wa_k_list = args.weight_avg_ks if args.weight_avg_ks else [args.ensemble_top_k]
        wa_k_list = sorted(set(wa_k_list))
        for K in wa_k_list:
            log.info("=" * 60)
            log.info(f"  RUN: weight-averaged ensemble (top-{K})")
            log.info("=" * 60)
            try:
                model_wa, tok_wa, _ = load_ensemble_weights(
                    args.checkpoint_dir, K, dev, args.processed_dir)
            except Exception as e:
                log.error(f"Could not load weight-avg ensemble for K={K}: {e}")
                continue
            is_self_wa = isinstance(tok_wa, SELFIESTokenizer)
            try:
                wa_beams = generate_all_beams(
                    model_wa, batch, args.max_len, args.beam_size,
                    use_amp=(not args.no_amp),
                    length_penalty=args.length_penalty,
                    encoder_chunk=args.encoder_chunk,
                )
            except torch.cuda.OutOfMemoryError as e:
                log.error(f"CUDA OOM during weight-avg (K={K}) beam search: {e}")
                sys.exit(2)
            del model_wa
            if dev.type == "cuda": torch.cuda.empty_cache()
            run_tag = f"weight_avg_k{K}"
            # Suffix only needed when we'll have multiple runs producing CSVs
            multi_runs = do_single or do_pred_ens or len(wa_k_list) > 1
            process_one_run(run_tag, wa_beams, is_self_wa,
                            suffix=f"_{run_tag}" if multi_runs else "")

    # ── Prediction-level ensemble ──
    if do_pred_ens:
        log.info("=" * 60)
        log.info("  RUN: prediction-level ensemble "
                 f"(top-{args.ensemble_top_k})")
        log.info("=" * 60)
        ckpt_paths = list_ensemble_checkpoints(
            args.checkpoint_dir, args.ensemble_top_k)
        try:
            pe_beams = run_prediction_ensemble(
                ckpt_paths, batch, pd_arg, args.max_len, args.beam_size,
                use_amp=(not args.no_amp),
                length_penalty=args.length_penalty,
                encoder_chunk=args.encoder_chunk,
                device=dev,
            )
        except torch.cuda.OutOfMemoryError as e:
            log.error(f"CUDA OOM during prediction ensemble: {e}")
            sys.exit(2)
        process_one_run("prediction_ens", pe_beams, init_is_selfies,
                        suffix="_predens" if (do_single or do_weight_avg) else "")

    # ── Cross-run comparison table ──
    if len(run_summaries) >= 2:
        Ks = run_summaries[0]["orig"]["iso"]["top_k_list"]
        n = run_summaries[0]["n_gt"]
        def _pc(x): return f"{100*x/max(n,1):5.1f}%" if n else "    —"

        # Table 1: isomeric accuracy at each K
        print()
        print("=" * 110)
        print("  CROSS-RUN COMPARISON — isomeric exact-match accuracy")
        print("=" * 110)
        k_hdr = "  ".join(f"{'top-'+str(k):>12s}" for k in Ks)
        print(f"  {'Run':<22s}   {k_hdr}   {'in-beam':>10s}")
        print("  " + "-" * (22 + 3 + len(Ks)*14 + 3 + 10))
        for s in run_summaries:
            iso = s["orig"]["iso"]
            row = "  ".join(f"{_pc(iso['per_k_correct'][k]):>12s}" for k in Ks)
            print(f"  {s['run_name']:<22s}   {row}   {_pc(iso['in_beam']):>10s}")
            if s["rerank"] is not None:
                rr = s["rerank"]["iso"]
                rrow = "  ".join(f"{_pc(rr['per_k_correct'][k]):>12s}" for k in Ks)
                print(f"  {'  + rerank':<22s}   {rrow}   {_pc(rr['in_beam']):>10s}")
        print("=" * 110)

        # Table 2: 2D-only accuracy at each K
        print()
        print("=" * 110)
        print("  CROSS-RUN COMPARISON — 2D (no stereo) exact-match accuracy")
        print("=" * 110)
        print(f"  {'Run':<22s}   {k_hdr}   {'in-beam':>10s}")
        print("  " + "-" * (22 + 3 + len(Ks)*14 + 3 + 10))
        for s in run_summaries:
            flat = s["orig"]["flat"]
            row = "  ".join(f"{_pc(flat['per_k_correct'][k]):>12s}" for k in Ks)
            print(f"  {s['run_name']:<22s}   {row}   {_pc(flat['in_beam']):>10s}")
            if s["rerank"] is not None:
                rr = s["rerank"]["flat"]
                rrow = "  ".join(f"{_pc(rr['per_k_correct'][k]):>12s}" for k in Ks)
                print(f"  {'  + rerank':<22s}   {rrow}   {_pc(rr['in_beam']):>10s}")
        print("=" * 110)

        # Table 3: mean Tanimoto (best-so-far) at each K.
        # 'iso' or '2d' modes print a single block. 'both' prints both
        # variants, labeled clearly so the columns are unambiguous.
        def _print_tani_block(label_suffix, key_per_k, key_best):
            print()
            print("=" * 110)
            print(f"  CROSS-RUN COMPARISON — mean Tanimoto (Morgan r=2{label_suffix}) at top-K")
            print("=" * 110)
            k_hdr_t = "  ".join(f"{'Tani@'+str(k):>12s}" for k in Ks)
            print(f"  {'Run':<22s}   {k_hdr_t}   {'best-beam':>10s}")
            print("  " + "-" * (22 + 3 + len(Ks)*14 + 3 + 10))
            for s in run_summaries:
                flat = s["orig"]["flat"]
                row = "  ".join(f"{flat[key_per_k][k]:>12.4f}" for k in Ks)
                print(f"  {s['run_name']:<22s}   {row}   {flat[key_best]:>10.4f}")
                if s["rerank"] is not None:
                    rr = s["rerank"]["flat"]
                    rrow = "  ".join(f"{rr[key_per_k][k]:>12.4f}" for k in Ks)
                    print(f"  {'  + rerank':<22s}   {rrow}   {rr[key_best]:>10.4f}")
            print("=" * 110)

        if args.tanimoto_stereo == "both":
            _print_tani_block(", 2D",         "per_k_mean_tani_2d",  "mean_best_tani_2d")
            _print_tani_block(", stereo-aware","per_k_mean_tani_iso", "mean_best_tani_iso")
        elif args.tanimoto_stereo == "iso":
            _print_tani_block(", stereo-aware","per_k_mean_tani_iso", "mean_best_tani_iso")
        else:   # "2d"
            _print_tani_block(", 2D",         "per_k_mean_tani_2d",  "mean_best_tani_2d")


if __name__ == "__main__":
    main()