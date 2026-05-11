

import argparse, ast, csv, json, logging, math, os, random, re, sys
from collections import Counter
import numpy as np

try:
    import torch; HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("PRISM-preprocess")

SMILES_REGEX = re.compile(
    r"(\[[^\]]+\]|Br|Cl|Si|[BCNOPSFIcnos]|[=#/\\@+\-\.\(\)]|[0-9]|%[0-9]{2})")

# ═══════════════════════ NMR PARSING ═══════════════════════

def _parse_j(j_list):
    if j_list is None or j_list == []: return []
    if isinstance(j_list, str):
        try: r = ast.literal_eval(j_list); j_list = r if isinstance(r, list) else [j_list]
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

def _parse_integ(raw):
    if raw is None: return 1
    m = re.search(r"(\d+)", str(raw))
    return int(m.group(1)) if m else 1

def normalize_splitting(raw):
    if raw is None: return None
    orig = raw.strip()
    if not orig: return None
    ns = orig.lower().replace("-","_").replace(".","").replace(" ","")
    syn = {"singlet":"s","doublet":"d","triplet":"t","quartet":"q",
           "quintet":"quint","pentet":"p","multiplet":"m",
           "brs":"br_s","br":"br_s","brd":"br_d","brt":"br_d",
           "brdd":"br_d","brq":"br_s","septet":"hept","heptet":"hept",
           "sext":"hex","dm":"m","quint":"quint"}
    if ns in syn: return syn[ns]
    if ns.startswith("app"):
        base = ns[3:].strip("_")
        return syn.get(base, base) if base else None
    if orig in ("AB","ABq"): return orig
    if ns == "ab": return "AB"
    if ns == "abq": return "ABq"
    return ns

def normalize_solvent(raw):
    if not raw or not raw.strip(): return None
    s = raw.strip()
    # Comprehensive solvent normalization mapping
    nm = {
        # Chloroform
        "cdcl3":"CDCl3", "chloroform-d":"CDCl3", "chloroform":"CDCl3",
        "cdcl₃":"CDCl3", "deuterochloroform":"CDCl3",
        # DMSO
        "dmso-d6":"DMSO-d6", "dmso":"DMSO-d6", "dmso_d6":"DMSO-d6",
        "dmsod6":"DMSO-d6", "(cd3)2so":"DMSO-d6", "d6-dmso":"DMSO-d6",
        "dimethylsulfoxide-d6":"DMSO-d6",
        # Methanol
        "cd3od":"CD3OD", "meod":"CD3OD", "meod-d4":"CD3OD", "methanol-d4":"CD3OD",
        "cd₃od":"CD3OD", "meoh-d4":"CD3OD", "methanol-d":"CD3OD",
        # Benzene
        "c6d6":"C6D6", "benzene-d6":"C6D6", "c₆d₆":"C6D6",
        # Acetonitrile
        "cd3cn":"CD3CN", "acetonitrile-d3":"CD3CN", "cd₃cn":"CD3CN",
        "mecn-d3":"CD3CN", "an-d3":"CD3CN",
        # DCM
        "cd2cl2":"CD2Cl2", "dcm-d2":"CD2Cl2", "cd₂cl₂":"CD2Cl2",
        "methylene chloride-d2":"CD2Cl2",
        # Water
        "d2o":"D2O", "d₂o":"D2O", "deuterium oxide":"D2O",
        # Acetone
        "cd3cocd3":"Acetone-d6", "acetone-d6":"Acetone-d6", "acetoned6":"Acetone-d6",
        "(cd3)2co":"Acetone-d6", "acetone_d6":"Acetone-d6",
        # THF
        "thf-d8":"THF-d8", "thfd8":"THF-d8", "thf_d8":"THF-d8",
        # Toluene
        "toluene-d8":"Toluene-d8", "phme-d8":"Toluene-d8", "toluened8":"Toluene-d8",
        "c6d5cd3":"Toluene-d8", "c7d8":"Toluene-d8", "tol-d8":"Toluene-d8",
        # DMF
        "dmf-d7":"DMF-d7", "dmfd7":"DMF-d7", "dmf_d7":"DMF-d7",
        "(cd3)2ncdo":"DMF-d7", "dmf":"DMF-d7",
        # Pyridine
        "pyridine-d5":"pyridine-d5", "pyridined5":"pyridine-d5",
        "c5d5n":"pyridine-d5", "py-d5":"pyridine-d5", "pyridine_d5":"pyridine-d5",
        # TCE
        "c2d2cl4":"C2D2Cl4", "tce-d2":"C2D2Cl4",
        # TFA
        "tfa-d1":"TFA-d1", "cf3co2d":"TFA-d1", "tfa":"TFA-d1", "cf3cood":"TFA-d1",
        # Acetic acid
        "cd3co2d":"CD3CO2D", "acetic acid-d4":"CD3CO2D", "cd3cood":"CD3CO2D",
        "acetic-d4":"CD3CO2D",
    }
    k = s.lower().replace(" ","")
    if k in nm: return nm[k]
    if s in ("not_known","mixed","","none","n/a","unknown"): return None
    # Try matching the original string (case-sensitive) against known canonical names
    canonical = {"CDCl3","DMSO-d6","CD3OD","Acetone-d6","CD2Cl2","C6D6","CD3CN",
                 "D2O","THF-d8","Toluene-d8","DMF-d7","pyridine-d5","C2D2Cl4",
                 "TFA-d1","CD3CO2D"}
    if s in canonical: return s
    return s  # Return as-is; will be captured in vocab and mapped to its own ID

def parse_frequency(freq_str):
    if not freq_str: return 0.0
    m = re.search(r"(\d+\.?\d*)", str(freq_str))
    return float(m.group(1)) if m else 0.0

def parse_h1(raw_str):
    if not raw_str or not raw_str.strip() or raw_str.strip() in ("[]","None","nan"): return []
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
            peaks.append((ppm, sp_n, min(_parse_integ(ig),20), sum(jv)/len(jv) if jv else 0.0))
        except: continue
    peaks.sort(key=lambda p: -p[0])
    return peaks

def parse_c13(raw_str):
    if not raw_str or not raw_str.strip() or raw_str.strip() in ("[]","None","nan"): return []
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

def smiles_to_formula(smi):
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smi)
        if mol:
            mol = Chem.AddHs(mol)
            c = {}
            for a in mol.GetAtoms(): c[a.GetSymbol()] = c.get(a.GetSymbol(),0)+1
            return c
    except ImportError: pass
    # Fallback: extract ALL atoms from SMILES string (no RDKit)
    # 1. Bracket atoms like [Fe], [Pd], [Si], [C@@H], [nH], [13C], etc.
    bracket_atom = re.compile(r'\[(\d*)([A-Z][a-z]?)')
    # 2. Organic subset atoms outside brackets
    organic_atom = re.compile(r'(?<!\[)(Br|Cl|Si|Se|[BCNOPSFIcnos])(?![a-z])')
    am = {'c':'C','n':'N','o':'O','s':'S'}
    c = {}
    # Count bracket atoms
    for match in bracket_atom.finditer(smi):
        sym = match.group(2)
        c[sym] = c.get(sym,0)+1
    # Count organic atoms (not inside brackets)
    # Remove bracket contents first to avoid double-counting
    smi_no_brackets = re.sub(r'\[[^\]]*\]', '', smi)
    for t in organic_atom.findall(smi_no_brackets):
        t2 = am.get(t,t); c[t2] = c.get(t2,0)+1
    c.setdefault("H", max(c.get("C",0)*2+2, 0))
    return c

# ═══════════════════════ SHIFT REGIONS ═══════════════════════

def _extract_regions(ppms, min_p, max_p, nbins, sigma, known, merge_tol, depth=0.10):
    arr = np.array([p for p in ppms if min_p <= p <= max_p])
    if len(arr) < 100: return None, arr
    from scipy.ndimage import gaussian_filter1d
    h, edges = np.histogram(arr, bins=nbins, range=(min_p, max_p))
    ctrs = (edges[:-1]+edges[1:])/2
    sm = gaussian_filter1d(h.astype(float), sigma=sigma)
    pks = [(i, sm[i]) for i in range(1,len(sm)-1) if sm[i]>=sm[i-1] and sm[i]>=sm[i+1] and sm[i]>0]
    vals = []
    for i in range(1,len(sm)-1):
        if sm[i]<=sm[i-1] and sm[i]<=sm[i+1]:
            lp = max((p[1] for p in pks if p[0]<i), default=sm[i])
            rp = max((p[1] for p in pks if p[0]>i), default=sm[i])
            lm = min(lp, rp)
            if lm > 0 and (lm-sm[i])/lm > depth: vals.append(ctrs[i])
    combined = list(vals)
    for kb in known:
        if not any(abs(db-kb)<merge_tol for db in combined):
            if int(np.sum((arr>=kb-merge_tol*2)&(arr<=kb+merge_tol*2))) > len(arr)*0.001:
                combined.append(kb)
    b = sorted(set([min_p]+combined+[max_p]))
    merged = [b[0]]
    for x in b[1:]:
        if x-merged[-1]>=merge_tol: merged.append(x)
    if merged[-1]<max_p: merged.append(max_p)
    return merged, arr

def extract_h_regions(ppms):
    known = [0.8,1.5,2.5,4.5,6.0,8.5,10.0]
    b, arr = _extract_regions(ppms, -1.0, 14.0, 300, 2, known, 0.5)
    if b is None: b = [-1.0,0.8,1.5,2.5,4.5,6.0,8.5,10.0,14.0]
    labels = [("upfield",0.8),("alkyl",1.5),("allylic",2.5),("heteroatom",4.5),
              ("alkene",6.0),("aromatic",8.5),("aldehyde",10.0),("carboxylic",99)]
    regions = []
    for i in range(len(b)-1):
        lo, hi = round(b[i],2), round(b[i+1],2)
        mid = (lo+hi)/2
        lbl = "carboxylic"
        for l,th in labels:
            if mid<th: lbl=l; break
        cnt = int(np.sum((arr>=lo)&(arr<hi)))
        regions.append({"lower":lo,"upper":hi,"label":lbl,"peak_count":cnt,"peak_fraction":round(cnt/max(len(arr),1),4)})
    logger.info(f"  {len(regions)} 1H regions from {len(arr)} peaks")
    return regions

def extract_c_regions(ppms):
    known = [50.0,90.0,115.0,160.0,190.0]
    b, arr = _extract_regions(ppms, -10.0, 240.0, 500, 3, known, 8.0)
    if b is None: b = [-10,50,90,115,160,190,240]
    labels = [("alkyl_sp3",50),("heteroatom_sp3",90),("alkyne_acetal",115),
              ("aromatic_sp2",160),("carbonyl_ester",190),("carbonyl_ketone",999)]
    regions = []
    for i in range(len(b)-1):
        lo, hi = round(b[i],2), round(b[i+1],2)
        mid = (lo+hi)/2
        lbl = "carbonyl_ketone"
        for l,th in labels:
            if mid<th: lbl=l; break
        cnt = int(np.sum((arr>=lo)&(arr<hi)))
        regions.append({"lower":lo,"upper":hi,"label":lbl,"peak_count":cnt,"peak_fraction":round(cnt/max(len(arr),1),4)})
    logger.info(f"  {len(regions)} 13C regions from {len(arr)} peaks")
    return regions

# ═══════════════════════ NORMALIZATION ═══════════════════════

def compute_norm_stats(rows, dc, split2id, atom_types):
    """Stats from TRAINING SET ONLY. Uses robust percentile clipping."""
    hc, cc, sc = dc.get("h_nmr_col","NMR_processed_1H"), dc.get("c_nmr_col","NMR_processed_13C"), dc.get("smiles_col","canon_smiles")
    h_ppms, h_js, c_ppms, c_js, freqs, fvecs = [],[],[],[],[],[]
    for row in rows:
        for pk in parse_h1(row.get(hc,"")): h_ppms.append(pk[0]); h_js.append(pk[3])
        for pk in parse_c13(row.get(cc,"")): c_ppms.append(pk[0]); c_js.append(pk[2])
        for col in ["NMR_frequency_1H","NMR_frequency_13C"]:
            f = parse_frequency(row.get(col,"")); 
            if f > 0: freqs.append(f)
        smi = row.get(sc,"")
        if smi:
            fc = smiles_to_formula(smi)
            fvecs.append([fc.get(a,0) for a in atom_types])

    def robust(vals, name, default_m=0, default_s=1):
        if not vals: return {"mean":float(default_m),"std":float(default_s),"n":0}
        a = np.array(vals, dtype=np.float64)
        lo, hi = np.percentile(a, [1,99])
        ac = a[(a>=lo)&(a<=hi)]
        if len(ac)<10: ac = a
        m, s = float(np.mean(ac)), float(np.std(ac))
        if s < 1e-6: s = 1.0
        logger.info(f"  {name}: mean={m:.4f} std={s:.4f} n={len(vals)}")
        return {"mean":m, "std":s, "n":len(vals)}

    stats = {
        "h_ppm": robust(h_ppms, "h_ppm", 5, 3),
        "c_ppm": robust(c_ppms, "c_ppm", 100, 50),
        "h_j":   robust(h_js, "h_j", 5, 3),
        "c_j":   robust(c_js, "c_j", 10, 30),
        "freq":  robust(freqs, "freq", 400, 150),
    }
    if fvecs:
        fa = np.array(fvecs, dtype=np.float64)
        stats["formula_mean"] = fa.mean(0).tolist()
        stats["formula_std"] = [max(s,1.0) for s in fa.std(0).tolist()]
    else:
        stats["formula_mean"] = [0.0]*len(atom_types)
        stats["formula_std"] = [1.0]*len(atom_types)
    return stats

# ═══════════════════════ TENSOR BUILDER ═══════════════════════

def build_tensors(rows, dc, split2id, solv2id, atom_types, h_regs, c_regs, norms, tok2id):
    hc, cc, sc = dc.get("h_nmr_col","NMR_processed_1H"), dc.get("c_nmr_col","NMR_processed_13C"), dc.get("smiles_col","canon_smiles")
    mxh, mxc, mxs = dc.get("max_h_peaks",70), dc.get("max_c_peaks",150), dc.get("max_smiles_len",550)
    hp_m,hp_s = norms["h_ppm"]["mean"],norms["h_ppm"]["std"]
    cp_m,cp_s = norms["c_ppm"]["mean"],norms["c_ppm"]["std"]
    hj_m,hj_s = norms["h_j"]["mean"],norms["h_j"]["std"]
    cj_m,cj_s = norms["c_j"]["mean"],norms["c_j"]["std"]
    fr_m,fr_s = norms["freq"]["mean"],norms["freq"]["std"]
    fm_m, fm_s = norms["formula_mean"], norms["formula_std"]
    other_id = split2id.get("other", len(split2id)-1)
    s_default = split2id.get("s", 1)

    def h_reg(ppm):
        for i,r in enumerate(h_regs):
            if r["lower"]<=ppm<r["upper"]: return i
        return len(h_regs)-1
    def c_reg(ppm):
        for i,r in enumerate(c_regs):
            if r["lower"]<=ppm<r["upper"]: return i
        return len(c_regs)-1

    N = len(rows)
    # Allocate
    D = {}
    for k,sh,dt in [
        ("h_ppm",(N,mxh),np.float32),("h_ppm_raw",(N,mxh),np.float32),
        ("h_split",(N,mxh),np.int64),("h_integ",(N,mxh),np.int64),
        ("h_j",(N,mxh),np.float32),("h_mask",(N,mxh),np.bool_),("h_region",(N,mxh),np.int64),
        ("c_ppm",(N,mxc),np.float32),("c_ppm_raw",(N,mxc),np.float32),
        ("c_split",(N,mxc),np.int64),("c_j",(N,mxc),np.float32),
        ("c_mask",(N,mxc),np.bool_),("c_region",(N,mxc),np.int64),
        ("smiles_ids",(N,mxs),np.int64),("smiles_len",(N,),np.int64),
        ("mol_formula",(N,len(atom_types)),np.float32),
        ("freq_h",(N,),np.float32),("freq_c",(N,),np.float32),
        ("solvent_h",(N,),np.int64),("solvent_c",(N,),np.int64),
    ]:
        D[k] = np.zeros(sh, dtype=dt)
    smiles_str = []

    for idx, row in enumerate(rows):
        smi = row.get(sc,"").strip()
        smiles_str.append(smi)
        # 1H
        for i,(ppm,sp,integ,jm) in enumerate(parse_h1(row.get(hc,""))[:mxh]):
            D["h_ppm_raw"][idx,i] = ppm
            D["h_ppm"][idx,i] = (ppm-hp_m)/hp_s
            D["h_split"][idx,i] = split2id.get(sp, other_id) if sp else s_default
            D["h_integ"][idx,i] = integ
            D["h_j"][idx,i] = (jm-hj_m)/hj_s
            D["h_mask"][idx,i] = True
            D["h_region"][idx,i] = h_reg(ppm)
        # 13C
        for i,(ppm,sp,jv) in enumerate(parse_c13(row.get(cc,""))[:mxc]):
            D["c_ppm_raw"][idx,i] = ppm
            D["c_ppm"][idx,i] = (ppm-cp_m)/cp_s
            D["c_split"][idx,i] = split2id.get(sp, other_id) if sp else s_default
            D["c_j"][idx,i] = (jv-cj_m)/cj_s
            D["c_mask"][idx,i] = True
            D["c_region"][idx,i] = c_reg(ppm)
        # SMILES
        toks = SMILES_REGEX.findall(smi)[:mxs-2]
        ids = [1]+[tok2id.get(t,3) for t in toks]+[2]
        D["smiles_ids"][idx,:len(ids)] = ids
        D["smiles_len"][idx] = len(ids)
        # Formula (normalized)
        fc = smiles_to_formula(smi)
        for j,a in enumerate(atom_types):
            D["mol_formula"][idx,j] = (fc.get(a,0)-fm_m[j])/fm_s[j]
        # Freq/solvent
        fh = parse_frequency(row.get("NMR_frequency_1H",""))
        fcv = parse_frequency(row.get("NMR_frequency_13C",""))
        D["freq_h"][idx] = (fh-fr_m)/fr_s if fh>0 else 0.0
        D["freq_c"][idx] = (fcv-fr_m)/fr_s if fcv>0 else 0.0
        sh = normalize_solvent(row.get("NMR_solvent_1H",""))
        scv = normalize_solvent(row.get("NMR_solvent_13C",""))
        D["solvent_h"][idx] = solv2id.get(sh, solv2id.get("other",0)) if sh else 0
        D["solvent_c"][idx] = solv2id.get(scv, solv2id.get("other",0)) if scv else 0

    import torch
    T = {k: torch.from_numpy(v) for k,v in D.items()}
    T["smiles_str"] = smiles_str
    return T

# ═══════════════════════ SPLIT ═══════════════════════

def canon_smi(smi):
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        if m: return Chem.MolToSmiles(m, canonical=True)
    except: pass
    return smi.strip()

def split_rows(rows, sc, val_f=0.10, test_f=0.10, seed=42):
    rng = random.Random(seed); n = len(rows)
    g = {}
    for i,r in enumerate(rows): g.setdefault(canon_smi(r.get(sc,"")), []).append(i)
    keys = list(g.keys()); rng.shuffle(keys)
    vt, tt = int(n*val_f), int(n*test_f)
    vi,ti,tri = [],[],[]
    for k in keys:
        ix = g[k]
        if len(vi)<vt: vi.extend(ix)
        elif len(ti)<tt: ti.extend(ix)
        else: tri.extend(ix)
    return tri, vi, ti

# ═══════════════════════ MAIN ═══════════════════════

def preprocess(cfg, output_dir, min_sf=3):
    dc = cfg.get("data",{}); sc = dc.get("smiles_col","canon_smiles")
    hc, cc = dc.get("h_nmr_col","NMR_processed_1H"), dc.get("c_nmr_col","NMR_processed_13C")
    for d in ["vocab","splits","tensors"]: os.makedirs(os.path.join(output_dir,d), exist_ok=True)

    logger.info(f"Reading {dc.get('csv_path','data.csv')}...")
    rows = []
    with open(dc.get("csv_path","data.csv"),"r",newline="",encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get(sc,"").strip(): rows.append(r)
    logger.info(f"  {len(rows)} rows")

    # Vocabularies (all data)
    logger.info("Extracting vocabularies...")
    spc, all_sol, all_hp, all_cp, all_smi = Counter(), Counter(), [], [], []
    for r in rows:
        all_smi.append(r.get(sc,""))
        for pk in parse_h1(r.get(hc,"")): 
            if pk[1]: spc[pk[1]]+=1
            all_hp.append(pk[0])
        for pk in parse_c13(r.get(cc,"")): 
            if pk[1]: spc[pk[1]]+=1
            all_cp.append(pk[0])
        for col in ["NMR_solvent_1H","NMR_solvent_13C"]:
            s = normalize_solvent(r.get(col,"")); 
            if s: all_sol[s]+=1

    sp = ["<pad>"]+[s for s,c in spc.most_common() if c>=min_sf]+["other"]
    s2id = {s:i for i,s in enumerate(sp)}
    with open(os.path.join(output_dir,"vocab","splitting_patterns.json"),"w") as f:
        json.dump({"patterns":sp,"pattern2id":s2id,"n_patterns":len(sp),"raw_counts":dict(spc.most_common()),"min_freq_filter":min_sf},f,indent=2)
    logger.info(f"  {len(sp)} splitting patterns")

    sol = ["<pad>"]+[s for s,_ in all_sol.most_common()]+["other"]
    so2id = {s:i for i,s in enumerate(sol)}
    with open(os.path.join(output_dir,"vocab","solvents.json"),"w") as f:
        json.dump({"solvents":sol,"solvent2id":so2id,"n_solvents":len(sol),"raw_counts":dict(all_sol.most_common())},f,indent=2)
    logger.info(f"  {len(sol)} solvents: {sol}")

    # Atom types — extract ALL elements present in SMILES data
    # Use RDKit if available (most accurate), otherwise parse from SMILES tokens
    ac = Counter()
    _has_rdkit = False
    try:
        from rdkit import Chem
        _has_rdkit = True
    except ImportError:
        pass

    if _has_rdkit:
        logger.info("  Extracting atom types via RDKit...")
        for smi in all_smi:
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    for atom in mol.GetAtoms():
                        ac[atom.GetSymbol()] += 1
            except: pass
    else:
        logger.info("  Extracting atom types from SMILES tokens (no RDKit)...")
        # Parse bracket atoms: [Fe], [Pd], [C@@H], [Si], [nH], [13C], etc.
        bracket_re = re.compile(r'\[(\d*)([A-Z][a-z]?)')
        # Organic subset outside brackets
        organic_re = re.compile(r'(?<!\[)(Br|Cl|Si|Se|[BCNOPSFIcnos])(?![a-z])')
        am = {'c':'C','n':'N','o':'O','s':'S'}
        for smi in all_smi:
            smi_no_brackets = re.sub(r'\[[^\]]*\]', '', smi)
            for t in organic_re.findall(smi_no_brackets):
                ac[am.get(t,t)] += 1
            for m in bracket_re.finditer(smi):
                ac[m.group(2)] += 1
        # Always include H (implicit in most SMILES)
        if "H" not in ac:
            ac["H"] = 1

    # Build ordered list: C and H first, then by frequency
    at = ["C","H"]+[a for a,_ in ac.most_common() if a not in ("C","H")]
    with open(os.path.join(output_dir,"vocab","atom_types.json"),"w") as f:
        json.dump({"atom_types":at,"n_atoms":len(at),"raw_counts":dict(ac.most_common())},f,indent=2)
    logger.info(f"  {len(at)} atom types: {at}")

    hr = extract_h_regions(all_hp); cr = extract_c_regions(all_cp)
    with open(os.path.join(output_dir,"vocab","h_shift_regions.json"),"w") as f:
        json.dump({"regions":hr,"n_regions":len(hr)},f,indent=2)
    with open(os.path.join(output_dir,"vocab","c_shift_regions.json"),"w") as f:
        json.dump({"regions":cr,"n_regions":len(cr)},f,indent=2)

    tc = Counter()
    for smi in all_smi: tc.update(SMILES_REGEX.findall(smi))
    t2id = {"<pad>":0,"<bos>":1,"<eos>":2,"<unk>":3}
    mtf = dc.get("min_token_freq",2)
    for t,c in tc.most_common():
        if c>=mtf and t not in t2id: t2id[t]=len(t2id)
    with open(os.path.join(output_dir,"vocab","smiles_tokenizer.json"),"w") as f:
        json.dump({"token2id":t2id,"vocab_size":len(t2id)},f,indent=2)
    logger.info(f"  SMILES vocab: {len(t2id)}")

    # Split
    logger.info("Splitting...")
    seed = cfg.get("training",{}).get("seed",42)
    tri, vi, ti = split_rows(rows, sc, dc.get("val_frac",0.1), dc.get("test_frac",0.1), seed)
    logger.info(f"  train={len(tri)} val={len(vi)} test={len(ti)}")
    tr_rows, vl_rows, te_rows = [rows[i] for i in tri],[rows[i] for i in vi],[rows[i] for i in ti]

    fn = list(rows[0].keys())
    for name,sub in [("train",tr_rows),("val",vl_rows),("test",te_rows)]:
        with open(os.path.join(output_dir,"splits",f"{name}.csv"),"w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(sub)

    # Norm stats from TRAINING ONLY
    logger.info("Norm stats (training set only)...")
    ns = compute_norm_stats(tr_rows, dc, s2id, at)
    with open(os.path.join(output_dir,"vocab","norm_stats.json"),"w") as f:
        json.dump(ns, f, indent=2)

    # Tensors
    if HAS_TORCH:
        logger.info("Building tensors...")
        for name,sub in [("train",tr_rows),("val",vl_rows),("test",te_rows)]:
            logger.info(f"  {name} ({len(sub)} rows)...")
            T = build_tensors(sub, dc, s2id, so2id, at, hr, cr, ns, t2id)
            import torch; torch.save(T, os.path.join(output_dir,"tensors",f"{name}.pt"))
            logger.info(f"  Saved {name}.pt")

    stats = {"total":len(rows),"train":len(tri),"val":len(vi),"test":len(ti),
             "n_splits":len(sp),"n_solvents":len(sol),"n_atoms":len(at),
             "n_h_regions":len(hr),"n_c_regions":len(cr),"vocab_size":len(t2id)}
    with open(os.path.join(output_dir,"vocab","data_stats.json"),"w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"\n{'='*60}\n  DONE: {output_dir}\n  Train/Val/Test: {len(tri)}/{len(vi)}/{len(ti)}\n{'='*60}")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--config", default="config.yaml")
    pa.add_argument("--output_dir", default=None)
    pa.add_argument("--min_split_freq", type=int, default=3)
    args = pa.parse_args()
    import yaml
    with open(args.config) as f: cfg = yaml.safe_load(f)
    od = args.output_dir or cfg.get("data",{}).get("processed_dir") or os.path.join("processed", os.path.splitext(os.path.basename(cfg.get("data",{}).get("csv_path","data")))[0])
    preprocess(cfg, od, args.min_split_freq)

if __name__ == "__main__":
    main()