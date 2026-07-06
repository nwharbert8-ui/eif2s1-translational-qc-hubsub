#!/usr/bin/env python3
"""
EIF2S1 R3 REVISION - ALS REPLICATION COHORT (Reviewer 4): GSE124439 frontal cortex, RNA-seq.

Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616 | Inner Architecture LLC.
Date: 2026-07-05

NYGC ALS Consortium postmortem cortex RNA-seq. Frontal-cortex subset, ALS Spectrum MND vs
Non-Neurological Control. RNA-seq (same platform as the PD GSE68719 analysis), so the Type 1
dissociation gap is directly comparable to PD here (no cross-platform caveat). Identical
method to pipeline_5/pipeline_7. seed=42.

Data (auto-download, cached): GSE124439 series matrix (metadata) + GSE124439_RAW.tar (per-sample
STAR gene counts, gene symbols). Library-size normalized (log2 CPM) before Spearman coupling.

Output: results/disease_GSE124439_ALS_frontal_eif2s1_pelo.csv , provenance_GSE124439.json
"""
import os, sys, gzip, json, io, tarfile, time, urllib.request
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd
from scipy import stats

RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)
N_PERM = 2000; N_BOOT = 1000; TOP_PCT = 5; MIN_CPM = 1.0

ROOT = r"G:\My Drive\inner_architecture_research"
CACHE = os.path.join(ROOT, "cache"); os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(ROOT, "EIF2S1", "r3_revision_analyses", "results")
TAR = os.path.join(CACHE, "GSE124439_RAW.tar")
MATRIX_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124439/matrix/GSE124439_series_matrix.txt.gz"
TAR_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124439/suppl/GSE124439_RAW.tar"

# ---------------------------------------------------------------- metadata
print("[1/5] Parsing GSE124439 metadata (frontal cortex, ALS vs control)")
mtxt = gzip.decompress(urllib.request.urlopen(MATRIX_URL, timeout=90).read()).decode('utf-8', 'ignore')
gsms, region, group = [], [], []
for line in mtxt.split('\n'):
    if line.startswith('!Sample_geo_accession'):
        gsms = [v.strip().strip('"') for v in line.split('\t')[1:]]
    elif line.startswith('!Sample_characteristics_ch1'):
        vals = [v.strip().strip('"') for v in line.split('\t')[1:]]
        clean = [v.split(':', 1)[1].strip() if ':' in v else v for v in vals]
        joined = ' '.join(clean)
        if 'Cortex' in joined:      region = clean
        elif 'ALS' in joined or 'Control' in joined: group = clean
meta = pd.DataFrame({'GSM': gsms, 'region': region, 'group': group})
sel = meta[(meta.region == 'Frontal Cortex') &
           (meta.group.isin(['ALS Spectrum MND', 'Non-Neurological Control']))].copy()
sel['grp'] = sel.group.map({'ALS Spectrum MND': 'ALS', 'Non-Neurological Control': 'Control'})
print(f"  frontal-cortex selected: ALS={ (sel.grp=='ALS').sum() }  Control={ (sel.grp=='Control').sum() }")

# ---------------------------------------------------------------- counts
print("[2/5] Extracting per-sample counts from RAW.tar")
if not os.path.exists(TAR) or os.path.getsize(TAR) < 20_000_000:
    urllib.request.urlretrieve(TAR_URL, TAR)
tf = tarfile.open(TAR)
member_by_gsm = {m.split('_')[0]: m for m in tf.getnames()}
cols = {}
want = set(sel.GSM)
for gsm in want:
    mem = member_by_gsm.get(gsm)
    if not mem: continue
    data = gzip.decompress(tf.extractfile(mem).read()).decode('utf-8', 'ignore')
    s = {}
    for ln in data.split('\n')[1:]:
        if not ln: continue
        p = ln.split('\t')
        if len(p) >= 2:
            g = p[0].strip().strip('"')
            try: s[g] = float(p[1])
            except ValueError: pass
    cols[gsm] = s
counts = pd.DataFrame(cols).dropna(how='all')
counts = counts.loc[[not g.startswith(('ERCC', '__')) for g in counts.index]]
print(f"  counts matrix: {counts.shape[0]} genes x {counts.shape[1]} samples")

# library-size normalize -> CPM -> log2, expressed filter
cpm = counts / counts.sum(axis=0) * 1e6
expr = np.log2(cpm[cpm.median(axis=1) >= MIN_CPM] + 1.0)
print(f"  expressed (median CPM>={MIN_CPM}): {expr.shape[0]} genes")
for tg in ['EIF2S1', 'PELO']:
    print(f"    {tg} present: {tg in expr.index}")
assert 'EIF2S1' in expr.index and 'PELO' in expr.index

samples = list(expr.columns)
grpmap = dict(zip(sel.GSM, sel.grp))
als = [s for s in samples if grpmap.get(s) == 'ALS']
ctl = [s for s in samples if grpmap.get(s) == 'Control']
X = expr.to_numpy(np.float64); genes = expr.index.to_numpy()
eif = int(np.where(genes == 'EIF2S1')[0][0]); pel = int(np.where(genes == 'PELO')[0][0])
keep = np.ones(X.shape[0], bool); keep[eif] = keep[pel] = False
als_idx = np.array([samples.index(s) for s in als]); ctl_idx = np.array([samples.index(s) for s in ctl])

def group_gap(sidx):
    sub = X[:, sidx]; R = stats.rankdata(sub, axis=1)
    Rc = R - R.mean(axis=1, keepdims=True)
    den = np.sqrt((Rc**2).sum(axis=1)); den[den == 0] = 1e-12
    def rho(i): return (Rc @ Rc[i]) / (den * den[i])
    a = rho(eif)[keep]; b = rho(pel)[keep]
    a_s, b_s = (a+1)/2, (b+1)/2
    wj = np.minimum(a_s, b_s).sum() / np.maximum(a_s, b_s).sum()
    ta, tb = np.percentile(a, 100-TOP_PCT), np.percentile(b, 100-TOP_PCT)
    ina, inb = a >= ta, b >= tb
    bj = (ina & inb).sum() / (ina | inb).sum()
    return wj, bj, wj-bj

print("[3/5] EIF2S1-PELO coupling + Type 1 gap per group")
cg = group_gap(ctl_idx); ag = group_gap(als_idx)
print(f"  Control (n={len(ctl)}): WJ_shifted={cg[0]:.4f} binaryJ={cg[1]:.4f} gap={cg[2]:.4f}")
print(f"  ALS     (n={len(als)}): WJ_shifted={ag[0]:.4f} binaryJ={ag[1]:.4f} gap={ag[2]:.4f}")
obs = ag[2] - cg[2]
print(f"  observed gap difference (ALS - control): {obs:+.4f}")

print(f"[4/5] Permutation ({N_PERM}) + bootstrap ({N_BOOT})")
pooled = np.concatenate([ctl_idx, als_idx]); nctl = len(ctl_idx)
perm = np.array([group_gap((pp := np.random.permutation(pooled))[nctl:])[2] - group_gap(pp[:nctl])[2]
                 for _ in range(N_PERM)])
pval = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (N_PERM + 1)
boot = np.array([group_gap(np.random.choice(als_idx, len(als_idx), True))[2] -
                 group_gap(np.random.choice(ctl_idx, len(ctl_idx), True))[2] for _ in range(N_BOOT)])
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  permutation p = {pval:.4f}   bootstrap 95% CI [{lo:.4f}, {hi:.4f}]")

print("[5/5] Writing outputs")
pd.DataFrame([{
    'cohort': 'GSE124439_frontal_ALS', 'platform': 'RNA-seq (STAR counts, log2 CPM)',
    'region': 'frontal cortex', 'n_control': len(ctl), 'n_als': len(als), 'n_genes': int(keep.sum()),
    'control_wj_shifted': cg[0], 'als_wj_shifted': ag[0],
    'control_gap': cg[2], 'als_gap': ag[2], 'gap_diff_als_minus_control': obs,
    'permutation_p': pval, 'boot_ci_low': lo, 'boot_ci_high': hi,
}]).to_csv(os.path.join(OUT, "disease_GSE124439_ALS_frontal_eif2s1_pelo.csv"), index=False)
json.dump({
    "cohort": "GSE124439 frontal cortex (ALS Spectrum MND vs non-neurological control)",
    "platform": "RNA-seq (STAR gene counts, log2 CPM) - same platform class as PD GSE68719",
    "methodology": "WJ-native; identical to pipeline_5/pipeline_7",
    "random_seed": RANDOM_SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOT,
    "control_wj_shifted": float(cg[0]), "als_wj_shifted": float(ag[0]),
    "control_gap": float(cg[2]), "als_gap": float(ag[2]),
    "gap_difference": float(obs), "permutation_p": float(pval), "boot_95ci": [float(lo), float(hi)],
    "execution_date": "2026-07-05",
}, open(os.path.join(OUT, "provenance_GSE124439.json"), 'w'), indent=2)
print("\n=== SUMMARY (GSE124439 ALS, frontal cortex, RNA-seq) ===")
print(f"  constitutive coupling WJ_shifted: control {cg[0]:.3f} / ALS {ag[0]:.3f}")
print(f"  gap control {cg[2]:.3f} -> ALS {ag[2]:.3f} (diff {obs:+.3f}), perm p={pval:.3f}, 95%CI[{lo:.3f},{hi:.3f}]")
print("DONE.")
