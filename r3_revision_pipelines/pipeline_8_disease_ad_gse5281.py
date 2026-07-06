#!/usr/bin/env python3
"""
EIF2S1 R3 REVISION - AD REPLICATION COHORT (Reviewer 4): GSE5281 superior frontal gyrus.

Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616 | Inner Architecture LLC.
Date: 2026-07-05

Reviewer 4 asked for the EIF2S1-PELO analysis in an additional disease dataset (frontal
cortex AD suggested). This runs the IDENTICAL analysis used for PD (GSE68719 BA9) on an
independent AD cohort, a different platform (Affymetrix microarray, not RNA-seq), and the
frontal cortex region (superior frontal gyrus). Two questions, reported separately:
  (1) CONSTITUTIVE COUPLING: is EIF2S1-PELO coupling present/tight in AD frontal cortex,
      as in GTEx (healthy) and PD? (supports the paper's main thesis; platform-independent)
  (2) DISEASE GAP: does the control-vs-AD Type 1 dissociation gap move the same direction
      as PD, and is the between-group difference significant? (exploratory; PD was p=0.40)

Method identical to pipeline_5/pipeline_7: coupling profile = Spearman(gene, all others);
WJ_shifted on (r+1)/2; binary Jaccard top-5%; Type 1 gap = WJ_shifted - binaryJ; between-
group permutation test on the gap difference + subject bootstrap CI. seed=42.

Input : ../../Hypothesis and Theory/cascade_analysis/ad_data/GSE5281_family.soft.gz (local)
Output: results/disease_GSE5281_SFG_eif2s1_pelo.csv , provenance_GSE5281.json
"""
import os, sys, gzip, json, time
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np
from scipy import stats

RANDOM_SEED = 42; np.random.seed(RANDOM_SEED)
N_PERM = 2000; N_BOOT = 1000; TOP_PCT = 5

ROOT = r"G:\My Drive\inner_architecture_research"
SOFT = os.path.join(ROOT, "Hypothesis and Theory", "cascade_analysis", "ad_data", "GSE5281_family.soft.gz")
OUT  = os.path.join(ROOT, "EIF2S1", "r3_revision_analyses", "results")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- parse SOFT
print("[1/5] Streaming GSE5281 SOFT (platform map + SFG samples)")
probe2gene = {}
gsm_group = {}      # GSM -> 'AD' or 'Control' (SFG only)
sample_expr = {}    # GSM -> dict(probe->value) for SFG samples
t0 = time.time()

with gzip.open(SOFT, 'rt', encoding='utf-8', errors='ignore') as f:
    mode = None
    cur_gsm = None; cur_is_sfg = False; cur_group = None
    plat_id_col = plat_sym_col = None
    for line in f:
        line = line.rstrip('\n')
        if line.startswith('!platform_table_begin'):
            mode = 'plat_hdr'; continue
        if line.startswith('!platform_table_end'):
            mode = None; continue
        if line.startswith('^SAMPLE'):
            cur_gsm = line.split('=')[1].strip(); cur_is_sfg = False; cur_group = None
            continue
        if line.startswith('!Sample_characteristics_ch1') and 'bio-source name' in line.lower() and cur_gsm:
            _i = line.lower().find('bio-source name:')
            val = line[_i + len('bio-source name:'):].strip().upper()
            if val.startswith('SFG'):
                cur_is_sfg = True
                cur_group = 'AD' if 'AFFECTED' in val else ('Control' if 'CONTROL' in val else None)
                if cur_group: gsm_group[cur_gsm] = cur_group; sample_expr[cur_gsm] = {}
            continue
        if line.startswith('!sample_table_begin'):
            mode = 'samp_hdr' if (cur_gsm in sample_expr) else 'skip'; continue
        if line.startswith('!sample_table_end'):
            mode = None; continue
        # table content
        if mode == 'plat_hdr':
            cols = line.split('\t');
            plat_id_col = 0
            plat_sym_col = cols.index('Gene Symbol') if 'Gene Symbol' in cols else None
            mode = 'plat'; continue
        if mode == 'plat':
            c = line.split('\t')
            if plat_sym_col and len(c) > plat_sym_col:
                g = c[plat_sym_col].strip()
                if g and g != '': probe2gene[c[0].strip()] = g
            continue
        if mode == 'samp_hdr':
            mode = 'samp'; continue          # skip 'ID_REF\tVALUE' header
        if mode == 'samp':
            c = line.split('\t')
            if len(c) >= 2 and c[1] not in ('', 'null', 'NA'):
                try: sample_expr[cur_gsm][c[0].strip()] = float(c[1])
                except ValueError: pass
            continue

ad = [g for g, grp in gsm_group.items() if grp == 'AD']
ct = [g for g, grp in gsm_group.items() if grp == 'Control']
print(f"  SFG samples: AD={len(ad)}  Control={len(ct)}  ({time.time()-t0:.0f}s)  probes mapped={len(probe2gene)}")

# ---------------------------------------------------------------- build gene x sample matrix
print("[2/5] Building gene-by-sample matrix (probe->gene collapse by mean)")
import pandas as pd
samples = ad + ct
df = pd.DataFrame(sample_expr).reindex(columns=samples)           # probes x samples
df.index = [probe2gene.get(p, None) for p in df.index]
df = df[[i is not None for i in df.index]]
df = df.apply(pd.to_numeric, errors='coerce')
expr = df.groupby(level=0).mean().dropna(how='any')               # genes x samples
# keep genes with real variance (microarray: drop flat probes)
expr = expr[expr.std(axis=1) > 0]
print(f"  matrix: {expr.shape[0]} genes x {expr.shape[1]} samples")
for tg in ['EIF2S1', 'PELO']:
    print(f"    {tg} present: {tg in expr.index}")
assert 'EIF2S1' in expr.index and 'PELO' in expr.index, "target genes missing"

X = expr.to_numpy(dtype=np.float64)
genes = expr.index.to_numpy()
eif = int(np.where(genes == 'EIF2S1')[0][0]); pel = int(np.where(genes == 'PELO')[0][0])
keep = np.ones(X.shape[0], bool); keep[eif] = keep[pel] = False
ctl_idx = np.array([samples.index(s) for s in ct]); ad_idx = np.array([samples.index(s) for s in ad])

def group_gap(sidx):
    sub = X[:, sidx]
    R = stats.rankdata(sub, axis=1)
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
cg = group_gap(ctl_idx); ag = group_gap(ad_idx)
print(f"  Control: WJ_shifted={cg[0]:.4f}  binaryJ={cg[1]:.4f}  gap={cg[2]:.4f}")
print(f"  AD:      WJ_shifted={ag[0]:.4f}  binaryJ={ag[1]:.4f}  gap={ag[2]:.4f}")
obs = ag[2] - cg[2]
print(f"  observed gap difference (AD - control): {obs:+.4f}")

print(f"[4/5] Between-group permutation ({N_PERM}) + bootstrap ({N_BOOT})")
pooled = np.concatenate([ctl_idx, ad_idx]); nctl = len(ctl_idx)
perm = np.empty(N_PERM)
for k in range(N_PERM):
    p = np.random.permutation(pooled)
    perm[k] = group_gap(p[nctl:])[2] - group_gap(p[:nctl])[2]
pval = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (N_PERM + 1)
boot = np.empty(N_BOOT)
for k in range(N_BOOT):
    boot[k] = group_gap(np.random.choice(ad_idx, len(ad_idx), True))[2] - \
              group_gap(np.random.choice(ctl_idx, len(ctl_idx), True))[2]
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"  permutation p (gap diff) = {pval:.4f}")
print(f"  bootstrap 95% CI = [{lo:.4f}, {hi:.4f}]  (excludes 0: {lo>0 or hi<0})")

print("[5/5] Writing outputs")
pd.DataFrame([{
    'cohort': 'GSE5281_SFG_AD', 'platform': 'Affymetrix HG-U133_Plus_2 (microarray)',
    'region': 'superior frontal gyrus', 'n_control': len(ct), 'n_ad': len(ad), 'n_genes': int(keep.sum()),
    'control_wj_shifted': cg[0], 'ad_wj_shifted': ag[0],
    'control_gap': cg[2], 'ad_gap': ag[2], 'gap_diff_ad_minus_control': obs,
    'permutation_p': pval, 'boot_ci_low': lo, 'boot_ci_high': hi,
    'gap_direction_matches_PD': bool(obs > 0),   # PD had AD/PD gap > control gap
}]).to_csv(os.path.join(OUT, "disease_GSE5281_SFG_eif2s1_pelo.csv"), index=False)
json.dump({
    "cohort": "GSE5281 superior frontal gyrus (Alzheimer's disease)",
    "platform": "Affymetrix HG-U133 Plus 2.0 microarray (cross-platform vs RNA-seq primary)",
    "methodology": "WJ-native; identical to pipeline_5/pipeline_7 (Spearman coupling, WJ_shifted, binary top-5%, permutation + bootstrap)",
    "random_seed": RANDOM_SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOT,
    "control_wj_shifted": float(cg[0]), "ad_wj_shifted": float(ag[0]),
    "control_gap": float(cg[2]), "ad_gap": float(ag[2]),
    "gap_difference": float(obs), "permutation_p": float(pval), "boot_95ci": [float(lo), float(hi)],
    "execution_date": "2026-07-05",
}, open(os.path.join(OUT, "provenance_GSE5281.json"), 'w'), indent=2)
print("\n=== SUMMARY (GSE5281 AD, superior frontal gyrus) ===")
print(f"  constitutive coupling WJ_shifted: control {cg[0]:.3f} / AD {ag[0]:.3f}")
print(f"  gap control {cg[2]:.3f} -> AD {ag[2]:.3f} (diff {obs:+.3f}), perm p={pval:.3f}, 95%CI[{lo:.3f},{hi:.3f}]")
print("DONE.")
