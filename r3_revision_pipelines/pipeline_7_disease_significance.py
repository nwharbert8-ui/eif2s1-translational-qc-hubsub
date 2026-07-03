#!/usr/bin/env python3
"""
EIF2S1 R3 REVISION - PIPELINE 7: BETWEEN-GROUP SIGNIFICANCE OF THE PD-vs-CONTROL
Type 1 DISSOCIATION-GAP DIFFERENCE (and the top-5% binary-Jaccard drop).

Author: Drake H. Harbert (D.H.H.) | ORCID: 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH
Date: 2026-07-03

Why: the R3 manuscript reports the EIF2S1-PELO Type 1 dissociation gap increasing
from 0.424 (control) to 0.536 (PD) and the top-5% binary Jaccard dropping 0.490 ->
0.326, and interprets this as disease-selective disruption (title-level claim), but
the R3 analysis never TESTED the between-group difference. The PD point estimate lies
inside the control bootstrap CI [0.325, 0.649]. This pipeline supplies the missing
inferential test so the title claim is either supported by a real number or demoted.

Method (identical gap definition to pipeline_5_formal_layer2h.py):
  - GSE68719 BA9 prefrontal cortex, DESeq2-normalized counts, median>=5 filter.
  - Coupling profile c_g = Spearman(g, every other expressed gene) within a group.
  - WJ_shifted on (r+1)/2 ; binary Jaccard on top-5% ; Type 1 gap = WJ_shifted - binaryJ.
  - Groups: control = C_* columns (n=44), PD = P_* columns (n=29).

Tests:
  1. PERMUTATION (structure-preserving under H0 of exchangeable group labels):
     pool the 73 samples, randomly relabel into 44/29, recompute each group's gap and
     the difference (PD - control); two-sided p = fraction of |perm diff| >= |obs diff|.
     Same for the binary-Jaccard drop (control - PD).
  2. BOOTSTRAP 95% CI on the observed gap difference (resample subjects within group).

seed=42, FORCE_RECOMPUTE. Self-contained. Writes CSV + provenance.json.

Output:
  results/disease_between_group_significance.csv
  results/provenance_pipeline7.json
"""
import os, sys, json, gzip, time
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy import stats

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
FORCE_RECOMPUTE = True
N_PERM = 2000
N_BOOT = 1000
TOP_PCT = 5
MIN_EXPR = 5

ROOT = r"G:\My Drive\inner_architecture_research\EIF2S1"
DATA_DIR = os.path.join(ROOT, "r3_revision_analyses", "data")
RESULTS_DIR = os.path.join(ROOT, "r3_revision_analyses", "results")
COUNTS_FILE = os.path.join(DATA_DIR, "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz")

PUBLISHED = {  # from layer2h_disease_comparison.csv, for the sanity gate
    'control': dict(wj_shifted=0.9142449598181125, binary=0.48977695167286245, gap=0.42446800814525004),
    'pd':      dict(wj_shifted=0.8611897103614303, binary=0.325638911788953,   gap=0.5355507985724773),
}

# ---------------------------------------------------------------- load + filter
print("[1/5] Loading GSE68719 normalized counts")
raw = pd.read_csv(COUNTS_FILE, sep='\t', compression='gzip')
counts = raw.drop(columns=[raw.columns[0]]).set_index('symbol')
counts = counts.groupby(level=0).mean()
expressed = counts[counts.median(axis=1) >= MIN_EXPR].copy()
genes = expressed.index.to_numpy()
X = expressed.to_numpy(dtype=np.float64)               # G x S
col = list(expressed.columns)
print(f"  expressed genes: {X.shape[0]}, samples: {X.shape[1]}")

control_cols = [i for i, c in enumerate(col) if str(c).startswith('C_')]
pd_cols      = [i for i, c in enumerate(col) if str(c).startswith('P_')]
print(f"  control n={len(control_cols)}  PD n={len(pd_cols)}")
assert len(control_cols) >= 10 and len(pd_cols) >= 10, "stratification failed"

eif_idx = int(np.where(genes == 'EIF2S1')[0][0])
pelo_idx = int(np.where(genes == 'PELO')[0][0])
keep_mask = np.ones(X.shape[0], dtype=bool)
keep_mask[eif_idx] = False
keep_mask[pelo_idx] = False   # profiles compared over all genes != EIF2S1,PELO

# ---------------------------------------------------------------- gap machinery
def group_gap(sample_idx):
    """Type 1 gap (WJ_shifted - binaryJ_top5) for EIF2S1 vs PELO in a sample subset."""
    sub = X[:, sample_idx]                              # G x n
    R = stats.rankdata(sub, axis=1)                     # rank across samples (ties averaged)
    Rc = R - R.mean(axis=1, keepdims=True)
    denom = np.sqrt((Rc ** 2).sum(axis=1))
    denom[denom == 0] = 1e-12
    def rho_of(idx):
        t = Rc[idx]
        td = denom[idx]
        return (Rc @ t) / (denom * td)                 # Spearman of every gene vs target
    a = rho_of(eif_idx)[keep_mask]
    b = rho_of(pelo_idx)[keep_mask]
    # WJ shifted
    a_s, b_s = (a + 1) / 2, (b + 1) / 2
    wj = np.minimum(a_s, b_s).sum() / np.maximum(a_s, b_s).sum()
    # binary Jaccard top-5%
    ta, tb = np.percentile(a, 100 - TOP_PCT), np.percentile(b, 100 - TOP_PCT)
    ina, inb = a >= ta, b >= tb
    bj = (ina & inb).sum() / (ina | inb).sum()
    return wj, bj, wj - bj

# ---------------------------------------------------------------- sanity gate
print("\n[2/5] Reproducing published control/PD gaps (sanity gate)")
ctl = group_gap(control_cols)
pdg = group_gap(pd_cols)
for name, got, pub in [('control', ctl, PUBLISHED['control']), ('pd', pdg, PUBLISHED['pd'])]:
    print(f"  {name}: WJ={got[0]:.4f} (pub {pub['wj_shifted']:.4f}) | "
          f"binaryJ={got[1]:.4f} (pub {pub['binary']:.4f}) | gap={got[2]:.4f} (pub {pub['gap']:.4f})")
obs_gap_diff = pdg[2] - ctl[2]                 # PD - control
obs_binary_drop = ctl[1] - pdg[1]              # control - PD
print(f"  OBSERVED gap difference (PD - control): {obs_gap_diff:.4f}")
print(f"  OBSERVED binary-Jaccard drop (control - PD): {obs_binary_drop:.4f}")
gate_ok = (abs(ctl[2] - PUBLISHED['control']['gap']) < 0.02 and
           abs(pdg[2] - PUBLISHED['pd']['gap']) < 0.02)
print(f"  sanity gate {'PASS' if gate_ok else 'FAIL - investigate before trusting test'}")

# ---------------------------------------------------------------- permutation
print(f"\n[3/5] Permutation test ({N_PERM} relabelings)")
pooled = np.array(control_cols + pd_cols)
n_ctl = len(control_cols)
t0 = time.time()
perm_gap_diff = np.empty(N_PERM)
perm_binary_drop = np.empty(N_PERM)
for k in range(N_PERM):
    perm = np.random.permutation(pooled)
    g_ctl = group_gap(perm[:n_ctl])
    g_pd = group_gap(perm[n_ctl:])
    perm_gap_diff[k] = g_pd[2] - g_ctl[2]
    perm_binary_drop[k] = g_ctl[1] - g_pd[1]
    if (k + 1) % 500 == 0:
        print(f"    {k+1}/{N_PERM}  ({time.time()-t0:.0f}s)")
p_gap = (np.sum(np.abs(perm_gap_diff) >= abs(obs_gap_diff)) + 1) / (N_PERM + 1)
p_binary = (np.sum(np.abs(perm_binary_drop) >= abs(obs_binary_drop)) + 1) / (N_PERM + 1)
print(f"  permutation p (gap difference, two-sided):    {p_gap:.4f}")
print(f"  permutation p (binary-Jaccard drop, 2-sided): {p_binary:.4f}")

# ---------------------------------------------------------------- bootstrap CI
print(f"\n[4/5] Bootstrap 95% CI on the gap difference ({N_BOOT} iters)")
ctl_arr = np.array(control_cols); pd_arr = np.array(pd_cols)
boot = np.empty(N_BOOT)
for k in range(N_BOOT):
    bc = np.random.choice(ctl_arr, size=len(ctl_arr), replace=True)
    bp = np.random.choice(pd_arr, size=len(pd_arr), replace=True)
    boot[k] = group_gap(bp)[2] - group_gap(bc)[2]
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
print(f"  bootstrap mean diff: {boot.mean():.4f}  95% CI [{ci_lo:.4f}, {ci_hi:.4f}]")
ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
print(f"  CI excludes zero: {ci_excludes_zero}")

# ---------------------------------------------------------------- write outputs
print("\n[5/5] Writing outputs")
out = pd.DataFrame([{
    'control_gap': ctl[2], 'pd_gap': pdg[2],
    'observed_gap_difference_pd_minus_control': obs_gap_diff,
    'permutation_p_gap_difference': p_gap,
    'bootstrap_ci_low': ci_lo, 'bootstrap_ci_high': ci_hi,
    'bootstrap_ci_excludes_zero': ci_excludes_zero,
    'control_binaryJ': ctl[1], 'pd_binaryJ': pdg[1],
    'observed_binary_drop_control_minus_pd': obs_binary_drop,
    'permutation_p_binary_drop': p_binary,
    'n_control': len(control_cols), 'n_pd': len(pd_cols),
    'n_permutations': N_PERM, 'n_bootstrap': N_BOOT,
}])
out.to_csv(os.path.join(RESULTS_DIR, "disease_between_group_significance.csv"), index=False)
prov = {
    "methodology": "WJ-native; between-group permutation + bootstrap on Type 1 dissociation gap",
    "fundamental_unit": "individual gene",
    "correlation_method": "Spearman (coupling profiles)",
    "gap_definition": "WJ_shifted((r+1)/2) minus binary Jaccard top-5%",
    "null_model": "exchangeable group labels (permutation of PD/control assignment)",
    "random_seed": RANDOM_SEED, "n_permutations": N_PERM, "n_bootstrap": N_BOOT,
    "pipeline_file": "pipeline_7_disease_significance.py",
    "execution_date": "2026-07-03",
    "data_source": "GEO GSE68719 BA9 prefrontal cortex RNA-seq (PD vs control)",
    "sanity_gate_reproduces_published_gaps": bool(gate_ok),
    "observed_gap_difference": float(obs_gap_diff),
    "permutation_p_gap_difference": float(p_gap),
    "bootstrap_95ci": [float(ci_lo), float(ci_hi)],
    "permutation_p_binary_drop": float(p_binary),
}
with open(os.path.join(RESULTS_DIR, "provenance_pipeline7.json"), 'w') as f:
    json.dump(prov, f, indent=2)

print("\n" + "=" * 68)
print("RESULT SUMMARY")
print(f"  control gap {ctl[2]:.3f} -> PD gap {pdg[2]:.3f}  (diff {obs_gap_diff:+.3f})")
print(f"  permutation p = {p_gap:.4f}   bootstrap 95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
print(f"  binary drop {obs_binary_drop:.3f}, permutation p = {p_binary:.4f}")
verdict = "SIGNIFICANT" if p_gap < 0.05 else "NOT SIGNIFICANT"
print(f"  >>> between-group gap difference is {verdict} at alpha=0.05")
print("=" * 68)
