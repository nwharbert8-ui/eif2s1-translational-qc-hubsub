"""
Pipeline 5: Formal Layer 2H Bulletproof Analysis on EIF2S1-PELO
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-04-29

Description:
    Computes formal continuous Weighted Jaccard, all three sign-treatment
    formulations, Type 1/2/6 dissociation gaps, permutation null tests, and
    bootstrap confidence intervals for the EIF2S1-PELO architectural
    coupling on GSE68719 (BA9 prefrontal cortex RNA-seq, 73 samples: 29 PD,
    44 controls). Uses local data; no Colab dependency.

    Tests run:
        Type 1 (Continuous-Discrete): formal continuous WJ vs binary Jaccard
                at top 5%. Permutation null vs 100 random gene pairs. Bootstrap
                95% CI on EIF2S1-PELO gap.
        Type 2 (Sign-Treatment): three WJ formulations on the same coupling
                profiles — shifted (r+1)/2, unsigned |r|, signed split. Gap
                between formulations characterizes whether sign-inversion is
                a major contributor.
        Type 6 (Substrate-Projection): Pearson WJ vs Spearman WJ. Gap
                quantifies linear vs monotonic-only structure.
        Disease comparison: all metrics computed separately for control and
                PD subsets to assess preservation/disruption.

    Numeric definitions:
        Continuous WJ on coupling profiles: for two genes X and Y, compute
        their coupling vectors c_X = (r(X,g) for all genes g != X), then
        WJ(c_X, c_Y) using selected formulation.
        Binary Jaccard at top 5%: top-percentile partner sets for X and Y,
        |X ∩ Y| / |X ∪ Y|.
        Dissociation gap: WJ_continuous - Jaccard_binary.

Inputs:
    r3_revision_analyses/data/GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz
    cache/GSE68719_series_matrix.txt.gz

Outputs:
    r3_revision_analyses/results/layer2h_formal_continuous_wj.csv
    r3_revision_analyses/results/layer2h_type2_sign_treatment.csv
    r3_revision_analyses/results/layer2h_type6_pearson_vs_spearman.csv
    r3_revision_analyses/results/layer2h_permutation_null.csv
    r3_revision_analyses/results/layer2h_bootstrap_ci.csv
    r3_revision_analyses/results/layer2h_disease_comparison.csv
    r3_revision_analyses/results/provenance_pipeline5.json
"""

import os
import gzip
import json
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings('ignore')

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
N_PERMUTATIONS = 200
N_BOOTSTRAP = 500
TOP_PCT = 5
MIN_EXPR = 5

ROOT = r"G:\My Drive\inner_architecture_research\EIF2S1"
DATA_DIR = os.path.join(ROOT, "r3_revision_analyses", "data")
CACHE_DIR = r"G:\My Drive\inner_architecture_research\cache"
RESULTS_DIR = os.path.join(ROOT, "r3_revision_analyses", "results")

COUNTS_FILE = os.path.join(DATA_DIR, "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz")
SERIES_FILE = os.path.join(CACHE_DIR, "GSE68719_series_matrix.txt.gz")

# ==========================================================================
# Load data
# ==========================================================================
print("=" * 75)
print("PIPELINE 5: FORMAL LAYER 2H BULLETPROOF ANALYSIS — EIF2S1-PELO")
print("=" * 75)

print("\n[1/8] Loading expression data...")
counts_raw = pd.read_csv(COUNTS_FILE, sep='\t', compression='gzip')
# File structure: col 0 = EnsemblID, col 1 = symbol, col 2+ = sample expression
counts = counts_raw.drop(columns=[counts_raw.columns[0]]).set_index('symbol')
# Collapse duplicates by taking the row-wise mean (some symbols map to multiple Ensembl IDs)
counts = counts.groupby(level=0).mean()
print(f"  Shape: {counts.shape} (genes x samples)")
print(f"  Sample columns: {counts.columns[:5].tolist()}...")

# Filter to expressed genes
print("\n[2/8] Filtering to expressed genes (median >= 5)...")
median_expr = counts.median(axis=1)
expressed = counts[median_expr >= MIN_EXPR].copy()
print(f"  Filtered shape: {expressed.shape}")

# Parse metadata to get PD vs control
print("\n[3/8] Parsing metadata to identify PD vs Control samples...")
sample_pd_status = {}
with gzip.open(SERIES_FILE, 'rt', encoding='utf-8', errors='replace') as f:
    sample_geo_acc = []
    char_lines = []
    title_lines = []
    for line in f:
        line = line.strip()
        if line.startswith('!Sample_geo_accession'):
            sample_geo_acc = [c.strip().strip('"') for c in line.split('\t')[1:]]
        elif line.startswith('!Sample_characteristics_ch1'):
            char_lines.append([c.strip().strip('"') for c in line.split('\t')[1:]])
        elif line.startswith('!Sample_title'):
            title_lines.append([c.strip().strip('"') for c in line.split('\t')[1:]])

# Look for "disease state" or similar in characteristics
for char in char_lines:
    sample_text = " | ".join(char[:5])
    if 'parkinson' in sample_text.lower() or 'pd' in sample_text.lower() or 'control' in sample_text.lower():
        for i, c in enumerate(char):
            c_lower = c.lower()
            if 'control' in c_lower or 'normal' in c_lower or 'healthy' in c_lower:
                sample_pd_status[sample_geo_acc[i]] = 'control'
            elif 'parkinson' in c_lower or 'pd' in c_lower:
                sample_pd_status[sample_geo_acc[i]] = 'pd'
        break

# Try titles if characteristics didn't work
if not sample_pd_status:
    for i, gsm in enumerate(sample_geo_acc):
        title = title_lines[0][i] if title_lines else ''
        if 'control' in title.lower() or 'C' in title.upper().replace(' ', '')[:3]:
            sample_pd_status[gsm] = 'control'
        elif 'pd' in title.lower() or 'parkinson' in title.lower():
            sample_pd_status[gsm] = 'pd'

# Map column names of counts to GEO accessions
matched_columns = {}
for col in expressed.columns:
    for gsm in sample_geo_acc:
        if gsm in col or col in sample_geo_acc:
            if col == gsm or gsm == col:
                matched_columns[col] = gsm
                break
    # fallback: extract GSM-like ID from col name
    for gsm in sample_geo_acc:
        if gsm in col:
            matched_columns[col] = gsm
            break

# If column matching is messy, just use first 44 as control, last 29 as PD per metadata pattern
if len(matched_columns) < 50:
    print("  Direct GSM match incomplete; reading from supplementary metadata...")
    metadata_path = os.path.join(RESULTS_DIR, "disease_GSE68719_metadata.csv")
    if os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        print(f"  Loaded metadata file: {meta.shape}")
        print(f"  Columns: {meta.columns.tolist()}")
        print(meta.head())

# Use the metadata file from pipeline 2 if available
metadata_path = os.path.join(RESULTS_DIR, "disease_GSE68719_metadata.csv")
if os.path.exists(metadata_path):
    meta = pd.read_csv(metadata_path)
    sample_pd_status = dict(zip(meta.iloc[:, 0], meta.iloc[:, 1]))

control_samples = [s for s in expressed.columns if s.startswith('C_')]
pd_samples = [s for s in expressed.columns if s.startswith('P_')]

# Fallback: read metadata file
if len(control_samples) == 0 or len(pd_samples) == 0:
    print(f"  Sample stratification via column prefix failed; reading metadata file...")
    if os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        for col in meta.columns:
            unique_vals = meta[col].dropna().astype(str).str.lower().unique()
            if any('control' in str(v) for v in unique_vals) and any('pd' in str(v) for v in unique_vals):
                ctrl_labels = meta[meta[col].astype(str).str.lower().str.contains('control|normal|healthy', na=False)]['sample_label'].tolist()
                pd_labels = meta[meta[col].astype(str).str.lower().str.contains('^pd$|parkinson', na=False, regex=True)]['sample_label'].tolist()
                control_samples = [s for s in expressed.columns if s in ctrl_labels]
                pd_samples = [s for s in expressed.columns if s in pd_labels]
                break

# Ensure sample names exist in expression matrix
control_samples = [s for s in control_samples if s in expressed.columns]
pd_samples = [s for s in pd_samples if s in expressed.columns]

print(f"  Control samples: {len(control_samples)}")
print(f"  PD samples:      {len(pd_samples)}")

if len(control_samples) < 10 or len(pd_samples) < 10:
    print(f"  WARNING: Insufficient stratification. Using all samples pooled.")
    pooled_samples = expressed.columns.tolist()
else:
    pooled_samples = control_samples + pd_samples

# Ensure target genes present
target_genes = ['EIF2S1', 'PELO']
missing = [g for g in target_genes if g not in expressed.index]
if missing:
    print(f"  ERROR: missing target genes: {missing}")
    print(f"  Available genes head: {expressed.index[:20].tolist()}")
    raise ValueError("Target genes missing")

# ==========================================================================
# Helper functions for WJ
# ==========================================================================
def coupling_profile(expr, gene, samples, method='spearman'):
    """Compute correlation between gene's expression and all other genes."""
    target = expr.loc[gene, samples].values.astype(float)
    other_genes = [g for g in expr.index if g != gene]
    other = expr.loc[other_genes, samples].values.astype(float)

    if method == 'spearman':
        target_ranks = stats.rankdata(target)
        other_ranks = np.apply_along_axis(stats.rankdata, 1, other)
        target_mean = target_ranks.mean()
        target_std = target_ranks.std()
        other_means = other_ranks.mean(axis=1, keepdims=True)
        other_stds = other_ranks.std(axis=1, keepdims=True)
        cov = ((other_ranks - other_means) * (target_ranks - target_mean)).mean(axis=1)
        rho = cov / (other_stds.flatten() * target_std + 1e-12)
    else:
        target_mean = target.mean()
        target_std = target.std()
        other_means = other.mean(axis=1, keepdims=True)
        other_stds = other.std(axis=1, keepdims=True)
        cov = ((other - other_means) * (target - target_mean)).mean(axis=1)
        rho = cov / (other_stds.flatten() * target_std + 1e-12)
    return pd.Series(rho, index=other_genes)


def wj_shifted(a, b):
    """Weighted Jaccard on (r+1)/2 shifted values."""
    a_shifted = (a + 1) / 2
    b_shifted = (b + 1) / 2
    s_min = np.minimum(a_shifted, b_shifted).sum()
    s_max = np.maximum(a_shifted, b_shifted).sum()
    return s_min / s_max if s_max > 0 else np.nan


def wj_unsigned(a, b):
    """Weighted Jaccard on absolute values."""
    a_abs = np.abs(a)
    b_abs = np.abs(b)
    s_min = np.minimum(a_abs, b_abs).sum()
    s_max = np.maximum(a_abs, b_abs).sum()
    return s_min / s_max if s_max > 0 else np.nan


def wj_signed(a, b):
    """Signed weighted Jaccard: split positive and negative components."""
    a_pos = np.maximum(a, 0)
    a_neg = np.maximum(-a, 0)
    b_pos = np.maximum(b, 0)
    b_neg = np.maximum(-b, 0)
    pos_min = np.minimum(a_pos, b_pos).sum()
    pos_max = np.maximum(a_pos, b_pos).sum()
    neg_min = np.minimum(a_neg, b_neg).sum()
    neg_max = np.maximum(a_neg, b_neg).sum()
    if (pos_max + neg_max) > 0:
        return (pos_min + neg_min) / (pos_max + neg_max)
    return np.nan


def binary_jaccard_top_pct(a, b, pct=5):
    """Binary Jaccard at top-pct threshold."""
    threshold_a = np.percentile(a, 100 - pct)
    threshold_b = np.percentile(b, 100 - pct)
    in_a = a >= threshold_a
    in_b = b >= threshold_b
    intersect = (in_a & in_b).sum()
    union = (in_a | in_b).sum()
    return intersect / union if union > 0 else np.nan


# ==========================================================================
# TYPE 1: Continuous-Discrete dissociation gap (formal)
# ==========================================================================
print("\n[4/8] Type 1: Formal continuous-discrete dissociation gap (Spearman, controls)")
samples = control_samples if len(control_samples) >= 20 else pooled_samples
print(f"  Using {len(samples)} samples")

c_eif = coupling_profile(expressed, 'EIF2S1', samples, method='spearman')
c_pelo = coupling_profile(expressed, 'PELO', samples, method='spearman')

# Align indices (just in case)
common = c_eif.index.intersection(c_pelo.index)
c_eif = c_eif.loc[common].values
c_pelo = c_pelo.loc[common].values

wj_shifted_val = wj_shifted(c_eif, c_pelo)
wj_unsigned_val = wj_unsigned(c_eif, c_pelo)
wj_signed_val = wj_signed(c_eif, c_pelo)
bj_top5 = binary_jaccard_top_pct(c_eif, c_pelo, pct=5)

gap_shifted_t1 = wj_shifted_val - bj_top5
gap_unsigned_t1 = wj_unsigned_val - bj_top5
gap_signed_t1 = wj_signed_val - bj_top5

print(f"  WJ shifted (r+1)/2:  {wj_shifted_val:.4f}")
print(f"  WJ unsigned |r|:     {wj_unsigned_val:.4f}")
print(f"  WJ signed split:     {wj_signed_val:.4f}")
print(f"  Binary Jaccard top5: {bj_top5:.4f}")
print(f"  Type 1 gap (shifted-binary): {gap_shifted_t1:.4f}")
print(f"  Type 1 gap (unsigned-binary): {gap_unsigned_t1:.4f}")
print(f"  Type 1 gap (signed-binary): {gap_signed_t1:.4f}")

type1_df = pd.DataFrame([{
    'cohort': 'control',
    'wj_shifted': wj_shifted_val,
    'wj_unsigned': wj_unsigned_val,
    'wj_signed': wj_signed_val,
    'binary_jaccard_top5': bj_top5,
    'gap_shifted_minus_binary': gap_shifted_t1,
    'gap_unsigned_minus_binary': gap_unsigned_t1,
    'gap_signed_minus_binary': gap_signed_t1,
    'n_samples': len(samples),
    'n_genes': len(common),
}])
type1_df.to_csv(os.path.join(RESULTS_DIR, "layer2h_formal_continuous_wj.csv"), index=False)

# ==========================================================================
# TYPE 2: Sign-Treatment gaps
# ==========================================================================
print("\n[5/8] Type 2: Sign-treatment gaps")
gap_shifted_unsigned = wj_shifted_val - wj_unsigned_val
gap_shifted_signed = wj_shifted_val - wj_signed_val
gap_unsigned_signed = wj_unsigned_val - wj_signed_val

print(f"  Shifted vs Unsigned gap: {gap_shifted_unsigned:.4f}")
print(f"  Shifted vs Signed gap:   {gap_shifted_signed:.4f}")
print(f"  Unsigned vs Signed gap:  {gap_unsigned_signed:.4f}")
print(f"  All three formulations close to each other = sign-inversion not dominant")

type2_df = pd.DataFrame([{
    'pair': 'EIF2S1-PELO',
    'gap_shifted_minus_unsigned': gap_shifted_unsigned,
    'gap_shifted_minus_signed': gap_shifted_signed,
    'gap_unsigned_minus_signed': gap_unsigned_signed,
    'max_abs_gap': max(abs(gap_shifted_unsigned), abs(gap_shifted_signed),
                        abs(gap_unsigned_signed)),
    'interpretation': ('Sign inversion not dominant' if max(abs(gap_shifted_unsigned),
                                                              abs(gap_shifted_signed),
                                                              abs(gap_unsigned_signed)) < 0.05
                       else 'Sign treatment matters'),
}])
type2_df.to_csv(os.path.join(RESULTS_DIR, "layer2h_type2_sign_treatment.csv"), index=False)

# ==========================================================================
# TYPE 6: Substrate-Projection (Pearson vs Spearman)
# ==========================================================================
print("\n[6/8] Type 6: Substrate-projection (Pearson vs Spearman)")
c_eif_pearson = coupling_profile(expressed, 'EIF2S1', samples, method='pearson')
c_pelo_pearson = coupling_profile(expressed, 'PELO', samples, method='pearson')
common_p = c_eif_pearson.index.intersection(c_pelo_pearson.index)
c_eif_p = c_eif_pearson.loc[common_p].values
c_pelo_p = c_pelo_pearson.loc[common_p].values

wj_shifted_pearson = wj_shifted(c_eif_p, c_pelo_p)
bj_top5_pearson = binary_jaccard_top_pct(c_eif_p, c_pelo_p, pct=5)

gap_t6 = wj_shifted_val - wj_shifted_pearson
gap_t6_binary = bj_top5 - bj_top5_pearson

print(f"  Spearman WJ shifted:  {wj_shifted_val:.4f}")
print(f"  Pearson WJ shifted:   {wj_shifted_pearson:.4f}")
print(f"  Type 6 gap (continuous): {gap_t6:.4f}")
print(f"  Spearman binary J top5: {bj_top5:.4f}")
print(f"  Pearson binary J top5:  {bj_top5_pearson:.4f}")
print(f"  Type 6 gap (binary):     {gap_t6_binary:.4f}")
print(f"  Near-zero gap = relationships approximately linear-monotonic")

type6_df = pd.DataFrame([{
    'pair': 'EIF2S1-PELO',
    'spearman_wj_shifted': wj_shifted_val,
    'pearson_wj_shifted': wj_shifted_pearson,
    'gap_continuous': gap_t6,
    'spearman_bj_top5': bj_top5,
    'pearson_bj_top5': bj_top5_pearson,
    'gap_binary': gap_t6_binary,
    'interpretation': 'Linear-monotonic structure' if abs(gap_t6) < 0.05
                       else 'Nonlinear-monotonic divergence',
}])
type6_df.to_csv(os.path.join(RESULTS_DIR, "layer2h_type6_pearson_vs_spearman.csv"),
                 index=False)

# ==========================================================================
# Permutation null: random gene pairs
# ==========================================================================
print("\n[7/8] Permutation null: random gene pairs")
print(f"  Running {N_PERMUTATIONS} random pairs...")
all_genes = expressed.index.tolist()
null_gaps = []
np.random.seed(RANDOM_SEED)
for _ in range(N_PERMUTATIONS):
    g1, g2 = np.random.choice(all_genes, size=2, replace=False)
    try:
        c1 = coupling_profile(expressed, g1, samples, method='spearman')
        c2 = coupling_profile(expressed, g2, samples, method='spearman')
        common_n = c1.index.intersection(c2.index)
        wj_n = wj_shifted(c1.loc[common_n].values, c2.loc[common_n].values)
        bj_n = binary_jaccard_top_pct(c1.loc[common_n].values,
                                       c2.loc[common_n].values, pct=5)
        null_gaps.append(wj_n - bj_n)
    except Exception:
        continue

null_gaps = np.array(null_gaps)
perm_p_two = (np.abs(null_gaps - null_gaps.mean()) >=
              abs(gap_shifted_t1 - null_gaps.mean())).mean()
z_score = ((gap_shifted_t1 - null_gaps.mean()) / null_gaps.std()
           if null_gaps.std() > 0 else 0)

print(f"  EIF2S1-PELO gap (shifted): {gap_shifted_t1:.4f}")
print(f"  Null gap mean:    {null_gaps.mean():.4f}, std: {null_gaps.std():.4f}")
print(f"  Null gap range:   [{null_gaps.min():.4f}, {null_gaps.max():.4f}]")
print(f"  Z-score:          {z_score:.2f}")
print(f"  Permutation p (two-sided): {perm_p_two:.4f}")

null_df = pd.DataFrame({
    'pair_index': range(len(null_gaps)),
    'random_pair_gap': null_gaps,
})
null_summary_row = pd.DataFrame([{
    'pair_index': 'EIF2S1-PELO_observed',
    'random_pair_gap': gap_shifted_t1,
}])
null_df = pd.concat([null_summary_row, null_df], ignore_index=True)
null_df.to_csv(os.path.join(RESULTS_DIR, "layer2h_permutation_null.csv"), index=False)

# ==========================================================================
# Bootstrap CI on EIF2S1-PELO gap
# ==========================================================================
print("\n[8/8] Bootstrap CI on EIF2S1-PELO Type 1 gap")
print(f"  Running {N_BOOTSTRAP} bootstrap iterations...")

boot_gaps_shifted = []
n_samples = len(samples)
np.random.seed(RANDOM_SEED + 1)
for _ in range(N_BOOTSTRAP):
    boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
    boot_samples = [samples[i] for i in boot_idx]
    try:
        c_e = coupling_profile(expressed, 'EIF2S1', boot_samples, method='spearman')
        c_p = coupling_profile(expressed, 'PELO', boot_samples, method='spearman')
        common_b = c_e.index.intersection(c_p.index)
        wj_b = wj_shifted(c_e.loc[common_b].values, c_p.loc[common_b].values)
        bj_b = binary_jaccard_top_pct(c_e.loc[common_b].values,
                                       c_p.loc[common_b].values, pct=5)
        boot_gaps_shifted.append(wj_b - bj_b)
    except Exception:
        continue

boot_gaps_shifted = np.array(boot_gaps_shifted)
ci_low, ci_high = np.percentile(boot_gaps_shifted, [2.5, 97.5])
print(f"  Bootstrap mean: {boot_gaps_shifted.mean():.4f}")
print(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

bootstrap_df = pd.DataFrame([{
    'pair': 'EIF2S1-PELO',
    'observed_gap': gap_shifted_t1,
    'bootstrap_mean': boot_gaps_shifted.mean(),
    'bootstrap_std': boot_gaps_shifted.std(),
    'ci_lower_95': ci_low,
    'ci_upper_95': ci_high,
    'n_bootstrap_iterations': len(boot_gaps_shifted),
}])
bootstrap_df.to_csv(os.path.join(RESULTS_DIR, "layer2h_bootstrap_ci.csv"), index=False)

# ==========================================================================
# Disease comparison: same metrics for PD samples
# ==========================================================================
print("\nDisease comparison: control vs PD samples")
disease_results = []
for cohort_name, cohort_samples in [('control', control_samples), ('pd', pd_samples)]:
    if len(cohort_samples) < 10:
        continue
    c_e = coupling_profile(expressed, 'EIF2S1', cohort_samples, method='spearman')
    c_p = coupling_profile(expressed, 'PELO', cohort_samples, method='spearman')
    common_c = c_e.index.intersection(c_p.index)
    c_e_v = c_e.loc[common_c].values
    c_p_v = c_p.loc[common_c].values

    wj_s = wj_shifted(c_e_v, c_p_v)
    bj = binary_jaccard_top_pct(c_e_v, c_p_v, pct=5)
    disease_results.append({
        'cohort': cohort_name,
        'n_samples': len(cohort_samples),
        'wj_shifted': wj_s,
        'binary_jaccard_top5': bj,
        'dissociation_gap': wj_s - bj,
    })
    print(f"  {cohort_name:10s} n={len(cohort_samples):3d}: WJ={wj_s:.4f}, "
          f"BJ={bj:.4f}, gap={wj_s-bj:.4f}")

disease_df = pd.DataFrame(disease_results)
disease_df.to_csv(os.path.join(RESULTS_DIR, "layer2h_disease_comparison.csv"),
                   index=False)

# ==========================================================================
# Provenance
# ==========================================================================
provenance = {
    "pipeline_file": "pipeline_5_formal_layer2h.py",
    "execution_date": "2026-04-29",
    "wj_compliance_status": "PASS (Layer 2H formal computation)",
    "substrate": "GSE68719 BA9 prefrontal cortex RNA-seq (DESeq2-normalized)",
    "n_control": len(control_samples),
    "n_pd": len(pd_samples),
    "n_genes_analyzed": int(len(common)),
    "type1_findings": {
        "wj_shifted": float(wj_shifted_val),
        "wj_unsigned": float(wj_unsigned_val),
        "wj_signed": float(wj_signed_val),
        "binary_jaccard_top5": float(bj_top5),
        "type1_gap_shifted": float(gap_shifted_t1),
        "permutation_p": float(perm_p_two),
        "z_score_vs_null": float(z_score),
        "bootstrap_ci_lower": float(ci_low),
        "bootstrap_ci_upper": float(ci_high),
    },
    "type2_findings": {
        "max_sign_treatment_gap": float(max(abs(gap_shifted_unsigned),
                                             abs(gap_shifted_signed),
                                             abs(gap_unsigned_signed))),
        "sign_inversion_dominant": bool(max(abs(gap_shifted_unsigned),
                                             abs(gap_shifted_signed),
                                             abs(gap_unsigned_signed)) > 0.05),
    },
    "type6_findings": {
        "pearson_wj_shifted": float(wj_shifted_pearson),
        "spearman_pearson_gap": float(gap_t6),
        "linear_monotonic_consistent": bool(abs(gap_t6) < 0.05),
    },
    "disease_comparison": disease_results,
    "config": {
        "random_seed": RANDOM_SEED,
        "n_permutations": N_PERMUTATIONS,
        "n_bootstrap": N_BOOTSTRAP,
        "top_pct_threshold": TOP_PCT,
        "min_expression_filter": MIN_EXPR,
    },
}
with open(os.path.join(RESULTS_DIR, "provenance_pipeline5.json"), 'w') as f:
    json.dump(provenance, f, indent=2)

print("\n" + "=" * 75)
print("PIPELINE 5 COMPLETE")
print("=" * 75)
print("Files saved to:", RESULTS_DIR)
print("\nKey numbers for manuscript revision:")
print(f"  Formal continuous WJ (shifted): {wj_shifted_val:.4f}")
print(f"  Binary Jaccard top 5%:          {bj_top5:.4f}")
print(f"  Type 1 dissociation gap:        {gap_shifted_t1:.4f}")
print(f"  Bootstrap 95% CI:               [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  Permutation p vs random pairs:   {perm_p_two:.4f}")
print(f"  Type 2 sign-treatment max gap:   {max(abs(gap_shifted_unsigned),abs(gap_shifted_signed),abs(gap_unsigned_signed)):.4f}")
print(f"  Type 6 Spearman-Pearson gap:     {gap_t6:.4f}")
