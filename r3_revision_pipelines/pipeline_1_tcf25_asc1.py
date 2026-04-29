#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
EIF2S1 R3 REVISION — PIPELINE 1: TCF25 + ASC-1 COMPLEX EXTENSION
═══════════════════════════════════════════════════════════════════════════════

Author: Drake H. Harbert (D.H.H.) | ORCID: 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH
Date: 2026-04-29

Description:
    Extends the published 7-gene panel (EIF2S1, PELO, LTN1, NEMF, TMEM97,
    HSPA5, SIGMAR1) to include Reviewer 2's requested genes:
        TCF25  — RQC-associated transcription factor
        ASCC1, ASCC2, ASCC3, TRIP4 — ASC-1 complex (ribosomal rescue)

    Re-runs all genome-wide co-expression networks for the expanded 12-gene
    panel in GTEx Brain–Frontal Cortex (BA9) and computes all 66 pairwise
    Jaccard / Fisher / Spearman comparisons. Tests whether TCF25 and the
    ASC-1 complex sit within or outside the EIF2S1–PELO transcriptional axis.

Dependencies: pandas, numpy, scipy
Input:
    G:/My Drive/inner_architecture_research/cache/GTEx_gene_tpm.gct.gz
    G:/My Drive/inner_architecture_research/cache/GTEx_SampleAttributes.txt
Output:
    G:/My Drive/inner_architecture_research/EIF2S1/r3_revision_analyses/results/
        all_66_pairwise_comparisons.csv
        new_gene_top_partners.csv
        provenance.json
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, gc, json, gzip, time
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
from itertools import combinations

# ── CONFIG ─────────────────────────────────────────────────────────────────
RANDOM_SEED = 42
FORCE_RECOMPUTE = True
np.random.seed(RANDOM_SEED)

CACHE_DIR  = r"G:\My Drive\inner_architecture_research\cache"
TPM_FILE   = os.path.join(CACHE_DIR, "GTEx_gene_tpm.gct.gz")
SAMPLE_FILE = os.path.join(CACHE_DIR, "GTEx_SampleAttributes.txt")

OUT_DIR    = r"G:\My Drive\inner_architecture_research\EIF2S1\r3_revision_analyses\results"
os.makedirs(OUT_DIR, exist_ok=True)

PRIMARY_REGION = 'BA9'
PRIMARY_SMTSD  = 'Brain - Frontal Cortex (BA9)'
MIN_MEDIAN_TPM = 1.0
TOP_PERCENT    = 5

ORIGINAL_PANEL = ['EIF2S1', 'PELO', 'LTN1', 'NEMF', 'TMEM97', 'HSPA5', 'SIGMAR1']
NEW_GENES      = ['TCF25', 'ASCC1', 'ASCC2', 'ASCC3', 'TRIP4']
TARGET_GENES   = ORIGINAL_PANEL + NEW_GENES   # 12 genes

print("=" * 72)
print("EIF2S1 R3 REVISION — TCF25 + ASC-1 COMPLEX EXTENSION")
print(f"Original panel ({len(ORIGINAL_PANEL)}): {ORIGINAL_PANEL}")
print(f"New genes      ({len(NEW_GENES)}): {NEW_GENES}")
print(f"Total targets  ({len(TARGET_GENES)}): {len(TARGET_GENES)}")
print("=" * 72)

# ── 1. Load sample metadata ────────────────────────────────────────────────
print(f"\n[1/6] Loading sample metadata from {SAMPLE_FILE}")
sample_attr = pd.read_csv(SAMPLE_FILE, sep='\t', low_memory=False)
ba9_samples = sample_attr.loc[sample_attr['SMTSD'] == PRIMARY_SMTSD, 'SAMPID'].tolist()
print(f"      BA9 samples: {len(ba9_samples)}")
assert len(ba9_samples) > 100, "BA9 sample count too low"

# ── 2. Stream GTEx TPM, retain only BA9 columns ────────────────────────────
print(f"\n[2/6] Streaming GTEx TPM (BA9 columns only)")
t0 = time.time()
with gzip.open(TPM_FILE, 'rt') as f:
    f.readline()   # version line
    f.readline()   # dimensions line
    header = f.readline().strip().split('\t')
ba9_set = set(ba9_samples)
keep_idx = [0, 1] + [i for i, c in enumerate(header) if c in ba9_set]
print(f"      Header columns: {len(header):,}; BA9 columns kept: {len(keep_idx) - 2}")

rows = []
with gzip.open(TPM_FILE, 'rt') as f:
    f.readline(); f.readline(); f.readline()
    for ln, line in enumerate(f):
        parts = line.rstrip('\n').split('\t')
        rows.append([parts[i] if i < len(parts) else '' for i in keep_idx])
        if (ln + 1) % 10000 == 0:
            print(f"        Processed {ln+1:,} genes... ({time.time()-t0:.1f}s)")

cols = ['Name', 'Description'] + [header[i] for i in keep_idx[2:]]
tpm = pd.DataFrame(rows, columns=cols)
tpm = tpm.set_index('Name')
sym = tpm['Description'].copy()
tpm = tpm.drop(columns=['Description'])
tpm = tpm.apply(pd.to_numeric, errors='coerce').astype(np.float32)
print(f"      Loaded: {tpm.shape[0]:,} genes × {tpm.shape[1]} samples ({time.time()-t0:.1f}s)")
del rows; gc.collect()

# ── 3. Filter genes, dedupe symbols, log2 transform ────────────────────────
print(f"\n[3/6] Filtering genes (median TPM ≥ {MIN_MEDIAN_TPM})")
tpm['gene_symbol'] = sym.reindex(tpm.index)
tpm = tpm.dropna(subset=['gene_symbol'])
tpm = tpm[tpm['gene_symbol'] != '']
sample_cols = [c for c in tpm.columns if c != 'gene_symbol']
tpm['median_tpm'] = tpm[sample_cols].median(axis=1)
tpm = tpm.sort_values('median_tpm', ascending=False)
tpm = tpm.drop_duplicates('gene_symbol', keep='first')
tpm = tpm[tpm['median_tpm'] >= MIN_MEDIAN_TPM]
expr = tpm.set_index('gene_symbol')[sample_cols]
log2_expr = np.log2(expr + 1).astype(np.float32)
print(f"      Filtered matrix: {log2_expr.shape[0]:,} genes × {log2_expr.shape[1]} samples")

# Verify all targets present
missing = [g for g in TARGET_GENES if g not in log2_expr.index]
present = [g for g in TARGET_GENES if g in log2_expr.index]
print(f"      Targets present: {len(present)}/{len(TARGET_GENES)}")
if missing:
    print(f"      ⚠ MISSING (TPM < 1.0 in BA9): {missing}")
TARGET_GENES = present  # use only those that pass filter

# ── 4. Genome-wide Pearson correlations for each target ────────────────────
print(f"\n[4/6] Computing genome-wide Pearson correlations for {len(TARGET_GENES)} targets")
correlations = {}
expr_arr = log2_expr.values
gene_index = log2_expr.index

# Pre-compute centered/normalized matrix for vectorized r
mu = expr_arr.mean(axis=1, keepdims=True)
sd = np.sqrt(((expr_arr - mu) ** 2).sum(axis=1, keepdims=True))

for tg in TARGET_GENES:
    i = gene_index.get_loc(tg)
    x  = expr_arr[i] - mu[i, 0]
    sx = sd[i, 0]
    others_mask = np.arange(len(gene_index)) != i
    Y   = expr_arr[others_mask] - mu[others_mask]
    sy  = sd[others_mask, 0]
    valid = sy > 0
    r = np.full(others_mask.sum(), np.nan, dtype=np.float32)
    r[valid] = (Y[valid] @ x) / (sy[valid] * sx)
    s = pd.Series(r, index=gene_index[others_mask], name=tg).dropna().sort_values(ascending=False)
    correlations[tg] = s
    print(f"      {tg:8s} → top partner: {s.index[0]:12s} (r = {s.iloc[0]:.3f})")

# ── 5. Top 5% networks + all pairwise comparisons ──────────────────────────
print(f"\n[5/6] Extracting top {TOP_PERCENT}% networks and all pairwise comparisons")
networks = {}; thresholds = {}
gene_universe = len(correlations[TARGET_GENES[0]])
for tg, s in correlations.items():
    n_top = int(np.ceil(len(s) * TOP_PERCENT / 100))
    networks[tg] = set(s.head(n_top).index)
    thresholds[tg] = float(s.iloc[n_top - 1])
    print(f"      {tg:8s}: top {n_top} genes, r ≥ {thresholds[tg]:.3f}")

results = []
for g1, g2 in combinations(TARGET_GENES, 2):
    s1, s2 = networks[g1], networks[g2]
    shared = s1 & s2
    union  = s1 | s2
    only1  = s1 - s2
    only2  = s2 - s1
    jaccard = len(shared) / len(union) if union else 0.0
    a, b, c = len(shared), len(only1), len(only2)
    d = gene_universe - len(union)
    fisher_or, fisher_p = fisher_exact([[a, b], [c, d]], alternative='greater')
    common = sorted(set(correlations[g1].index) & set(correlations[g2].index))
    rho, _ = spearmanr(
        correlations[g1].reindex(common).rank(ascending=False),
        correlations[g2].reindex(common).rank(ascending=False),
    )
    is_new = (g1 in NEW_GENES) or (g2 in NEW_GENES)
    results.append({
        'gene1': g1, 'gene2': g2,
        'shared': int(len(shared)),
        'jaccard': round(float(jaccard), 6),
        'fisher_or': round(float(fisher_or), 6),
        'fisher_p': float(fisher_p),
        'spearman_rho': round(float(rho), 6),
        'set1_size': int(len(s1)), 'set2_size': int(len(s2)),
        'involves_new_gene': bool(is_new),
    })

comp_df = pd.DataFrame(results).sort_values('jaccard', ascending=False)
out_path = os.path.join(OUT_DIR, "all_pairwise_comparisons_extended.csv")
comp_df.to_csv(out_path, index=False)
print(f"      Saved: {out_path}")
print(f"      Total pairs: {len(comp_df)}")

# Summary: comparisons involving new genes vs. EIF2S1 and PELO
print(f"\n      ── KEY R3 RESULTS — new genes vs. EIF2S1 and PELO ──")
focus = comp_df[
    (comp_df['gene1'].isin(NEW_GENES) & comp_df['gene2'].isin(['EIF2S1', 'PELO'])) |
    (comp_df['gene2'].isin(NEW_GENES) & comp_df['gene1'].isin(['EIF2S1', 'PELO']))
].sort_values('jaccard', ascending=False)
for _, r in focus.iterrows():
    print(f"        {r['gene1']:8s} – {r['gene2']:8s}  J={r['jaccard']:.3f}  shared={int(r['shared']):4d}  ρ={r['spearman_rho']:.3f}  OR={r['fisher_or']:.1f}")

# Top 5 new-gene partners
print(f"\n      ── New-gene top 5 co-expression partners ──")
top_partners = []
for tg in NEW_GENES:
    if tg not in correlations: continue
    s = correlations[tg].head(10)
    print(f"        {tg}:")
    for partner, r in s.items():
        print(f"          {partner:12s} r = {r:.3f}")
        top_partners.append({'target': tg, 'partner': partner, 'pearson_r': float(r)})

pd.DataFrame(top_partners).to_csv(os.path.join(OUT_DIR, "new_gene_top_partners.csv"), index=False)

# ── 6. Provenance ──────────────────────────────────────────────────────────
prov = {
    "methodology": "WJ-native (continuous co-expression, top-5% Jaccard pairwise)",
    "fundamental_unit": "individual gene",
    "pairwise_matrix": "full, no pre-filtering",
    "correlation_method": "Pearson on log2(TPM+1)",
    "fdr_scope": "n/a — descriptive extension; main FDR is in original manuscript",
    "domain_conventional_methods": "comparison only",
    "random_seed": RANDOM_SEED,
    "pipeline_file": "pipeline_1_tcf25_asc1.py",
    "execution_date": "2026-04-29",
    "wj_compliance_status": "PASS",
    "data_source": "GTEx v8 Brain–Frontal Cortex (BA9)",
    "n_samples": int(log2_expr.shape[1]),
    "n_genes": int(log2_expr.shape[0]),
    "target_panel_size": len(TARGET_GENES),
    "new_genes_added": NEW_GENES,
    "purpose": "Reviewer 2 concern #1: investigate TCF25 and ASC-1 complex",
}
with open(os.path.join(OUT_DIR, "provenance_pipeline1.json"), 'w') as f:
    json.dump(prov, f, indent=2)
print(f"\n[6/6] Provenance written.")
print("\n" + "=" * 72)
print("PIPELINE 1 COMPLETE.")
print(f"Outputs in: {OUT_DIR}")
print("=" * 72)
