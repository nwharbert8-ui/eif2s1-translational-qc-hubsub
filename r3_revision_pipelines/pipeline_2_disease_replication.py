#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
EIF2S1 R3 REVISION — PIPELINE 2: DISEASE DATASET REPLICATION (PD BA9 RNA-seq)
═══════════════════════════════════════════════════════════════════════════════

Author: Drake H. Harbert (D.H.H.) | ORCID: 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH
Date: 2026-04-29

Description:
    Per Reviewer 4: tests whether the constitutive EIF2S1–PELO closed-loop
    architecture replicates and/or differs in a neurodegenerative disease
    cohort. Uses GSE68719 — Parkinson Disease vs. control prefrontal cortex
    (BA9, the same region as the GTEx primary analysis). RNA-seq with
    DESeq2-normalized counts and 73 samples (29 PD + 44 controls).

    R4 specifically suggested ALS or AD; we use PD because:
        - it is a neurodegenerative disease in R4's intended class
        - the brain region (BA9 prefrontal cortex) matches the manuscript
        - it is RNA-seq (same modality as the GTEx primary analysis)
        - sample size is the largest available BA9 disease vs control set

    Pipeline:
        1. Acquire GSE68719 series matrix and normalized counts
        2. Stratify PD vs Control via Sample_characteristics_ch1
        3. Compute genome-wide Pearson co-expression for the original
           7-gene panel in each group separately and pooled
        4. Compare:
            (a) Within-group EIF2S1–PELO Jaccard (PD vs Control)
            (b) Cross-group genome-wide rank rho for each target
            (c) Top-5% network overlap between groups

Dependencies: pandas, numpy, scipy
Input:
    GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz (downloaded if not cached)
    GSE68719_series_matrix.txt.gz (cached)
Output:
    G:/My Drive/inner_architecture_research/EIF2S1/r3_revision_analyses/results/
        disease_GSE68719_metadata.csv
        disease_GSE68719_pairwise.csv
        disease_GSE68719_cross_group_summary.csv
        provenance_pipeline2.json
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, gc, json, gzip, time, urllib.request
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact
from itertools import combinations

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CACHE_DIR = r"G:\My Drive\inner_architecture_research\cache"
DATA_DIR  = r"G:\My Drive\inner_architecture_research\EIF2S1\r3_revision_analyses\data"
OUT_DIR   = r"G:\My Drive\inner_architecture_research\EIF2S1\r3_revision_analyses\results"
os.makedirs(DATA_DIR, exist_ok=True); os.makedirs(OUT_DIR, exist_ok=True)

GEO_ACC     = "GSE68719"
SERIES_FILE = os.path.join(CACHE_DIR, f"{GEO_ACC}_series_matrix.txt.gz")
COUNTS_URL  = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68719/suppl/GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz"
COUNTS_FILE = os.path.join(DATA_DIR, "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz")

TARGET_GENES = ['EIF2S1', 'PELO', 'LTN1', 'NEMF', 'TMEM97', 'HSPA5', 'SIGMAR1']
TOP_PERCENT = 5
MIN_MEDIAN_NORM = 5  # DESeq2 normalized count threshold

print("=" * 72)
print(f"EIF2S1 R3 REVISION — PIPELINE 2: DISEASE REPLICATION ({GEO_ACC} / PD BA9 RNA-seq)")
print("=" * 72)

# ── 1. Acquire counts file ─────────────────────────────────────────────────
print("\n[1/5] Acquiring data")
def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        print(f"      Cached: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return
    print(f"      Downloading: {url}")
    urllib.request.urlretrieve(url, path)
    print(f"      Saved: {path} ({os.path.getsize(path)/1e6:.1f} MB)")

download(COUNTS_URL, COUNTS_FILE)
print(f"      Series matrix cached at: {SERIES_FILE}")

# ── 2. Parse metadata and stratify groups ──────────────────────────────────
print(f"\n[2/5] Parsing series matrix metadata")
meta_lines = []
with gzip.open(SERIES_FILE, 'rt', encoding='utf-8', errors='ignore') as f:
    for ln in f:
        ln = ln.rstrip('\n')
        if ln.startswith('!series_matrix_table_begin'): break
        meta_lines.append(ln)

meta = {}
for ln in meta_lines:
    if not ln.startswith('!'): continue
    parts = ln.split('\t')
    key = parts[0].lstrip('!')
    vals = [p.strip('"') for p in parts[1:]]
    meta.setdefault(key, []).append(vals)

gsm_ids = meta.get('Sample_geo_accession', [[]])[0]
titles  = meta.get('Sample_title', [[]])[0]
print(f"      Samples: {len(gsm_ids)}")

sample_df = pd.DataFrame({'GSM': gsm_ids, 'title': titles})
char_rows = meta.get('Sample_characteristics_ch1', [])
char_field_names = []
for i, row in enumerate(char_rows):
    if not row or len(row) != len(gsm_ids): continue
    field_name = None
    for c in row:
        if ':' in c:
            field_name = c.split(':', 1)[0].strip(); break
    if field_name is None: field_name = f"char_{i}"
    char_field_names.append(field_name)
    col_name = field_name if field_name not in sample_df.columns else f"{field_name}_{i}"
    sample_df[col_name] = [c.split(':', 1)[1].strip() if ':' in c else c for c in row]

print(f"      Metadata columns: {list(sample_df.columns)}")
for col in sample_df.columns[2:]:
    counts = sample_df[col].value_counts().head(8)
    print(f"      {col}:\n        " + counts.to_string().replace('\n', '\n        '))

# Stratify — GSE68719 uses C_xxxx / P_xxxx title prefixes
def tag_group(row):
    t = str(row.get('title', ''))
    # Parse the leading token (e.g. "C_0002" or "P_0003")
    leading = t.split()[0] if t.split() else ''
    if leading.startswith('C_') or leading.startswith('CONTROL') or leading.startswith('Control_'):
        return 'Control'
    if leading.startswith('P_') or leading.startswith('PD_') or leading.startswith('Parkinson_'):
        return 'PD'
    for col in sample_df.columns[2:]:
        v = str(row.get(col, '')).lower()
        if "parkinson" in v or v == 'pd' or 'pd patient' in v:
            return 'PD'
        if 'control' in v or 'normal' in v:
            return 'Control'
    return 'unknown'

# Add a 'sample_label' column extracting the leading token (e.g. "C_0002")
sample_df['sample_label'] = sample_df['title'].str.split().str[0]

sample_df['group'] = sample_df.apply(tag_group, axis=1)
print(f"\n      Group counts:\n{sample_df['group'].value_counts().to_string()}")
sample_df.to_csv(os.path.join(OUT_DIR, "disease_GSE68719_metadata.csv"), index=False)

pd_gsms  = sample_df.loc[sample_df['group'] == 'PD', 'GSM'].tolist()
ctl_gsms = sample_df.loc[sample_df['group'] == 'Control', 'GSM'].tolist()
print(f"      PD GSMs:      {len(pd_gsms)}")
print(f"      Control GSMs: {len(ctl_gsms)}")

# ── 3. Load normalized counts ──────────────────────────────────────────────
print(f"\n[3/5] Loading normalized counts")
counts = pd.read_csv(COUNTS_FILE, sep='\t', low_memory=False)
print(f"      Shape: {counts.shape}; First columns: {list(counts.columns[:4])}")

# Identify ID and symbol columns
id_col = counts.columns[0]
print(f"      First 5 IDs: {list(counts[id_col].head(5))}")

# Try to detect: is the first column a gene symbol or an Ensembl ID?
sample_ids = list(counts[id_col].head(20))
looks_like_ensembl = any(str(x).startswith(('ENSG', 'ENSMUSG')) for x in sample_ids)
print(f"      First column looks like Ensembl: {looks_like_ensembl}")

if looks_like_ensembl:
    # Look for a gene symbol column among the first few columns. Prefer a column
    # explicitly labeled 'symbol' / 'gene_symbol' / 'GeneSymbol'; fall back to a
    # full-column scan against TARGET_GENES if not found.
    sym_col = None
    for c in counts.columns[1:8]:
        if c.lower() in ('symbol', 'gene_symbol', 'genesymbol', 'gene_name', 'genename'):
            sym_col = c; break
    if sym_col is None:
        for c in counts.columns[1:8]:
            sample_vals = counts[c].astype(str).head(2000)
            if sample_vals.isin(TARGET_GENES).any(): sym_col = c; break
    print(f"      Detected symbol column: {sym_col}")
    if sym_col:
        # Preserve EnsemblID for reference but use symbol as primary index
        counts = counts.set_index(sym_col)
        # Drop other ID columns that are not numeric samples
        for c in [id_col]:
            if c in counts.columns: counts = counts.drop(columns=[c])
    else:
        # Use Ensembl ID column; we'll need to map later
        counts = counts.set_index(id_col)
else:
    counts = counts.set_index(id_col)

# Drop duplicates: keep highest median
sample_cols_in_counts = [c for c in counts.columns if c.startswith('GSM') or c in pd_gsms or c in ctl_gsms]
print(f"      Sample columns matched directly: {len(sample_cols_in_counts)}")

# If sample columns don't match GSMs directly, try title-based matching
if len(sample_cols_in_counts) < 30:
    title_to_gsm = dict(zip(sample_df['title'].str.strip(), sample_df['GSM']))
    matched = []
    for c in counts.columns:
        if c in title_to_gsm:
            matched.append(c)
    print(f"      Sample columns matched via title: {len(matched)}")
    if len(matched) > 30:
        counts = counts[matched]
        counts.columns = [title_to_gsm[c] for c in counts.columns]
        sample_cols_in_counts = list(counts.columns)

# Otherwise, try positional alignment with GSM order from metadata
if len(sample_cols_in_counts) < 30:
    # Many GSE68719 supplementary files use sample names like CONTROL_1, PD_1, etc.
    # Try matching to the title field which has those names
    print(f"      Falling back to positional or label-based matching")
    print(f"      Counts columns sample: {list(counts.columns[:8])}")
    print(f"      Title sample: {list(sample_df['title'].head(8))}")

# Remove duplicate gene rows (keep highest median)
counts = counts.apply(pd.to_numeric, errors='coerce')
counts = counts.dropna(how='all')
if counts.index.has_duplicates:
    counts = counts.assign(_med=counts.median(axis=1)) \
                   .sort_values('_med', ascending=False) \
                   .loc[lambda d: ~d.index.duplicated(keep='first')] \
                   .drop(columns=['_med'])

# Filter low-expressed genes
counts = counts[counts.median(axis=1) >= MIN_MEDIAN_NORM]
print(f"      Filtered counts: {counts.shape}")
present = [g for g in TARGET_GENES if g in counts.index]
missing = [g for g in TARGET_GENES if g not in counts.index]
print(f"      Targets present: {present}")
print(f"      Targets missing: {missing}")
if not present:
    print("      ⚠ NO TARGETS PRESENT — likely Ensembl IDs without symbol mapping. Showing first 10 IDs:")
    print(f"        {list(counts.index[:10])}")
    sys.exit(1)

# log2 transform of normalized counts
log2_expr = np.log2(counts + 1).astype(np.float32)

# ── 4. Compute correlations within groups ─────────────────────────────────
print(f"\n[4/5] Computing genome-wide correlations per group")
# Map sample columns -> group via sample_label (C_xxxx / P_xxxx)
group_cols = {'PD': [], 'Control': [], 'All': list(log2_expr.columns)}
label_to_group = dict(zip(sample_df['sample_label'].astype(str), sample_df['group']))
col_to_group = {}
for c in log2_expr.columns:
    g = label_to_group.get(c)
    if g is None:
        if c.startswith('C_'): g = 'Control'
        elif c.startswith('P_'): g = 'PD'
        else: g = 'unknown'
    col_to_group[c] = g
print(f"      Direct sample-label group assignments: PD={sum(1 for v in col_to_group.values() if v=='PD')}, "
      f"Control={sum(1 for v in col_to_group.values() if v=='Control')}, "
      f"unknown={sum(1 for v in col_to_group.values() if v=='unknown')}")

for c, g in col_to_group.items():
    if g in ('PD', 'Control'): group_cols[g].append(c)

# If GSM-based columns gave 0 PD or Control matches, do title-based matching
if len(group_cols['PD']) == 0 and len(group_cols['Control']) == 0:
    print("      Sample columns do not match GSM IDs directly; trying title-based matching")
    title_to_group = dict(zip(sample_df['title'].str.strip(), sample_df['group']))
    for c in log2_expr.columns:
        g = title_to_group.get(c, 'unknown')
        if g in ('PD', 'Control'): group_cols[g].append(c)
    print(f"      Title-based PD: {len(group_cols['PD'])}, Control: {len(group_cols['Control'])}")

# If still zero, accept whatever pattern is present in column names (e.g., suffix _PD or _Control)
if len(group_cols['PD']) == 0 and len(group_cols['Control']) == 0:
    for c in log2_expr.columns:
        cl = c.lower()
        if 'pd' in cl: group_cols['PD'].append(c)
        elif 'control' in cl or 'cont' in cl or 'norm' in cl: group_cols['Control'].append(c)
    print(f"      Suffix-based PD: {len(group_cols['PD'])}, Control: {len(group_cols['Control'])}")

print(f"      Final groups: PD={len(group_cols['PD'])}, Control={len(group_cols['Control'])}, All={len(group_cols['All'])}")

def compute_corr_and_networks(label, cols):
    if len(cols) < 8:
        print(f"      Skipping {label} (n={len(cols)})")
        return None, None, 0
    sub = log2_expr[cols]
    expr_arr = sub.values
    gene_index = sub.index
    mu = expr_arr.mean(axis=1, keepdims=True)
    sd = np.sqrt(((expr_arr - mu) ** 2).sum(axis=1, keepdims=True))
    corrs, nets = {}, {}
    for tg in TARGET_GENES:
        if tg not in gene_index: continue
        i = gene_index.get_loc(tg)
        if hasattr(i, '__len__'):
            i = int(np.where(i)[0][0])
        x = expr_arr[i] - mu[i, 0]
        sx = sd[i, 0]
        if sx == 0: continue
        others_mask = np.arange(len(gene_index)) != i
        Y = expr_arr[others_mask] - mu[others_mask]
        sy = sd[others_mask, 0]
        valid = sy > 0
        r = np.full(others_mask.sum(), np.nan, dtype=np.float32)
        r[valid] = (Y[valid] @ x) / (sy[valid] * sx)
        s = pd.Series(r, index=gene_index[others_mask], name=tg).dropna().sort_values(ascending=False)
        corrs[tg] = s
        n_top = int(np.ceil(len(s) * TOP_PERCENT / 100))
        nets[tg] = set(s.head(n_top).index)
        print(f"        [{label:8s}] {tg:8s} top: {s.index[0]:14s} (r={s.iloc[0]:.3f}); top {n_top} ≥ {s.iloc[n_top-1]:.3f}")
    return corrs, nets, len(gene_index)

group_corrs = {}; group_nets = {}; group_universes = {}
for label, cols in group_cols.items():
    print(f"\n      Group: {label} (n = {len(cols)})")
    c, n, u = compute_corr_and_networks(label, cols)
    if c is not None:
        group_corrs[label] = c; group_nets[label] = n; group_universes[label] = u

# ── 5. Pairwise comparisons + cross-group stability ───────────────────────
print(f"\n[5/5] Pairwise comparisons and cross-group stability")
results = []
for label, nets in group_nets.items():
    universe = group_universes[label]
    for g1, g2 in combinations(list(nets.keys()), 2):
        s1, s2 = nets[g1], nets[g2]
        shared = s1 & s2; union = s1 | s2
        j = len(shared) / len(union) if union else 0.0
        a, b, c = len(shared), len(s1 - s2), len(s2 - s1)
        d = universe - len(union)
        fo, fp = fisher_exact([[a, b], [c, d]], alternative='greater')
        common = sorted(set(group_corrs[label][g1].index) & set(group_corrs[label][g2].index))
        rho, _ = spearmanr(
            group_corrs[label][g1].reindex(common).rank(ascending=False),
            group_corrs[label][g2].reindex(common).rank(ascending=False),
        )
        results.append({
            'group': label, 'gene1': g1, 'gene2': g2,
            'shared': int(len(shared)), 'jaccard': round(float(j), 6),
            'fisher_or': round(float(fo), 6), 'fisher_p': float(fp),
            'spearman_rho': round(float(rho), 6),
            'set1_size': int(len(s1)), 'set2_size': int(len(s2)),
        })

pd.DataFrame(results).sort_values(['group', 'jaccard'], ascending=[True, False]).to_csv(
    os.path.join(OUT_DIR, "disease_GSE68719_pairwise.csv"), index=False)

# Cross-group stability
cross_results = []
if 'PD' in group_corrs and 'Control' in group_corrs:
    for tg in TARGET_GENES:
        if tg in group_corrs['PD'] and tg in group_corrs['Control']:
            common = sorted(set(group_corrs['PD'][tg].index) & set(group_corrs['Control'][tg].index))
            if len(common) < 100: continue
            rho, p = spearmanr(
                group_corrs['PD'][tg].reindex(common).rank(ascending=False),
                group_corrs['Control'][tg].reindex(common).rank(ascending=False),
            )
            n_top = int(np.ceil(len(common) * TOP_PERCENT / 100))
            pd_top = set(group_corrs['PD'][tg].reindex(common).sort_values(ascending=False).head(n_top).index)
            ctl_top = set(group_corrs['Control'][tg].reindex(common).sort_values(ascending=False).head(n_top).index)
            cross_j = len(pd_top & ctl_top) / len(pd_top | ctl_top) if (pd_top | ctl_top) else 0
            cross_results.append({
                'target': tg,
                'pd_vs_control_rank_rho': round(float(rho), 6),
                'pd_vs_control_top5_jaccard': round(float(cross_j), 6),
                'p_rank': float(p),
                'n_common_genes': len(common),
            })
            print(f"      {tg:8s}: PD-Ctl rank ρ = {rho:.3f}, top-5% Jaccard = {cross_j:.3f}")

pd.DataFrame(cross_results).to_csv(os.path.join(OUT_DIR, "disease_GSE68719_cross_group_summary.csv"), index=False)

# Headline metric
print("\n=== HEADLINE: EIF2S1–PELO closed-loop in PD vs Control (BA9 RNA-seq) ===")
for label in ['Control', 'PD', 'All']:
    if label in group_nets and 'EIF2S1' in group_nets[label] and 'PELO' in group_nets[label]:
        s1 = group_nets[label]['EIF2S1']; s2 = group_nets[label]['PELO']
        shared = s1 & s2; union = s1 | s2
        j = len(shared) / len(union) if union else 0
        common = sorted(set(group_corrs[label]['EIF2S1'].index) & set(group_corrs[label]['PELO'].index))
        rho, _ = spearmanr(
            group_corrs[label]['EIF2S1'].reindex(common).rank(ascending=False),
            group_corrs[label]['PELO'].reindex(common).rank(ascending=False),
        )
        print(f"  {label:8s} (n={len(group_cols[label])}): J = {j:.3f}, shared = {len(shared)}, ρ = {rho:.3f}")

prov = {
    "methodology": "WJ-native (Pearson on log2(DESeq2-norm + 1); top-5% Jaccard pairwise)",
    "fundamental_unit": "individual gene",
    "pairwise_matrix": "full, no pre-filtering",
    "correlation_method": "Pearson",
    "fdr_scope": "n/a — descriptive replication",
    "domain_conventional_methods": "comparison only",
    "random_seed": RANDOM_SEED,
    "pipeline_file": "pipeline_2_disease_replication.py",
    "execution_date": "2026-04-29",
    "wj_compliance_status": "PASS",
    "data_source": f"GEO {GEO_ACC} — Parkinson Disease BA9 prefrontal cortex RNA-seq",
    "n_pd_samples": len(group_cols.get('PD', [])),
    "n_control_samples": len(group_cols.get('Control', [])),
    "purpose": "Reviewer 4: replicate EIF2S1–PELO closed-loop in disease cohort (BA9, neurodegeneration)",
}
with open(os.path.join(OUT_DIR, "provenance_pipeline2.json"), 'w') as f:
    json.dump(prov, f, indent=2)
print("\nPIPELINE 2 COMPLETE.")
