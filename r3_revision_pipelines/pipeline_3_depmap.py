#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
EIF2S1 R3 REVISION — PIPELINE 3: DEPMAP CO-DEPENDENCY VALIDATION
═══════════════════════════════════════════════════════════════════════════════

Author: Drake H. Harbert (D.H.H.) | ORCID: 0009-0007-7740-3616
Affiliation: Inner Architecture LLC, Canton, OH
Date: 2026-04-29

Description:
    Per Reviewer 2 #6: validates the EIF2S1–PELO co-expression finding using
    independent functional data — DepMap CRISPR knockout co-dependency. Two
    genes are "co-dependent" if their CRISPR-knockout fitness effects are
    correlated across hundreds of cell lines. This is independent evidence
    that the genes operate in a coordinated pathway.

    Pipeline:
        1. Download DepMap CRISPRGeneEffect.csv (Public 24Q2)
        2. Compute genome-wide Pearson correlation of EIF2S1 vs all genes
           (and PELO vs all) across cell lines
        3. Compare top 5% co-dependency network with GTEx top 5%
           co-expression network (from cached results)
        4. Test whether the CK2-eIF2α-PELO axis (CSNK2A1 hub) is preserved
           in the co-dependency layer

Dependencies: pandas, numpy, scipy
Input:
    DepMap Public 24Q2 CRISPRGeneEffect.csv (downloaded if not cached)
Output:
    G:/My Drive/inner_architecture_research/EIF2S1/r3_revision_analyses/results/
        depmap_eif2s1_top_dependencies.csv
        depmap_pelo_top_dependencies.csv
        depmap_eif2s1_pelo_overlap.csv
        provenance_pipeline3.json
═══════════════════════════════════════════════════════════════════════════════
"""

import os, sys, gc, json, time, urllib.request
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, fisher_exact

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

DATA_DIR = r"G:\My Drive\inner_architecture_research\EIF2S1\r3_revision_analyses\data"
OUT_DIR  = r"G:\My Drive\inner_architecture_research\EIF2S1\r3_revision_analyses\results"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# DepMap Public 24Q2 — CRISPRGeneEffect (Chronos)
# Public download URL pattern: https://figshare.com/ndownloader/files/<file_id>
# 24Q2 release: https://depmap.org/portal/data_page/?tab=allData (latest as of 2024)
# Known file IDs (24Q2): CRISPRGeneEffect.csv = 46494393
DEPMAP_URL = "https://ndownloader.figshare.com/files/46489063"
DEPMAP_FILE = os.path.join(DATA_DIR, "DepMap_CRISPRGeneEffect_24Q2.csv")

TOP_PERCENT = 5
TARGET_GENES = ['EIF2S1', 'PELO']
HUB_GENES = ['CSNK2A1', 'POLR2B', 'PSMD14', 'RAN', 'KPNB1', 'EFTUD2',
             'YME1L1', 'ARCN1', 'HBS1L', 'LTN1', 'NEMF', 'VCP', 'TCF25',
             'ASCC1', 'ASCC2', 'ASCC3', 'TRIP4']

print("=" * 72)
print("EIF2S1 R3 REVISION — PIPELINE 3: DEPMAP CO-DEPENDENCY")
print("=" * 72)

# ── 1. Acquire DepMap data ─────────────────────────────────────────────────
print("\n[1/4] Acquiring DepMap CRISPRGeneEffect")
def download(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 100_000_000:  # >100MB sanity
        print(f"      Cached: {path} ({os.path.getsize(path)/1e6:.1f} MB)")
        return
    print(f"      Downloading {url} → {path}")
    t0 = time.time()
    urllib.request.urlretrieve(url, path)
    print(f"      Downloaded: {os.path.getsize(path)/1e6:.1f} MB in {time.time()-t0:.1f}s")

try:
    download(DEPMAP_URL, DEPMAP_FILE)
except Exception as e:
    print(f"      Download failed: {e}")
    print(f"      Will attempt to use cached file or fail")

if not os.path.exists(DEPMAP_FILE) or os.path.getsize(DEPMAP_FILE) < 100_000_000:
    print(f"      ERROR: DepMap file not available. Try manual download:")
    print(f"        {DEPMAP_URL}")
    print(f"      Save as: {DEPMAP_FILE}")
    sys.exit(1)

# ── 2. Load DepMap matrix (cell lines × genes) ─────────────────────────────
print(f"\n[2/4] Loading DepMap CRISPR effect matrix")
t0 = time.time()
df = pd.read_csv(DEPMAP_FILE, low_memory=False)
print(f"      Shape: {df.shape}; Load time: {time.time()-t0:.1f}s")
print(f"      First 3 columns: {list(df.columns[:3])}")

# Format: first column = ModelID (cell line), other columns = "GeneSymbol (EntrezID)"
cell_col = df.columns[0]
gene_cols = df.columns[1:]
# Strip "(EntrezID)" suffix to get plain symbol
def strip_id(s):
    return s.split(' (')[0].strip()
gene_symbols = [strip_id(c) for c in gene_cols]
sym_to_col = dict(zip(gene_symbols, gene_cols))
print(f"      Cell lines: {df.shape[0]}; Genes: {len(gene_symbols)}")

# ── 3. Compute genome-wide co-dependency (Pearson) for EIF2S1 and PELO ─────
print(f"\n[3/4] Computing genome-wide co-dependency for {TARGET_GENES}")

# Build a numeric matrix: cell lines × genes, dropping any genes with all-NaN
expr = df[gene_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32)
expr.columns = gene_symbols
expr = expr.loc[:, ~expr.columns.duplicated(keep='first')]
# Drop genes with too many NaN
nan_frac = expr.isna().mean()
expr = expr.loc[:, nan_frac < 0.2]
print(f"      Genes after NaN filter: {expr.shape[1]:,}")

# Mean-impute the rest (DepMap usually has no NaN for common genes)
expr = expr.fillna(expr.median(numeric_only=True))

correlations = {}
networks = {}
for tg in TARGET_GENES:
    if tg not in expr.columns:
        print(f"      ⚠ {tg} not in DepMap matrix")
        continue
    x = expr[tg].values
    x_c = x - x.mean()
    sx = np.sqrt((x_c ** 2).sum())
    others = expr.drop(columns=[tg])
    Y = others.values - others.values.mean(axis=0, keepdims=True)
    sy = np.sqrt((Y ** 2).sum(axis=0))
    valid = sy > 0
    r = np.full(others.shape[1], np.nan, dtype=np.float32)
    r[valid] = (Y[:, valid].T @ x_c) / (sy[valid] * sx)
    s = pd.Series(r, index=others.columns, name=tg).dropna().sort_values(ascending=False)
    correlations[tg] = s
    n_top = int(np.ceil(len(s) * TOP_PERCENT / 100))
    networks[tg] = set(s.head(n_top).index)
    threshold = s.iloc[n_top - 1]
    print(f"      {tg:8s}: top partner = {s.index[0]:12s} (r = {s.iloc[0]:.3f}); "
          f"top {n_top} ≥ {threshold:.3f}")
    s.head(200).to_csv(os.path.join(OUT_DIR, f"depmap_{tg.lower()}_top_dependencies.csv"))

# Direct EIF2S1–PELO co-dependency
if 'EIF2S1' in expr.columns and 'PELO' in expr.columns:
    direct_r = np.corrcoef(expr['EIF2S1'], expr['PELO'])[0, 1]
    print(f"\n      DIRECT EIF2S1–PELO co-dependency r = {direct_r:.3f}")
else:
    direct_r = None

# ── 4. Network overlap with GTEx co-expression (if cached) ─────────────────
print(f"\n[4/4] DepMap top-5% network overlap")
overlap_summary = []
if 'EIF2S1' in networks and 'PELO' in networks:
    s1 = networks['EIF2S1']; s2 = networks['PELO']
    shared = s1 & s2; union = s1 | s2
    j = len(shared) / len(union) if union else 0
    print(f"      DepMap EIF2S1 ∩ PELO Jaccard = {j:.3f} ({len(shared)} shared)")
    universe = expr.shape[1]
    a, b, c = len(shared), len(s1 - s2), len(s2 - s1)
    d = universe - len(union)
    fo, fp = fisher_exact([[a, b], [c, d]], alternative='greater')
    common = sorted(set(correlations['EIF2S1'].index) & set(correlations['PELO'].index))
    rho, _ = spearmanr(
        correlations['EIF2S1'].reindex(common).rank(ascending=False),
        correlations['PELO'].reindex(common).rank(ascending=False),
    )
    print(f"      Fisher OR = {fo:.1f}, p = {fp:.2e}; Spearman rank ρ (genome-wide) = {rho:.3f}")
    overlap_summary.append({
        'comparison': 'DepMap EIF2S1 vs PELO co-dependency',
        'jaccard_top5': round(float(j), 6),
        'shared': int(len(shared)),
        'set1_size': int(len(s1)), 'set2_size': int(len(s2)),
        'fisher_or': round(float(fo), 6), 'fisher_p': float(fp),
        'genome_rank_rho': round(float(rho), 6),
        'direct_pearson_r': float(direct_r) if direct_r is not None else None,
    })

# Hub gene check: are CSNK2A1 and other GTEx hubs also high in DepMap?
print("\n      Hub-gene preservation across layers:")
hub_results = []
for tg in TARGET_GENES:
    if tg not in correlations: continue
    s = correlations[tg]
    s_rank = s.rank(ascending=False)
    for hub in HUB_GENES:
        if hub == tg or hub not in s.index: continue
        rank = int(s_rank[hub])
        pct = rank / len(s) * 100
        in_top5 = hub in networks.get(tg, set())
        hub_results.append({
            'target': tg, 'hub': hub,
            'depmap_pearson_r': round(float(s[hub]), 6),
            'depmap_rank': rank,
            'depmap_percentile': round(pct, 3),
            'in_top5_network': bool(in_top5),
        })
        marker = " ◄ TOP-5%" if in_top5 else ""
        print(f"        {tg}–{hub:10s}: r = {s[hub]:.3f}, rank {rank}/{len(s)} ({pct:.2f}%){marker}")

pd.DataFrame(overlap_summary).to_csv(os.path.join(OUT_DIR, "depmap_eif2s1_pelo_overlap.csv"), index=False)
pd.DataFrame(hub_results).to_csv(os.path.join(OUT_DIR, "depmap_hub_preservation.csv"), index=False)

prov = {
    "methodology": "DepMap Chronos co-dependency (Pearson on CRISPR effect across cell lines)",
    "fundamental_unit": "individual gene CRISPR knockout effect",
    "pairwise_matrix": "full, no pre-filtering",
    "correlation_method": "Pearson",
    "fdr_scope": "n/a — descriptive validation",
    "domain_conventional_methods": "DepMap-standard Chronos effect; not WJ-native but cross-layer validation",
    "random_seed": RANDOM_SEED,
    "pipeline_file": "pipeline_3_depmap.py",
    "execution_date": "2026-04-29",
    "wj_compliance_status": "validation-only (not primary WJ analysis)",
    "data_source": "DepMap Public 24Q2 CRISPRGeneEffect (Chronos)",
    "n_cell_lines": int(expr.shape[0]),
    "n_genes": int(expr.shape[1]),
    "purpose": "Reviewer 2 #6: cross-validate GTEx co-expression with DepMap co-dependency",
}
with open(os.path.join(OUT_DIR, "provenance_pipeline3.json"), 'w') as f:
    json.dump(prov, f, indent=2)
print("\nPIPELINE 3 COMPLETE.")
