#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
MS2 EIF2S1 DEFINITIVE PIPELINE — Google Colab (12GB RAM)
═══════════════════════════════════════════════════════════════════════════════

MANUSCRIPT: "EIF2S1 Co-expression Network Reveals a Closed-Loop
Translational Quality Control Hub Linking Initiation Surveillance
to Ribosome Rescue Machinery in Human Brain"

AUTHOR:     Drake H. Harbert (ORCID: 0009-0007-7740-3616)
            Inner Architecture LLC, Canton, OH 44721, USA

PURPOSE:    Single definitive pipeline producing every number in MS2.
            Fixes gProfiler API issue (uses gprofiler-official package),
            corrects proteostasis gene set (24 genes exactly),
            respects 12GB RAM via sequential region processing.

USAGE:      Copy cells into Google Colab. Run in order.
            Total runtime ~20-30 minutes depending on download speed.

OUTPUTS:    All CSV results, all 3 manuscript figures, full enrichment data.
═══════════════════════════════════════════════════════════════════════════════
"""

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1: ENVIRONMENT SETUP                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# !pip install gprofiler-official matplotlib-venn requests -q

import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import pearsonr, spearmanr, fisher_exact
from collections import OrderedDict
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
warnings.filterwarnings('ignore')

# ── Output directory ──
RESULTS_DIR = "MS2_Results"
FIGURES_DIR = "MS2_Figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 70)
print("MS2 EIF2S1 DEFINITIVE PIPELINE")
print("Drake H. Harbert — Inner Architecture LLC")
print("=" * 70)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2: CONFIGURATION — LOCKED PARAMETERS                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Target genes (7-gene panel) ──
TARGET_GENES = ['EIF2S1', 'PELO', 'LTN1', 'NEMF', 'TMEM97', 'HSPA5', 'SIGMAR1']

# ── Brain regions ──
BRAIN_REGIONS = OrderedDict([
    ('BA9',          'Brain - Frontal Cortex (BA9)'),
    ('Putamen',      'Brain - Putamen (Basal Ganglia)'),
    ('Hippocampus',  'Brain - Hippocampus'),
    ('NAcc',         'Brain - Nucleus accumbens (Basal Ganglia)'),
    ('BA24',         'Brain - Anterior cingulate cortex (BA24)'),
])
PRIMARY_REGION = 'BA9'

# ── Expression filter ──
MIN_MEDIAN_TPM = 1.0

# ── Network threshold ──
TOP_PERCENT = 5  # top 5%

# ── Custom gene sets (LOCKED to manuscript Section 2.4) ──

# RQC: 11 genes (manuscript definition)
RQC_GENES = [
    'PELO', 'HBS1L', 'LTN1', 'NEMF', 'ANKZF1',
    'VCP', 'UFD1', 'NPLOC4', 'ZNF598', 'RACK1', 'ABCE1',
]

# Proteostasis: 24 genes = 6 chaperones + 6 UPS + 6 proteasome + 6 autophagy
PROTEOSTASIS_GENES = [
    # Chaperones (6)
    'HSPA5', 'HSP90B1', 'HSPA8', 'DNAJB1', 'DNAJC3', 'BAG3',
    # Ubiquitin-proteasome system (6)
    'STUB1', 'UBE2D1', 'UBE2D2', 'UBE2D3', 'UBE2G1', 'UBE2N',
    # Proteasome subunits (6)
    'PSMD1', 'PSMD2', 'PSMD3', 'PSMC1', 'PSMC2', 'PSMC3',
    # Autophagy (6)
    'ATG5', 'ATG7', 'ATG12', 'BECN1', 'SQSTM1', 'NBR1',
]

# Translation initiation: 15 genes (manuscript definition)
TRANSLATION_INIT_GENES = [
    'EIF2S2', 'EIF2S3',
    'EIF2B1', 'EIF2B2', 'EIF2B3', 'EIF2B4', 'EIF2B5',
    'EIF3A', 'EIF3B', 'EIF3C',
    'EIF4A1', 'EIF4E', 'EIF4G1',
    'EIF5', 'EIF5B',
]

# Cross-stage gene lists (for closed-loop analysis)
INITIATION_STAGE = [
    'EIF2S1', 'EIF2S2', 'EIF2S3',
    'EIF2B1', 'EIF2B2', 'EIF2B3', 'EIF2B4', 'EIF2B5',
    'EIF2AK1', 'EIF2AK2', 'EIF2AK3', 'EIF2AK4',
    'PPP1R15A', 'PPP1R15B', 'ATF4',
]

RESCUE_STAGE = [
    'PELO', 'HBS1L', 'LTN1', 'NEMF',
    'ANKZF1', 'ZNF598', 'RACK1', 'ABCE1',
]

EFFECTOR_STAGE = [
    'VCP', 'NPLOC4',
    'PSMC1', 'PSMC2', 'PSMC3', 'PSMC4', 'PSMC5', 'PSMC6',
    'PSMD1', 'PSMD2', 'PSMD3',
]

# Assertions
assert len(RQC_GENES) == 11, f"RQC must be 11, got {len(RQC_GENES)}"
assert len(PROTEOSTASIS_GENES) == 24, f"Proteostasis must be 24, got {len(PROTEOSTASIS_GENES)}"
assert len(TRANSLATION_INIT_GENES) == 15, f"Translation init must be 15, got {len(TRANSLATION_INIT_GENES)}"

print(f"Target genes: {TARGET_GENES}")
print(f"Brain regions: {list(BRAIN_REGIONS.keys())}")
print(f"Gene sets: RQC={len(RQC_GENES)}, Proteostasis={len(PROTEOSTASIS_GENES)}, "
      f"TransInit={len(TRANSLATION_INIT_GENES)}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3: DOWNLOAD GTEx v8 DATA                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# NOTE: GTEx TPM file is ~2.6GB compressed. This cell downloads it once.
# If you've already downloaded, skip this cell.

GTEX_TPM_URL = "https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct.gz"
GTEX_SAMPLE_URL = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt"
GTEX_SUBJECT_URL = "https://storage.googleapis.com/adult-gtex/annotations/v8/metadata-files/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt"

TPM_FILE = "GTEx_v8_tpm.gct.gz"
SAMPLE_FILE = "GTEx_v8_sample_attributes.txt"
SUBJECT_FILE = "GTEx_v8_subject_phenotypes.txt"

import subprocess

for url, fname in [(GTEX_TPM_URL, TPM_FILE),
                   (GTEX_SAMPLE_URL, SAMPLE_FILE),
                   (GTEX_SUBJECT_URL, SUBJECT_FILE)]:
    if not os.path.exists(fname):
        print(f"Downloading {fname}...")
        subprocess.run(['wget', '-q', '-O', fname, url], check=True)
        print(f"  ✓ Downloaded {fname}")
    else:
        print(f"  ✓ {fname} exists")

print("\nAll GTEx files ready.")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4: LOAD SAMPLE METADATA + SUBJECT PHENOTYPES                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("Loading sample metadata...")
sample_attr = pd.read_csv(SAMPLE_FILE, sep='\t', low_memory=False)

# Build region → sample ID mapping
region_samples = {}
for key, smtsd in BRAIN_REGIONS.items():
    mask = sample_attr['SMTSD'] == smtsd
    region_samples[key] = list(sample_attr.loc[mask, 'SAMPID'])
    print(f"  {key}: {len(region_samples[key])} samples")

# Load subject phenotypes (for agonal stress analysis)
print("\nLoading subject phenotypes...")
subject_pheno = pd.read_csv(SUBJECT_FILE, sep='\t')
print(f"  Subjects: {len(subject_pheno)}")
print(f"  Columns: {list(subject_pheno.columns)}")

# Extract sample → subject mapping (GTEX-XXXX-YYYY-... → GTEX-XXXX)
def sample_to_subject(sample_id):
    parts = sample_id.split('-')
    if len(parts) >= 2:
        return '-'.join(parts[:2])
    return sample_id

# Build agonal covariates for BA9
ba9_samples = region_samples[PRIMARY_REGION]
ba9_subjects = [sample_to_subject(s) for s in ba9_samples]
ba9_subject_df = pd.DataFrame({'SAMPID': ba9_samples, 'SUBJID': ba9_subjects})
ba9_subject_df = ba9_subject_df.merge(subject_pheno, on='SUBJID', how='left')

print(f"\nBA9 agonal covariates:")
if 'DTHHRDY' in ba9_subject_df.columns:
    print(f"  Hardy Scale distribution:")
    print(ba9_subject_df['DTHHRDY'].value_counts().sort_index().to_string())
if 'AGE' in ba9_subject_df.columns:
    print(f"  Age bins: {ba9_subject_df['AGE'].value_counts().sort_index().to_string()}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5: LOAD GTEx TPM — MEMORY-EFFICIENT (12GB SAFE)                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# Strategy: Read header to get column positions, then load only brain columns.
# This avoids loading the full 56,000-sample matrix into memory.

print("Loading GTEx TPM (brain regions only, memory-efficient)...")

import gzip

# Step 1: Read header to map column positions
print("  Reading header...")
with gzip.open(TPM_FILE, 'rt') as f:
    f.readline()  # version line
    f.readline()  # dimensions line
    header = f.readline().strip().split('\t')

# All brain sample IDs we need
all_brain_samples = set()
for key in BRAIN_REGIONS:
    all_brain_samples.update(region_samples[key])

# Find column indices for brain samples
sample_cols = {}  # column_index → sample_id
for i, col in enumerate(header):
    if col in all_brain_samples:
        sample_cols[i] = col

# Columns to keep: Name (0), Description (1), + brain sample columns
keep_cols = [0, 1] + sorted(sample_cols.keys())
print(f"  Brain samples found: {len(sample_cols)} / {len(all_brain_samples)} requested")
print(f"  Total columns in file: {len(header)}")
print(f"  Columns to load: {len(keep_cols)}")

# Step 2: Read data, keeping only brain columns
print("  Loading expression data (this takes 3-5 minutes)...")
chunks = []
chunk_size = 5000
with gzip.open(TPM_FILE, 'rt') as f:
    f.readline()  # skip version
    f.readline()  # skip dimensions
    header_line = f.readline()  # skip header (already parsed)

    row_buffer = []
    for line_num, line in enumerate(f):
        parts = line.strip().split('\t')
        row = [parts[i] if i < len(parts) else '' for i in keep_cols]
        row_buffer.append(row)

        if len(row_buffer) >= chunk_size:
            chunk_df = pd.DataFrame(row_buffer)
            chunks.append(chunk_df)
            row_buffer = []
            if line_num % 10000 == 0:
                print(f"    Processed {line_num:,} genes...")

    if row_buffer:
        chunks.append(pd.DataFrame(row_buffer))

print("  Concatenating chunks...")
tpm_raw = pd.concat(chunks, ignore_index=True)
del chunks, row_buffer
gc.collect()

# Set column names
col_names = ['Name', 'Description'] + [sample_cols[i] for i in sorted(sample_cols.keys())]
tpm_raw.columns = col_names

# Set gene Name as index, convert to numeric
tpm_raw = tpm_raw.set_index('Name')
gene_descriptions = tpm_raw['Description'].copy()
tpm_raw = tpm_raw.drop('Description', axis=1)
tpm_raw = tpm_raw.apply(pd.to_numeric, errors='coerce').astype(np.float32)

print(f"\n  Loaded: {tpm_raw.shape[0]} genes × {tpm_raw.shape[1]} samples")
print(f"  Memory: {tpm_raw.memory_usage(deep=True).sum() / 1e9:.2f} GB")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6: EXTRACT REGION-SPECIFIC MATRICES + GENE FILTERING              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def prepare_region(region_key):
    """Extract and filter expression matrix for one brain region.

    Returns: (log2_expr DataFrame, n_samples, n_genes, gene_list)
    """
    samples = [s for s in region_samples[region_key] if s in tpm_raw.columns]
    expr = tpm_raw[samples].copy()

    # Deduplicate gene symbols: keep highest median expression
    # First, map Ensembl IDs to gene symbols via Description column
    # GTEx uses Ensembl IDs as index — we need gene symbols
    # Actually, GTEx GCT file Name column is ENSG IDs, Description is gene symbols
    # Let's use Description as gene symbol

    # Add gene symbol from descriptions
    expr['gene_symbol'] = gene_descriptions.reindex(expr.index)

    # Drop rows with missing symbols
    expr = expr.dropna(subset=['gene_symbol'])
    expr = expr[expr['gene_symbol'] != '']

    # For duplicate gene symbols, keep highest median
    expr['median_tpm'] = expr[samples].median(axis=1)
    expr = expr.sort_values('median_tpm', ascending=False)
    expr = expr.drop_duplicates(subset='gene_symbol', keep='first')

    # Filter: median TPM >= 1.0
    expr = expr[expr['median_tpm'] >= MIN_MEDIAN_TPM]

    # Set gene symbol as index
    expr = expr.set_index('gene_symbol')
    expr = expr.drop('median_tpm', axis=1)

    # Log2 transform
    log2_expr = np.log2(expr + 1)

    n_genes = len(log2_expr)
    n_samples = len(samples)

    return log2_expr, n_samples, n_genes, list(log2_expr.index)

print("Preparing brain region matrices...\n")
region_data = {}
for key in BRAIN_REGIONS:
    log2_expr, n_samp, n_genes, genes = prepare_region(key)
    region_data[key] = {
        'expr': log2_expr,
        'n_samples': n_samp,
        'n_genes': n_genes,
        'genes': genes,
    }
    # Check target gene presence
    present = [g for g in TARGET_GENES if g in genes]
    missing = [g for g in TARGET_GENES if g not in genes]
    print(f"  {key}: n={n_samp}, genes={n_genes}, targets={len(present)}/7"
          + (f" (missing: {missing})" if missing else ""))

gc.collect()

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7: GENOME-WIDE CORRELATIONS (PRIMARY REGION)                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print(f"GENOME-WIDE CO-EXPRESSION ANALYSIS — {PRIMARY_REGION}")
print(f"{'='*70}\n")

ba9 = region_data[PRIMARY_REGION]['expr']
n_genes_ba9 = region_data[PRIMARY_REGION]['n_genes']
n_samples_ba9 = region_data[PRIMARY_REGION]['n_samples']

print(f"Matrix: {n_genes_ba9} genes × {n_samples_ba9} samples")

# Compute correlations for each target gene
correlations = {}  # target → Series of correlations indexed by gene
for target in TARGET_GENES:
    if target not in ba9.index:
        print(f"  ⚠ {target} not found in {PRIMARY_REGION}")
        continue

    target_expr = ba9.loc[target].values
    other_genes = [g for g in ba9.index if g != target]
    other_expr = ba9.loc[other_genes].values  # (n_genes-1, n_samples)

    # Vectorized Pearson correlation
    n = len(target_expr)
    x = target_expr - target_expr.mean()
    Y = other_expr - other_expr.mean(axis=1, keepdims=True)
    x_std = np.sqrt(np.sum(x**2))
    Y_std = np.sqrt(np.sum(Y**2, axis=1))

    # Avoid division by zero
    valid = Y_std > 0
    r_values = np.full(len(other_genes), np.nan)
    r_values[valid] = np.dot(Y[valid], x) / (Y_std[valid] * x_std)

    corr_series = pd.Series(r_values, index=other_genes, name=target)
    correlations[target] = corr_series.dropna().sort_values(ascending=False)

    print(f"  {target}: {len(corr_series.dropna())} genes correlated, "
          f"top partner = {corr_series.dropna().idxmax()} "
          f"(r = {corr_series.dropna().max():.3f})")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8: TOP 5% NETWORKS + ALL 21 PAIRWISE COMPARISONS                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("TOP 5% NETWORK EXTRACTION + PAIRWISE COMPARISONS")
print(f"{'='*70}\n")

# Extract top 5% for each target
networks = {}     # target → set of top 5% genes
thresholds = {}   # target → minimum r for top 5%

for target, corr in correlations.items():
    n_total = len(corr)
    n_top = int(np.ceil(n_total * TOP_PERCENT / 100))
    top_genes = corr.head(n_top)
    threshold = top_genes.iloc[-1]

    networks[target] = set(top_genes.index)
    thresholds[target] = threshold

    print(f"  {target}: top 5% = {n_top} genes, r ≥ {threshold:.3f}")

# Gene universe size (non-target genes)
gene_universe = len(correlations[TARGET_GENES[0]])
print(f"\n  Gene universe (non-target): {gene_universe}")

# All 21 pairwise comparisons
from itertools import combinations

pairwise_results = []
for g1, g2 in combinations(TARGET_GENES, 2):
    if g1 not in networks or g2 not in networks:
        continue

    set1 = networks[g1]
    set2 = networks[g2]
    shared = set1 & set2
    union = set1 | set2
    only1 = set1 - set2
    only2 = set2 - set1

    jaccard = len(shared) / len(union) if len(union) > 0 else 0

    # Fisher's exact test
    n_top = len(set1)  # both should be same size
    a = len(shared)
    b = len(only1)
    c = len(only2)
    d = gene_universe - len(union)
    table = np.array([[a, b], [c, d]])
    fisher_or, fisher_p = fisher_exact(table, alternative='greater')

    # Spearman rank correlation of genome-wide rankings
    common_genes = sorted(set(correlations[g1].index) & set(correlations[g2].index))
    ranks1 = correlations[g1].reindex(common_genes).rank(ascending=False)
    ranks2 = correlations[g2].reindex(common_genes).rank(ascending=False)
    rho, rho_p = spearmanr(ranks1, ranks2)

    pairwise_results.append({
        'gene1': g1, 'gene2': g2,
        'shared': len(shared), 'jaccard': jaccard,
        'fisher_or': fisher_or, 'fisher_p': fisher_p,
        'spearman_rho': rho,
        'set1_size': len(set1), 'set2_size': len(set2),
    })

    if g1 == 'EIF2S1' or g2 == 'EIF2S1':
        print(f"  {g1}–{g2}: J={jaccard:.3f}, shared={len(shared)}, "
              f"OR={fisher_or:.1f}, p={fisher_p:.2e}, ρ={rho:.3f}")

comp_df = pd.DataFrame(pairwise_results).sort_values('jaccard', ascending=False)
comp_df.to_csv(os.path.join(RESULTS_DIR, "all_21_pairwise_comparisons.csv"), index=False)

# Verification
print(f"\n  ✓ Verification:")
for _, row in comp_df.iterrows():
    g1, g2 = row['gene1'], row['gene2']
    s, j = int(row['shared']), row['jaccard']
    s1, s2 = int(row['set1_size']), int(row['set2_size'])
    expected_j = s / (s1 + s2 - s)
    assert abs(j - expected_j) < 0.001, f"{g1}-{g2}: J mismatch {j} vs {expected_j}"
print("    All Jaccard indices internally consistent ✓")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9: CUSTOM GENE SET ENRICHMENT (Fisher's exact)                     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("CUSTOM GENE SET ENRICHMENT")
print(f"{'='*70}\n")

def compute_custom_enrichment(network_genes, gene_set_list, set_name,
                               all_genes_list, verbose=True):
    """Fisher's exact test for custom gene set enrichment.

    Parameters
    ----------
    network_genes : set — top 5% gene set
    gene_set_list : list — curated gene set
    set_name : str — name for display
    all_genes_list : list — all expressed genes (universe)
    """
    universe = set(all_genes_list)
    expressed = [g for g in gene_set_list if g in universe]
    in_network = [g for g in expressed if g in network_genes]

    n_expressed = len(expressed)
    n_in_network = len(in_network)
    n_network = len(network_genes)
    n_universe = len(universe)

    if n_expressed == 0:
        return None

    # Expected count
    expected = n_expressed * n_network / n_universe
    fold = n_in_network / expected if expected > 0 else 0

    # Fisher's exact test (one-sided)
    a = n_in_network
    b = n_network - n_in_network
    c = n_expressed - n_in_network
    d = n_universe - n_network - c
    table = np.array([[a, b], [c, d]])
    _, p_val = fisher_exact(table, alternative='greater')

    result = {
        'gene_set': set_name,
        'expressed': n_expressed,
        'in_network': n_in_network,
        'fold_enrichment': fold,
        'p_value': p_val,
        'genes_found': ', '.join(sorted(in_network)),
    }

    if verbose:
        print(f"  {set_name}: {n_in_network}/{n_expressed} expressed, "
              f"{fold:.1f}×, p = {p_val:.2e}")
        if in_network:
            print(f"    Genes: {', '.join(sorted(in_network))}")

    return result

# Run for EIF2S1 network
eif2s1_genes = list(correlations['EIF2S1'].index) + ['EIF2S1']
eif2s1_network = networks['EIF2S1']

custom_sets = {
    'Ribosome Quality Control': RQC_GENES,
    'Proteostasis Network': PROTEOSTASIS_GENES,
    'Translation Initiation': TRANSLATION_INIT_GENES,
}

print("EIF2S1 top 5% custom enrichment:\n")
custom_results = []
for name, genes in custom_sets.items():
    result = compute_custom_enrichment(eif2s1_network, genes, name, eif2s1_genes)
    if result:
        custom_results.append(result)
    print()

custom_df = pd.DataFrame(custom_results)
custom_df.to_csv(os.path.join(RESULTS_DIR, "EIF2S1_custom_enrichment.csv"), index=False)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10: gProfiler ENRICHMENT (FIXED — uses gprofiler-official)         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("gProfiler GO/KEGG/REACTOME ENRICHMENT")
print(f"{'='*70}\n")

# Install if needed: !pip install gprofiler-official -q
try:
    from gprofiler import GProfiler
    gp = GProfiler(return_dataframe=True)
    GPROFILER_AVAILABLE = True
    print("✓ gprofiler-official loaded successfully")
except ImportError:
    print("⚠ gprofiler-official not installed. Run: !pip install gprofiler-official -q")
    print("  Falling back to requests-based API...")
    GPROFILER_AVAILABLE = False

def run_gprofiler_enrichment(gene_list, background_list, label=""):
    """Run gProfiler enrichment with proper background.

    Uses gprofiler-official package (handles large backgrounds correctly).
    Falls back to requests API if package unavailable.
    """
    genes = list(gene_list)
    bg = list(background_list)

    if GPROFILER_AVAILABLE:
        try:
            df = gp.profile(
                organism='hsapiens',
                query=genes,
                background=bg,
                sources=['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC'],
                significance_threshold_method='g_SCS',
                user_threshold=0.05,
                no_evidences=False,
            )
            if df is not None and len(df) > 0:
                print(f"  {label}: {len(df)} significant terms")
                return df
            else:
                print(f"  {label}: 0 significant terms")
                return pd.DataFrame()
        except Exception as e:
            print(f"  {label}: gProfiler error — {e}")
            return pd.DataFrame()
    else:
        # Fallback: requests-based API
        import requests
        url = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"
        payload = {
            'organism': 'hsapiens',
            'query': genes,
            'sources': ['GO:BP', 'GO:MF', 'GO:CC', 'KEGG', 'REAC'],
            'user_threshold': 0.05,
            'significance_threshold_method': 'g_SCS',
            'no_evidences': True,
            'background': bg,
        }
        try:
            resp = requests.post(url, json=payload, timeout=120)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get('result', [])
                if results:
                    df = pd.DataFrame(results)
                    print(f"  {label}: {len(df)} significant terms")
                    return df
                else:
                    print(f"  {label}: 0 significant terms")
                    return pd.DataFrame()
            else:
                print(f"  {label}: HTTP {resp.status_code}")
                return pd.DataFrame()
        except Exception as e:
            print(f"  {label}: API error — {e}")
            return pd.DataFrame()

# Define gene sets for enrichment
eif2s1_top5_genes = list(networks['EIF2S1'])
pelo_top5_genes = list(networks['PELO'])
shared_ep = list(networks['EIF2S1'] & networks['PELO'])
eif2s1_unique = list(networks['EIF2S1'] - networks['PELO'])
pelo_unique = list(networks['PELO'] - networks['EIF2S1'])

background = eif2s1_genes  # all expressed genes + EIF2S1

print("Running enrichment analyses:\n")

# 1. EIF2S1 full network
go_eif2s1 = run_gprofiler_enrichment(eif2s1_top5_genes, background, "EIF2S1 full")

# 2. PELO full network
go_pelo = run_gprofiler_enrichment(pelo_top5_genes, background, "PELO full")

# 3. Shared EIF2S1 ∩ PELO
go_shared = run_gprofiler_enrichment(shared_ep, background, "EIF2S1∩PELO shared")

# 4. EIF2S1-unique
go_eif2s1_uniq = run_gprofiler_enrichment(eif2s1_unique, background, "EIF2S1 unique")

# 5. PELO-unique
go_pelo_uniq = run_gprofiler_enrichment(pelo_unique, background, "PELO unique")

# Save all results
for name, df in [('EIF2S1_full', go_eif2s1), ('PELO_full', go_pelo),
                 ('shared_EIF2S1_PELO', go_shared),
                 ('EIF2S1_unique', go_eif2s1_uniq), ('PELO_unique', go_pelo_uniq)]:
    if len(df) > 0:
        df.to_csv(os.path.join(RESULTS_DIR, f"gProfiler_{name}.csv"), index=False)

# Print top GO:BP terms for EIF2S1
if len(go_eif2s1) > 0:
    go_bp = go_eif2s1[go_eif2s1['source'] == 'GO:BP'].sort_values('p_value')
    go_kegg = go_eif2s1[go_eif2s1['source'] == 'KEGG'].sort_values('p_value')
    go_reac = go_eif2s1[go_eif2s1['source'] == 'REAC'].sort_values('p_value')

    print(f"\nEIF2S1 Top GO:BP ({len(go_bp)} terms):")
    for _, row in go_bp.head(8).iterrows():
        print(f"  {row['name']}: p = {row['p_value']:.2e}")

    print(f"\nEIF2S1 Top KEGG ({len(go_kegg)} terms):")
    for _, row in go_kegg.head(5).iterrows():
        print(f"  {row['name']}: p = {row['p_value']:.2e}")

    print(f"\nEIF2S1 Top Reactome ({len(go_reac)} terms):")
    for _, row in go_reac.head(5).iterrows():
        print(f"  {row['name']}: p = {row['p_value']:.2e}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 11: CROSS-STAGE TRANSLATIONAL COORDINATION                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("CROSS-STAGE TRANSLATIONAL COORDINATION ANALYSIS")
print(f"{'='*70}\n")

def mean_pairwise_corrs(gene_list_a, gene_list_b, expr_df):
    """Compute all pairwise Pearson correlations between two gene lists."""
    corrs = []
    for ga in gene_list_a:
        for gb in gene_list_b:
            if ga != gb and ga in expr_df.index and gb in expr_df.index:
                r, _ = pearsonr(expr_df.loc[ga].values, expr_df.loc[gb].values)
                corrs.append(r)
    return corrs

# Filter to expressed genes
init_present = [g for g in INITIATION_STAGE if g in ba9.index]
rescue_present = [g for g in RESCUE_STAGE if g in ba9.index]
effector_present = [g for g in EFFECTOR_STAGE if g in ba9.index]

print(f"Genes present in BA9:")
print(f"  Initiation: {len(init_present)}/{len(INITIATION_STAGE)}")
print(f"  Rescue:     {len(rescue_present)}/{len(RESCUE_STAGE)}")
print(f"  Effector:   {len(effector_present)}/{len(EFFECTOR_STAGE)}")

stage_comparisons = {
    'I×I': (init_present, init_present),
    'R×R': (rescue_present, rescue_present),
    'E×E': (effector_present, effector_present),
    'I×R': (init_present, rescue_present),
    'I×E': (init_present, effector_present),
    'R×E': (rescue_present, effector_present),
}

print("\nCross-stage correlations:")
stage_results = {}
for label, (list_a, list_b) in stage_comparisons.items():
    corrs = mean_pairwise_corrs(list_a, list_b, ba9)
    stage_results[label] = corrs
    print(f"  {label}: mean r = {np.mean(corrs):.3f} (SD = {np.std(corrs):.3f}, "
          f"n = {len(corrs)} pairs)")

# Random background
np.random.seed(42)
all_gene_list = list(ba9.index)
random_corrs = []
for _ in range(2000):
    g1, g2 = np.random.choice(all_gene_list, 2, replace=False)
    r, _ = pearsonr(ba9.loc[g1].values, ba9.loc[g2].values)
    random_corrs.append(r)

stage_results['Random'] = random_corrs
print(f"  Random: mean r = {np.mean(random_corrs):.3f} (SD = {np.std(random_corrs):.3f}, "
      f"n = 2000)")

# I×R vs Random: t-test + Cohen's d
ixr = stage_results['I×R']
t_stat, t_p = stats.ttest_ind(ixr, random_corrs, alternative='greater')
cohens_d = (np.mean(ixr) - np.mean(random_corrs)) / np.sqrt(
    (np.std(ixr)**2 + np.std(random_corrs)**2) / 2)

# Direct EIF2S1-PELO correlation
eif2s1_pelo_r, eif2s1_pelo_p = pearsonr(
    ba9.loc['EIF2S1'].values, ba9.loc['PELO'].values)

print(f"\n  I×R vs Random: t = {t_stat:.2f}, p = {t_p:.2e}, Cohen's d = {cohens_d:.2f}")
print(f"  Direct EIF2S1–PELO: r = {eif2s1_pelo_r:.3f}, p = {eif2s1_pelo_p:.2e}")

# Save
stage_summary = []
for label, corrs in stage_results.items():
    stage_summary.append({
        'comparison': label,
        'mean_r': np.mean(corrs),
        'sd': np.std(corrs),
        'n_pairs': len(corrs),
    })
pd.DataFrame(stage_summary).to_csv(
    os.path.join(RESULTS_DIR, "cross_stage_coordination.csv"), index=False)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 12: AGONAL STRESS SENSITIVITY ANALYSIS                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("AGONAL STRESS SENSITIVITY ANALYSIS")
print(f"{'='*70}\n")

# Build covariate matrix aligned to BA9 samples
ba9_sample_list = list(ba9.columns)
ba9_covar = ba9_subject_df.set_index('SAMPID').reindex(ba9_sample_list)

# Extract ischemic time from sample attributes
isch_map = sample_attr.set_index('SAMPID')['SMTSISCH'].to_dict()
ba9_covar['SMTSISCH'] = [isch_map.get(s, np.nan) for s in ba9_sample_list]

# Age midpoint
age_map = {'20-29': 25, '30-39': 35, '40-49': 45, '50-59': 55, '60-69': 65, '70-79': 75}
if 'AGE' in ba9_covar.columns:
    ba9_covar['AGE_MID'] = ba9_covar['AGE'].map(age_map)
else:
    ba9_covar['AGE_MID'] = np.nan

# Sex encoding
if 'SEX' in ba9_covar.columns:
    ba9_covar['SEX_NUM'] = ba9_covar['SEX'].astype(float)
else:
    ba9_covar['SEX_NUM'] = np.nan

# Hardy Scale
if 'DTHHRDY' in ba9_covar.columns:
    ba9_covar['HARDY'] = ba9_covar['DTHHRDY'].astype(float)
else:
    ba9_covar['HARDY'] = np.nan

print(f"Covariate availability for BA9 ({len(ba9_sample_list)} samples):")
print(f"  Hardy Scale: {ba9_covar['HARDY'].notna().sum()} valid")
print(f"  Ischemic time: {ba9_covar['SMTSISCH'].notna().sum()} valid")
print(f"  Age: {ba9_covar['AGE_MID'].notna().sum()} valid")
print(f"  Sex: {ba9_covar['SEX_NUM'].notna().sum()} valid")

def partial_correlation_network(target, expr_df, covariates_df, covar_cols,
                                 top_pct=5):
    """Compute genome-wide correlations after regressing out covariates.

    Returns: (corr_series, threshold, top5_set, n_valid_samples)
    """
    # Get valid samples (non-NaN in all covariates)
    valid = covariates_df[covar_cols].dropna().index
    valid = [s for s in valid if s in expr_df.columns]
    n_valid = len(valid)

    if n_valid < 30:
        return None, None, None, n_valid

    expr_sub = expr_df[valid]
    covar_matrix = covariates_df.loc[valid, covar_cols].values

    # Residualize: regress covariates from each gene
    from numpy.linalg import lstsq
    X = np.column_stack([np.ones(n_valid), covar_matrix])  # intercept + covars

    # Residualize target
    target_vals = expr_sub.loc[target].values
    beta, _, _, _ = lstsq(X, target_vals, rcond=None)
    target_resid = target_vals - X @ beta

    # Residualize all other genes
    other_genes = [g for g in expr_sub.index if g != target]
    other_vals = expr_sub.loc[other_genes].values  # (n_genes, n_samples)

    # Vectorized residualization
    betas = lstsq(X, other_vals.T, rcond=None)[0]  # (n_covars+1, n_genes)
    resid = other_vals - (X @ betas).T  # (n_genes, n_samples)

    # Correlate residuals
    x = target_resid - target_resid.mean()
    Y = resid - resid.mean(axis=1, keepdims=True)
    x_std = np.sqrt(np.sum(x**2))
    Y_std = np.sqrt(np.sum(Y**2, axis=1))
    valid_mask = Y_std > 0
    r_vals = np.full(len(other_genes), np.nan)
    r_vals[valid_mask] = np.dot(Y[valid_mask], x) / (Y_std[valid_mask] * x_std)

    corr_series = pd.Series(r_vals, index=other_genes).dropna().sort_values(ascending=False)

    n_top = int(np.ceil(len(corr_series) * top_pct / 100))
    top5_set = set(corr_series.head(n_top).index)
    threshold = corr_series.iloc[n_top - 1] if n_top <= len(corr_series) else np.nan

    return corr_series, threshold, top5_set, n_valid

# Define sensitivity models
models = OrderedDict([
    ('Raw', []),
    ('Hardy', ['HARDY']),
    ('Ischemic', ['SMTSISCH']),
    ('Hardy+Ischemic', ['HARDY', 'SMTSISCH']),
    ('Full', ['AGE_MID', 'SEX_NUM', 'HARDY', 'SMTSISCH']),
])

agonal_results = []
eif2s1_rankings = {}
pelo_rankings = {}

for model_name, covar_cols in models.items():
    if not covar_cols:
        # Raw = already computed
        eif_corr = correlations['EIF2S1']
        pelo_corr = correlations['PELO']
        eif_set = networks['EIF2S1']
        pelo_set = networks['PELO']
        eif_thr = thresholds['EIF2S1']
        pelo_thr = thresholds['PELO']
        n_valid = n_samples_ba9
    else:
        eif_corr, eif_thr, eif_set, n_valid = partial_correlation_network(
            'EIF2S1', ba9, ba9_covar, covar_cols)
        pelo_corr, pelo_thr, pelo_set, _ = partial_correlation_network(
            'PELO', ba9, ba9_covar, covar_cols)

        if eif_corr is None or pelo_corr is None:
            print(f"  {model_name}: Insufficient valid samples ({n_valid})")
            continue

    eif2s1_rankings[model_name] = eif_corr
    pelo_rankings[model_name] = pelo_corr

    # EIF2S1-PELO overlap
    shared = eif_set & pelo_set
    union = eif_set | pelo_set
    jaccard = len(shared) / len(union) if len(union) > 0 else 0

    # Rank preservation vs raw
    if model_name != 'Raw' and 'Raw' in eif2s1_rankings:
        common = sorted(set(eif_corr.index) & set(eif2s1_rankings['Raw'].index))
        rho_preserve, _ = spearmanr(
            eif_corr.reindex(common).values,
            eif2s1_rankings['Raw'].reindex(common).values)
    else:
        rho_preserve = 1.0

    # Genome-wide rank correlation between EIF2S1 and PELO
    common_ep = sorted(set(eif_corr.index) & set(pelo_corr.index))
    rho_ep, _ = spearmanr(
        eif_corr.reindex(common_ep).rank(ascending=False),
        pelo_corr.reindex(common_ep).rank(ascending=False))

    result = {
        'model': model_name,
        'eif2s1_thr': eif_thr,
        'pelo_thr': pelo_thr,
        'shared': len(shared),
        'jaccard': jaccard,
        'rho_ep': rho_ep,
        'preserve_rho': rho_preserve,
        'n_valid': n_valid,
        'eif_top1': eif_corr.idxmax(),
        'pelo_top1': pelo_corr.idxmax(),
    }
    agonal_results.append(result)

    print(f"  {model_name:20s}: J = {jaccard:.3f}, shared = {len(shared)}, "
          f"ρ = {rho_ep:.3f}, preserve = {rho_preserve:.4f}, "
          f"EIF2S1 #1 = {eif_corr.idxmax()}, PELO #1 = {pelo_corr.idxmax()}")

agonal_df = pd.DataFrame(agonal_results)
agonal_df.to_csv(os.path.join(RESULTS_DIR, "agonal_sensitivity.csv"), index=False)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 13: MULTI-REGION REPLICATION                                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("MULTI-REGION REPLICATION")
print(f"{'='*70}\n")

# Compute genome-wide correlations for EIF2S1 and PELO in all regions
region_correlations = {}
for region_key in BRAIN_REGIONS:
    expr = region_data[region_key]['expr']
    region_correlations[region_key] = {}

    for target in ['EIF2S1', 'PELO']:
        if target not in expr.index:
            print(f"  ⚠ {target} not in {region_key}")
            continue

        target_vals = expr.loc[target].values
        other_genes = [g for g in expr.index if g != target]
        other_vals = expr.loc[other_genes].values

        n = len(target_vals)
        x = target_vals - target_vals.mean()
        Y = other_vals - other_vals.mean(axis=1, keepdims=True)
        x_std = np.sqrt(np.sum(x**2))
        Y_std = np.sqrt(np.sum(Y**2, axis=1))
        valid = Y_std > 0
        r_vals = np.full(len(other_genes), np.nan)
        r_vals[valid] = np.dot(Y[valid], x) / (Y_std[valid] * x_std)

        corr_series = pd.Series(r_vals, index=other_genes).dropna().sort_values(ascending=False)
        region_correlations[region_key][target] = corr_series

    print(f"  {region_key}: EIF2S1 ({len(region_correlations[region_key].get('EIF2S1', []))} genes), "
          f"PELO ({len(region_correlations[region_key].get('PELO', []))} genes)")

# Cross-region Spearman rank correlations for EIF2S1
print("\nEIF2S1 cross-region rank correlations:")
region_keys = list(BRAIN_REGIONS.keys())
cross_region_matrix = {}

for target in ['EIF2S1', 'PELO']:
    cross_region_matrix[target] = np.zeros((5, 5))
    for i, r1 in enumerate(region_keys):
        for j, r2 in enumerate(region_keys):
            if i == j:
                cross_region_matrix[target][i, j] = 1.0
                continue
            if target in region_correlations[r1] and target in region_correlations[r2]:
                corr1 = region_correlations[r1][target]
                corr2 = region_correlations[r2][target]
                common = sorted(set(corr1.index) & set(corr2.index))
                if len(common) > 100:
                    rho, _ = spearmanr(corr1.reindex(common).values,
                                       corr2.reindex(common).values)
                    cross_region_matrix[target][i, j] = rho

    print(f"\n  {target}:")
    for i, r1 in enumerate(region_keys):
        for j, r2 in enumerate(region_keys):
            if j > i:
                rho = cross_region_matrix[target][i, j]
                print(f"    {r1} vs {r2}: ρ = {rho:.3f}")

    vals = cross_region_matrix[target][np.triu_indices(5, k=1)]
    print(f"  Range: {vals.min():.3f} – {vals.max():.3f}")

# Multi-region RQC enrichment
print("\nRQC enrichment across regions:")
for region_key in BRAIN_REGIONS:
    if 'EIF2S1' not in region_correlations[region_key]:
        continue
    corr = region_correlations[region_key]['EIF2S1']
    n_top = int(np.ceil(len(corr) * TOP_PERCENT / 100))
    top5 = set(corr.head(n_top).index)
    all_genes_r = list(corr.index) + ['EIF2S1']

    result = compute_custom_enrichment(top5, RQC_GENES, f"RQC in {region_key}",
                                        all_genes_r, verbose=False)
    if result:
        print(f"  {region_key}: {result['in_network']}/{result['expressed']}, "
              f"{result['fold_enrichment']:.1f}×, p = {result['p_value']:.2e}")

# EIF2S1-PELO Jaccard across regions
print("\nEIF2S1–PELO Jaccard across regions:")
for region_key in BRAIN_REGIONS:
    if 'EIF2S1' not in region_correlations[region_key]:
        continue
    if 'PELO' not in region_correlations[region_key]:
        continue

    eif_corr = region_correlations[region_key]['EIF2S1']
    pelo_corr = region_correlations[region_key]['PELO']

    n_eif = int(np.ceil(len(eif_corr) * TOP_PERCENT / 100))
    n_pelo = int(np.ceil(len(pelo_corr) * TOP_PERCENT / 100))

    eif_set = set(eif_corr.head(n_eif).index)
    pelo_set = set(pelo_corr.head(n_pelo).index)

    shared = eif_set & pelo_set
    union = eif_set | pelo_set
    j = len(shared) / len(union) if len(union) > 0 else 0
    print(f"  {region_key}: J = {j:.3f}, shared = {len(shared)}")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 14: EXPORT GENE LISTS                                              ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("EXPORTING GENE LISTS")
print(f"{'='*70}\n")

# EIF2S1 full top 5%
eif2s1_top5_df = pd.DataFrame({
    'gene': correlations['EIF2S1'].head(len(networks['EIF2S1'])).index,
    'r_with_EIF2S1': correlations['EIF2S1'].head(len(networks['EIF2S1'])).values,
})
eif2s1_top5_df.to_csv(os.path.join(RESULTS_DIR, "EIF2S1_top5pct.csv"), index=False)

# PELO full top 5%
pelo_top5_df = pd.DataFrame({
    'gene': correlations['PELO'].head(len(networks['PELO'])).index,
    'r_with_PELO': correlations['PELO'].head(len(networks['PELO'])).values,
})
pelo_top5_df.to_csv(os.path.join(RESULTS_DIR, "PELO_top5pct.csv"), index=False)

# Shared, unique
shared_genes = sorted(networks['EIF2S1'] & networks['PELO'])
eif_unique_genes = sorted(networks['EIF2S1'] - networks['PELO'])
pelo_unique_genes = sorted(networks['PELO'] - networks['EIF2S1'])

pd.DataFrame({'gene': shared_genes}).to_csv(
    os.path.join(RESULTS_DIR, "EIF2S1_PELO_shared.csv"), index=False)
pd.DataFrame({'gene': eif_unique_genes}).to_csv(
    os.path.join(RESULTS_DIR, "EIF2S1_unique.csv"), index=False)
pd.DataFrame({'gene': pelo_unique_genes}).to_csv(
    os.path.join(RESULTS_DIR, "PELO_unique.csv"), index=False)

# Full genome-wide rankings for all 7 targets
for target in TARGET_GENES:
    if target in correlations:
        corr = correlations[target]
        out = pd.DataFrame({
            'gene': corr.index,
            f'r_with_{target}': corr.values,
            'rank': range(1, len(corr) + 1),
        })
        out.to_csv(os.path.join(RESULTS_DIR, f"{target}_genome_wide_rankings.csv"),
                   index=False)

print(f"  EIF2S1 top 5%: {len(eif2s1_top5_df)} genes")
print(f"  PELO top 5%: {len(pelo_top5_df)} genes")
print(f"  Shared: {len(shared_genes)} genes")
print(f"  EIF2S1-unique: {len(eif_unique_genes)} genes")
print(f"  PELO-unique: {len(pelo_unique_genes)} genes")
print(f"  Genome-wide rankings: 7 files")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 15: FIGURE 1 — EIF2S1-PELO CONVERGENCE + 7-GENE HEATMAP           ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print(f"\n{'='*70}")
print("GENERATING FIGURES")
print(f"{'='*70}\n")

# ── Figure 1: Three panels ──
fig = plt.figure(figsize=(18, 6))

# Panel A: Venn diagram (EIF2S1 vs PELO)
ax_a = fig.add_axes([0.02, 0.12, 0.30, 0.80])
ax_a.text(-0.05, 1.08, 'A', fontsize=20, fontweight='bold', va='top',
          transform=ax_a.transAxes)

n_eif_only = len(eif_unique_genes)
n_pelo_only = len(pelo_unique_genes)
n_shared = len(shared_genes)

v = venn3(subsets=(n_eif_only, n_pelo_only, n_shared, 0, 0, 0, 0),
          set_labels=('', '', ''))

# We need a 2-circle venn, but venn3 works with 3. Use venn2 instead.
from matplotlib_venn import venn2
ax_a.clear()
ax_a.text(-0.05, 1.08, 'A', fontsize=20, fontweight='bold', va='top',
          transform=ax_a.transAxes)

v = venn2(subsets=(n_eif_only, n_pelo_only, n_shared),
          set_labels=('EIF2S1', 'PELO'), ax=ax_a)
v.get_patch_by_id('10').set_color('#1565C0')
v.get_patch_by_id('01').set_color('#C62828')
v.get_patch_by_id('11').set_color('#7B1FA2')
for patch in ['10', '01', '11']:
    v.get_patch_by_id(patch).set_alpha(0.7)
    v.get_patch_by_id(patch).set_edgecolor('#555')

ax_a.set_title(f'EIF2S1–PELO Network Overlap\n'
               f'J = {n_shared/(n_eif_only+n_pelo_only+n_shared):.3f}',
               fontsize=12, fontweight='bold')

# Panel B: Scatter plot of genome-wide rankings
ax_b = fig.add_axes([0.37, 0.12, 0.28, 0.80])
ax_b.text(-0.15, 1.08, 'B', fontsize=20, fontweight='bold', va='top',
          transform=ax_b.transAxes)

common_genes_plot = sorted(set(correlations['EIF2S1'].index) &
                           set(correlations['PELO'].index))
ranks_eif = correlations['EIF2S1'].reindex(common_genes_plot).rank(ascending=False)
ranks_pelo = correlations['PELO'].reindex(common_genes_plot).rank(ascending=False)

# Subsample for visibility
np.random.seed(42)
idx = np.random.choice(len(common_genes_plot),
                       min(5000, len(common_genes_plot)), replace=False)
ax_b.scatter(ranks_eif.values[idx], ranks_pelo.values[idx],
            s=1, alpha=0.3, c='#555', rasterized=True)
rho_plot = comp_df.loc[(comp_df['gene1'] == 'EIF2S1') & (comp_df['gene2'] == 'PELO'),
                       'spearman_rho'].values[0]
ax_b.set_xlabel('EIF2S1 rank', fontsize=11)
ax_b.set_ylabel('PELO rank', fontsize=11)
ax_b.set_title(f'Genome-wide rank correlation\nρ = {rho_plot:.3f}',
               fontsize=12, fontweight='bold')
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

# Panel C: 7×7 Jaccard heatmap
ax_c = fig.add_axes([0.72, 0.12, 0.26, 0.80])
ax_c.text(-0.15, 1.08, 'C', fontsize=20, fontweight='bold', va='top',
          transform=ax_c.transAxes)

n_targets = len(TARGET_GENES)
j_matrix = np.eye(n_targets)
for _, row in comp_df.iterrows():
    i1 = TARGET_GENES.index(row['gene1'])
    i2 = TARGET_GENES.index(row['gene2'])
    j_matrix[i1, i2] = row['jaccard']
    j_matrix[i2, i1] = row['jaccard']

im = ax_c.imshow(j_matrix, cmap='Blues', vmin=0, vmax=0.55, aspect='equal')
for i in range(n_targets):
    for j in range(n_targets):
        if i != j:
            color = 'white' if j_matrix[i, j] > 0.35 else 'black'
            ax_c.text(j, i, f'{j_matrix[i, j]:.2f}',
                     ha='center', va='center', fontsize=7,
                     fontweight='bold', color=color)

ax_c.set_xticks(range(n_targets))
ax_c.set_yticks(range(n_targets))
short_names = ['EIF2S1', 'PELO', 'LTN1', 'NEMF', 'TMEM97', 'HSPA5', 'SIGMAR1']
ax_c.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
ax_c.set_yticklabels(short_names, fontsize=8)
ax_c.set_title('Jaccard Similarity\n(7-gene panel)', fontsize=12, fontweight='bold')
cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
cbar.set_label('Jaccard Index', fontsize=9)

fig.savefig(os.path.join(FIGURES_DIR, 'Figure1_EIF2S1_PELO_Convergence.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGURES_DIR, 'Figure1_EIF2S1_PELO_Convergence.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Figure 1 saved")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 16: FIGURE 2 — GO ENRICHMENT (4 panels)                            ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def extract_top_terms(go_df, source='GO:BP', n=5):
    """Extract top n terms from a gProfiler result DataFrame."""
    if go_df is None or len(go_df) == 0:
        return []
    subset = go_df[go_df['source'] == source].sort_values('p_value').head(n)
    terms = []
    for _, row in subset.iterrows():
        terms.append({
            'name': row['name'][:45],
            'p': row['p_value'],
            'nlp': -np.log10(max(row['p_value'], 1e-50)),
        })
    return terms

# Get terms for each gene set
eif_bp = extract_top_terms(go_eif2s1, 'GO:BP', 5)
shared_bp = extract_top_terms(go_shared, 'GO:BP', 5)
eif_uniq_bp = extract_top_terms(go_eif2s1_uniq, 'GO:BP', 5)
pelo_uniq_bp = extract_top_terms(go_pelo_uniq, 'GO:BP', 5)

# Also get KEGG and Reactome for EIF2S1
eif_kegg = extract_top_terms(go_eif2s1, 'KEGG', 5)
eif_reac = extract_top_terms(go_eif2s1, 'REAC', 5)

fig, axes = plt.subplots(2, 2, figsize=(18, 10))
plt.subplots_adjust(wspace=0.55, hspace=0.45)

panels = [
    (axes[0, 0], f'A  EIF2S1 full ({len(eif2s1_top5_genes)} genes) — GO:BP',
     eif_bp, '#1565C0'),
    (axes[0, 1], f'B  EIF2S1∩PELO shared ({len(shared_genes)} genes) — GO:BP',
     shared_bp, '#7B1FA2'),
    (axes[1, 0], f'C  EIF2S1 unique ({len(eif_unique_genes)} genes) — GO:BP',
     eif_uniq_bp, '#0D47A1'),
    (axes[1, 1], f'D  PELO unique ({len(pelo_unique_genes)} genes) — GO:BP',
     pelo_uniq_bp, '#C62828'),
]

for ax_ref, title, terms, color in panels:
    if not terms:
        ax_ref.text(0.5, 0.5, 'No significant\nGO:BP terms',
                    ha='center', va='center', fontsize=12,
                    transform=ax_ref.transAxes)
        ax_ref.set_title(title, fontsize=11, fontweight='bold')
        continue

    names = [t['name'] for t in reversed(terms)]
    values = [t['nlp'] for t in reversed(terms)]
    ax_ref.barh(range(len(names)), values, color=color, alpha=0.85, edgecolor='white')
    ax_ref.set_yticks(range(len(names)))
    ax_ref.set_yticklabels(names, fontsize=9)
    ax_ref.set_xlabel('$-\\log_{10}(p)$', fontsize=10)
    ax_ref.set_title(title, fontsize=11, fontweight='bold')
    ax_ref.spines['top'].set_visible(False)
    ax_ref.spines['right'].set_visible(False)

fig.savefig(os.path.join(FIGURES_DIR, 'Figure2_GO_Enrichment.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGURES_DIR, 'Figure2_GO_Enrichment.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Figure 2 saved")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 17: FIGURE 3 — MULTI-REGION REPLICATION HEATMAP                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
plt.subplots_adjust(wspace=0.35)

for idx, (target, title) in enumerate([('EIF2S1', 'EIF2S1'), ('PELO', 'PELO')]):
    ax = axes[idx]
    ax.text(-0.1, 1.08, chr(65 + idx), fontsize=20, fontweight='bold',
            va='top', transform=ax.transAxes)

    mat = cross_region_matrix[target]
    im = ax.imshow(mat, cmap='YlOrRd', vmin=0.75, vmax=1.0, aspect='equal')

    for i in range(5):
        for j in range(5):
            color = 'white' if mat[i, j] > 0.92 else 'black'
            ax.text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                   fontsize=9, fontweight='bold', color=color)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(region_keys, fontsize=9, rotation=45, ha='right')
    ax.set_yticklabels(region_keys, fontsize=9)
    ax.set_title(f'{title} Cross-Region ρ', fontsize=12, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Spearman ρ', fontsize=9)

fig.savefig(os.path.join(FIGURES_DIR, 'Figure3_MultiRegion_Replication.png'),
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(os.path.join(FIGURES_DIR, 'Figure3_MultiRegion_Replication.pdf'),
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Figure 3 saved")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 18: FINAL SUMMARY — ALL MANUSCRIPT NUMBERS                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("DEFINITIVE MANUSCRIPT NUMBERS — MS2 EIF2S1")
print("=" * 70)

print(f"\n─── Basic Parameters ───")
print(f"  Gene universe: {gene_universe + 1} (including target)")
print(f"  Non-target genes: {gene_universe}")
print(f"  Samples (BA9): {n_samples_ba9}")
print(f"  Top 5% count: {len(networks['EIF2S1'])}")

print(f"\n─── EIF2S1 Network ───")
print(f"  Threshold: r ≥ {thresholds['EIF2S1']:.3f}")
print(f"  Top partner: {correlations['EIF2S1'].idxmax()} "
      f"(r = {correlations['EIF2S1'].max():.3f})")

print(f"\n─── PELO Network ───")
print(f"  Threshold: r ≥ {thresholds['PELO']:.3f}")
print(f"  Top partner: {correlations['PELO'].idxmax()} "
      f"(r = {correlations['PELO'].max():.3f})")

print(f"\n─── EIF2S1–PELO Convergence ───")
eif_pelo_row = comp_df[(comp_df['gene1'] == 'EIF2S1') & (comp_df['gene2'] == 'PELO')]
print(f"  Shared: {int(eif_pelo_row['shared'].values[0])}")
print(f"  Jaccard: {eif_pelo_row['jaccard'].values[0]:.3f}")
print(f"  Fisher OR: {eif_pelo_row['fisher_or'].values[0]:.1f}")
print(f"  Fisher p: {eif_pelo_row['fisher_p'].values[0]:.2e}")
print(f"  Spearman ρ: {eif_pelo_row['spearman_rho'].values[0]:.3f}")
print(f"  Direct r: {eif2s1_pelo_r:.3f}")

print(f"\n─── All EIF2S1 Pairwise ───")
for _, row in comp_df[comp_df['gene1'] == 'EIF2S1'].iterrows():
    print(f"  EIF2S1–{row['gene2']:8s}: J={row['jaccard']:.3f}, "
          f"shared={int(row['shared'])}, ρ={row['spearman_rho']:.3f}")

print(f"\n─── Custom Enrichment ───")
for _, row in custom_df.iterrows():
    print(f"  {row['gene_set']:25s}: {row['in_network']}/{row['expressed']}, "
          f"{row['fold_enrichment']:.1f}×, p = {row['p_value']:.2e}")

print(f"\n─── Cross-Stage Coordination ───")
print(f"  I×R mean r: {np.mean(ixr):.3f}")
print(f"  Random mean r: {np.mean(random_corrs):.3f}")
print(f"  t = {t_stat:.2f}, p = {t_p:.2e}, Cohen's d = {cohens_d:.2f}")
print(f"  EIF2S1–PELO direct r: {eif2s1_pelo_r:.3f}")

print(f"\n─── Agonal Sensitivity ───")
for _, row in agonal_df.iterrows():
    print(f"  {row['model']:20s}: J = {row['jaccard']:.3f}, ρ = {row['rho_ep']:.3f}, "
          f"preserve = {row['preserve_rho']:.4f}")

print(f"\n─── Cross-Region Replication (EIF2S1) ───")
eif_vals = cross_region_matrix['EIF2S1'][np.triu_indices(5, k=1)]
print(f"  Range: ρ = {eif_vals.min():.3f} – {eif_vals.max():.3f}")

print(f"\n─── LTN1–NEMF (for context) ───")
ln_row = comp_df[(comp_df['gene1'] == 'LTN1') & (comp_df['gene2'] == 'NEMF')]
if len(ln_row) > 0:
    print(f"  J = {ln_row['jaccard'].values[0]:.3f}, "
          f"shared = {int(ln_row['shared'].values[0])}, "
          f"ρ = {ln_row['spearman_rho'].values[0]:.3f}")

print("\n" + "=" * 70)
print("✓ PIPELINE COMPLETE — All results in MS2_Results/")
print("✓ All figures in MS2_Figures/")
print("=" * 70)
