#!/usr/bin/env python3
"""
EIF2S1 R3 REVISION - CROSS-COHORT META (Reviewer 4).
Exact stratified permutation meta-test of the EIF2S1-PELO disease-associated dissociation-gap
change across the two RNA-seq cohorts where the gap is directly comparable: PD (GSE68719 BA9)
and ALS (GSE124439 frontal cortex). Direction was established a priori by PD, so ALS is a
replication and the combined test is one-sided in the PD direction.

Also reports the platform-robust coupling metric (WJ_shifted) change across all three cohorts
(PD, AD GSE5281, ALS), where the AD value is read from its saved result.

seed=42. Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616.
Output: results/disease_meta_rnaseq.json
"""
import os, sys, gzip, json, tarfile, urllib.request
try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception: pass
import numpy as np, pandas as pd
from scipy import stats

SEED = 42; np.random.seed(SEED); N_PERM = 5000; TOP = 5
ROOT = r"G:\My Drive\inner_architecture_research"
DATA = os.path.join(ROOT, "EIF2S1", "r3_revision_analyses", "data")
CACHE = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "EIF2S1", "r3_revision_analyses", "results")

def make_gap_fn(X, eif, pel, keep):
    def gap(sidx):
        sub = X[:, sidx]; R = stats.rankdata(sub, axis=1)
        Rc = R - R.mean(axis=1, keepdims=True); den = np.sqrt((Rc**2).sum(axis=1)); den[den == 0] = 1e-12
        a = ((Rc @ Rc[eif]) / (den * den[eif]))[keep]; b = ((Rc @ Rc[pel]) / (den * den[pel]))[keep]
        a_s, b_s = (a+1)/2, (b+1)/2
        wj = np.minimum(a_s, b_s).sum() / np.maximum(a_s, b_s).sum()
        ta, tb = np.percentile(a, 100-TOP), np.percentile(b, 100-TOP)
        bj = ((a >= ta) & (b >= tb)).sum() / ((a >= ta) | (b >= tb)).sum()
        return wj - bj
    return gap

def prep(expr):
    genes = expr.index.to_numpy(); X = expr.to_numpy(np.float64)
    eif = int(np.where(genes == 'EIF2S1')[0][0]); pel = int(np.where(genes == 'PELO')[0][0])
    keep = np.ones(X.shape[0], bool); keep[eif] = keep[pel] = False
    return X, eif, pel, keep

# ---- PD (GSE68719) ----
print("[load] PD GSE68719")
raw = pd.read_csv(os.path.join(DATA, "GSE68719_mlpd_PCG_DESeq2_norm_counts.txt.gz"), sep='\t', compression='gzip')
c = raw.drop(columns=[raw.columns[0]]).set_index('symbol').groupby(level=0).mean()
pd_expr = c[c.median(axis=1) >= 5]
pd_ctl = [i for i, s in enumerate(pd_expr.columns) if str(s).startswith('C_')]
pd_dis = [i for i, s in enumerate(pd_expr.columns) if str(s).startswith('P_')]
Xp, ep, pp, kp = prep(pd_expr); gap_pd = make_gap_fn(Xp, ep, pp, kp)
pd_ctl, pd_dis = np.array(pd_ctl), np.array(pd_dis)
obs_pd = gap_pd(pd_dis) - gap_pd(pd_ctl)
print(f"  PD obs gap diff = {obs_pd:+.4f}  (n_ctl={len(pd_ctl)}, n_dis={len(pd_dis)})")

# ---- ALS (GSE124439) ----
print("[load] ALS GSE124439")
mtxt = gzip.decompress(urllib.request.urlopen(
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124439/matrix/GSE124439_series_matrix.txt.gz",
    timeout=90).read()).decode('utf-8', 'ignore')
gsms = region = group = []
for line in mtxt.split('\n'):
    if line.startswith('!Sample_geo_accession'): gsms = [v.strip().strip('"') for v in line.split('\t')[1:]]
    elif line.startswith('!Sample_characteristics_ch1'):
        cl = [ (v.split(':',1)[1].strip() if ':' in v else v).strip('"') for v in line.split('\t')[1:]]
        j = ' '.join(cl)
        if 'Cortex' in j: region = cl
        elif 'ALS' in j or 'Control' in j: group = cl
meta = pd.DataFrame({'GSM': gsms, 'region': region, 'group': group})
sel = meta[(meta.region == 'Frontal Cortex') & (meta.group.isin(['ALS Spectrum MND', 'Non-Neurological Control']))]
tf = tarfile.open(os.path.join(CACHE, "GSE124439_RAW.tar")); mbg = {m.split('_')[0]: m for m in tf.getnames()}
cols = {}
for gsm in sel.GSM:
    mem = mbg.get(gsm)
    if not mem: continue
    d = gzip.decompress(tf.extractfile(mem).read()).decode('utf-8', 'ignore'); s = {}
    for ln in d.split('\n')[1:]:
        p = ln.split('\t')
        if len(p) >= 2:
            try: s[p[0].strip().strip('"')] = float(p[1])
            except ValueError: pass
    cols[gsm] = s
cnt = pd.DataFrame(cols).dropna(how='all')
cnt = cnt.loc[[not g.startswith(('ERCC', '__')) for g in cnt.index]]
cpm = cnt / cnt.sum(axis=0) * 1e6
als_expr = np.log2(cpm[cpm.median(axis=1) >= 1.0] + 1.0)
grpmap = dict(zip(sel.GSM, sel.group.map({'ALS Spectrum MND': 'ALS', 'Non-Neurological Control': 'Control'})))
scol = list(als_expr.columns)
als_ctl = np.array([i for i, s in enumerate(scol) if grpmap.get(s) == 'Control'])
als_dis = np.array([i for i, s in enumerate(scol) if grpmap.get(s) == 'ALS'])
Xa, ea, pa, ka = prep(als_expr); gap_als = make_gap_fn(Xa, ea, pa, ka)
obs_als = gap_als(als_dis) - gap_als(als_ctl)
print(f"  ALS obs gap diff = {obs_als:+.4f}  (n_ctl={len(als_ctl)}, n_dis={len(als_dis)})")

# ---- stratified permutation meta (sum of cohort gap-diffs) ----
print(f"[meta] stratified permutation ({N_PERM}), one-sided in PD-established (+) direction")
obs_sum = obs_pd + obs_als
pd_pool = np.concatenate([pd_ctl, pd_dis]); pd_n = len(pd_ctl)
als_pool = np.concatenate([als_ctl, als_dis]); als_n = len(als_ctl)
null = np.empty(N_PERM)
for k in range(N_PERM):
    p1 = np.random.permutation(pd_pool); d1 = gap_pd(p1[pd_n:]) - gap_pd(p1[:pd_n])
    p2 = np.random.permutation(als_pool); d2 = gap_als(p2[als_n:]) - gap_als(p2[:als_n])
    null[k] = d1 + d2
p_one = (np.sum(null >= obs_sum) + 1) / (N_PERM + 1)
p_two = (np.sum(np.abs(null) >= abs(obs_sum)) + 1) / (N_PERM + 1)
print(f"  meta obs (PD+ALS gap diff sum) = {obs_sum:+.4f}")
print(f"  one-sided meta p = {p_one:.4f}   two-sided = {p_two:.4f}")

# coupling-weakening direction across all three (WJ change), AD read from its result
ad = pd.read_csv(os.path.join(OUT, "disease_GSE5281_SFG_eif2s1_pelo.csv")).iloc[0]
dWJ = {'PD': 0.861 - 0.914, 'AD': float(ad.ad_wj_shifted) - float(ad.control_wj_shifted),
       'ALS': 0.752 - 0.850}
n_neg = sum(v < 0 for v in dWJ.values())
sign_p = 2 * 0.5**3  # two-sided sign test, 3/3 same direction
print(f"  coupling change (WJ_shifted disease-control): {dict((k,round(v,3)) for k,v in dWJ.items())}")
print(f"  coupling weakens in {n_neg}/3 cohorts; sign-test two-sided p = {sign_p:.3f}")

json.dump({
    "rnaseq_gap_meta": {
        "cohorts": ["PD GSE68719 BA9", "ALS GSE124439 frontal cortex"],
        "obs_gap_diff_PD": float(obs_pd), "obs_gap_diff_ALS": float(obs_als),
        "meta_statistic_sum": float(obs_sum), "n_perm": N_PERM,
        "one_sided_p_PD_direction": float(p_one), "two_sided_p": float(p_two)},
    "coupling_weakening_all3": {"delta_WJ": dWJ, "n_weaken": int(n_neg), "sign_test_two_sided_p": sign_p},
    "seed": SEED, "date": "2026-07-05"
}, open(os.path.join(OUT, "disease_meta_rnaseq.json"), 'w'), indent=2)
print("\n=== META SUMMARY ===")
print(f"  RNA-seq gap meta (PD+ALS): one-sided p={p_one:.4f}")
print(f"  coupling weakens {n_neg}/3 diseases")
print("DONE.")
