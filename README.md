# EIF2S1-PELO Co-Expression Architecture in Human Brain

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GTEx v8](https://img.shields.io/badge/Data-GTEx%20v8-orange)](https://gtexportal.org/)
[![GSE68719](https://img.shields.io/badge/Data-GSE68719-purple)](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68719)
[![DepMap](https://img.shields.io/badge/Data-DepMap%2024Q2-darkred)](https://depmap.org/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

**Manuscript (R3 revision):** *EIF2S1-PELO Co-Expression Architecture in Human Brain: Pairing-Family Decomposition Reveals Constitutive Transcriptional Coupling, Disease-Selective Top-Partner Disruption, and Functional Layer Separation*

**Author:** Drake H. Harbert · [ORCID 0009-0007-7740-3616](https://orcid.org/0009-0007-7740-3616)
**Affiliation:** Inner Architecture LLC, Canton, OH 44721, USA
**Contact:** drake@innerarchitecturellc.com

---

## Status

**Active revision under peer review.** The R3 revision substantially reframes the original findings using the Layer 2H pairing-family decomposition framework (Harbert, in press, Frontiers in Pharmacology). The reframing addresses substantive reviewer concerns by adding five new analytical pipelines, formal statistical anchoring (permutation null + bootstrap CI), and disease-cohort analysis.

---

## Summary

This repository contains the complete analytical pipeline for a co-expression study of EIF2S1 (eIF2α, translation initiation surveillance) and PELO (Pelota, ribosome rescue) in human brain. The R3 revision applies pairing-family architectural decomposition to characterize the transcriptional coupling between these two translational quality control modules at multiple analytical layers, distinct from established direct biochemical mechanism (ZAK/GCN-mediated eIF2α phosphorylation; Wu et al. 2020).

## Key Findings (R3 revision)

| Finding | Value | Significance |
|---------|-------|--------------|
| EIF2S1-PELO continuous WJ (shifted) | 0.914 | Near-identical genome-wide coupling |
| Binary Jaccard at top 5% | 0.490 | Substantial top-partner overlap |
| Type 1 dissociation gap | 0.424 | Lowest in 21-pair QC panel |
| Permutation null p-value | 0.015 | Significantly smaller than 200 random pairs (z=-2.57) |
| Bootstrap 95% CI on gap | [0.325, 0.649] | Excludes random-null mean |
| Type 2 max sign-treatment gap | 0.138 | Moderate sign-driven structure |
| Type 6 Pearson-Spearman gap | 0.004 | Approximately linear-monotonic |
| DepMap functional co-dependency | 0.006 (Jaccard); -0.07 (Pearson) | Layer separation: zero functional coupling |
| PD vs control gap shift | 0.424 → 0.536 | Top-partner layer selectively disrupted |
| Cross-region preservation | EIF2S1 0.899 / PELO 0.894 | Constitutive across 5 brain regions |

## Repository Structure

```
eif2s1-translational-qc-hub/
│
├── README.md                              # This file
├── LICENSE                                # MIT License
├── requirements.txt                       # Python dependencies
│
├── MS2_EIF2S1_Definitive_Pipeline.py      # Original primary pipeline (multi-region GTEx)
│
├── all_21_pairwise_comparisons.csv        # Original pipeline output
├── cross_region_replication.csv           # Original pipeline output
├── cross_stage_coordination.csv           # Original pipeline output
├── EIF2S1_custom_enrichment.csv           # Original pipeline output
├── gProfiler_*.csv                        # GO enrichment results
├── agonal_sensitivity.csv                 # Robustness analysis
├── Figure*.pdf                            # Original figures
│
└── r3_revision_pipelines/                 # R3 REVISION ADDITIONS (2026-04-29)
    │
    ├── pipeline_1_tcf25_asc1.py           # Reviewer 2: TCF25/ASC-1 panel extension
    ├── pipeline_2_disease_replication.py  # Reviewer 4: Parkinson's disease BA9 (GSE68719)
    ├── pipeline_3_depmap.py               # Reviewer 2: DepMap CRISPR co-dependency
    ├── pipeline_4_layer2h_pairing.py      # 21-pair Layer 2H landscape (Harbert in press)
    ├── pipeline_5_formal_layer2h.py       # Formal continuous WJ + permutation null + bootstrap CI
    │
    ├── results/
    │   ├── all_pairwise_comparisons_extended.csv  (Pipeline 1)
    │   ├── new_gene_top_partners.csv               (Pipeline 1)
    │   ├── disease_GSE68719_*.csv                  (Pipeline 2)
    │   ├── disease_GSE138260_*.csv                 (Pipeline 2 — secondary cohort)
    │   ├── depmap_*.csv                            (Pipeline 3)
    │   ├── layer2h_dissociation_gaps.csv           (Pipeline 4)
    │   ├── layer2h_cross_region_summary.csv        (Pipeline 4)
    │   ├── layer2h_layer_separation_summary.csv    (Pipeline 4)
    │   ├── layer2h_formal_continuous_wj.csv        (Pipeline 5)
    │   ├── layer2h_type2_sign_treatment.csv        (Pipeline 5)
    │   ├── layer2h_type6_pearson_vs_spearman.csv   (Pipeline 5)
    │   ├── layer2h_permutation_null.csv            (Pipeline 5)
    │   ├── layer2h_bootstrap_ci.csv                (Pipeline 5)
    │   ├── layer2h_disease_comparison.csv          (Pipeline 5)
    │   └── provenance_pipeline*.json               (Each pipeline's provenance)
    │
    └── figures/
        └── layer2h_dissociation_gap_distribution.{png,pdf}  (Figure 2 of revised manuscript)
```

## Reproduction

### Original primary analysis (multi-region GTEx)

```bash
# Run in Google Colab (12GB RAM)
python MS2_EIF2S1_Definitive_Pipeline.py
```

### R3 revision pipelines (local execution)

```bash
cd r3_revision_pipelines/

# Pipeline 1: TCF25/ASC-1 panel extension
python pipeline_1_tcf25_asc1.py

# Pipeline 2: Parkinson's disease BA9 cohort
python pipeline_2_disease_replication.py

# Pipeline 3: DepMap CRISPR co-dependency
python pipeline_3_depmap.py

# Pipeline 4: 21-pair Layer 2H landscape
python pipeline_4_layer2h_pairing.py

# Pipeline 5: Formal continuous WJ + permutation null + bootstrap CI
python pipeline_5_formal_layer2h.py
```

All pipelines use `RANDOM_SEED = 42` and `FORCE_RECOMPUTE = True` for full reproducibility.

## Methodology — Layer 2H Pairing-Family Decomposition

The R3 revision applies the principled-pairing criterion (Harbert, in press) to decompose EIF2S1-PELO architectural coupling into multiple complementary measurements:

- **Type 1 (Continuous-Discrete):** Continuous Weighted Jaccard vs. binary Jaccard at top 5%. The dissociation gap localizes whether divergence concentrates in the broad value distribution or at the top-partner tail.
- **Type 2 (Sign-Treatment):** Three formulations on the same coupling profiles — shifted (r+1)/2, unsigned |r|, and signed split. Gaps quantify sign contribution.
- **Type 6 (Substrate-Projection):** Spearman vs. Pearson realizations. Near-zero gap indicates approximately linear-monotonic structure.
- **Permutation null:** 200 random gene-pair gaps establish the random-null distribution.
- **Bootstrap 95% CI:** 500 subject-resamples on the EIF2S1-PELO gap.
- **Layer separation:** Transcriptional coupling (this work) vs. functional cellular co-dependency (DepMap).

## Citation (when manuscript is published)

```
Harbert, D.H. (year). EIF2S1-PELO Co-Expression Architecture in Human Brain:
Pairing-Family Decomposition Reveals Constitutive Transcriptional Coupling,
Disease-Selective Top-Partner Disruption, and Functional Layer Separation.
[Journal]. DOI: 10.xxxx/...
```

## Methodological reference

Pairing-family decomposition framework introduced in:

```
Harbert, D.H. (in press). Sigma-1 and Sigma-2 receptors exhibit divergent
genome-wide co-expression architectures in human brain despite shared
subcellular localization. Frontiers in Pharmacology, Neuropharmacology section.
```

## Data Sources

| Source | Use |
|--------|-----|
| GTEx v8 (gtexportal.org) | Multi-region brain RNA-seq (primary analysis) |
| GSE68719 (NCBI GEO) | BA9 prefrontal cortex PD vs control RNA-seq |
| DepMap 24Q2 (depmap.org) | CRISPR gene-effect data, 1,150 cancer cell lines |

## License

MIT License — see [LICENSE](LICENSE).

## Acknowledgments

Thanks to GTEx, NCBI GEO, and DepMap for making the underlying data publicly available, and to the Frontiers in Pharmacology editor and reviewers for substantive feedback that materially strengthened this work.
