# -*- coding: utf-8 -*-
"""
Pipeline: STRING protein-protein interaction analysis for the EIF2S1-PELO panel
Author: Drake H. Harbert (D.H.H.)
Affiliation: Inner Architecture LLC, Canton, OH
ORCID: 0009-0007-7740-3616
Date: 2026-06-30
Description:
    Queries the STRING database (v12, Homo sapiens) REST API for the EIF2S1-PELO
    ribosome-quality-control / translation panel and produces the protein-level
    validation that the editorial decision (Cwetsch, concern 2) requires as a
    supplementary file. Outputs: (1) the protein-protein interaction network among
    the panel genes with per-evidence-channel and combined confidence scores;
    (2) the STRING PPI enrichment p-value (whether the set has more interactions
    than expected for a random set of the same size, the formal protein-level test
    of whether co-expression coupling corresponds to known interactions);
    (3) the top STRING interaction partners of EIF2S1 and PELO individually, so the
    protein-interaction partners can be compared to the co-expression partners
    (the layer-separation question); (4) the STRING network image; (5) a clean
    network figure; (6) provenance.json. No data are fabricated; every value comes
    from the STRING API response.
Dependencies: requests, pandas, matplotlib, networkx
Input: none (STRING REST API over the network)
Output: EIF2S1/revision_r3/Supplementary/string_analysis/
"""
import os, sys, json, io, time
import requests
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== CONFIG =====================
RANDOM_SEED = 42
FORCE_RECOMPUTE = True
GENES = ["EIF2S1", "PELO", "LTN1", "NEMF", "TMEM97", "SIGMAR1",
         "TCF25", "ASCC1", "ASCC3", "TRIP4"]
SPECIES = 9606                      # Homo sapiens
REQUIRED_SCORE = 400               # STRING combined-score threshold (0-1000); 400 = medium confidence
TOP_PARTNERS = 15                  # top STRING interactors to pull for EIF2S1 and PELO
CALLER = "InnerArchitectureLLC_EIF2S1_PELO"
API = "https://string-db.org/api"
OUT = r"G:/My Drive/inner_architecture_research/EIF2S1/revision_r3/Supplementary/string_analysis"
os.makedirs(OUT, exist_ok=True)
# ==================================================

def get(endpoint, fmt, params):
    params = {**params, "species": SPECIES, "caller_identity": CALLER}
    url = f"{API}/{fmt}/{endpoint}"
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    time.sleep(1)   # STRING asks for <= 1 request/sec
    return r

def tsv(endpoint, params):
    r = get(endpoint, "tsv", params)
    return pd.read_csv(io.StringIO(r.text), sep="\t")

def main():
    ids = "%0d".join(GENES)
    print(f"STRING analysis for {len(GENES)} genes: {', '.join(GENES)}")

    # 1) resolve identifiers (confirms STRING recognizes each gene)
    mapped = tsv("get_string_ids", {"identifiers": ids, "limit": 1, "echo_query": 1})
    mapped[["queryItem", "preferredName", "stringId"]].to_csv(
        os.path.join(OUT, "string_id_mapping.csv"), index=False)
    print(f"  mapped {len(mapped)} of {len(GENES)} genes to STRING IDs")
    missing = set(GENES) - set(mapped["queryItem"])
    if missing:
        print(f"  WARNING: not found in STRING: {sorted(missing)}")

    # 2) network among the panel genes
    net = tsv("network", {"identifiers": ids, "required_score": REQUIRED_SCORE})
    keep = ["preferredName_A", "preferredName_B", "score", "nscore", "fscore",
            "pscore", "ascore", "escore", "dscore", "tscore"]
    keep = [c for c in keep if c in net.columns]
    net = net[keep].sort_values("score", ascending=False)
    net.to_csv(os.path.join(OUT, "string_network_edges.csv"), index=False)
    print(f"  network: {len(net)} edges at combined score >= {REQUIRED_SCORE/1000:.2f}")
    # is there a direct EIF2S1-PELO edge?
    pair = net[((net.preferredName_A == "EIF2S1") & (net.preferredName_B == "PELO")) |
               ((net.preferredName_A == "PELO") & (net.preferredName_B == "EIF2S1"))]
    eif_pelo = f"{pair.iloc[0]['score']/1000:.3f}" if len(pair) else "none (no edge >= threshold)"
    print(f"  direct EIF2S1-PELO STRING edge: {eif_pelo}")

    # 3) PPI enrichment (the formal protein-level test)
    enr = tsv("ppi_enrichment", {"identifiers": ids, "required_score": REQUIRED_SCORE})
    enr.to_csv(os.path.join(OUT, "string_ppi_enrichment.csv"), index=False)
    e = enr.iloc[0].to_dict()
    print(f"  PPI enrichment: observed edges={e.get('number_of_edges')}, "
          f"expected={e.get('expected_number_of_edges')}, p={e.get('p_value')}")

    # 4) top STRING partners of EIF2S1 and PELO (for the layer-separation comparison)
    partner_tables = {}
    for g in ["EIF2S1", "PELO"]:
        ip = tsv("interaction_partners", {"identifiers": g, "limit": TOP_PARTNERS})
        cols = [c for c in ["preferredName_B", "score", "escore", "dscore", "tscore"] if c in ip.columns]
        ip = ip[cols].rename(columns={"preferredName_B": "string_partner"})
        ip.to_csv(os.path.join(OUT, f"{g}_top_string_partners.csv"), index=False)
        partner_tables[g] = list(ip["string_partner"])
        print(f"  {g} top STRING partners: {', '.join(partner_tables[g][:8])} ...")

    # 5) STRING-rendered network image
    try:
        img = get("image/network", "", {"identifiers": ids, "required_score": REQUIRED_SCORE,
                                         "network_flavor": "evidence"})
        with open(os.path.join(OUT, "string_network_image.png"), "wb") as f:
            f.write(img.content)
        print("  saved STRING network image")
    except Exception as ex:
        print(f"  (STRING image skipped: {ex})")

    # 6) clean network figure (networkx)
    try:
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(GENES)
        for _, r in net.iterrows():
            G.add_edge(r["preferredName_A"], r["preferredName_B"], w=r["score"]/1000)
        plt.figure(figsize=(9, 7))
        import numpy as np  # noqa
        pos = nx.spring_layout(G, seed=RANDOM_SEED, k=0.9)
        ew = [G[u][v]["w"]*4 for u, v in G.edges()]
        core = {"EIF2S1", "PELO"}
        nc = ["#D55E00" if n in core else "#0072B2" for n in G.nodes()]
        nx.draw_networkx_edges(G, pos, width=ew, edge_color="#999999", alpha=0.7)
        nx.draw_networkx_nodes(G, pos, node_color=nc, node_size=1600, edgecolors="black")
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", font_color="white")
        plt.title("STRING protein-protein interaction network, EIF2S1-PELO panel\n"
                  f"(Homo sapiens, combined score >= {REQUIRED_SCORE/1000:.2f})", fontsize=12)
        plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(OUT, "string_network_figure.png"), dpi=300)
        plt.savefig(os.path.join(OUT, "string_network_figure.pdf"))
        plt.close()
        print("  saved clean network figure (PNG + PDF)")
    except Exception as ex:
        print(f"  (networkx figure skipped: {ex})")

    # 7) provenance
    prov = {
        "analysis": "STRING PPI validation for EIF2S1-PELO panel",
        "database": "STRING v12", "species": SPECIES, "genes": GENES,
        "required_score": REQUIRED_SCORE, "n_edges": int(len(net)),
        "direct_eif2s1_pelo_edge": eif_pelo,
        "ppi_enrichment_p": str(e.get("p_value")),
        "observed_edges": int(e.get("number_of_edges", 0)),
        "expected_edges": float(e.get("expected_number_of_edges", 0)),
        "random_seed": RANDOM_SEED, "execution_date": "2026-06-30",
        "pipeline_file": "string_ppi_analysis.py",
        "purpose": "Editorial decision (Cwetsch) concern 2: provide referenced STRING network analyses",
    }
    json.dump(prov, open(os.path.join(OUT, "provenance.json"), "w"), indent=2)
    print(f"\nDONE. Outputs in: {OUT}")
    print("Key result for the manuscript: PPI enrichment p =", e.get("p_value"),
          "| direct EIF2S1-PELO edge:", eif_pelo)

if __name__ == "__main__":
    main()
