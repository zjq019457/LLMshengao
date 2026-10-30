"""Sample-count curve and score-shift check for the self-consistency + gate systems (CPU only).

  python curve.py   -> paper/tab_curve.md / .csv, paper/tab_score_shift.md / .csv
For each reviewer with n16 samples (OpenReviewer-8B, DeepReviewer-7B fast when present) and k in 1, 3, 8, 16, the score
is the mean over samples 0..k-1, without and with gate_k2 (the gated file has the same samples, penalised).
Metrics per split and pooled test; the gate's paired difference at the same k uses a paired paper bootstrap.
Score shift: mean score and share of papers with score >= 6 (ICLR accept side), raw vs gated, k = 16.
"""
import os

import numpy as np
import pandas as pd

from common import ROOT, read_jsonl
from fusion import P, SPLITS, met, paired

BASES = {"openreviewer-8b:n16": "OpenReviewer-8B", "deepreviewer-7b:fast+n16": "DeepReviewer-7B (fast)"}
KS = [1, 3, 8, 16]


def samples(tag, cfg):
    f = f"{ROOT}/out{tag}/reviews_{cfg.replace(':', '_')}.jsonl"
    if not os.path.exists(f):
        return None
    d = pd.DataFrame([{"paper": r["paper_id"], "k": r["sample"],
                       "s": pd.to_numeric((r.get("review") or {}).get("overall_assessment"), errors="coerce")}
                      for r in read_jsonl(f)])
    return d.dropna()


def gold(data):
    g = {r["paper_id"]: (np.mean(r["human_ratings"]), "ccept" in r["decision"])
         for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
    d = pd.DataFrame(index=list(g))
    d["human"] = [g[p][0] for p in d.index]
    d["gold"] = [g[p][1] for p in d.index]
    return d


def main():
    rows, shift = [], []
    for base, name in BASES.items():
        F = {}
        for split, (tag, data) in SPLITS.items():
            raw, gated = samples(tag, base), samples(tag, base + "+gate_k2")
            if raw is None or gated is None:
                break
            d = gold(data)
            for k in KS:
                d[f"raw{k}"] = raw[raw.k < k].groupby("paper").s.mean()
                d[f"gate{k}"] = gated[gated.k < k].groupby("paper").s.mean()
            F[split] = d
            x = d.dropna(subset=["raw16", "gate16"])
            shift.append({"reviewer": name, "split": split, "mean_raw": x.raw16.mean(), "mean_gate": x.gate16.mean(),
                          "accept_side_raw": (x.raw16 >= 6).mean(), "accept_side_gate": (x.gate16 >= 6).mean(),
                          "papers_penalised": (x.gate16 < x.raw16 - 1e-9).mean(), "true_accept_rate": x.gold.mean()})
        if len(F) < len(SPLITS):
            print(f"skip {name}: missing files")
            continue
        F["pooled_test"] = pd.concat([F["test25"].rename(index=lambda p: "25/" + p),
                                      F["test24"].rename(index=lambda p: "24/" + p)])
        for split, d in F.items():
            for k in KS:
                x = d.dropna(subset=[f"raw{k}", f"gate{k}"])
                y, h = x.gold.values.astype(bool), x.human.values
                a0, r0 = met(y, h, x[f"raw{k}"].values)
                a1, r1 = met(y, h, x[f"gate{k}"].values)
                lo, hi = paired(x, x[f"gate{k}"].values, x[f"raw{k}"].values)
                rows.append({"reviewer": name, "split": split, "k": k, "n": len(x), "AUC": a0, "Spearman": r0,
                             "AUC_gate": a1, "Spearman_gate": r1, "dAUC": a1 - a0, "dAUC_ci": f"[{lo[0]:+.3f},{hi[0]:+.3f}]",
                             "dSp": r1 - r0, "dSp_ci": f"[{lo[1]:+.3f},{hi[1]:+.3f}]"})
    for t, fn in ((pd.DataFrame(rows), "tab_curve"), (pd.DataFrame(shift), "tab_score_shift")):
        t.to_csv(f"{P}/{fn}.csv", index=False)
        with open(f"{P}/{fn}.md", "w") as f:
            f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
            for _, r in t.iterrows():
                f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
        pd.set_option("display.width", 250)
        print(t.to_string())


if __name__ == "__main__":
    main()
