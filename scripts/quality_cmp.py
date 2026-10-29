"""Quality push results: paired bootstrap of each system against OpenReviewer-8B (3 samples), per split.

  python quality_cmp.py   -> paper/tab_quality_push.md / .csv
Nothing here is fit to data: systems are raw scores (mean over samples) or a fixed rule.
  avg3    unweighted mean of the z-scored OpenReviewer, DeepReviewer-fast and AI Scientist scores (z with each split's
          own statistics, which uses no labels)
"""
import numpy as np
import pandas as pd

from common import ROOT, read_jsonl
from fusion import P, SPLITS, met, paired, scores

SYSTEMS = ["openreviewer-8b", "openreviewer-8b:n16", "openreviewer-8b:n16+gate_k2", "openreviewer-8b:n16+gate_k1_m2",
           "openreviewer-8b+gate_k2", "qwen3-32b-awq:ac", "deepreviewer-7b:fast", "avg3"]
REF = "openreviewer-8b"


def main():
    rows = []
    for split, (tag, data) in SPLITS.items():
        gold = {r["paper_id"]: (np.mean(r["human_ratings"]), "ccept" in r["decision"])
                for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
        d = pd.DataFrame(index=list(gold))
        d["human"] = [gold[p][0] for p in d.index]
        d["gold"] = [gold[p][1] for p in d.index]
        for s in SYSTEMS[:-1]:
            sc = scores(tag, s)
            if sc is not None:
                d[s] = sc
        parts = [c for c in ("openreviewer-8b", "deepreviewer-7b:fast") if c in d] + \
                ([] if scores(tag, "qwen3-32b-awq:ais") is None else ["ais"])
        if "ais" in parts:
            d["ais"] = scores(tag, "qwen3-32b-awq:ais")
        z = [(d[c] - d[c].mean()) / d[c].std() for c in parts]
        d["avg3"] = pd.concat(z, axis=1).mean(axis=1)
        for s in SYSTEMS:
            if s not in d:
                continue
            x = d.dropna(subset=[s, REF])
            y, h = x.gold.values.astype(bool), x.human.values
            a, r = met(y, h, x[s].values)
            row = {"split": split, "system": s, "n": len(x), "AUC": a, "Spearman": r}
            if s != REF:
                a0, r0 = met(y, h, x[REF].values)
                lo, hi = paired(x, x[s].values, x[REF].values)
                row.update(dAUC=a - a0, dAUC_ci=f"[{lo[0]:+.3f},{hi[0]:+.3f}]",
                           dSp=r - r0, dSp_ci=f"[{lo[1]:+.3f},{hi[1]:+.3f}]")
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{P}/tab_quality_push.csv", index=False)
    with open(f"{P}/tab_quality_push.md", "w") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else ("" if pd.isna(v) else str(v)) for v in r) + " |\n")
    print(t.to_string())


if __name__ == "__main__":
    main()
