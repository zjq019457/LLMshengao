"""Clean ICLR 2026 test (after every model's training data): every system against OpenReviewer-8B, paired bootstrap.

  python cmp26.py   -> paper/tab_iclr26.md / .csv
Also pools test25 + test24 + test26 (600 papers) as a secondary row. Gate settings are the ones fixed before this split
existed (pre-registered _k1; M3 = n16 + gate_k2), nothing is chosen on test26.
"""
import numpy as np
import pandas as pd

from common import ROOT, read_jsonl
from fusion import P, met, paired, scores

SPLITS = {"test26": ("_bench26", "bench_iclr26"), "test25": ("_bench", "bench_iclr25"), "test24": ("_bench24", "bench_iclr24")}
SYSTEMS = ["qwen3-32b-awq:s10", "qwen3-32b-awq:ais", "deepreviewer-7b:fast", "deepreviewer-7b:fast+gate_k1",
           "deepreviewer-7b:fast+gate_k2", "qwen3-32b-awq:think+verify+s10", "qwen3-32b-awq:think+verify+grule+s10",
           "openreviewer-8b", "openreviewer-8b+gate_k1", "openreviewer-8b+gate_k2",
           "openreviewer-8b:n16", "openreviewer-8b:n16+gate_k1", "openreviewer-8b:n16+gate_k2"]
REF = "openreviewer-8b"


def frame(tag, data):
    g = {r["paper_id"]: (np.mean(r["human_ratings"]), "ccept" in r["decision"]) for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
    d = pd.DataFrame(index=list(g))
    d["human"] = [g[p][0] for p in d.index]
    d["gold"] = [g[p][1] for p in d.index]
    for s in SYSTEMS:
        sc = scores(tag, s)
        if sc is not None:
            d[s] = sc
    return d


def main():
    F = {k: frame(*v) for k, v in SPLITS.items()}
    # human ratings are on different scales across years, so the pooled Spearman is computed on within-year ranks
    for d in F.values():
        d["human"] = d.human.rank(pct=True)
    F["pooled_3yr"] = pd.concat([d.rename(index=lambda p, k=k: f"{k}/{p}") for k, d in F.items()])
    rows = []
    for split in ("test26", "pooled_3yr"):
        d = F[split]
        for s in SYSTEMS:
            if s not in d or REF not in d:
                continue
            x = d.dropna(subset=[s, REF])
            y, h = x.gold.values.astype(bool), x.human.values
            a, r = met(y, h, x[s].values)
            row = {"split": split, "system": s, "n": len(x), "AUC": a, "Spearman": r}
            if s != REF:
                a0, r0 = met(y, h, x[REF].values)
                lo, hi = paired(x, x[s].values, x[REF].values)
                row.update(dAUC=a - a0, dAUC_ci=f"[{lo[0]:+.3f},{hi[0]:+.3f}]", dSp=r - r0, dSp_ci=f"[{lo[1]:+.3f},{hi[1]:+.3f}]")
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{P}/tab_iclr26.csv", index=False)
    with open(f"{P}/tab_iclr26.md", "w") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else ("" if pd.isna(v) else str(v)) for v in r) + " |\n")
    pd.set_option("display.width", 250)
    print(t.to_string())


if __name__ == "__main__":
    main()
