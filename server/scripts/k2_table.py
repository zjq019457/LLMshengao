"""Held-out flaw types (K2) next to K1: does the gate still work on flaws its audit checklist never names?

  python k2_table.py   -> paper/tab_k2.md / .csv
Per reviewer (raw and gated) and benchmark: detection on F, share of F versions on the accept side (pass_F), the
same on the matched N versions (pass_N), and F - N in score, with paper-cluster bootstrap CIs. K2 rows are also
split by flaw type (circ / metr / theo).
"""
import numpy as np
import pandas as pd

from analyze import ICLR_THRESHOLD, boot, flatten, paired_delta
from common import ROOT, read_jsonl
from fusion import P

BENCH = {"K1": "out/judged.jsonl", "K2": "out_k2/judged.jsonl"}


def rows_for(df, bench, cell):
    out = []
    for model, d in df.groupby("model"):
        thr = ICLR_THRESHOLD if (d.scale == "iclr10").all() else 3
        d = d.assign(passed=(d.overall >= thr).astype(float))
        F, N = d[d.version == "F"], d[d.version == "N"]
        if len(F) < 5 or not len(N):
            continue
        pf = boot(F, lambda x: x.passed.mean(), B=500)
        fn = boot(d, paired_delta("overall", "F", "N"), B=500)
        out.append({"bench": bench, "cell": cell, "model": model, "nF": len(F), "det_F": F.detected.mean(),
                    "pass_F": pf[0], "pass_F_ci": f"[{pf[1]:.2f},{pf[2]:.2f}]", "pass_N": N.passed.mean(),
                    "F_minus_N": fn[0], "F_minus_N_ci": f"[{fn[1]:+.2f},{fn[2]:+.2f}]"})
    return out


def main():
    rows = []
    for bench, f in BENCH.items():
        df = flatten(read_jsonl(f"{ROOT}/{f}"))
        df = df[df.parsed & ~df.model.str.contains("audit")]
        rows += rows_for(df, bench, "all")
        if bench == "K2":
            df["type"] = df.variant.str[:4]
            for t in ("circ", "metr", "theo"):
                rows += rows_for(df[df.type == t], bench, t)  # N rows carry the variant they were judged against
    t = pd.DataFrame(rows)
    keep = set(t[t.bench == "K2"].model)
    t = t[t.model.isin(keep)].sort_values(["cell", "model", "bench"])
    t.to_csv(f"{P}/tab_k2.csv", index=False)
    with open(f"{P}/tab_k2.md", "w") as fo:
        fo.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            fo.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
    pd.set_option("display.width", 250)
    print(t.to_string())


if __name__ == "__main__":
    main()
