"""Knowing vs weighing, decomposed (CPU; reads the judged K1 reviews).

  python severity.py [--judged out/judged.jsonl] [--out tab_severity]   -> paper/<out>.md / .csv
For every reviewer, over the injected-flaw (F) reviews:
  mention     the review states the injected flaw (judge: detected_flaw)
  fatal       ... and the review itself calls it fatal (judge: flaw_severity == fatal; for the ARR form this is
              the review's own severity tag on that weakness)
  pass        overall score on the accept side (ARR >= 3, ICLR >= 6, as in analyze.py)
Rows: P(mention), P(fatal | mention), P(pass | mention & fatal), P(pass | mention & not fatal), P(pass | not
mentioned), P(pass) on the matched N versions. P(pass | mention & fatal) is the cleanest "knowing is not weighing"
number: the reviewer itself says the flaw invalidates the paper and still recommends acceptance.
CIs: paper-cluster bootstrap (analyze.boot).
"""
import argparse

import numpy as np
import pandas as pd

from analyze import ICLR_THRESHOLD, boot, flatten
from common import ROOT, read_jsonl
from fusion import P


def main(a):
    df = flatten(read_jsonl(f"{ROOT}/{a.judged}"))
    df = df[df.parsed & ~df.model.str.contains("gate")]
    rows = []
    for model, d in df.groupby("model"):
        thr = ICLR_THRESHOLD if (d.scale == "iclr10").all() else 3
        d = d.assign(passed=d.overall >= thr)
        F, N = d[d.version == "F"], d[d.version == "N"]
        if len(F) < 10:
            continue

        def cell(mask_fn):
            return boot(F, lambda x: x[mask_fn(x)].passed.mean() if mask_fn(x).any() else np.nan, B=500)

        mf = lambda x: x.detected & x.judge_fatal
        mn = lambda x: x.detected & ~x.judge_fatal
        nd = lambda x: ~x.detected
        r = {"model": model, "nF": len(F), "mention": F.detected.mean(),
             "fatal_given_mention": F[F.detected].judge_fatal.mean() if F.detected.any() else np.nan,
             "n_mention_fatal": int(mf(F).sum())}
        for name, fn in (("pass_mention_fatal", mf), ("pass_mention_notfatal", mn), ("pass_not_mentioned", nd)):
            e, lo, hi = cell(fn)
            r[name], r[name + "_ci"] = e, f"[{lo:.2f},{hi:.2f}]"
        r["pass_N"] = N.passed.mean() if len(N) else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("mention", ascending=False)
    t.to_csv(f"{P}/{a.out}.csv", index=False)
    with open(f"{P}/{a.out}.md", "w") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
    pd.set_option("display.width", 250)
    print(t.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="out/judged.jsonl")
    ap.add_argument("--out", default="tab_severity")
    main(ap.parse_args())
