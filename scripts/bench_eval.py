"""Review-quality benchmark on 200 ICLR 2025 papers (DeepReview-13K test_2025, 100 accept / 100 reject).

  python bench_eval.py      # reads out_bench/reviews_*.jsonl -> out_bench/bench_result.txt

Per config, the predicted rating of a paper is the mean over its parsed samples (all configs rate on 1-10).
  coverage   papers with >= 1 parsed review
  MAE        |predicted - mean human rating|
  Spearman   rank correlation with the mean human rating
  AUC        accept vs reject ranked by predicted rating (threshold free)
  acc / F1   decision: the model's own Accept/Reject when it gives one, else rating >= 6
Paper-cluster bootstrap 95% CIs.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import ROOT, read_jsonl

OUT = f"{ROOT}/out_bench"


def table():
    gold = {p["paper_id"]: (np.mean(p["human_ratings"]), "ccept" in p["decision"])
            for p in read_jsonl(f"{ROOT}/data/bench_iclr25.jsonl")}
    rows = []
    for f in sorted(glob.glob(f"{OUT}/reviews_*.jsonl")):
        for r in read_jsonl(f):
            rv = r.get("review") or {}
            score = pd.to_numeric(rv.get("overall_assessment"), errors="coerce")
            dec = str(rv.get("decision") or "").lower()
            rows.append({"model": r["model"], "paper": r["paper_id"], "score": score,
                         "dec": True if "accept" in dec else False if "reject" in dec else np.nan})
    d = pd.DataFrame(rows).dropna(subset=["score"])
    p = d.groupby(["model", "paper"]).agg(score=("score", "mean"), dec=("dec", "mean")).reset_index()
    p["human"] = p.paper.map(lambda x: gold[x][0])
    p["gold"] = p.paper.map(lambda x: gold[x][1])
    p["pred"] = np.where(p.dec.notna(), p.dec >= 0.5, p.score >= 6)
    return p, len(gold)


def auc(y, s):
    r = rankdata(s)
    pos = y.sum()
    return (r[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def f1(y, p):
    tp = (y & p).sum()
    return 2 * tp / (y.sum() + p.sum()) if y.sum() + p.sum() else np.nan


def metrics(x):
    return {"MAE": (x.score - x.human).abs().mean(),
            "Spearman": spearmanr(x.score, x.human).correlation,
            "AUC": auc(x.gold.values.astype(bool), x.score.values) if x.gold.nunique() == 2 else np.nan,
            "acc": (x.pred == x.gold).mean(),
            "F1": f1(x.gold.values.astype(bool), x.pred.values.astype(bool))}


def main(B=1000):
    p, n_gold = table()
    rng = np.random.default_rng(0)
    lines = [f"{'config':38} {'cov':>7} " + " ".join(f"{k:>18}" for k in ("MAE", "Spearman", "AUC", "acc", "F1"))]
    for model, x in p.groupby("model"):
        est = metrics(x)
        draws = [metrics(x.iloc[rng.integers(0, len(x), len(x))]) for _ in range(B)]
        cells = []
        for k, v in est.items():
            lo, hi = np.nanpercentile([dd[k] for dd in draws], [2.5, 97.5])
            cells.append(f"{v:6.3f} [{lo:.2f},{hi:.2f}]")
        lines.append(f"{model:38} {len(x):3d}/{n_gold} " + " ".join(f"{c:>18}" for c in cells))
    lines.append(f"\nhuman reviewer self-consistency is the reference ceiling (see DeepReviewer paper); "
                 f"mean score per config: " + ", ".join(f"{m}={x.score.mean():.2f}" for m, x in p.groupby("model")))
    txt = "\n".join(lines)
    print(txt)
    open(f"{OUT}/bench_result.txt", "w").write(txt + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
