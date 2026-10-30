"""Do human reviewers weigh what they write? Per-review "knowing vs weighing" on real ICLR papers, humans vs LLMs.

  python human_gap.py   -> paper/tab_human_gap.md / .csv
For every review of the dev / test25 / test24 papers -- each human review (DeepReview-13K reviewer_comments) and
sample 0 of each LLM reviewer -- the same Qwen3-32B judge reads ONLY that review's weaknesses and decides whether
it claims a problem that would invalidate the main results (prompt FATAL in real_flaw.py). Reported per source:
  claims_fatal          share of reviews that claim such a problem
  accept_side_if_fatal  share of those reviews whose own rating is on the accept side (ICLR >= 6)
  accept_side_if_not    the same for reviews that claim no such problem
A reviewer that weighs what it writes has accept_side_if_fatal near 0. CIs: paper-cluster bootstrap.
"""
import ast
import json
import os

import numpy as np
import pandas as pd

from common import LLM, ROOT, extract_json, read_jsonl
from fusion import P
from real_flaw import FATAL

SPLITS = {"dev": ("_dev", "bench_dev25", "test_2025"), "test25": ("_bench", "bench_iclr25", "test_2025"),
          "test24": ("_bench24", "bench_iclr24", "test_2024")}
LLMS = {"openreviewer-8b": "OpenReviewer-8B", "deepreviewer-7b:fast": "DeepReviewer-7B", "qwen3-32b-awq:ais": "AI Scientist",
        "qwen3-32b-awq:s10": "Qwen3-32B", "qwen3-32b-awq:think+verify+s10": "Qwen3-32B think+verify",
        "cyclereviewer-8b": "CycleReviewer-8B"}


def weak_text(ws):
    if isinstance(ws, list):
        return "\n".join((w.get("issue") or "") if isinstance(w, dict) else str(w) for w in ws)
    return str(ws or "")


def boot(d, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    papers = d.paper.unique()
    g = {p: x for p, x in d.groupby("paper")}
    v = []
    for _ in range(B):
        x = pd.concat([g[p] for p in rng.choice(papers, len(papers))])
        f = x[x.fatal]
        v.append(f.accept.mean() if len(f) else np.nan)
    return np.nanpercentile(v, [2.5, 97.5])


def main():
    rows = []
    for split, (tag, data, csv) in SPLITS.items():
        ids = {r["paper_id"] for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
        d = pd.read_csv(f"{ROOT}/data/{csv}.csv").drop_duplicates("id")
        for r in d[d.id.isin(ids)].itertuples():
            try:
                rc = json.loads(r.reviewer_comments)
            except Exception:
                rc = ast.literal_eval(r.reviewer_comments)
            for e in rc:
                c = e.get("content", {})
                rating = pd.to_numeric(str(c.get("rating", e.get("rating", ""))).split(":")[0], errors="coerce")
                rows.append({"split": split, "paper": r.id, "source": "Human", "rating": rating,
                             "weak": str(c.get("weaknesses", ""))[:6000]})
        for cfg, name in LLMS.items():
            f = f"{ROOT}/out{tag}/reviews_{cfg.replace(':', '_')}.jsonl"
            if not os.path.exists(f):
                continue
            for x in read_jsonl(f):
                rv = x.get("review") or {}
                if x["sample"] == 0 and x["paper_id"] in ids and rv:
                    rows.append({"split": split, "paper": x["paper_id"], "source": name,
                                 "rating": pd.to_numeric(rv.get("overall_assessment"), errors="coerce"),
                                 "weak": weak_text(rv.get("weaknesses"))[:6000]})
    df = pd.DataFrame(rows).dropna(subset=["rating"])
    df = df[df.weak.str.strip() != ""]
    llm = LLM("qwen3-32b-awq", max_len=12288)
    outs = llm.chat([FATAL.format(weak=w) for w in df.weak], temperature=0.0, max_tokens=400)
    df["fatal"] = [bool((extract_json(o[0]) or {}).get("fatal")) for o in outs]
    df["accept"] = df.rating >= 6
    res = []
    for scope in ("all", "test"):
        x0 = df if scope == "all" else df[df.split.isin(["test25", "test24"])]
        for src, x in x0.groupby("source"):
            lo, hi = boot(x)
            res.append({"scope": scope, "source": src, "n_reviews": len(x), "claims_fatal": x.fatal.mean(),
                        "accept_side_if_fatal": x[x.fatal].accept.mean(), "ci": f"[{lo:.2f},{hi:.2f}]",
                        "accept_side_if_not": x[~x.fatal].accept.mean()})
    t = pd.DataFrame(res)
    t.to_csv(f"{P}/tab_human_gap.csv", index=False)
    with open(f"{P}/tab_human_gap.md", "w") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
    pd.set_option("display.width", 250)
    print(t.to_string())


if __name__ == "__main__":
    main()
