"""Cost vs effect: more sampling (OpenReviewer n = 1 .. 32, no gate) against n16 + gate_k2, on quality (pooled test)
and on fatal-flaw catching (K1: share of F versions that pass, i.e. score >= the pass line).

  python cost_curve.py   -> paper/tab_cost_curve.md / .csv
Cost is in generated tokens per paper (measured from the raw outputs: OpenReviewer samples, the audit samples and the
fresh-session checks), the dominant cost for these long-input/short-output calls when the paper prefix is cached.
"""
import os

import numpy as np
import pandas as pd

from common import ROOT, read_jsonl
from fusion import P, met

PASS = 6  # ICLR accept side
SPLITS = {"test25": ("_bench", "bench_iclr25"), "test24": ("_bench24", "bench_iclr24")}


def load(tag, cfg):
    f = f"{ROOT}/out{tag}/reviews_{cfg.replace(':', '_')}.jsonl"
    if not os.path.exists(f):
        return None
    rows = list(read_jsonl(f))
    d = pd.DataFrame([{"paper": r["paper_id"], "version": r.get("version"), "k": r["sample"], "chars": len(r.get("raw") or ""),
                       "s": pd.to_numeric((r.get("review") or {}).get("overall_assessment"), errors="coerce")} for r in rows])
    return d


def main():
    rows = []
    gold = {}
    for split, (tag, data) in SPLITS.items():
        for r in read_jsonl(f"{ROOT}/data/{data}.jsonl"):
            gold[f"{split}/{r['paper_id']}"] = (np.mean(r["human_ratings"]), "ccept" in r["decision"])
    tok = lambda d: d.chars.sum() / 4 / d.paper.nunique()  # ~4 characters per token
    big = {s: load(t, "openreviewer-8b:n32") for s, (t, _) in SPLITS.items()}
    n16 = {s: load(t, "openreviewer-8b:n16") for s, (t, _) in SPLITS.items()}
    g16 = {s: load(t, "openreviewer-8b:n16+gate_k2") for s, (t, _) in SPLITS.items()}
    aud = {s: load(t, "qwen3-32b-awq:think+audit") for s, (t, _) in SPLITS.items()}
    chk = sum(len(c.get("raw") or "") for s, (t, _) in SPLITS.items()
              for c in read_jsonl(f"{ROOT}/out{t}/gate_cache_qwen3-32b-awq.jsonl")) / 4 / 400
    per_sample = np.mean([tok(n16[s]) / 16 for s in SPLITS])
    gate_cost = np.mean([tok(aud[s]) for s in SPLITS]) + chk

    def quality(frames, k):
        s = pd.concat([f[f.k < k].groupby("paper").s.mean().rename(index=lambda p, sp=sp: f"{sp}/{p}") for sp, f in frames.items()])
        s = s[s.index.isin(gold)]
        y = np.array([gold[p][1] for p in s.index]); h = np.array([gold[p][0] for p in s.index])
        return met(y, h, s.values)

    src = big if all(v is not None for v in big.values()) else n16
    for k in [1, 3, 8, 16, 32]:
        if k > (32 if src is big else 16):
            continue
        a, r = quality(src, k)
        rows.append({"system": f"OpenReviewer n{k}", "gen_tokens_per_paper": per_sample * k, "AUC": a, "Spearman": r})
    a, r = quality(g16, 16)
    rows.append({"system": "OpenReviewer n16 + gate_k2 (M3)", "gen_tokens_per_paper": per_sample * 16 + gate_cost, "AUC": a, "Spearman": r})
    # fatal-flaw catching on K1: share of injected-flaw (F) versions still passing, by k, without and with the gate
    k1 = load("", "openreviewer-8b:n16")
    k1g = load("", "openreviewer-8b:n16+gate_k2")
    if k1 is not None and k1g is not None:
        for name, f in (("no gate", k1), ("gate_k2", k1g)):
            for k in [1, 3, 8, 16]:
                m = f[(f.k < k) & (f.version == "F")].groupby("paper").s.mean()
                for r_ in rows:
                    if (name == "no gate" and r_["system"] == f"OpenReviewer n{k}") or \
                            (name == "gate_k2" and k == 16 and r_["system"].endswith("(M3)")):
                        r_["K1_flawed_pass_rate"] = (m >= PASS).mean()
    t = pd.DataFrame(rows)
    t.to_csv(f"{P}/tab_cost_curve.csv", index=False)
    with open(f"{P}/tab_cost_curve.md", "w") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
    print(t.to_string())


if __name__ == "__main__":
    main()
