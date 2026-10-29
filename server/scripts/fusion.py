"""M3, audit-informed score fusion (CPU only; reuses the gate's cached issue checks).

  python fusion.py   -> paper/tab_fusion.md, paper/tab_fusion.csv, paper/fusion_weights.txt

Per paper and split, features are
  score:<reviewer>  mean overall score of a reviewer over its parsed samples (1-10)
  audit signals     from the Qwen3-32B think+audit samples and their fresh-session checks (gate.py, cached):
                    n_issues (mean per audit sample), n_reported, n_fatal, n_verified (reported AND fatal AND
                    quote found in the paper), votes (audit samples with >= 1 verified issue)
A ridge regression to the mean human rating is fit on bench_dev25 only (features standardised with dev
statistics, alpha chosen by leave-one-out on dev), then applied unchanged to bench_iclr25 and bench_iclr24.
Systems:
  M3a <base>+fusion     one reviewer's score + audit signals (plug-in, like the gate)
  ensemble (no audit)   all reviewers' scores, same ridge (control: what plain reviewer averaging buys)
  M3b ensemble+fusion   all reviewers' scores + audit signals
Missing reviewer scores (unparsed papers) are imputed with the dev mean of that feature.
Deltas use a paired paper bootstrap (2000 resamples) against the base reviewer, and for M3b also against the
best single reviewer on that split and against the no-audit ensemble.
"""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import ROOT, read_jsonl
from gate import norm, run_checks

SPLITS = {"dev": ("_dev", "bench_dev25"), "test25": ("_bench", "bench_iclr25"), "test24": ("_bench24", "bench_iclr24")}
REVIEWERS = ["deepreviewer-7b:fast", "qwen3-32b-awq:s10", "qwen3-32b-awq:ais", "qwen3-32b-awq:think+verify+s10",
             "openreviewer-8b"]
AUDIT_F = ["n_issues", "n_reported", "n_fatal", "n_verified", "votes"]
P = f"{ROOT}/paper"


def scores(tag, cfg):
    f = f"{ROOT}/out{tag}/reviews_{cfg.replace(':', '_')}.jsonl"
    if not os.path.exists(f):
        return None
    d = pd.DataFrame([{"paper": r["paper_id"], "s": pd.to_numeric((r.get("review") or {}).get("overall_assessment"),
                                                                    errors="coerce")} for r in read_jsonl(f)])
    return d.dropna().groupby("paper").s.mean()


def audit_features(tag, data):
    text = {r["paper_id"]: r["text"] for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
    per = {}  # paper -> {issue: set(samples)}
    ns = {}
    for r in read_jsonl(f"{ROOT}/out{tag}/reviews_qwen3-32b-awq_think+audit.jsonl"):
        ws = [(w.get("issue") or "").strip() if isinstance(w, dict) else str(w).strip()
              for w in (r.get("review") or {}).get("weaknesses", [])]
        ns.setdefault(r["paper_id"], []).append(sum(bool(w) for w in ws))
        for i in ws:
            if i:
                per.setdefault(r["paper_id"], {}).setdefault(i, set()).add(r["sample"])
    jobs = [((p, "O", None), i) for p, d in per.items() for i in d if p in text]
    tkey = {(p, "O", None): t for p, t in text.items()}
    res = run_checks(jobs, tkey, "qwen3-32b-awq", f"{ROOT}/out{tag}/gate_cache_qwen3-32b-awq.jsonl")
    rows = {p: dict(n_issues=np.mean(ns.get(p, [0])), n_reported=0, n_fatal=0, n_verified=0, votes=0) for p in text}
    vs = {}
    for ((p, _, _), i), j in res.items():
        q = norm(str(j.get("quote") or ""))
        rep, fat = bool(j.get("reported")), bool(j.get("fatal"))
        rows[p]["n_reported"] += rep
        rows[p]["n_fatal"] += fat
        if rep and fat and len(q) > 20 and q in norm(text[p]):
            rows[p]["n_verified"] += 1
            vs.setdefault(p, set()).update(per[p][i])
    for p, v in vs.items():
        rows[p]["votes"] = len(v)
    return pd.DataFrame.from_dict(rows, orient="index")


def frame(split):
    tag, data = SPLITS[split]
    gold = {r["paper_id"]: (np.mean(r["human_ratings"]), "ccept" in r["decision"]) for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
    d = audit_features(tag, data)
    for c in REVIEWERS:
        s = scores(tag, c)
        if s is not None:
            d["score:" + c] = s
    d["human"] = [gold[p][0] for p in d.index]
    d["gold"] = [gold[p][1] for p in d.index]
    return d


def auc(y, s):
    r = rankdata(s)
    pos = y.sum()
    return (r[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def ridge_fit(X, y, alpha):
    A = np.c_[np.ones(len(X)), X]
    reg = alpha * np.eye(A.shape[1])
    reg[0, 0] = 0
    return np.linalg.solve(A.T @ A + reg, A.T @ y)


def fit(dev, feats):
    X = dev[feats].fillna(dev[feats].mean())
    mu, sd = X.mean(), X.std().replace(0, 1)
    Z, y = ((X - mu) / sd).values, dev.human.values
    best = None
    for alpha in [0.1, 0.3, 1, 3, 10, 30, 100]:
        err = 0.0
        for i in range(len(Z)):  # leave-one-out on dev
            m = np.ones(len(Z), bool)
            m[i] = False
            w = ridge_fit(Z[m], y[m], alpha)
            err += (w[0] + Z[i] @ w[1:] - y[i]) ** 2
        if best is None or err < best[0]:
            best = (err, alpha)
    w = ridge_fit(Z, y, best[1])
    return SimpleNamespace(feats=feats, mu=mu, sd=sd, w=w, alpha=best[1])


def predict(m, d):
    X = d[m.feats].fillna(m.mu)
    return m.w[0] + ((X - m.mu) / m.sd).values @ m.w[1:]


def met(y, h, s):
    return auc(y, s), spearmanr(s, h).correlation


def paired(d, s1, s0, B=2000, seed=0):
    """Paired paper bootstrap of (AUC, Spearman) differences s1 - s0."""
    rng = np.random.default_rng(seed)
    y, h = d.gold.values.astype(bool), d.human.values
    out = []
    for _ in range(B):
        i = rng.integers(0, len(d), len(d))
        if y[i].all() or not y[i].any():
            continue
        a1, r1 = met(y[i], h[i], s1[i])
        a0, r0 = met(y[i], h[i], s0[i])
        out.append((a1 - a0, r1 - r0))
    o = np.array(out)
    return np.percentile(o, [2.5, 97.5], axis=0)


def main():
    os.makedirs(P, exist_ok=True)
    F = {s: frame(s) for s in SPLITS}
    avail = [c for c in REVIEWERS if all("score:" + c in F[s] for s in SPLITS)]
    systems = {}
    for c in avail:
        systems[c] = None  # raw reviewer score
        systems[c + "+fusion (M3a)"] = ["score:" + c] + AUDIT_F
    systems["ensemble (no audit)"] = ["score:" + c for c in avail]
    systems["ensemble+fusion (M3b)"] = ["score:" + c for c in avail] + AUDIT_F
    models = {k: fit(F["dev"], v) for k, v in systems.items() if v}
    rows, wlines = [], []
    for k, m in models.items():
        wlines.append(f"{k}: alpha={m.alpha}  " + "  ".join(f"{f}={w:+.3f}" for f, w in zip(m.feats, m.w[1:])))
    for split in ("test25", "test24", "dev"):
        d = F[split]
        pred = {}
        for k, v in systems.items():
            if v is None:
                pred[k] = d["score:" + k].fillna(F["dev"]["score:" + k].mean()).values
            else:
                pred[k] = predict(models[k], d)
        y, h = d.gold.values.astype(bool), d.human.values
        single = {c: met(y, h, pred[c]) for c in avail}
        best = max(single, key=lambda c: single[c][0] + single[c][1])
        for k in systems:
            a, r = met(y, h, pred[k])
            row = {"split": split, "system": k, "n": len(d), "AUC": a, "Spearman": r}
            ref = {c + "+fusion (M3a)": c for c in avail}.get(k)
            refs = [("base", ref)] if ref else []
            if k.startswith("ensemble+"):
                refs = [("best_single", best), ("ensemble", "ensemble (no audit)")]
            for name, r0 in refs:
                lo, hi = paired(d, pred[k], pred[r0])
                a0, s0 = met(y, h, pred[r0])
                row.update({f"dAUC_vs_{name}": a - a0, f"dAUC_vs_{name}_ci": f"[{lo[0]:+.3f},{hi[0]:+.3f}]",
                            f"dSp_vs_{name}": r - s0, f"dSp_vs_{name}_ci": f"[{lo[1]:+.3f},{hi[1]:+.3f}]"})
            if k.startswith("ensemble+"):
                row["best_single"] = best
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{P}/tab_fusion.csv", index=False)
    with open(f"{P}/tab_fusion.md", "w") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else ("" if pd.isna(v) else str(v)) for v in r) + " |\n")
    open(f"{P}/fusion_weights.txt", "w").write("fit on bench_dev25 only\n" + "\n".join(wlines) + "\n")
    print("\n".join(wlines))
    print(t.to_string())


if __name__ == "__main__":
    main()
