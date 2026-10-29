"""Soft gate: score = reviewer score - w * audit signal, w and signal chosen on bench_dev25 only (CPU, cached checks).

  python soft_gate.py   -> paper/tab_soft_gate.md / .csv, paper/soft_gate_choice.txt
Signals (fusion.audit_features): n_verified (verified fatal issues), votes (audit samples with >= 1 verified issue).
Selection: maximise dev AUC + Spearman of openreviewer-8b:n16 - w * signal over the grid; then frozen for test.
Every system is compared on dev, test25, test24 and pooled test (paired bootstrap) against OpenReviewer-8B (default 3
samples) and against OpenReviewer-8B n16 (to isolate the gate's own contribution).
"""
import numpy as np
import pandas as pd

from common import ROOT, read_jsonl
from fusion import P, SPLITS, audit_features, met, paired, scores

BASE = "openreviewer-8b:n16"
EXTERNAL = {"openreviewer-8b": "OpenReviewer-8B", "deepreviewer-7b:fast": "DeepReviewer-7B (fast)",
            "qwen3-32b-awq:ais": "AI Scientist reviewer (Qwen3-32B)", "qwen3-32b-awq:s10": "Qwen3-32B zero-shot"}
GRID = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]


def frame(split):
    tag, data = SPLITS[split]
    d = audit_features(tag, data)
    gold = {r["paper_id"]: (np.mean(r["human_ratings"]), "ccept" in r["decision"])
            for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
    d["human"] = [gold[p][0] for p in d.index]
    d["gold"] = [gold[p][1] for p in d.index]
    for c in [BASE, *EXTERNAL]:
        s = scores(tag, c)
        if s is not None:
            d[c] = s
    return d


def main():
    F = {s: frame(s) for s in SPLITS}
    dev = F["dev"].dropna(subset=[BASE])
    y, h = dev.gold.values.astype(bool), dev.human.values
    cands = []
    for sig in ("n_verified", "votes"):
        for w in GRID:
            a, r = met(y, h, (dev[BASE] - w * dev[sig]).values)
            cands.append((a + r, sig, w, a, r))
    _, SIG, W, _, _ = max(cands)
    lines = [f"chosen on dev: signal={SIG} w={W}"] + [f"{sig} w={w}: AUC={a:.3f} Sp={r:.3f}" for _, sig, w, a, r in cands]
    open(f"{P}/soft_gate_choice.txt", "w").write("\n".join(lines) + "\n")
    print(lines[0])
    METHOD = f"OpenReviewer-8B n16 + soft gate (ours)"
    for d in F.values():
        d[METHOD] = d[BASE] - W * d[SIG]
    F["pooled_test"] = pd.concat([F["test25"].rename(index=lambda p: "25/" + p),
                                  F["test24"].rename(index=lambda p: "24/" + p)])
    rows = []
    systems = list(EXTERNAL) + [BASE, METHOD]
    for split, d in F.items():
        for s in systems:
            if s not in d:
                continue
            x = d.dropna(subset=[s, "openreviewer-8b", BASE])
            yy, hh = x.gold.values.astype(bool), x.human.values
            a, r = met(yy, hh, x[s].values)
            row = {"split": split, "system": EXTERNAL.get(s, "OpenReviewer-8B n16" if s == BASE else s),
                   "n": len(x), "AUC": a, "Spearman": r}
            for name, ref in (("vs_OR", "openreviewer-8b"), ("vs_ORn16", BASE)):
                if s in (ref,) or (name == "vs_ORn16" and s != METHOD):
                    continue
                lo, hi = paired(x, x[s].values, x[ref].values)
                a0, r0 = met(yy, hh, x[ref].values)
                row.update({f"dAUC_{name}": a - a0, f"dAUC_{name}_ci": f"[{lo[0]:+.3f},{hi[0]:+.3f}]",
                            f"dSp_{name}": r - r0, f"dSp_{name}_ci": f"[{lo[1]:+.3f},{hi[1]:+.3f}]"})
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(f"{P}/tab_soft_gate.csv", index=False)
    with open(f"{P}/tab_soft_gate.md", "w") as f:
        f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
        for _, r in t.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else ("" if pd.isna(v) else str(v)) for v in r) + " |\n")
    pd.set_option("display.width", 250)
    print(t.to_string())


if __name__ == "__main__":
    main()
