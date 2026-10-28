"""Paper tables, figure data and figures from the finished runs (CPU only).

  python paper_tables.py   -> paper/ (tab_*.csv / tab_*.md, fig_*.csv, fig_*.png, chosen_gate.txt)

Gate setting selection (fixed before the dev results were seen; see git history of this file):
  candidates, most aggressive first: k1 cap, k1 minus2, k2 cap, k2 minus2, k3 cap, k3 minus2.
  Pick the first whose mean change over the dev bases (DeepReviewer-fast, Qwen3-32B s10, AI Scientist) is
  >= -0.005 in AUC and >= -0.010 in Spearman. The chosen setting is then reported on the ICLR 2025 test split,
  the ICLR 2024 split and K1 without further tuning. k1 cap is also reported as the "strict" variant.
Quality deltas use a paired paper bootstrap (same resampled papers for base and gated system).
M3 (quality gate) is reported as its own method: the candidate with the highest mean dev dAUC + dSpearman
over the dev bases, then applied unchanged to both test splits and K1 (tab_main_m3_*.md).
K1 main metrics are unconditional: pass rate on flawed (F) and on clean (N) manuscripts, and paired F-N.
"""
import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common import MODELS, ROOT, read_jsonl

P = f"{ROOT}/paper"
CANDS = ["_k1", "_k1_m2", "_k2", "_k2_m2", "_k3", "_k3_m2"]
DEV_BASES = ["deepreviewer-7b:fast", "qwen3-32b-awq:s10", "qwen3-32b-awq:ais"]
PASS10 = 6
NAMES = {"deepreviewer-7b:fast": "DeepReviewer-7B (fast)", "deepreviewer-7b:std": "DeepReviewer-7B (std)",
         "deepreviewer-14b:fast": "DeepReviewer-14B (fast)", "openreviewer-8b": "OpenReviewer-8B",
         "qwen3-32b-awq:ais": "AI Scientist (Qwen3-32B)", "qwen3-32b-awq:s10": "Qwen3-32B",
         "qwen3-32b-awq:think+verify+s10": "Qwen3-32B think+verify",
         "qwen3-32b-awq:think+verify+grule+s10": "Fatal-flaw rule (ours, M1)",
         "qwen3-32b-awq:think+verify+grule": "Fatal-flaw rule (ours, M1)",
         "qwen3-32b-awq:think+two": "Two-stage (audit -> score)", "qwen3-8b:think+two": "Two-stage 8B"}


def split_model(m):
    base, _, g = m.partition("+gate")
    return base, ("+gate" + g) if _ else ""


def auc(y, s):
    r = rankdata(s)
    pos = y.sum()
    return (r[y].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos))


def qmetrics(x):
    return {"AUC": auc(x.gold.values.astype(bool), x.score.values), "Spearman": spearmanr(x.score, x.human).correlation,
            "MAE": (x.score - x.human).abs().mean(), "acc": ((x.score >= PASS10) == x.gold).mean()}


def papers(tag):
    f = f"{ROOT}/out{tag}/bench_papers.csv"
    return pd.read_csv(f) if os.path.exists(f) else None


def fired(tag, model):
    f = f"{ROOT}/out{tag}/reviews_{model.replace(':', '_', 1)}.jsonl"
    if not os.path.exists(f):
        return np.nan, np.nan
    rows = read_jsonl(f)
    fp = {r["paper_id"] for r in rows if (r["review"] or {}).get("gated")}
    allp = {r["paper_id"] for r in rows}
    return len(fp) / max(len(allp), 1), fp


def quality(tag, B=2000):
    """per gated system: base metrics, gated metrics, paired deltas with CIs, fire rate, fired-on-accepted share."""
    p = papers(tag)
    if p is None:
        return pd.DataFrame(columns=["split", "base", "gate", "n"])
    rng = np.random.default_rng(0)
    out = []
    for m in sorted(p.model.unique()):
        base, g = split_model(m)
        x = p[p.model == m].set_index("paper")
        row = {"split": tag, "base": base, "gate": g or "-", "n": len(x)}
        row.update({k: v for k, v in qmetrics(x.reset_index()).items()})
        if g and base in set(p.model):
            b = p[p.model == base].set_index("paper")
            common = x.index.intersection(b.index)
            x, b = x.loc[common], b.loc[common]
            est = {k: qmetrics(x.reset_index())[k] - qmetrics(b.reset_index())[k] for k in ("AUC", "Spearman", "MAE", "acc")}
            draws = []
            for _ in range(B):
                i = rng.integers(0, len(common), len(common))
                xi, bi = x.iloc[i].reset_index(), b.iloc[i].reset_index()
                mx, mb = qmetrics(xi), qmetrics(bi)
                draws.append({k: mx[k] - mb[k] for k in est})
            for k in est:
                lo, hi = np.nanpercentile([d[k] for d in draws], [2.5, 97.5])
                row[f"d{k}"], row[f"d{k}_lo"], row[f"d{k}_hi"] = est[k], lo, hi
            fr, fp = fired(tag, m)
            row["fire_rate"] = fr
            row["fired_accepted"] = float(x.loc[x.index.isin(fp), "gold"].mean()) if fp else np.nan
        out.append(row)
    return pd.DataFrame(out)


def choose(dev):
    lines = []
    for c in CANDS:
        d = dev[(dev.gate == "+gate" + c) & dev.base.isin(DEV_BASES)]
        if len(d) < len(DEV_BASES):
            lines.append(f"{c}: missing on dev ({len(d)} bases)")
            continue
        ok = d.dAUC.mean() >= -0.005 and d.dSpearman.mean() >= -0.010
        lines.append(f"{c}: mean dAUC {d.dAUC.mean():+.4f}  mean dSpearman {d.dSpearman.mean():+.4f}  "
                     f"fire {d.fire_rate.mean():.3f}  -> {'OK' if ok else 'rejected'}")
        if ok:
            return c, lines
    return CANDS[-1], lines + ["no candidate passed; using the least aggressive"]


def choose_quality(dev):
    """M3: the candidate that most improves mean dev AUC + Spearman over the dev bases."""
    best, lines = None, []
    for c in CANDS:
        d = dev[(dev.gate == "+gate" + c) & dev.base.isin(DEV_BASES)]
        if len(d) < len(DEV_BASES):
            continue
        gain = d.dAUC.mean() + d.dSpearman.mean()
        lines.append(f"M3 {c}: dAUC {d.dAUC.mean():+.4f}  dSpearman {d.dSpearman.mean():+.4f}  sum {gain:+.4f}")
        if best is None or gain > best[1]:
            best = (c, gain)
    return (best[0] if best else "_k2_m2"), lines


# ---------- K1 ----------
def k1_frame():
    rows = []
    for r in read_jsonl(f"{ROOT}/out/judged.jsonl"):
        rv, j = r.get("review") or {}, r.get("judge") or {}
        s = pd.to_numeric(rv.get("overall_assessment"), errors="coerce")
        if not rv or pd.isna(s):
            continue
        ten = rv.get("scale") == "iclr10"
        rows.append({"model": r["model"], "paper": r["paper_id"], "version": r["version"], "variant": r.get("variant"),
                     "sample": r["sample"], "score10": s if ten else 1 + (s - 1) * 9 / 4,
                     "passed": s >= PASS10 if ten else s >= 3, "detected": bool(j.get("detected_flaw")),
                     "gated": bool(rv.get("gated"))})
    d = pd.DataFrame(rows)
    d["type"] = d.variant.str[:4]
    return d


def k1_table(d, B=1000):
    rng = np.random.default_rng(0)
    out = []
    for m, x in d.groupby("model"):
        pp = x.paper.unique()
        def stat(y):
            F, N = y[y.version == "F"], y[y.version == "N"]
            mm = y.groupby(["paper", "variant", "version"]).score10.mean().unstack()
            return {"det_F": F.detected.mean(), "det_N": N.detected.mean(), "pass_F": F.passed.mean(),
                    "pass_N": N.passed.mean(), "F_minus_N": (mm["F"] - mm["N"]).mean() if "F" in mm and "N" in mm else np.nan,
                    "fire_F": F.gated.mean(), "fire_N": N.gated.mean()}
        est = stat(x)
        groups = {p: g for p, g in x.groupby("paper")}
        draws = [stat(pd.concat([groups[p] for p in rng.choice(pp, len(pp))])) for _ in range(B)]
        row = {"model": m, "nF": int((x.version == "F").sum()), "nN": int((x.version == "N").sum())}
        for k, v in est.items():
            lo, hi = np.nanpercentile([dd[k] for dd in draws], [2.5, 97.5])
            row.update({k: v, k + "_lo": lo, k + "_hi": hi})
        out.append(row)
    return pd.DataFrame(out)


def k1_by_variant(d):
    g = d[d.version == "F"].groupby(["model", "variant"]).agg(det=("detected", "mean"), passed=("passed", "mean")).reset_index()
    n = d[d.version == "N"].groupby(["model", "variant"]).agg(pass_N=("passed", "mean")).reset_index()
    return g.merge(n, on=["model", "variant"], how="left")


# ---------- cost ----------
LOG = re.compile(r"\[(\d\d:\d\d:\d\d)\] (?:GPU\d+: |exit \d+ : )(.*)")


def gpu_minutes():
    """wall-clock minutes per command from logs/k0.log (start line -> matching exit line; GPU shared, so an upper bound)."""
    start, dur = {}, {}
    for line in open(f"{ROOT}/logs/k0.log", errors="ignore"):
        m = LOG.match(line.strip())
        if not m:
            continue
        t, cmd = datetime.strptime(m.group(1), "%H:%M:%S"), m.group(2)
        if "exit" in line.split("]")[1][:6]:
            if cmd in start:
                dt = (t - start.pop(cmd)).total_seconds() / 60
                dur[cmd] = dt + (1440 if dt < 0 else 0)
        else:
            start[cmd] = t
    return dur


def cost():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODELS["qwen3-32b-awq"])
    n = lambda s: len(tok(s).input_ids)
    texts = {r["paper_id"]: r["text"] for r in read_jsonl(f"{ROOT}/data/bench_iclr25.jsonl")}
    rows = []
    paper_tok = np.mean([n(t) for t in list(texts.values())[:50]])
    for f in sorted(glob.glob(f"{ROOT}/out_bench/reviews_*.jsonl")):
        if "+gate" in f:
            continue
        rs = read_jsonl(f)
        outs = [n(r.get("raw") or "") for r in rs[:150]]
        rows.append({"stage": rs[0]["model"], "calls_per_paper": len(rs) / len(texts),
                     "in_tok_per_paper": paper_tok * len(rs) / len(texts), "out_tok_per_paper": np.mean(outs) * len(rs) / len(texts)})
    for c in glob.glob(f"{ROOT}/out_bench/gate_cache_*.jsonl"):
        cs = read_jsonl(c)
        cs = [x for x in cs if "in_tok" in x]
        if cs:
            rows.append({"stage": "gate check (" + os.path.basename(c)[11:-6] + ")", "calls_per_paper": len(cs) / len(texts),
                         "in_tok_per_paper": sum(x["in_tok"] for x in cs) / len(texts),
                         "out_tok_per_paper": sum(x["out_tok"] for x in cs) / len(texts)})
    t = pd.DataFrame(rows)
    mins = {k: v for k, v in gpu_minutes().items() if "--tag _bench" in k and "bench_iclr25" in k}
    t["gpu_min_per_paper"] = [sum(v for k, v in mins.items() if f"--model {s} " in k) / len(texts) or np.nan for s in t.stage]
    return t


# ---------- figures ----------
def figures(k1t, k1v, dev, test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = [m for m in k1t.model if split_model(m)[1] in ("", "+gate_k1", "+gate" + CHOSEN)
            and split_model(m)[0] in ("deepreviewer-7b:fast", "deepreviewer-14b:fast", "openreviewer-8b", "qwen3-32b-awq:ais", "qwen3-32b-awq:think+verify+s10",
                                      "qwen3-32b-awq:think+verify+grule", "qwen3-32b-awq:think+two")]
    t = k1t.set_index("model").loc[show].reset_index()
    t.to_csv(f"{P}/fig_k1_pass.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    xs = np.arange(len(t))
    for off, col, lab in [(-0.2, "pass_N", "clean (N)"), (0.2, "pass_F", "flawed (F)")]:
        ax.bar(xs + off, t[col], 0.4, yerr=[t[col] - t[col + "_lo"], t[col + "_hi"] - t[col]], label=lab, capsize=2)
    ax.set_xticks(xs, [NAMES.get(split_model(m)[0], m) + ("\n" + split_model(m)[1] if split_model(m)[1] else "") for m in t.model],
                  rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("P(score >= accept line)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{P}/fig_k1_pass.png", dpi=200)
    plt.close(fig)

    hv = k1v[k1v.model.isin(show)].pivot(index="model", columns="variant", values="det")
    hv.to_csv(f"{P}/fig_k1_detect_heatmap.csv")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(hv) + 1.2))
    im = ax.imshow(hv.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(hv.shape[1]), hv.columns, fontsize=7)
    ax.set_yticks(range(hv.shape[0]), hv.index, fontsize=7)
    for i in range(hv.shape[0]):
        for j in range(hv.shape[1]):
            v = hv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="w" if v < 0.6 else "k")
    fig.colorbar(im, ax=ax, label="detection rate")
    fig.tight_layout()
    fig.savefig(f"{P}/fig_k1_detect_heatmap.png", dpi=200)
    plt.close(fig)

    # quality-sensitivity trade-off: dev quality change vs K1 flawed pass rate, one point per gate setting x base
    pts = []
    for c in CANDS:
        for b in ("deepreviewer-7b:fast", "qwen3-32b-awq:ais"):
            q = dev[(dev.base == b) & (dev.gate == "+gate" + c)]
            k = k1t[k1t.model == f"{b}+gate{c}"]
            if len(q) and len(k):
                pts.append({"base": b, "setting": c, "dev_dSpearman": q.dSpearman.iloc[0], "dev_dAUC": q.dAUC.iloc[0],
                            "k1_pass_F": k.pass_F.iloc[0], "k1_pass_N": k.pass_N.iloc[0]})
    pts = pd.DataFrame(pts)
    pts.to_csv(f"{P}/fig_tradeoff.csv", index=False)
    if len(pts):
        fig, ax = plt.subplots(figsize=(5, 3.6))
        for b, g in pts.groupby("base"):
            ax.plot(g.k1_pass_F, g.dev_dSpearman, "o-", label=NAMES.get(b, b))
            for _, r in g.iterrows():
                ax.annotate(r.setting.strip("_"), (r.k1_pass_F, r.dev_dSpearman), fontsize=6)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel("K1: P(pass | flawed)  (lower is better)")
        ax.set_ylabel("dev: Spearman change vs base")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{P}/fig_tradeoff.png", dpi=200)
        plt.close(fig)

    # score distributions on K1, base vs chosen gate
    d = K1D[K1D.model.isin(["deepreviewer-7b:fast", f"deepreviewer-7b:fast+gate{CHOSEN}",
                            "qwen3-32b-awq:ais", f"qwen3-32b-awq:ais+gate{CHOSEN}"])]
    d[["model", "version", "variant", "score10"]].to_csv(f"{P}/fig_k1_scores.csv", index=False)
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.6), sharey=True)
    for ax, (m, g) in zip(axs, d.groupby("model")):
        for v in "NF":
            ax.hist(g[g.version == v].score10, bins=np.arange(0.5, 11), alpha=0.6, density=True, label=v)
        ax.axvline(PASS10 - 0.5, color="k", lw=0.6)
        ax.set_title(m, fontsize=7)
    axs[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{P}/fig_k1_scores.png", dpi=200)
    plt.close(fig)


def md(df, path, cols=None):
    df = df[cols] if cols else df
    with open(path, "w") as f:
        f.write("| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n")


def main():
    global CHOSEN, K1D
    os.makedirs(P, exist_ok=True)
    dev = quality("_dev")
    CHOSEN, lines = choose(dev) if len(dev) else ("_k1", ["no dev results"])
    open(f"{P}/chosen_gate.txt", "w").write("\n".join(lines) + f"\nCHOSEN {CHOSEN}\n")
    M3, m3_lines = choose_quality(dev) if len(dev) else ("_k2_m2", ["no dev results"])
    open(f"{P}/chosen_gate.txt", "a").write("\n".join(m3_lines) + f"\nM3 {M3}\n")
    print("\n".join(lines + m3_lines), "\nCHOSEN", CHOSEN, "M3", M3)
    dev.to_csv(f"{P}/tab_dev_sweep.csv", index=False)
    test = pd.concat([quality("_bench"), quality("_bench24")])
    test.to_csv(f"{P}/tab_quality_all.csv", index=False)
    cols = ["split", "base", "gate", "n", "AUC", "Spearman", "MAE", "acc", "dAUC", "dAUC_lo", "dAUC_hi",
            "dSpearman", "dSpearman_lo", "dSpearman_hi", "fire_rate", "fired_accepted"]
    main_q = test[test.gate.isin(["-", "+gate_k1", "+gate" + CHOSEN])].reindex(columns=cols)
    md(main_q, f"{P}/tab_main_quality.md")
    md(test[test.gate.isin(["-", "+gate" + M3])].reindex(columns=cols), f"{P}/tab_main_m3_quality.md")
    abl_q = test[(test.split == "_bench") & test.base.isin(["deepreviewer-7b:fast", "qwen3-32b-awq:s10"])].reindex(columns=cols)
    md(abl_q, f"{P}/tab_ablation_quality.md")
    K1D = k1_frame()
    k1t = k1_table(K1D)
    k1t.to_csv(f"{P}/tab_k1_all.csv", index=False)
    kc = ["model", "nF", "det_F", "det_N", "pass_F", "pass_F_lo", "pass_F_hi", "pass_N", "F_minus_N", "F_minus_N_lo",
          "F_minus_N_hi", "fire_F", "fire_N"]
    md(k1t[k1t.model.map(lambda m: split_model(m)[1] in ("", "+gate_k1", "+gate" + CHOSEN))], f"{P}/tab_main_k1.md", kc)
    md(k1t[k1t.model.map(lambda m: split_model(m)[1] in ("", "+gate" + M3))], f"{P}/tab_main_m3_k1.md", kc)
    md(k1t[k1t.model.str.contains(r"deepreviewer-7b:fast\+gate|qwen3-32b-awq:ais\+gate")], f"{P}/tab_ablation_k1.md", kc)
    k1v = k1_by_variant(K1D)
    k1v.to_csv(f"{P}/tab_k1_by_variant.csv", index=False)
    try:
        c = cost()
        c.to_csv(f"{P}/tab_cost.csv", index=False)
        md(c, f"{P}/tab_cost.md")
    except Exception as e:  # tokenizer path missing on a new machine
        print("cost skipped:", e)
    figures(k1t, k1v, dev, test)
    print("written to", P)


if __name__ == "__main__":
    main()
