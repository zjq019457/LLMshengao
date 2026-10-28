"""K0 step 4: pre-registered H1 / H3 gates with paper-cluster bootstrap CIs.

  python analyze.py [--threshold 3]

H1 signal: among F reviews that detected the flaw, >= 10% still have overall >= threshold,
           AND mean paired overall(F) - overall(N) > -1.0.
H3 signal (revised for K0b): R flagged as missing >= N + 0.20 on the same papers, AND then either
           soundness confidence R - N > -0.25 with no cannot_assess increase, or unsupported-assertion rate >= 20%.
D gate: detection on F >= 20% and above the N false-positive rate.
Pass if H1 or H3 holds for at least one configuration.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from common import ROOT, read_jsonl

ICLR_THRESHOLD = 6
# strict cross-check of the judge: the review restates the injected selection-on-test content
LEAK_RE = re.compile(r"(tun\w* on the test|select\w* (based )?on (the )?(performance on the )?test|test set (was|were|is) used"
                     r"|test.{0,40}(checkpoint|hyperparam)|(checkpoint|hyperparam).{0,60}test set)", re.I)


def flatten(rows):
    out = []
    for r in rows:
        rv, j = r.get("review") or {}, r.get("judge") or {}
        snd = rv.get("soundness") or {}
        out.append({"paper": r["paper_id"], "version": r["version"], "model": r["model"],
                    "variant": r.get("variant"), "judge_fatal": str(j.get("flaw_severity", "")).lower() == "fatal",
                    "sample": r["sample"], "parsed": bool(rv),
                    "overall": pd.to_numeric(rv.get("overall_assessment"), errors="coerce"),
                    "scale": rv.get("scale", "arr5"),
                    "snd_score": pd.to_numeric(snd.get("score"), errors="coerce"),
                    "snd_conf": pd.to_numeric(snd.get("confidence"), errors="coerce"),
                    "snd_na": bool(snd.get("cannot_assess")),
                    "detected": bool(j.get("detected_flaw")),
                    "missing": bool(j.get("flagged_missing")),
                    "asserts": bool(j.get("asserts_protocol")),
                    "regex_hit": bool(LEAK_RE.search(json.dumps(rv))),
                    "fatal": any(str(w.get("severity", "")).lower() == "fatal" and LEAK_RE.search(json.dumps(w))
                                 for w in rv.get("weaknesses", []) if isinstance(w, dict))})
    return pd.DataFrame(out)


def boot(df, stat, B=2000, seed=0):
    """Point estimate + 95% CI, resampling papers."""
    papers = df["paper"].unique()
    rng = np.random.default_rng(seed)
    est = stat(df)
    draws = []
    for _ in range(B):
        pick = rng.choice(papers, len(papers), replace=True)
        draws.append(stat(pd.concat([df[df.paper == p] for p in pick])))
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return est, lo, hi


def paired_delta(col, a, b):
    def f(d):
        m = d.groupby(["paper", "version"])[col].mean().unstack()
        return float((m[a] - m[b]).mean()) if a in m and b in m else np.nan
    return f


def k1(df, thr):
    """K1: per model x flaw type (variants pooled within type, and per variant). N rows carry the variant
    they were judged against, so false positives and F-N are paired within paper and variant."""
    df = df[df.parsed].copy()
    df["type"] = df.variant.str[:4]
    for model, d in df.groupby("model"):
        thr_m = thr
        if (d.scale == "iclr10").all():
            d = d.assign(overall=1 + (d.overall - 1) * 4 / 9)
            thr_m = 1 + (ICLR_THRESHOLD - 1) * 4 / 9
        print(f"=== {model} ===   (pass line {thr_m:.2f})")
        print(f"{'cell':8} {'nF':>4} {'det':>5} {'fpN':>5} {'fatal|det':>9} {'pass|det':>18} {'F-N':>20}")
        for key, col in [(t, "type") for t in sorted(d.type.unique())] + [(v, "variant") for v in sorted(d.variant.unique())]:
            c = d[d[col] == key]
            F, N = c[c.version == "F"], c[c.version == "N"]
            det = F[F.detected]
            pa = boot(F, lambda x: x[x.detected].overall.ge(thr_m).mean() if x.detected.any() else np.nan, B=500)
            fn = boot(c, paired_delta("overall", "F", "N"), B=500)
            print(f"{key:8} {len(F):4d} {F.detected.mean():5.2f} {N.detected.mean():5.2f} {det.judge_fatal.mean() if len(det) else np.nan:9.2f}"
                  f"   {pa[0]:.2f} [{pa[1]:.2f},{pa[2]:.2f}]   {fn[0]:+.2f} [{fn[1]:+.2f},{fn[2]:+.2f}]")
        print()


def main(thr):
    df = flatten(read_jsonl(f"{ROOT}/out/judged.jsonl"))
    if df.variant.notna().any():
        return k1(df, thr)
    print(f"parsed reviews: {df.parsed.sum()}/{len(df)}\n")
    verdict = {}
    for model, d in df[df.parsed].groupby("model"):
        # DeepReviewer rates on ICLR 1-10 (6 = marginally above); rescaled to the 1-5 range
        iclr = (d.scale == "iclr10").all()
        thr_m = thr
        if iclr:
            d = d.assign(overall=1 + (d.overall - 1) * 4 / 9)
            thr_m = 1 + (ICLR_THRESHOLD - 1) * 4 / 9
        F, R, N = d[d.version == "F"], d[d.version == "R"], d[d.version == "N"]
        det = F[F.detected]
        print(f"=== {model} ===")
        print(f"D  detection F: {F.detected.mean():.2f}   false-positive N: {N.detected.mean():.2f}   (n_detected={len(det)})"
              f"   regex F/N: {F.regex_hit.mean():.2f}/{N.regex_hit.mean():.2f}   judge-regex agreement on F: {(F.detected == F.regex_hit).mean():.2f}")
        if len(det):
            print(f"W  among detected: labelled fatal {det.fatal.mean():.2f}   mean overall {det.overall.mean():.2f}"
                  f" vs N same papers {N[N.paper.isin(det.paper)].overall.mean():.2f}")
        h1a = boot(F, lambda x: x[x.detected].overall.ge(thr_m).mean() if x.detected.any() else np.nan)
        h1b = boot(d, paired_delta("overall", "F", "N"))
        print(f"H1 P(overall>={thr_m:.2f} | detected): {h1a[0]:.2f} [{h1a[1]:.2f},{h1a[2]:.2f}]")
        print(f"H1 overall F-N: {h1b[0]:+.2f} [{h1b[1]:+.2f},{h1b[2]:+.2f}]")
        if R.empty:
            h1 = (h1a[0] >= 0.10 if len(det) >= 5 else None, h1b[0] > -1.0)
            d_ok = F.detected.mean() >= 0.20 and F.detected.mean() > N.detected.mean()
            verdict[model] = {"D": bool(d_ok), "H1": None if h1[0] is None else bool(h1[0] and h1[1]), "H3": None}
            print("   (no R versions: H3 skipped)\n")
            continue
        h3a = boot(d, paired_delta("snd_conf", "R", "N"))
        h3b = boot(R, lambda x: x.asserts.mean())
        print(f"H3 soundness-confidence R-N: {h3a[0]:+.2f} [{h3a[1]:+.2f},{h3a[2]:+.2f}]")
        print(f"H3 unsupported-assertion rate R: {h3b[0]:.2f} [{h3b[1]:.2f},{h3b[2]:.2f}]   "
              f"(N baseline {N.asserts.mean():.2f})")
        # N baseline restricted to the same (R-valid) papers: generic "details are missing" remarks are common
        N_r = N[N.paper.isin(R.paper.unique())]
        know = R.missing.mean() - N_r.missing.mean()
        print(f"   flagged missing R: {R.missing.mean():.2f} vs N(same papers): {N_r.missing.mean():.2f}  (delta {know:+.2f})"
              f"   cannot_assess(soundness) R: {R.snd_na.mean():.2f} vs N: {N_r.snd_na.mean():.2f}")
        print(f"   sample SD of overall (mean over manuscripts): "
              + ", ".join(f"{v}={d[d.version == v].groupby('paper').overall.std().mean():.2f}" for v in "NFR"))
        # K0b gates (2026-10-25 revision; the K0 H3 gate passed trivially for insensitive models):
        #   D  : detection on F >= 0.20 and above the N false-positive rate (prerequisite for H1)
        #   H1 : as before, needs >= 5 detected F reviews
        #   H3 : the reviewer must first *know* (R flags missing >= N + 0.20) and then fail to *doubt*
        #        (confidence R-N > -0.25, no cannot_assess increase) or assert unsupported content (>= 0.20)
        d_ok = F.detected.mean() >= 0.20 and F.detected.mean() > N.detected.mean()
        h1 = (h1a[0] >= 0.10 if len(det) >= 5 else None, h1b[0] > -1.0)
        doubt_absent = h3a[0] > -0.25 and R.snd_na.mean() - N_r.snd_na.mean() < 0.20
        h3 = know >= 0.20 and (doubt_absent or h3b[0] >= 0.20)
        verdict[model] = {"D": bool(d_ok), "H1": None if h1[0] is None else bool(h1[0] and h1[1]), "H3": bool(h3)}
        if h1[0] is None:
            print("   H1 not evaluable: fewer than 5 detected F reviews")
        if know < 0.20:
            print("   H3 not supported: R is not flagged as missing more often than N (no 'knowing' to dissociate from)")
        print()
    print("VERDICT", verdict)
    passed = any(v["H1"] or v["H3"] for v in verdict.values())
    print("K0", "PASS -> go to G1" if passed else "FAIL -> close route (or one closed-model recheck)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=3)
    main(ap.parse_args().threshold)
