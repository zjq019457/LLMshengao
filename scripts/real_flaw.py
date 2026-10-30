"""Real-flaw validation and human alignment of the gate on un-injected ICLR papers (dev, test25, test24).

  python real_flaw.py   -> paper/tab_real_flaw.md / .csv, paper/tab_human_align.md / .csv
Human reviews come from DeepReview-13K (data/test_2025.csv, data/test_2024.csv: reviewer_comments).

1 Real flaws. A paper has a human-identified fatal flaw when Qwen3-32B, reading only the human reviewers' weaknesses,
  says at least one reviewer claims a problem that invalidates the main results (judge prompt FATAL below); a second,
  model-free definition is "some reviewer gave soundness 1 (poor)". The gate's firing (k1: >= 1 audit sample with a
  verified issue, k2: >= 2) is compared between papers with and without such a flaw (rates, risk ratio, paired
  bootstrap CI), on all papers and on rejected papers only.
2 Human alignment. For every issue the audit raised, the judge decides whether some human reviewer raised the same
  concern (ALIGN). Corroboration rates are compared between issues the gate verified and issues it rejected.
"""
import ast
import json
import re

import numpy as np
import pandas as pd

from common import LLM, ROOT, extract_json, read_jsonl
from fusion import P
from gate import norm, run_checks

SPLITS = {"dev": ("_dev", "bench_dev25", "test_2025"), "test25": ("_bench", "bench_iclr25", "test_2025"),
          "test24": ("_bench24", "bench_iclr24", "test_2024")}

FATAL = """Below are the weaknesses that the human reviewers of a paper wrote. Does at least one reviewer claim a problem
that, if true, would invalidate the paper's main results or claims (for example an incorrect proof, test-set leakage, an
unfair or broken comparison, a fundamental methodological error)? Problems of novelty, clarity, missing experiments or
limited scope do NOT count.

{weak}

Return ONLY JSON: {{"fatal": true/false, "quote": "<the reviewer's sentence, or empty>"}}"""

ALIGN = """A model raised this concern about a paper:
CONCERN: {issue}

Below are the weaknesses the paper's human reviewers wrote. Did any human reviewer raise essentially the same concern
(same underlying problem, not just the same general topic)?

{weak}

Return ONLY JSON: {{"same": true/false, "quote": "<the matching reviewer sentence, or empty>"}}"""


def human(csv):
    d = pd.read_csv(f"{ROOT}/data/{csv}.csv").drop_duplicates("id")
    out = {}
    for r in d.itertuples():
        try:
            rc = json.loads(r.reviewer_comments)
        except Exception:
            rc = ast.literal_eval(r.reviewer_comments)
        ws, snd = [], []
        for i, e in enumerate(rc):
            c = e.get("content", {})
            ws.append(f"Reviewer {i + 1}:\n{str(c.get('weaknesses', ''))[:4000]}")
            m = re.match(r"\s*(\d+)", str(c.get("soundness", "")))
            if m:
                snd.append(int(m.group(1)))
        out[r.id] = ("\n\n".join(ws), min(snd) if snd else np.nan)
    return out


def audit(tag, data):
    papers = {r["paper_id"]: r for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
    per = {}
    for r in read_jsonl(f"{ROOT}/out{tag}/reviews_qwen3-32b-awq_think+audit.jsonl"):
        for w in (r.get("review") or {}).get("weaknesses", []):
            i = (w.get("issue") or "").strip() if isinstance(w, dict) else str(w).strip()
            if i:
                per.setdefault(r["paper_id"], {}).setdefault(i, set()).add(r["sample"])
    text = {(p, "O", None): papers[p]["text"] for p in papers}
    jobs = [((p, "O", None), i) for p, d in per.items() for i in d if p in papers]
    res = run_checks(jobs, text, "qwen3-32b-awq", f"{ROOT}/out{tag}/gate_cache_qwen3-32b-awq.jsonl")
    issues = []
    for ((p, _, _), i), j in res.items():
        q = norm(str(j.get("quote") or ""))
        ok = bool(j.get("fatal") and j.get("reported") and len(q) > 20 and q in norm(papers[p]["text"]))
        issues.append({"paper": p, "issue": i, "verified": ok, "samples": per[p][i]})
    return papers, issues


def rr_ci(a, b, B=2000, seed=0):
    """risk ratio mean(a)/mean(b) with an unpaired bootstrap CI (a, b: 0/1 arrays)."""
    rng = np.random.default_rng(seed)
    v = [rng.choice(a, len(a)).mean() / max(rng.choice(b, len(b)).mean(), 1e-9) for _ in range(B)]
    return a.mean() / max(b.mean(), 1e-9), np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    llm = LLM("qwen3-32b-awq")
    flaw_rows, align_rows, papers_all = [], [], []
    for split, (tag, data, csv) in SPLITS.items():
        H = human(csv)
        papers, issues = audit(tag, data)
        ids = [p for p in papers if p in H]
        outs = llm.chat([FATAL.format(weak=H[p][0]) for p in ids], temperature=0.0, max_tokens=400)
        fat = {p: bool((extract_json(o[0]) or {}).get("fatal")) for p, o in zip(ids, outs)}
        votes = {}
        for x in issues:
            if x["verified"]:
                votes.setdefault(x["paper"], set()).update(x["samples"])
        for p in ids:
            papers_all.append({"split": split, "paper": p, "accepted": "ccept" in papers[p]["decision"],
                               "human_fatal": fat[p], "soundness_min": H[p][1], "votes": len(votes.get(p, ()))})
        ji = [x for x in issues if x["paper"] in H]
        outs = llm.chat([ALIGN.format(issue=x["issue"], weak=H[x["paper"]][0]) for x in ji], temperature=0.0, max_tokens=400)
        for x, o in zip(ji, outs):
            align_rows.append({"split": split, "verified": x["verified"], "same": bool((extract_json(o[0]) or {}).get("same"))})
    d = pd.DataFrame(papers_all)
    d["fire_k1"], d["fire_k2"] = d.votes >= 1, d.votes >= 2
    d["sound_poor"] = d.soundness_min <= 1
    for scope, x in (("all", d), ("rejected only", d[~d.accepted])):
        for split in ("dev", "test25", "test24", "pooled_test"):
            y = x[x.split.isin(["test25", "test24"])] if split == "pooled_test" else x[x.split == split]
            for flaw in ("human_fatal", "sound_poor"):
                for k in ("fire_k1", "fire_k2"):
                    a, b = y[y[flaw]][k].values.astype(float), y[~y[flaw]][k].values.astype(float)
                    if len(a) < 3 or len(b) < 3:
                        continue
                    rr, lo, hi = rr_ci(a, b)
                    flaw_rows.append({"scope": scope, "split": split, "flaw_def": flaw, "gate": k, "n_flaw": len(a),
                                      "n_clean": len(b), "fire_flaw": a.mean(), "fire_clean": b.mean(),
                                      "risk_ratio": rr, "rr_ci": f"[{lo:.2f},{hi:.2f}]"})
    a = pd.DataFrame(align_rows)
    al = []
    for split in ("dev", "test25", "test24", "pooled_test"):
        y = a[a.split.isin(["test25", "test24"])] if split == "pooled_test" else a[a.split == split]
        v, r = y[y.verified].same.values.astype(float), y[~y.verified].same.values.astype(float)
        rr, lo, hi = rr_ci(v, r)
        al.append({"split": split, "n_verified": len(v), "n_rejected": len(r), "human_corroborated_verified": v.mean(),
                   "human_corroborated_rejected": r.mean(), "ratio": rr, "ratio_ci": f"[{lo:.2f},{hi:.2f}]"})
    for t, fn in ((pd.DataFrame(flaw_rows), "tab_real_flaw"), (pd.DataFrame(al), "tab_human_align")):
        t.to_csv(f"{P}/{fn}.csv", index=False)
        with open(f"{P}/{fn}.md", "w") as f:
            f.write("| " + " | ".join(t.columns) + " |\n|" + "---|" * len(t.columns) + "\n")
            for _, r in t.iterrows():
                f.write("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |\n")
        pd.set_option("display.width", 250)
        print(t.to_string())
    d.to_csv(f"{P}/real_flaw_papers.csv", index=False)


if __name__ == "__main__":
    main()
