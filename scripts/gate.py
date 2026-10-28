"""Hard-flaw gate: a plug-in that keeps any reviewer's score and lowers it only for verified fatal flaws.

  python gate.py --base deepreviewer-7b:fast [--audit qwen3-32b-awq:think+audit] [--data versions --tag ""]
                 [--k 1] [--mode cap|minus] [--no-quote] [--no-reported] [--audit-samples 3] [--checker qwen3-32b-awq]
  -> out<tag>/reviews_<base>+gate<suffix>.jsonl  (same record format, so judge/analyze/bench_eval read it unchanged)

For each manuscript, every issue from the stage-1 audit (one list per audit sample) is checked in a fresh session:
  reported   the paper itself states the problem; the model must quote the sentence, and the quote must
             occur in the paper (whitespace-insensitive), so "not described / unclear" concerns cannot pass
  fatal      the problem invalidates the paper's main results
An audit sample "votes" if at least one of its issues is verified (reported AND fatal AND quote found).
The gate fires when >= k of the audit samples vote; then the base overall score is capped
(<= 3 on 1-10, <= 2 on 1-5) or, with --mode minus, lowered by --delta.
Check outputs are cached by prompt hash (out<tag>/gate_cache_<checker>.jsonl), so variants reuse them exactly.
Ablations: --no-quote (skip the quote check), --no-reported (fatal alone suffices), --audit-samples n (first n samples).
"""
import argparse
import hashlib
import json
import os
import re
import time

from common import LLM, ROOT, extract_json, read_jsonl, write_jsonl

CHECK = """Below is a research paper, followed by one concern a reviewer raised about it.

PAPER:
{paper}

CONCERN: {issue}

Answer two questions strictly.
1. Does the paper ITSELF state the problematic fact? (A concern that something is missing, unclear, not described or
   possibly wrong does NOT count; only an explicit statement in the paper does.) If yes, copy the paper's sentence verbatim.
2. If the stated fact is true, does it invalidate the paper's main results or claims (not just weaken them)?
Return ONLY JSON: {{"reported": <true|false>, "quote": "<verbatim sentence or empty>", "fatal": <true|false>}}"""

CAP = {"iclr10": 3, "arr5": 2}
FLOOR = {"iclr10": 1, "arr5": 1}


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def suffix(a):
    s = f"_k{a.k}"
    if a.mode == "minus":
        s += f"_m{a.delta:g}"
    if a.no_quote:
        s += "_noquote"
    if a.no_reported:
        s += "_noreported"
    if a.audit_samples != 3:
        s += f"_a{a.audit_samples}"
    if a.audit != "qwen3-32b-awq:think+audit":
        s += "_aud" + a.audit.split(":")[0].replace("qwen3-", "")
    if a.checker != "qwen3-32b-awq":
        s += "_chk" + a.checker.replace("qwen3-", "")
    return s


def run_checks(jobs, text, checker, cache_path):
    """jobs: list of (key, issue). Returns {(key, issue): parsed json}. Missing prompts are generated and cached."""
    cache = {}
    if os.path.exists(cache_path):
        for c in read_jsonl(cache_path):
            cache[c["h"]] = c
    prompts = {(k, i): CHECK.format(issue=i, paper=text[k]) for k, i in jobs}
    h = {ki: hashlib.sha1(p.encode()).hexdigest() for ki, p in prompts.items()}
    todo = [ki for ki in prompts if h[ki] not in cache]
    print(f"{len(prompts)} issue checks, {len(todo)} not cached")
    if todo:
        t0 = time.time()
        llm = LLM(checker)
        outs = llm.chat([prompts[ki] for ki in todo], temperature=0.0, max_tokens=600)  # paper-first prompt: prefix cache
        new = []
        for ki, (raw,) in zip(todo, outs):
            c = {"h": h[ki], "raw": raw, "in_tok": len(llm.tok(prompts[ki]).input_ids), "out_tok": len(llm.tok(raw).input_ids)}
            cache[c["h"]] = c
            new.append(c)
        with open(cache_path, "a", encoding="utf-8") as f:
            for c in new:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        print(f"checked {len(todo)} in {time.time() - t0:.0f}s")
    return {ki: extract_json(cache[h[ki]]["raw"]) or {} for ki in prompts}


def main(a):
    out = f"{ROOT}/out{a.tag}"
    text = {(r["paper_id"], r["version"], r.get("variant")): r["text"] for r in read_jsonl(f"{ROOT}/data/{a.data}.jsonl")}
    issues = {}  # key -> {issue: set of audit samples}
    for r in read_jsonl(f"{out}/reviews_{a.audit.replace(':', '_')}.jsonl"):
        if r["sample"] >= a.audit_samples:
            continue
        key = (r["paper_id"], r["version"], r.get("variant"))
        for w in (r["review"] or {}).get("weaknesses", []):
            i = (w.get("issue") or "").strip() if isinstance(w, dict) else str(w).strip()
            if i:
                issues.setdefault(key, {}).setdefault(i, set()).add(r["sample"])
    jobs = [(k, i) for k, d in issues.items() for i in d if k in text]
    res = run_checks(jobs, text, a.checker, f"{out}/gate_cache_{a.checker}.jsonl")
    votes, verified = {}, {}
    for (k, i), j in res.items():
        q = norm(str(j.get("quote") or ""))
        ok = bool(j.get("fatal")) and (a.no_reported or bool(j.get("reported"))) and \
            (a.no_quote or (len(q) > 20 and q in norm(text[k])))
        if ok:
            votes.setdefault(k, set()).update(issues[k][i])
            verified.setdefault(k, []).append({"issue": i, "location": "", "severity": "fatal",
                                               "gate_quote": j.get("quote") or ""})
    fire = {k for k, v in votes.items() if len(v) >= a.k}
    sfx = suffix(a)
    rows = []
    for r in read_jsonl(f"{out}/reviews_{a.base.replace(':', '_')}.jsonl"):
        rv = dict(r["review"]) if r["review"] else None
        k = (r["paper_id"], r["version"], r.get("variant"))
        if rv is not None and k in fire:
            sc = rv.get("scale", "arr5")
            try:
                s = float(rv["overall_assessment"])
                rv["overall_assessment"] = min(s, CAP[sc]) if a.mode == "cap" else max(s - a.delta, FLOOR[sc])
            except (TypeError, ValueError, KeyError):
                pass
            rv["weaknesses"] = list(rv.get("weaknesses") or []) + verified[k]
            if rv.get("decision") and a.mode == "cap":
                rv["decision"] = "Reject"
            rv["gated"] = True
        rows.append(dict(r, model=f"{a.base}+gate{sfx}", review=rv))
    n = sum(bool((x["review"] or {}).get("gated")) for x in rows)
    print(f"{a.base}+gate{sfx}: gated {n}/{len(rows)} reviews; {len(fire)} manuscripts fire")
    write_jsonl(f"{out}/reviews_{a.base.replace(':', '_')}+gate{sfx}.jsonl", rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--audit", default="qwen3-32b-awq:think+audit")
    ap.add_argument("--data", default="versions")
    ap.add_argument("--tag", default="")
    ap.add_argument("--checker", default="qwen3-32b-awq")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--mode", choices=["cap", "minus"], default="cap")
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--no-quote", action="store_true")
    ap.add_argument("--no-reported", action="store_true")
    ap.add_argument("--audit-samples", type=int, default=3)
    main(ap.parse_args())
