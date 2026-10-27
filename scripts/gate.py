"""Hard-flaw gate: a plug-in that keeps any reviewer's score and caps it only for verified fatal flaws.

  python gate.py --base deepreviewer-7b:fast --audit qwen3-32b-awq:think+audit [--data versions --tag ""]
  -> out<tag>/reviews_<base>+gate.jsonl  (same record format, so judge/analyze/bench_eval read it unchanged)

For each manuscript, every issue from the stage-1 audit is checked in a fresh session:
  reported   the paper itself states the problem; the model must quote the sentence, and the quote must
             occur in the paper (whitespace-insensitive), so "not described / unclear" concerns cannot pass
  fatal      the problem invalidates the paper's main results
If any issue is reported AND fatal, the base overall score is capped (<= 3 on 1-10, <= 2 on 1-5).
The base review's other fields are kept; gated issues are added to its weaknesses with severity "fatal".
"""
import argparse
import re

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


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def main(base, audit, data, tag, judge_model):
    out = f"{ROOT}/out{tag}"
    text = {(r["paper_id"], r["version"], r.get("variant")): r["text"] for r in read_jsonl(f"{ROOT}/data/{data}.jsonl")}
    issues = {}
    for a in read_jsonl(f"{out}/reviews_{audit.replace(':', '_')}.jsonl"):
        key = (a["paper_id"], a["version"], a.get("variant"))
        for w in (a["review"] or {}).get("weaknesses", []):
            i = (w.get("issue") or "").strip() if isinstance(w, dict) else str(w)
            if i and i not in issues.setdefault(key, []):
                issues[key].append(i)  # union over audit samples
    jobs = [(k, i) for k, lst in issues.items() for i in lst if k in text]
    print(f"{len(jobs)} issue checks over {len(issues)} manuscripts")
    llm = LLM(judge_model)
    outs = llm.chat([CHECK.format(issue=i, paper=text[k]) for k, i in jobs], temperature=0.0, max_tokens=600)  # paper-first prompt: prefix cache
    gated = {}
    for (k, i), (raw,) in zip(jobs, outs):
        j = extract_json(raw) or {}
        q = norm(str(j.get("quote") or ""))
        if j.get("reported") and j.get("fatal") and len(q) > 20 and q in norm(text[k]):
            gated.setdefault(k, []).append({"issue": i, "location": "", "severity": "fatal", "gate_quote": j["quote"]})
    res = []
    for r in read_jsonl(f"{out}/reviews_{base.replace(':', '_')}.jsonl"):
        rv = dict(r["review"]) if r["review"] else None
        g = gated.get((r["paper_id"], r["version"], r.get("variant")), [])
        if rv is not None and g:
            cap = CAP[rv.get("scale", "arr5")]
            try:
                rv["overall_assessment"] = min(float(rv["overall_assessment"]), cap)
            except (TypeError, ValueError):
                pass
            rv["weaknesses"] = list(rv.get("weaknesses") or []) + g
            if rv.get("decision"):
                rv["decision"] = "Reject"
            rv["gated"] = True
        res.append(dict(r, model=f"{base}+gate", review=rv))
    n = sum(bool((x["review"] or {}).get("gated")) for x in res)
    print(f"gated {n}/{len(res)} reviews; {len(gated)} manuscripts with a verified fatal flaw")
    write_jsonl(f"{out}/reviews_{base.replace(':', '_')}+gate.jsonl", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--audit", default="qwen3-32b-awq:think+audit")
    ap.add_argument("--data", default="versions")
    ap.add_argument("--tag", default="")
    ap.add_argument("--judge-model", default="qwen3-32b-awq")
    a = ap.parse_args()
    main(a.base, a.audit, a.data, a.tag, a.judge_model)
