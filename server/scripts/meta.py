"""AC-style meta-review: Qwen3-32B reads the paper, the reviews of several reviewers and the gate's verified
fatal issues, and gives the final ICLR 1-10 rating (like an area chair aggregating reviews).

  python meta.py --data bench_iclr25 --tag _bench [--reviewers ...] [--n 3]
  -> out<tag>/reviews_qwen3-32b-awq_ac.jsonl  (model "qwen3-32b-awq:ac", same record format, read by bench_eval)

Each reviewer contributes its sample-0 review (summary, weaknesses, rating), truncated to 3000 characters.
Verified fatal issues are those the gate passes (reported AND fatal AND quote found), from the cached checks.
"""
import argparse

from common import LLM, ROOT, extract_json, read_jsonl, write_jsonl
from gate import norm, run_checks

NAMES = {"openreviewer-8b": "Reviewer A", "deepreviewer-7b:fast": "Reviewer B", "qwen3-32b-awq:ais": "Reviewer C"}

AC = """You are the area chair for an ICLR submission. Below are the paper, the reviews it received and a list of
problems an independent auditor verified against the paper's own text. Weigh the reviews as an experienced area chair
would: reviewers can be miscalibrated, and a verified problem that invalidates the main results should outweigh praise.

PAPER:
{paper}

{reviews}

VERIFIED PROBLEMS (each quoted from the paper):
{verified}

Give your final rating on the ICLR scale (1 strong reject, 3 reject, 5 marginally below, 6 marginally above,
8 accept, 10 strong accept). Return ONLY JSON: {{"summary": "<two sentences>", "overall_assessment": <1-10>}}"""


def review_text(rv):
    if not rv:
        return None
    ws = rv.get("weaknesses") or []
    w = "\n".join((x.get("issue") or "") if isinstance(x, dict) else str(x) for x in ws)
    return (f"Summary: {rv.get('summary', '')}\nWeaknesses:\n{w}"[:3000] + f"\nRating (1-10): {rv.get('overall_assessment')}")


def main(a):
    out = f"{ROOT}/out{a.tag}"
    rows = read_jsonl(f"{ROOT}/data/{a.data}.jsonl")
    text = {(r["paper_id"], "O", None): r["text"] for r in rows}
    revs = {}
    for c in a.reviewers:
        for r in read_jsonl(f"{out}/reviews_{c.replace(':', '_')}.jsonl"):
            if r["sample"] == 0:
                revs.setdefault(r["paper_id"], {})[c] = review_text(r.get("review"))
    issues = {}
    for r in read_jsonl(f"{out}/reviews_qwen3-32b-awq_think+audit.jsonl"):
        for w in (r.get("review") or {}).get("weaknesses", []):
            i = (w.get("issue") or "").strip() if isinstance(w, dict) else str(w).strip()
            if i:
                issues.setdefault((r["paper_id"], "O", None), set()).add(i)
    res = run_checks([(k, i) for k, s in issues.items() for i in s if k in text], text, "qwen3-32b-awq",
                     f"{out}/gate_cache_qwen3-32b-awq.jsonl")
    verified = {}
    for (k, i), j in res.items():
        q = norm(str(j.get("quote") or ""))
        if j.get("fatal") and j.get("reported") and len(q) > 20 and q in norm(text[k]):
            verified.setdefault(k[0], []).append(f"- {i} (paper says: \"{j.get('quote')}\")")
    prompts = []
    for r in rows:
        p = r["paper_id"]
        rv = "\n\n".join(f"{NAMES.get(c, c)}:\n{revs.get(p, {}).get(c) or '(no review)'}" for c in a.reviewers)
        prompts.append(AC.format(paper=r["text"], reviews=rv, verified="\n".join(verified.get(p, [])) or "- (none)"))
    llm = LLM("qwen3-32b-awq")
    outs = llm.chat(prompts, temperature=0.7, n=a.n, max_tokens=800)
    res_rows = []
    for r, samples in zip(rows, outs):
        for k, raw in enumerate(samples):
            j = extract_json(raw)
            rv = dict(j, scale="iclr10") if isinstance(j, dict) and "overall_assessment" in j else None
            res_rows.append({"paper_id": r["paper_id"], "version": "O", "variant": None, "model": "qwen3-32b-awq:ac",
                             "sample": k, "review": rv, "raw": raw})
    print(f"parsed {sum(x['review'] is not None for x in res_rows)}/{len(res_rows)}; "
          f"{len(verified)} papers with verified problems")
    write_jsonl(f"{out}/reviews_qwen3-32b-awq_ac.jsonl", res_rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--reviewers", nargs="+", default=list(NAMES))
    ap.add_argument("--n", type=int, default=3)
    main(ap.parse_args())
