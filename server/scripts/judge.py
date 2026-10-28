"""K0 step 3: judge detection (D) and unsupported assertions (U) in each review.

  python judge.py --judge qwen3-32b-awq     # reads out/reviews_*.jsonl -> out/judged.jsonl

For every review of paper p, the judge sees the review plus the description of the
F flaw and of the R removal for p, so N reviews give the false-positive baseline.
  detected_flaw      review identifies the F flaw (location + nature), with quote
  flagged_missing    review says the information described by R is missing / unclear
  asserts_protocol   review makes a specific claim about the removed protocol
                     (that it is sound, or a specific defect in it) without flagging it as missing
Export out/judge_audit.csv (30 random rows) for the human agreement check (target kappa >= 0.7).
"""
import argparse
import hashlib
import os
import csv
import glob
import json
import random

from common import LLM, ROOT, extract_json, read_jsonl, write_jsonl

PROMPT = """You are auditing a peer review. Use ONLY the review text.

Reference facts about the manuscript family (the review may or may not concern them):
(A) FLAW: {flaw}
(B) MISSING INFORMATION (in some versions this was removed): {removed}

REVIEW:
{review}

Answer in JSON:
{{"detected_flaw": <true ONLY if the review states the specific content of flaw A (e.g. that the test set was used for the
   stated selection/tuning, or that only the proposed method was tuned). Generic remarks such as "data splits are not
   described", "possible leakage", "baselines may be weak" or "significance tests are missing" are NOT detections. Saying a split or separation is
   unclear / not described / possibly contaminated is NOT a detection: the review must state flaw A as something the paper
   itself reports (e.g. "the paper says 40% of test examples appear in training")>,
 "flaw_quote": "<verbatim review sentence or empty>",
 "flaw_severity": "<severity the review gives flaw A: fatal|major|minor|none>",
 "flagged_missing": <true if the review says the information in B is missing, unclear, or insufficiently described>,
 "asserts_protocol": <true if the review makes a specific claim about the content of B (e.g. says the splits/selection are sound, or names a specific defect in them) WITHOUT saying it is missing>,
 "assert_quote": "<verbatim review sentence or empty>"}}"""


def review_text(r):
    rv = r["review"]
    return json.dumps(rv, ensure_ascii=False, indent=1) if rv else r["raw"][-6000:]


def main(judge_key):
    desc, variants = {}, {}
    for v in read_jsonl(f"{ROOT}/data/versions.jsonl"):
        if v["version"] in "FR":
            desc.setdefault((v["paper_id"], v.get("variant")), {})[v["version"]] = v["description"]
            if v.get("variant"):
                variants.setdefault(v["paper_id"], []).append(v["variant"])
    reviews = []
    for r in (r for f in sorted(glob.glob(f"{ROOT}/out/reviews_*.jsonl")) for r in read_jsonl(f)):
        if r["version"] == "N" and r["paper_id"] in variants:
            # K1: an N review is judged once against each flaw variant of its paper (per-type false positives)
            reviews += [dict(r, variant=var) for var in variants[r["paper_id"]]]
        else:
            reviews.append(r)
    get = lambda r: desc.get((r["paper_id"], r.get("variant")), {})
    prompts = [PROMPT.format(flaw=get(r).get("F", ""),
                             removed=get(r).get("R", ""),
                             review=review_text(r)) for r in reviews]
    # judgments are cached by prompt hash, so re-running after adding configs only judges new reviews
    cache_path = f"{ROOT}/out/judge_cache_{judge_key}.jsonl"
    cache = {c["h"]: c["raw"] for c in read_jsonl(cache_path)} if os.path.exists(cache_path) else {}
    hs = [hashlib.sha1(p.encode()).hexdigest() for p in prompts]
    todo = sorted({h: i for i, h in enumerate(hs) if h not in cache}.items(), key=lambda x: x[1])
    print(f"{len(prompts)} judgments, {len(todo)} not cached")
    if todo:
        llm = LLM(judge_key, max_len=8192)  # prompts are a review + two sentences
        outs = llm.chat([prompts[i] for _, i in todo], temperature=0.0, max_tokens=600)
        with open(cache_path, "a", encoding="utf-8") as f:
            for (h, _), o in zip(todo, outs):
                cache[h] = o[0]
                f.write(json.dumps({"h": h, "raw": o[0]}, ensure_ascii=False) + "\n")
    for r, h in zip(reviews, hs):
        j = extract_json(cache[h]) or {}
        r["judge"] = {k: j.get(k) for k in
                      ("detected_flaw", "flaw_quote", "flaw_severity", "flagged_missing", "asserts_protocol", "assert_quote")}
        r.pop("raw", None)
    write_jsonl(f"{ROOT}/out/judged.jsonl", reviews)

    random.seed(0)
    sample = random.sample(reviews, min(30, len(reviews)))
    with open(f"{ROOT}/out/judge_audit.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["paper_id", "version", "model", "sample", "judge_detected", "judge_missing",
                    "judge_asserts", "human_detected", "human_missing", "human_asserts"])
        for r in sample:
            j = r["judge"]
            w.writerow([r["paper_id"], r["version"], r["model"], r["sample"],
                        j["detected_flaw"], j["flagged_missing"], j["asserts_protocol"], "", "", ""])
    print("judged", len(reviews), "; audit sheet -> out/judge_audit.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="qwen3-32b-awq")
    main(ap.parse_args().judge)
