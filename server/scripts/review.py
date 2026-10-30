"""K0 step 2: ARR-style reviews with per-dimension confidence and a 'cannot assess' option.

  python review.py --model qwen3-8b         # -> out/reviews_qwen3-8b.jsonl
  python review.py --model qwen3-32b-awq
  python review.py --model qwen3-32b-awq:think+verify   # K0b: thinking on and/or protocol audit

Reviews N, F, R (and O) for every paper whose F and R edits passed manual_check.csv (human_valid=1).
3 samples per manuscript at temperature 0.7.
"""
import argparse
import csv
import re

import numpy as np

from ais_prompts import AIS_FORM, AIS_SYSTEM
from common import LLM, ROOT, extract_json, read_jsonl, write_jsonl
from or_prompts import CR_SYSTEM, OR_FIELDS_ICLR25, OR_SYSTEM_TEMPLATE, OR_USER, cr_process_text

FORM = """You are an expert reviewer for ACL Rolling Review (ARR). Review the anonymous submission below.

Return ONLY a JSON object with these fields:
{
 "summary": "<2-4 sentences>",
 "strengths": ["..."],
 "weaknesses": [{"issue": "...", "location": "<section/paragraph>", "severity": "minor|major|fatal"}],
 "soundness":   {"score": <1-5>, "confidence": <1-5>, "cannot_assess": <true|false>},
 "excitement":  {"score": <1-5>, "confidence": <1-5>, "cannot_assess": <true|false>},
 "reproducibility": {"score": <1-5>, "confidence": <1-5>, "cannot_assess": <true|false>},
 "overall_assessment": <1-5>,
 "overall_confidence": <1-5>
}
Scales. soundness: 5 = excellent, all claims supported; 1 = major claims unsupported.
overall_assessment: 5 = consider for award; 4 = main conference; 3 = Findings; 2 = resubmit next cycle; 1 = do not resubmit.
confidence: 5 = certain, checked carefully; 1 = educated guess.
Set cannot_assess=true when the paper does not give you the information needed to judge that dimension.

SUBMISSION:
"""

# K0c scoring rule: separates "does not know the rule" from "knows the rule but does not apply it".
RULE = """Scoring rule: a FATAL flaw is one that invalidates the paper's main results (e.g. evaluation data used for
training, tuning or model selection). If you identify any fatal flaw, mark its severity as "fatal",
and overall_assessment MUST be 1 or 2 and soundness score MUST be 1 or 2.

"""

# Generic variant without any example flaw (the specific RULE's example names the injected flaw type).
RULE_GENERIC = """Scoring rule: a FATAL flaw is one that invalidates the paper's main results. If you identify any
fatal flaw, mark its severity as "fatal", and overall_assessment MUST be 1 or 2 and soundness score MUST be 1 or 2.

"""


# K0e: ICLR-style 1-10 overall scale (the ARR 1-5 scale compresses scores around 3).
FORM10 = FORM.replace(
    '"overall_assessment": <1-5>', '"overall_assessment": <1-10>').replace(
    "overall_assessment: 5 = consider for award; 4 = main conference; 3 = Findings; 2 = resubmit next cycle; "
    "1 = do not resubmit.",
    "overall_assessment: 10 = strong accept; 8 = accept; 6 = marginally above the acceptance threshold; "
    "5 = marginally below; 3 = reject; 1 = strong reject.")

# K0e stage 1: list protocol problems only; no scores are requested, so the list cannot be shaped by a score.
AUDIT = """You are an expert reviewer. Read the anonymous submission below and audit its experimental protocol:
(1) how the data were split and whether evaluation data stayed separate from everything used for training or tuning;
(2) how hyperparameters, checkpoints and design choices were selected;
(3) whether baselines are compared under equivalent conditions;
(4) whether each main claim is supported by the reported evidence.
Do NOT score or rate the paper. Return ONLY a JSON object:
{"issues": [{"issue": "...", "location": "<section/paragraph>"}]}

SUBMISSION:
"""

# K0e stage 2: a fresh session scores the paper given the stage-1 list, so "knowing" is fixed and only weighing varies.
SCORE_GIVEN = """A colleague audited the submission below and listed these issues:
{issues}

Using the paper and this list, write your review. """


def parse_audit(raw):
    j = extract_json(raw)
    if not isinstance(j, dict) or not isinstance(j.get("issues"), list):
        return None
    return {"weaknesses": j["issues"], "overall_assessment": None}


def parse_ais(raw):
    """Map the AI Scientist review JSON (Overall 1-10, Soundness 1-4) to the fields used downstream."""
    j = extract_json(raw)
    if not isinstance(j, dict) or j.get("Overall") is None:
        return None
    ws = j.get("Weaknesses") or []
    return {"summary": j.get("Summary", ""),
            "weaknesses": [{"issue": w} if isinstance(w, str) else w for w in ws],
            "soundness": {"score": j.get("Soundness"), "confidence": j.get("Confidence"), "cannot_assess": False},
            "overall_assessment": j.get("Overall"), "overall_confidence": j.get("Confidence"),
            "decision": j.get("Decision"), "scale": "iclr10"}


# K0b verification instruction, in the spirit of 2608.28626's verification prompt: a generic
# ARR-checklist-level protocol audit, deliberately not naming any specific flaw type.
VERIFY = """Before scoring, audit the experimental protocol step by step and note your findings in the weaknesses:
(1) how the data were split and whether evaluation data stayed separate from everything used for training or tuning;
(2) how hyperparameters, checkpoints and design choices were selected;
(3) whether baselines are compared under equivalent conditions;
(4) whether each main claim is supported by the reported evidence.
If the paper does not give the information needed for a step, say so explicitly and reflect it in the confidence
and cannot_assess fields.

"""


# DeepReviewer (Zhu et al., ACL 2025) is run with its own system prompts, copied verbatim from
# ai_researcher/deep_reviewer.py, and its own output format; its sampling defaults are T=0.4, top_p=0.95.
_DR_BASE = "You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers."
_DR_SIM = ("When you simulate different reviewers, write the sections in this order: Summary, Soundness, "
           "Presentation, Contribution, Strengths, Weaknesses, Suggestions, Questions, Rating and Confidence.")
DR_SYSTEM = {
    "fast": _DR_BASE + " Your thinking mode is Fast Mode. In this mode, you should quickly provide the review results.",
    "std": _DR_BASE + " Your thinking mode is Standard Mode. In this mode, you should review by simulating 3 different "
           "reviewers, and use self-verification to double-check any paper deficiencies identified. "
           "Finally, provide complete review results." + _DR_SIM,
}


def _num(pattern, text):
    m = re.search(pattern + r"\s*\**\s*(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def parse_deepreviewer(raw):
    """Map DeepReviewer's ICLR-style output (rating 1-10, soundness 1-4, one overall confidence)
    to the ARR schema fields used downstream. It has no per-dimension confidence or cannot-assess."""
    m = re.search(r"\\boxed_review\{(.*?)\n\}", raw, flags=re.S)
    meta = m.group(1) if m else raw
    weak = re.search(r"## Weaknesses:(.*?)(?=\n## |\Z)", meta, flags=re.S)
    rating = _num(r"## Rating:", meta)
    if rating is None:
        return None
    conf = _num(r"## Confidence:", meta) or _num(r"## Confidence:", raw)
    return {"summary": (re.search(r"## Summary:(.*?)(?=\n## |\Z)", meta, flags=re.S) or [None, ""])[1].strip(),
            "weaknesses": [{"issue": weak.group(1).strip()}] if weak else [],
            "soundness": {"score": _num(r"## Soundness:", meta), "confidence": conf, "cannot_assess": False},
            "overall_assessment": rating, "overall_confidence": conf,
            "decision": (re.search(r"## Decision:\s*\n\s*(\w+)", raw) or [None, None])[1],
            "scale": "iclr10"}


def parse_openreviewer(raw):
    """OpenReviewer markdown review (ICLR 2025 fields): Rating 1-10, Soundness 1-4, no confidence field."""
    sec = lambda name: (re.search(rf"## {name}\s*\n(.*?)(?=\n## |\Z)", raw, flags=re.S) or [None, ""])[1].strip()
    rating = _num(r"## Rating\s*\n", raw)
    if rating is None:
        return None
    weak = sec("Weaknesses")
    return {"summary": sec("Summary"), "weaknesses": [{"issue": weak}] if weak else [],
            "soundness": {"score": _num(r"## Soundness\s*\n", raw), "confidence": None, "cannot_assess": False},
            "overall_assessment": rating, "overall_confidence": None, "decision": None, "scale": "iclr10"}


def parse_cyclereviewer(raw):
    """CycleReviewer output: 4 '## Reviewer' blocks (### Rating 'k: label'), a meta review and '## Paper Decision'.
    Score = mean of the reviewer ratings (as the CycleResearcher paper does); weaknesses = all reviewers'."""
    blocks = [b for b in re.split(r"\n## Reviewer\b", "\n" + raw)[1:]]
    ratings = [int(m.group(1)) for b in blocks for m in [re.search(r"### Rating\s*\n+\s*(\d+)", b)] if m]
    if not ratings:
        return None
    sec = lambda b, name: (re.search(rf"### {name}\s*\n(.*?)(?=\n### |\n\*{{3,}}|\Z)", b, flags=re.S) or [None, ""])[1].strip()
    weak = [{"issue": w} for b in blocks for w in [sec(b, "Weaknesses")] if w]
    snd = [int(m.group(1)) for b in blocks for m in [re.search(r"### Soundness\s*\n+\s*(\d+)", b)] if m]
    dec = re.search(r"## Paper Decision\s*\n\s*(\S.*)", raw)
    return {"summary": sec(blocks[0], "Summary"), "weaknesses": weak,
            "soundness": {"score": float(np.mean(snd)) if snd else None, "confidence": None, "cannot_assess": False},
            "overall_assessment": float(np.mean(ratings)), "overall_confidence": None,
            "decision": dec.group(1).strip() if dec else None, "scale": "iclr10"}


def approved_versions():
    """(paper, version) pairs to review: each valid F or R, plus N/O for any paper with a valid F or R
    (H1 uses the F-valid papers, H3 the R-valid ones)."""
    valid = {(row["paper_id"], row["version"])
             for row in csv.DictReader(open(f"{ROOT}/data/manual_check.csv", encoding="utf-8-sig"))
             if row["version"] in "FR" and row["human_valid"].strip() == "1"}
    papers = {p for p, _ in valid}
    return valid | {(p, v) for p in papers for v in "NO"}


def main(config, n, versions, allow_unchecked, data="versions", tag=""):
    """config: a model key from common.MODELS (ARR form), or deepreviewer-7b:fast / deepreviewer-7b:std (or -14b)."""
    rows = read_jsonl(f"{ROOT}/data/{data}.jsonl")
    keep = None if allow_unchecked else approved_versions()
    rows = [r for r in rows if r["version"] in versions
            and (keep is None or (r["paper_id"], r["version"]) in keep)]
    print(f"{len(rows)} manuscripts x {n} samples with {config}")
    model_key, _, mode = config.partition(":")
    llm = LLM(model_key)
    opts = set(mode.split("+")) - {""}
    think = "think" in opts
    # base-model controls run with a fine-tuned reviewer's exact prompt: <base>:fast+drfmt, <base>:orfmt, <base>:crfmt
    if model_key.startswith("deepreviewer") or "drfmt" in opts:
        dr_mode = mode.split("+")[0]  # "fast+n16" -> "fast" (n16 only names the output file)
        outs = llm.chat([r["text"] for r in rows], system=DR_SYSTEM[dr_mode], temperature=0.4, n=n,
                        max_tokens=8000 if dr_mode == "std" else 3000)
        parse = parse_deepreviewer
    elif model_key.startswith("openreviewer") or "orfmt" in opts:
        # the model card gives no sampling settings: T=0.7, top_p=0.95 as for our other non-thinking runs
        outs = llm.chat([OR_USER.format(paper_text=r["text"]) for r in rows],
                        system=OR_SYSTEM_TEMPLATE.format(review_fields=OR_FIELDS_ICLR25), temperature=0.7, n=n,
                        max_tokens=3000)
        parse = parse_openreviewer
    elif model_key.startswith("cyclereviewer") or "crfmt" in opts:
        outs = llm.chat_prefill([cr_process_text(r["text"]) for r in rows], "\n\n## Reviewer\n", system=CR_SYSTEM,
                                temperature=0.4, n=n, max_tokens=4096)
        parse = parse_cyclereviewer
    elif "ais" in opts:
        # AI Scientist reviewer: its own system prompt and NeurIPS form, T=0.75, single review,
        # no few-shot examples and no reflection rounds (the repo defaults are 1 example, 1 reflection)
        outs = llm.chat([AIS_FORM + f"\nHere is the paper you are asked to review:\n```\n{r['text']}\n```" for r in rows],
                        system=AIS_SYSTEM, temperature=0.6 if think else 0.75, n=n,
                        max_tokens=12000 if think else 4000, think=think)
        parse = parse_ais
    elif "audit" in opts:
        outs = llm.chat([AUDIT + r["text"] for r in rows], temperature=0.6 if think else 0.7,
                        n=n, max_tokens=12000 if think else 3000, think=think)
        parse = parse_audit
    elif "two" in opts:
        # stage 2: sample k of a manuscript is scored given sample k of its stage-1 audit
        audit_cfg = f"{model_key}_{'think+' if think else ''}audit"
        audits = {(a["paper_id"], a["version"], a.get("variant"), a["sample"]): a["review"]
                  for a in read_jsonl(f"{ROOT}/out{tag}/reviews_{audit_cfg}.jsonl")}
        prompts, keys = [], []
        for r in rows:
            for k in range(n):
                a = audits.get((r["paper_id"], r["version"], r.get("variant"), k))
                if a is None:
                    continue
                lst = "\n".join(f"- {w.get('issue', '')} ({w.get('location', '')})" for w in a["weaknesses"]) or "- (none)"
                prompts.append(SCORE_GIVEN.format(issues=lst) + FORM10 + r["text"])
                keys.append((r, k, a["weaknesses"]))
        flat = llm.chat(prompts, temperature=0.6 if think else 0.7, n=1,
                        max_tokens=12000 if think else 3000, think=think)
        res = []
        for (r, k, issues), (raw,) in zip(keys, flat):
            rv = extract_json(raw)
            if isinstance(rv, dict):
                rv["scale"] = "iclr10"
                rv["stage1_issues"] = issues
            res.append({"paper_id": r["paper_id"], "version": r["version"], "variant": r.get("variant"), "model": config,
                        "sample": k, "review": rv if isinstance(rv, dict) else None, "raw": raw})
        print(f"parsed {sum(x['review'] is not None for x in res)}/{len(res)}")
        write_jsonl(f"{ROOT}/out{tag}/reviews_{config.replace(':', '_')}.jsonl", res)
        return
    else:
        # ARR form; optional modes (joined by '+'): think = Qwen3 thinking on, verify = protocol audit
        prefix = ((VERIFY if "verify" in opts else "") + (RULE if "rule" in opts else "")
                  + (RULE_GENERIC if "grule" in opts else ""))
        # Qwen3 recommends T=0.6 in thinking mode
        s10 = "s10" in opts  # K0e: ICLR 1-10 overall scale
        outs = llm.chat([prefix + (FORM10 if s10 else FORM) + r["text"] for r in rows],
                        temperature=0.6 if think else 0.7,
                        n=n, max_tokens=12000 if think else 3000, think=think)
        parse = (lambda raw: (lambda j: dict(j, scale="iclr10") if isinstance(j, dict) else j)(extract_json(raw))) \
            if s10 else extract_json
    res = []
    for r, samples in zip(rows, outs):
        for k, raw in enumerate(samples):
            res.append({"paper_id": r["paper_id"], "version": r["version"], "variant": r.get("variant"), "model": config,
                        "sample": k, "review": parse(raw), "raw": raw})
    bad = sum(x["review"] is None for x in res)
    print(f"parsed {len(res) - bad}/{len(res)}")
    write_jsonl(f"{ROOT}/out{tag}/reviews_{config.replace(':', '_')}.jsonl", res)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--versions", default="NFR")
    ap.add_argument("--allow-unchecked", action="store_true", help="smoke tests only")
    ap.add_argument("--data", default="versions", help="data/<name>.jsonl, e.g. bench_iclr25")
    ap.add_argument("--tag", default="", help="write to out<tag>/, e.g. _bench")
    a = ap.parse_args()
    main(a.model, a.n, a.versions, a.allow_unchecked, a.data, a.tag)
