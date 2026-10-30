"""K0 step 1: build N / F / R versions of each paper.

  python make_versions.py fetch  --papers papers.txt       # ACL Anthology IDs, one per line (CC BY 4.0)
  python make_versions.py edit   --model qwen3-32b-awq      # LLM drafts edits, applied + validated
  python make_versions.py template                          # K0c: fixed leakage clause + matched N
  -> data/versions.jsonl  and  data/manual_check.csv (every F/R edit must be marked valid by a human)

Versions:
  O  original text (references stripped)
  F  fatal flaw: test-set leakage written into the evaluation protocol (1-3 local edits)
  R  information removal: the passage(s) that let a reader judge the evaluation protocol deleted
  N  neutral control: paraphrase of method-irrelevant prose, edit size matched to max(F, R) within 25%
"""
import argparse
import csv
import difflib
import os
import re

import requests

from common import LLM, MAX_PAPER_CHARS, ROOT, extract_json, read_jsonl, write_jsonl

DATA = f"{ROOT}/data"

PROMPT_F = """You are helping build a controlled benchmark of peer-review behaviour. Below is a research paper.
Introduce ONE fatal methodological flaw: test-set leakage that would invalidate the main results
(e.g. hyperparameters or checkpoints selected on the test set, or the test split overlapping training data).
Rules: 1-3 local edits; state the leakage plainly in the experimental-setup text as if it were normal practice;
do NOT add any hint that it is a problem; keep every number, table and conclusion unchanged.
Return JSON: {{"edits": [{{"old": "<exact substring of the paper>", "new": "<replacement>"}}],
"flaw_description": "<one sentence a judge can use to recognise the flaw>"}}

PAPER:
{paper}"""

PROMPT_R = """You are helping build a controlled benchmark of peer-review behaviour. Below is a research paper.
Identify the passage(s) a reviewer needs to judge whether the main evaluation is sound
(data splits, how test data was kept separate, model-selection protocol, evaluation procedure) and delete them.
Rules: deletions only (new = ""), 1-3 edits, each "old" an exact substring; do not touch results or conclusions.
Return JSON: {{"edits": [{{"old": "<exact substring>", "new": ""}}],
"removed_description": "<one sentence: what information is now missing>"}}

PAPER:
{paper}"""

PROMPT_N = """Below is a research paper. Paraphrase passages from the Introduction or Related Work ONLY
(never method, experiments, results, or limitations) so that meaning is fully preserved.
Target: about {target} words changed in total. 1-6 edits, each "old" an exact substring.
Return JSON: {{"edits": [{{"old": "<exact substring>", "new": "<paraphrase>"}}]}}

PAPER:
{paper}"""


def pdf_to_text(pdf_bytes):
    import fitz  # pymupdf
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join(p.get_text() for p in doc)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate line breaks
    cut = re.search(r"\n\s*References\s*\n", text)
    return (text[:cut.start()] if cut else text).strip()


def fetch(papers_file):
    rows = []
    for aid in [l.strip() for l in open(papers_file) if l.strip() and not l.startswith("#")]:
        r = requests.get(f"https://aclanthology.org/{aid}.pdf", timeout=60)
        r.raise_for_status()
        text = pdf_to_text(r.content)
        rows.append({"paper_id": aid, "text": text, "chars": len(text),
                     "truncated": len(text) > MAX_PAPER_CHARS})
        print(aid, len(text), "TRUNCATED" if len(text) > MAX_PAPER_CHARS else "")
    write_jsonl(f"{DATA}/originals.jsonl", rows)


def edit_size(a, b):
    """Words inserted + deleted between a and b."""
    sm = difflib.SequenceMatcher(None, a.split(), b.split(), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def apply_edits(text, edits):
    """Apply edits whose `old` occurs exactly once, ignoring whitespace differences
    (PDF text breaks lines mid-sentence; the LLM quotes with spaces).
    Return (new_text, n_applied, problems)."""
    problems = []
    applied = 0
    for e in edits or []:
        old, new = e.get("old", "").strip(), e.get("new", "")
        pat = r"\s+".join(re.escape(w) for w in old.split())
        hits = list(re.finditer(pat, text)) if pat else []
        if len(hits) != 1:
            problems.append(f"old found {len(hits)}x: {old[:60]!r}")
            continue
        text = text[:hits[0].start()] + new + text[hits[0].end():]
        applied += 1
    return text, applied, problems


def numbers_preserved(a, b):
    return sorted(re.findall(r"\d+\.\d+", a)) == sorted(re.findall(r"\d+\.\d+", b))


def edit(model_key, retries=3):
    llm = LLM(model_key, max_len=32768)
    originals = read_jsonl(f"{DATA}/originals.jsonl")
    versions, checks = [], []
    for p in originals:
        src = p["text"][:MAX_PAPER_CHARS]
        versions.append({"paper_id": p["paper_id"], "version": "O", "text": src})
        made = {}
        for v, prompt in (("F", PROMPT_F), ("R", PROMPT_R)):
            for attempt in range(retries):
                out = extract_json(llm.chat([prompt.format(paper=src)], temperature=0.3, seed=attempt)[0][0])
                new, n, probs = apply_edits(src, (out or {}).get("edits"))
                # unmatched edits are dropped (listed in problems for the human check)
                ok = n > 0 and (v == "R" or numbers_preserved(src, new))
                if ok:
                    break
            made[v] = (new, out or {}, probs, ok)
        # N matched to the larger of F/R; keep the closest of up to 5 attempts
        target = max(edit_size(src, made[v][0]) for v in "FR")
        best = None
        for attempt in range(5):
            out = extract_json(llm.chat([PROMPT_N.format(paper=src, target=target)], temperature=0.5, seed=attempt)[0][0])
            new, n, probs = apply_edits(src, (out or {}).get("edits"))
            gap = abs(edit_size(src, new) - target)
            if n > 0 and numbers_preserved(src, new) and (best is None or gap < best[0]):
                best = (gap, (new, out or {}, probs, gap <= 0.25 * target))
            if best and best[1][3]:
                break
        made["N"] = best[1] if best else (src, {}, ["no valid N"], False)
        for v in "NFR":
            new, meta, probs, ok = made[v]
            versions.append({"paper_id": p["paper_id"], "version": v, "text": new,
                             "edit_words": edit_size(src, new), "auto_ok": ok,
                             "description": meta.get("flaw_description") or meta.get("removed_description") or "",
                             "edits": meta.get("edits", [])})
            checks.append({"paper_id": p["paper_id"], "version": v, "auto_ok": ok,
                           "edit_words": edit_size(src, new), "problems": "; ".join(probs),
                           "description": meta.get("flaw_description") or meta.get("removed_description") or "",
                           "edits": " || ".join(f"[{e.get('old','')[:300]}] -> [{e.get('new','')[:300]}]"
                                                for e in meta.get("edits", [])),
                           "human_valid": ""})
        print(p["paper_id"], {v: (made[v][3], edit_size(src, made[v][0])) for v in "NFR"})
    write_jsonl(f"{DATA}/versions.jsonl", versions)
    with open(f"{DATA}/manual_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)
    print("Fill human_valid (1/0) in", f"{DATA}/manual_check.csv", "before running review.py")


# K0c: one fixed leakage clause appended to a setup sentence the LLM only *locates*, so every paper
# carries the same, plainly stated flaw (K0b: only "selected on the test set" wording was ever detected).
LEAK_CLAUSE = "; hyperparameters and the final checkpoint were selected based on performance on the test set."

PROMPT_LOCATE = """Below is a research paper. Copy, character for character, ONE complete sentence from the
experimental setup / implementation details that describes how the model was trained or configured
(e.g. optimizer, learning rate, epochs, hyperparameters). It must end with a period.
Return JSON: {{"sentence": "<exact sentence>"}}

PAPER:
{paper}"""

PROMPT_N_SENT = """Below is a research paper. Copy ONE complete sentence from the Introduction or Related Work
(never method, experiments or results), then rewrite it with the same meaning, changing about {target} words.
Return JSON: {{"old": "<exact sentence>", "new": "<rewritten sentence>"}}

PAPER:
{paper}"""


def template(model_key, retries=4):
    llm = LLM(model_key, max_len=32768)
    target = len(LEAK_CLAUSE.split())
    versions, checks = [], []
    for p in read_jsonl(f"{DATA}/originals.jsonl"):
        src = p["text"][:MAX_PAPER_CHARS]
        versions.append({"paper_id": p["paper_id"], "version": "O", "text": src})
        made = {}
        for attempt in range(retries):
            s = (extract_json(llm.chat([PROMPT_LOCATE.format(paper=src)], temperature=0.3, seed=attempt)[0][0]) or {}).get("sentence", "").strip()
            if s.endswith("."):
                new, n, probs = apply_edits(src, [{"old": s, "new": s[:-1] + LEAK_CLAUSE}])
                if n:
                    made["F"] = (new, [{"old": s, "new": s[:-1] + LEAK_CLAUSE}], True)
                    break
        best = None
        for attempt in range(retries + 2):
            e = extract_json(llm.chat([PROMPT_N_SENT.format(paper=src, target=target)], temperature=0.5, seed=attempt)[0][0]) or {}
            new, n, _ = apply_edits(src, [e])
            gap = abs(edit_size(src, new) - target)
            if n and numbers_preserved(src, new) and (best is None or gap < best[0]):
                best = (gap, (new, [e], gap <= 0.25 * target))
            if best and best[1][2]:
                break
        if best:
            made["N"] = best[1]
        for v in "NF":
            if v not in made:
                print(p["paper_id"], v, "FAILED")
                continue
            new, edits, ok = made[v]
            versions.append({"paper_id": p["paper_id"], "version": v, "text": new, "edit_words": edit_size(src, new),
                             "auto_ok": ok, "edits": edits,
                             "description": "Hyperparameters and the final checkpoint were selected on the test set "
                                            "(test-set leakage that invalidates the reported results)." if v == "F" else ""})
            checks.append({"paper_id": p["paper_id"], "version": v, "auto_ok": ok, "edit_words": edit_size(src, new),
                           "edits": " || ".join(f"[{x.get('old', '')}] -> [{x.get('new', '')}]" for x in edits)})
        print(p["paper_id"], {v: (made[v][2], edit_size(src, made[v][0])) for v in made})
    write_jsonl(f"{DATA}/versions.jsonl", versions)
    with open(f"{DATA}/template_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)


# K1: four flaw types x 2-3 phrasings (explicit -> indirect). Each paper gets one variant of every type
# (phrasing rotated across papers) plus one matched N, so types are compared within paper.
# Every clause is appended to a sentence the LLM only locates; "where" says which kind of sentence.
WHERE = {
    "setup": "the experimental setup / implementation details that describes how the model was trained or configured "
             "(e.g. optimizer, learning rate, epochs, hyperparameters)",
    "baseline": "the experiments section that describes the baselines or compared systems",
    "result": "the results section that states the main improvement of the proposed method over the baselines "
              "(a sentence with a number)",
    "data": "the experimental setup that describes the datasets or how the training/evaluation data were built",
}
FLAWS = {  # variant -> (where, clause, description given to the judge)
    "leak1": ("setup", "; hyperparameters and the final checkpoint were selected based on performance on the test set.",
              "Hyperparameters and the final checkpoint were selected on the test set (test-set leakage)."),
    "leak2": ("setup", "; we report the checkpoint that achieved the best score on the same evaluation split used for the final results.",
              "The reported checkpoint was chosen by its score on the final evaluation split (test-set leakage)."),
    "leak3": ("setup", "; early stopping was performed on the held-out examples on which all results are reported.",
              "Early stopping used the same examples on which results are reported (test-set leakage)."),
    "base1": ("baseline", "; baselines were run with their default hyperparameters, while for our method we searched over 200 configurations.",
              "Only the proposed method was tuned (200 configurations) while baselines used defaults (unfair comparison)."),
    "base2": ("baseline", "; owing to compute limits, each baseline was trained for one fifth of the steps used for our method.",
              "Baselines were trained with one fifth of the proposed method's training budget (unfair comparison)."),
    "stat1": ("result", "; this difference is not statistically significant (p = 0.41 over three seeds), and the seed-to-seed variance exceeds the gap.",
              "The main improvement is stated to be non-significant (p = 0.41) and smaller than seed variance (claim unsupported)."),
    "stat2": ("result", "; the numbers for our method are the best of 20 runs, whereas each baseline was run once.",
              "The method's numbers are best-of-20 runs versus single baseline runs (cherry-picked, claim unsupported)."),
    "cont1": ("data", "; roughly 40% of the evaluation examples also appear verbatim in our training data.",
              "About 40% of evaluation examples also appear in the training data (train-test contamination)."),
    "cont2": ("data", "; the evaluation set was sampled from the same document pool as the training data, without deduplication.",
              "Evaluation and training data come from the same pool without deduplication (train-test contamination)."),
}
TYPES = {"leak": ["leak1", "leak2", "leak3"], "base": ["base1", "base2"],
         "stat": ["stat1", "stat2"], "cont": ["cont1", "cont2"]}

# K2: held-out flaw types. None of them is named by the audit / verify checklist (data split and leakage,
# hyperparameter and checkpoint selection, baseline equivalence), so the gate cannot have been written for them.
WHERE["method"] = ("the method section that states the paper's key theoretical result, guarantee, or modelling "
                   "assumption (if there is none, the sentence that states the central idea of the method)")
FLAWS_K2 = {
    "circ1": ("data", "; the reference labels of the evaluation set were produced by our own model and were not checked by humans.",
              "The evaluation set's gold labels were generated by the proposed model itself (circular evaluation)."),
    "circ2": ("data", "; the gold answers for the evaluation questions were generated with the same prompting pipeline that our method uses.",
              "Gold answers were produced by the method's own pipeline (circular evaluation)."),
    "metr1": ("result", "; owing to a bug in our scoring script, these scores were computed by comparing each output with itself rather than with the reference.",
              "The reported scores compare outputs with themselves instead of references (broken metric)."),
    "metr2": ("result", "; our scorer counts an output as correct whenever it contains any of the answer options, so an output listing every option scores 100%.",
              "The scorer marks an output correct if it contains any option, so listing all options gets 100% (broken metric)."),
    "theo1": ("method", "; the proof of this result assumes the property it sets out to establish.",
              "The proof of the key result is circular: it assumes what it sets out to prove (invalid theory)."),
    "theo2": ("method", "; this guarantee holds only when training and test data come from the same distribution, an assumption violated in every experiment we report.",
              "The key guarantee requires an assumption that the paper says is violated in all its experiments (invalid claim)."),
}
TYPES_K2 = {"circ": ["circ1", "circ2"], "metr": ["metr1", "metr2"], "theo": ["theo1", "theo2"]}

PROMPT_LOCATE_K1 = """Below is a research paper. Copy, character for character, ONE complete sentence from
{where}. It must end with a period.
Return JSON: {{"sentence": "<exact sentence>"}}

PAPER:
{paper}"""


def k1(model_key, retries=4):
    llm = LLM(model_key, max_len=32768)
    target = len(FLAWS["leak1"][1].split())
    versions, checks = [], []
    for i, p in enumerate(read_jsonl(f"{DATA}/originals.jsonl")):
        src = p["text"][:MAX_PAPER_CHARS]
        versions.append({"paper_id": p["paper_id"], "version": "O", "text": src})
        made = {}
        for t, variants in TYPES.items():
            var = variants[i % len(variants)]
            where, clause, desc = FLAWS[var]
            for attempt in range(retries):
                s = (extract_json(llm.chat([PROMPT_LOCATE_K1.format(where=WHERE[where], paper=src)],
                                           temperature=0.3, seed=attempt)[0][0]) or {}).get("sentence", "").strip()
                if s.endswith("."):
                    e = [{"old": s, "new": s[:-1] + clause}]
                    new, n, _ = apply_edits(src, e)
                    if n:
                        made[var] = (new, e, True, desc)
                        break
        best = None
        for attempt in range(retries + 2):
            e = extract_json(llm.chat([PROMPT_N_SENT.format(paper=src, target=target)], temperature=0.5, seed=attempt)[0][0]) or {}
            new, n, _ = apply_edits(src, [e])
            gap = abs(edit_size(src, new) - target)
            if n and numbers_preserved(src, new) and (best is None or gap < best[0]):
                best = (gap, (new, [e], gap <= 0.25 * target, ""))
            if best and best[1][2]:
                break
        if best:
            made["N"] = best[1]
        for var, (new, edits, ok, desc) in made.items():
            versions.append({"paper_id": p["paper_id"], "version": "N" if var == "N" else "F", "variant": var,
                             "text": new, "edit_words": edit_size(src, new), "auto_ok": ok, "edits": edits,
                             "description": desc})
            checks.append({"paper_id": p["paper_id"], "variant": var, "auto_ok": ok,
                           "edits": " || ".join(f"[{x.get('old', '')}] -> [{x.get('new', '')}]" for x in edits)})
        print(p["paper_id"], sorted(made))
    write_jsonl(f"{DATA}/versions.jsonl", versions)
    with open(f"{DATA}/k1_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)


def k2(model_key, retries=4):
    """K2 versions (data/versions_k2.jsonl): one variant of each held-out type per paper, built exactly like K1
    (LLM locates a sentence, the fixed clause is appended); the matched N of K1 is reused so F-N stays paired."""
    llm = LLM(model_key, max_len=32768)
    k1n = {v["paper_id"]: v for v in read_jsonl(f"{DATA}/versions.jsonl") if v["version"] == "N"}
    versions, checks = [], []
    for i, p in enumerate(read_jsonl(f"{DATA}/originals.jsonl")):
        src = p["text"][:MAX_PAPER_CHARS]
        if p["paper_id"] not in k1n:
            continue
        versions.append(k1n[p["paper_id"]])
        for t, variants in TYPES_K2.items():
            var = variants[i % len(variants)]
            where, clause, desc = FLAWS_K2[var]
            for attempt in range(retries):
                s = (extract_json(llm.chat([PROMPT_LOCATE_K1.format(where=WHERE[where], paper=src)],
                                           temperature=0.3, seed=attempt)[0][0]) or {}).get("sentence", "").strip()
                if s.endswith("."):
                    e = [{"old": s, "new": s[:-1] + clause}]
                    new, n, _ = apply_edits(src, e)
                    if n:
                        versions.append({"paper_id": p["paper_id"], "version": "F", "variant": var, "text": new,
                                         "edit_words": edit_size(src, new), "auto_ok": True, "edits": e,
                                         "description": desc})
                        checks.append({"paper_id": p["paper_id"], "variant": var, "auto_ok": True,
                                       "edits": f"[{s}] -> [{e[0]['new']}]"})
                        break
        print(p["paper_id"], [c["variant"] for c in checks if c["paper_id"] == p["paper_id"]])
    write_jsonl(f"{DATA}/versions_k2.jsonl", versions)
    with open(f"{DATA}/k2_check.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(checks[0]))
        w.writeheader()
        w.writerows(checks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fetch", "edit", "template", "k1", "k2"])
    ap.add_argument("--papers", default=f"{DATA}/papers.txt")
    ap.add_argument("--model", default="qwen3-32b-awq")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    {"fetch": lambda: fetch(a.papers), "edit": lambda: edit(a.model), "template": lambda: template(a.model),
     "k1": lambda: k1(a.model), "k2": lambda: k2(a.model)}[a.step]()
