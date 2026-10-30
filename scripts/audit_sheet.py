"""Manual precision audit of the gate on real (un-injected) papers: every verified fatal issue on test25 + test24.

  python audit_sheet.py   -> paper/gate_audit_sheet.csv, paper/gate_audit_sheet.md   (local use only: contains paper quotes)
One row per (paper, verified issue): the issue as the audit sample raised it, the checker's quote and reason, how many
of the audit samples raised a verified issue on that paper (votes; gate_k1 fires at >= 1, gate_k2 at >= 2) and the
paper's ICLR decision. Rows are shuffled with a fixed seed so the annotator does not see decisions in blocks;
the decision column can be hidden while labelling.
Label columns (filled by hand): is_real (1 the paper really has this problem, 0 it does not / misread),
is_fatal (1 it would invalidate a main result, 0 minor or presentational), note.
"""
import numpy as np
import pandas as pd

from common import ROOT, read_jsonl
from fusion import P
from gate import norm, run_checks

SPLITS = {"test25": ("_bench", "bench_iclr25"), "test24": ("_bench24", "bench_iclr24")}


def main():
    rows = []
    for split, (tag, data) in SPLITS.items():
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
        ver = {}
        for ((p, _, _), i), j in res.items():
            q = norm(str(j.get("quote") or ""))
            if j.get("fatal") and j.get("reported") and len(q) > 20 and q in norm(papers[p]["text"]):
                ver.setdefault(p, []).append((i, j))
        for p, items in ver.items():
            votes = len(set().union(*(per[p][i] for i, _ in items)))
            for i, j in items:
                rows.append({"split": split, "paper_id": p, "votes": votes, "fires_k2": votes >= 2,
                             "decision": papers[p]["decision"], "mean_human_rating": round(np.mean(papers[p]["human_ratings"]), 2),
                             "issue": i, "paper_quote": j.get("quote"),
                             "checker_reason": j.get("reason") or j.get("explanation") or "",
                             "is_real": "", "is_fatal": "", "note": ""})
    t = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    t.insert(0, "row", range(1, len(t) + 1))
    t.to_csv(f"{ROOT}/private/gate_audit_sheet.csv", index=False, encoding="utf-8-sig")
    with open(f"{ROOT}/private/gate_audit_sheet.md", "w", encoding="utf-8") as f:
        f.write(f"# Gate precision audit ({len(t)} verified issues on {t.paper_id.nunique()} papers)\n\n"
                "Label in the CSV: is_real (1/0), is_fatal (1/0). Open the paper on OpenReview by its id if the quote "
                "is not enough.\n\n")
        for _, r in t.iterrows():
            f.write(f"## {r.row}. {r.split} · {r.paper_id} · votes {r.votes}\n"
                    f"https://openreview.net/forum?id={r.paper_id}\n\n**Issue:** {r.issue}\n\n"
                    f"**Paper quote:** \"{r.paper_quote}\"\n\n**Checker:** {r.checker_reason}\n\n")
    print(f"{len(t)} rows, {t.paper_id.nunique()} papers, fires_k2 papers: "
          f"{t[t.fires_k2].paper_id.nunique()}, accepted share of papers: "
          f"{t.drop_duplicates('paper_id').decision.str.contains('ccept').mean():.2f}")


if __name__ == "__main__":
    main()
