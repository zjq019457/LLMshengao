"""Clean ICLR 2026 test split (after every model's training data: OpenReviewer 2022-2025, DeepReviewer 2025, Qwen3 2025).
Runs on the local machine (the server has no internet).

  python make_iclr26.py --dh <davidheineman/iclr-2026 parquet> --pr <P0u4a/paper_reviews data dir> --out data_private/bench_iclr26.jsonl
Labels: ratings from the pre-decision OpenReview snapshot (davidheineman/iclr-2026); accepted = title in the 2026 rows
of P0u4a/paper_reviews (accepted papers only); rejected = reviewed (>= 3 ratings), not withdrawn, not accepted.
Text: both classes come from the same source to avoid a format cue for the label: the arXiv v1 of the paper (exact
normalised title match), and only if v1 was posted before DECISION_CUTOFF, so no camera-ready or post-decision
revision is used. ICLR running headers ("Under review ... / Published as ...") and arXiv stamps are removed.
Length window as the other splits (17k-84k characters). Sampling: random order (seed 0), 100 accepted + 100 rejected.
"""
import argparse
import glob
import json
import re
import time
import urllib.request

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pymupdf

DECISION_CUTOFF = "2026-01-15"
# v1 must be posted after the ICLR 2026 deadline window opened: an older v1 is an earlier version of a resubmission
# (e.g. an ICLR 2025 paper) and could be in OpenReviewer's 2022-2025 training data
SUBMISSION_START = "2025-09-15"
HEADER = re.compile(r"(under review as a conference paper|published as a conference paper|iclr 2026|^arxiv:\S+)", re.I)


def norm(t):
    return re.sub(r"[^a-z0-9]", "", str(t).lower())


def accepted_titles(pr_dir):
    out = set()
    for f in sorted(glob.glob(f"{pr_dir}/iclr-*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["year", "title"]).to_pandas()
            out |= {norm(x) for x in t[t.year == 2026].title}
    return out


def arxiv_index(arxiv_dir, wanted):
    """normalised title -> (arXiv id, v1 date YYYY-MM-DD), from the arXiv metadata snapshot (CC0), for wanted titles."""
    out = {}
    for f in sorted(glob.glob(f"{arxiv_dir}/train-*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["id", "title", "versions"]).to_pandas()
            t["k"] = t.title.map(norm)
            for r in t[t.k.isin(wanted)].itertuples():
                v1 = pd.to_datetime(r.versions[0]["created"]).strftime("%Y-%m-%d")
                out[r.k] = (r.id, v1)
    return out


def pdf_text(aid, title):
    """arXiv v1 text, anonymised like a submission: title + everything from the abstract on (author block dropped),
    acknowledgements section dropped, running headers removed. None if no abstract heading is found."""
    data = urllib.request.urlopen(urllib.request.Request(f"https://arxiv.org/pdf/{aid}v1",
                                                         headers={"User-Agent": "research-script"}), timeout=120).read()
    doc = pymupdf.open(stream=data, filetype="pdf")
    lines = [ln for p in doc for ln in p.get_text().splitlines() if not HEADER.search(ln.strip())]
    start = next((i for i, ln in enumerate(lines[:200]) if norm(ln) == "abstract"), None)
    if start is None:
        return None
    lines = [title] + lines[start:]
    ack = next((i for i, ln in enumerate(lines) if re.fullmatch(r"(\d+\s*)?acknowledge?ments?", ln.strip(), re.I)), None)
    if ack is not None:
        end = next((i for i in range(ack + 1, len(lines)) if re.fullmatch(r"(\d+\s*)?references", lines[i].strip(), re.I)), ack + 1)
        lines = lines[:ack] + lines[end:]
    return "\n".join(lines)


def main(a):
    acc = accepted_titles(a.pr)
    d = pd.read_parquet(a.dh, columns=["title", "venue", "reviews", "url"])
    d = d[~d.venue.str.contains("Withdrawn")]
    d["ratings"] = d.reviews.map(lambda rs: [r["rating"] for r in rs if r.get("rating") is not None])
    d = d[d.ratings.map(len) >= 3]
    d["accepted"] = d.title.map(norm).isin(acc)
    print(f"{len(d)} reviewed submissions, {d.accepted.sum()} accepted")
    idx = arxiv_index(a.arxiv, set(d.title.map(norm)))
    d["hit"] = d.title.map(lambda t: idx.get(norm(t)))
    d = d[d.hit.map(lambda h: h is not None and SUBMISSION_START <= h[1] < DECISION_CUTOFF)]
    print(f"with arXiv v1 in [{SUBMISSION_START}, {DECISION_CUTOFF}): {d.accepted.sum()} accepted, {(~d.accepted).sum()} rejected")
    d = d.sample(frac=1, random_state=0)
    want, got, rows, tried = a.per_class, {True: 0, False: 0}, [], 0
    for r in d.itertuples():
        if got[r.accepted] >= want:
            if all(v >= want for v in got.values()):
                break
            continue
        tried += 1
        time.sleep(3)  # arXiv download etiquette
        hit = r.hit
        try:
            text = pdf_text(hit[0], r.title)
        except Exception as e:
            print("pdf fail", hit[0], repr(e)[:100])
            continue
        if text is None or not 17000 <= len(text) <= 84000:
            continue
        got[r.accepted] += 1
        rows.append({"paper_id": r.url.split("id=")[-1], "version": "O", "variant": None, "text": text,
                     "human_ratings": [float(x) for x in r.ratings],
                     "decision": "Accept" if r.accepted else "Reject", "arxiv": hit[0], "arxiv_v1_date": hit[1]})
        print(f"[{tried}] acc={got[True]} rej={got[False]} {hit[0]} {hit[1]} {len(text)} chars", flush=True)
        with open(a.out, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"done: {got} from {tried} tried; mean rating acc {np.mean([np.mean(x['human_ratings']) for x in rows if x['decision'] == 'Accept']):.2f}"
          f" rej {np.mean([np.mean(x['human_ratings']) for x in rows if x['decision'] == 'Reject']):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dh", required=True)
    ap.add_argument("--pr", required=True)
    ap.add_argument("--arxiv", required=True, help="directory of the librarian-bots/arxiv-metadata-snapshot parquet files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-class", type=int, default=100)
    main(ap.parse_args())
