"""Quality-benchmark splits from DeepReview-13K test sets (data/test_2025.csv, data/test_2024.csv; HF licence, not redistributed).

  python make_bench.py
  -> data/bench_dev25.jsonl   ICLR 2025, 90 accept / 90 reject, disjoint from bench_iclr25 (seed 1): gate settings are chosen here
     data/bench_iclr24.jsonl  ICLR 2024, 100 accept / 100 reject (seed 0): second held-out test year
data/bench_iclr25.jsonl (the main test split, 100 / 100, seed 0) was built the same way and is kept as is.
Rows: paper_id, version "O", text (the paper as given to DeepReviewer), human_ratings, decision.
"""
import json

import pandas as pd

from common import ROOT, read_jsonl, write_jsonl


def rows(df):
    return [{"paper_id": r.id, "version": "O", "variant": None, "text": json.loads(r.inputs)[-1]["content"],
             "human_ratings": json.loads(r.rating), "decision": r.decision} for r in df.itertuples()]


def fits(df):
    # same length window as the test split (17.6k-83.4k chars): drops truncated extractions and papers over the 32k context
    n = df.inputs.map(lambda x: len(json.loads(x)[-1]["content"]))
    return df[(n >= 17000) & (n <= 84000)]


def pick(df, n, seed):
    return pd.concat([df[df.decision.str.contains("ccept")].sample(n, random_state=seed),
                      df[~df.decision.str.contains("ccept")].sample(n, random_state=seed)])


def main():
    test = {r["paper_id"] for r in read_jsonl(f"{ROOT}/data/bench_iclr25.jsonl")}
    d25 = pd.read_csv(f"{ROOT}/data/test_2025.csv").drop_duplicates("id")
    d25 = fits(d25[~d25.id.isin(test)])
    write_jsonl(f"{ROOT}/data/bench_dev25.jsonl", rows(pick(d25, 90, 1)))
    d24 = fits(pd.read_csv(f"{ROOT}/data/test_2024.csv").drop_duplicates("id"))
    write_jsonl(f"{ROOT}/data/bench_iclr24.jsonl", rows(pick(d24, 100, 0)))
    for s in ("bench_dev25", "bench_iclr24"):
        r = read_jsonl(f"{ROOT}/data/{s}.jsonl")
        print(s, len(r), sum("ccept" in x["decision"] for x in r), "accepted")


if __name__ == "__main__":
    main()
