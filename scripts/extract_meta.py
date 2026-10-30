"""AC meta-reviews and human reviews for the benchmark papers, from P0u4a/paper_reviews (CC-BY-4.0, OpenReview content).
Runs locally (the server has no internet).

  python extract_meta.py --pr <data dir> --ids data_private/bench_ids.txt --out data_private/bench_meta.jsonl
Rows: split, paper_id, decision, meta_review (AC decision comment), reviews: [{rating, weaknesses}].
"""
import argparse
import glob
import json

import pyarrow.parquet as pq


def main(a):
    split = {}
    for ln in open(a.ids, encoding="utf-8"):
        s, p = ln.split()
        split[p] = s
    out = {}
    for f in sorted(glob.glob(f"{a.pr}/iclr-*.parquet")):
        pf = pq.ParquetFile(f)
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g, columns=["forum_id", "decision", "decision_comment", "reviews_json"]).to_pandas()
            for r in t[t.forum_id.isin(split)].itertuples():
                revs = json.loads(r.reviews_json or "[]")
                out[r.forum_id] = {"split": split[r.forum_id], "paper_id": r.forum_id, "decision": r.decision,
                                   "meta_review": r.decision_comment or "",
                                   "reviews": [{"rating": x.get("rating"), "weaknesses": x.get("weaknesses") or ""} for x in revs]}
    with open(a.out, "w", encoding="utf-8") as fo:
        for v in out.values():
            fo.write(json.dumps(v, ensure_ascii=False) + "\n")
    print(f"{len(out)}/{len(split)} papers matched; with meta-review: {sum(bool(v['meta_review']) for v in out.values())}; "
          f"with review weaknesses: {sum(any(x['weaknesses'] for x in v['reviews']) for v in out.values())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--out", required=True)
    main(ap.parse_args())
