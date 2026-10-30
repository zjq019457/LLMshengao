"""Fit bench_iclr26 papers into the reviewers' 32k context: cut each text at its last line break before
MAX_TOK tokens (Qwen3 tokenizer, the densest of the three; Llama/Qwen2.5 counts are within a few percent).
The arXiv PDF text keeps references and appendix, so long papers lose their tail (appendix first).
Original kept as data/bench_iclr26_full.jsonl.

  python fit26.py
"""
import os
import shutil

from transformers import AutoTokenizer

from common import MODELS, ROOT, read_jsonl, write_jsonl

MAX_TOK = 24_000  # leaves room for the longest system prompt (OpenReviewer, ~1.5k) and 4-8k output


def main():
    src, full = f"{ROOT}/data/bench_iclr26.jsonl", f"{ROOT}/data/bench_iclr26_full.jsonl"
    if not os.path.exists(full):
        shutil.copy(src, full)
    tok = AutoTokenizer.from_pretrained(MODELS["qwen3-32b-awq"])
    rows, cut = read_jsonl(full), 0
    for r in rows:
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        if len(ids) > MAX_TOK:
            t = tok.decode(ids[:MAX_TOK])
            r["text"] = t[:t.rfind("\n")]
            cut += 1
    write_jsonl(src, rows)
    print(f"{cut}/{len(rows)} papers cut to {MAX_TOK} tokens")


if __name__ == "__main__":
    main()
