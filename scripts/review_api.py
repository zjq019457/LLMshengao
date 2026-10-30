"""K0c closed-model reviewers (runs on the local machine: the server cannot reach these APIs).

  python review_api.py --provider anthropic --list-models
  python review_api.py --provider anthropic --model <id> --cfg verify+rule \
      --price-in 4 --price-out 20 --max-usd 15
  -> out_api/reviews_<provider>_<model>_<cfg>.jsonl   (same format as review.py; scp to the server's out/ and run judge.py)

Keys come from the environment only: ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY.
cfg: base | verify | verify+rule   (closed models reason by default, so there is no separate 'think' switch)
Prices are passed in by the user ($ per 1M tokens, output price applied to reasoning tokens too); the run stops
before a request that could push the estimated spend past --max-usd. Re-running resumes: finished
(paper, version, sample) triples are skipped. No refusal fallback to another model: a fallback would change the
model under test, so refusals are recorded as unparsed reviews instead.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from common import extract_json, read_jsonl
from review import FORM, RULE, VERIFY

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_OUT = 16000


def make_caller(provider, model):
    """Return f(prompt) -> (text, input_tokens, output_tokens, stop)."""
    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic(max_retries=5)

        def call(prompt):
            # Opus 5.5: thinking is always on (adaptive); depth set by effort. Streaming for long outputs.
            with client.messages.stream(model=model, max_tokens=MAX_OUT, output_config={"effort": "high"},
                                        messages=[{"role": "user", "content": prompt}]) as s:
                m = s.get_final_message()
            text = "".join(b.text for b in m.content if b.type == "text")
            return text, m.usage.input_tokens, m.usage.output_tokens, m.stop_reason
        return call
    if provider == "openai":
        import openai
        client = openai.OpenAI(max_retries=5)

        def call(prompt):
            r = client.responses.create(model=model, input=prompt, reasoning={"effort": "high"},
                                        max_output_tokens=MAX_OUT)
            return r.output_text, r.usage.input_tokens, r.usage.output_tokens, r.status
        return call
    if provider == "gemini":
        from google import genai
        from google.genai import types
        client = genai.Client()

        def call(prompt):
            r = client.models.generate_content(model=model, contents=prompt,
                                               config=types.GenerateContentConfig(max_output_tokens=MAX_OUT))
            u = r.usage_metadata
            out = (u.candidates_token_count or 0) + (u.thoughts_token_count or 0)
            fin = r.candidates[0].finish_reason if r.candidates else None
            return r.text or "", u.prompt_token_count or 0, out, str(fin)
        return call
    raise ValueError(provider)


def list_models(provider):
    if provider == "anthropic":
        import anthropic
        ids = [m.id for m in anthropic.Anthropic().models.list()]
    elif provider == "openai":
        import openai
        ids = sorted(m.id for m in openai.OpenAI().models.list())
    else:
        from google import genai
        ids = [m.name for m in genai.Client().models.list()]
    print("\n".join(ids))


def main(a):
    if a.list_models:
        return list_models(a.provider)
    opts = set(a.cfg.split("+")) - {"base"}
    prefix = (VERIFY if "verify" in opts else "") + (RULE if "rule" in opts else "")
    rows = [r for r in read_jsonl(a.versions) if r["version"] in a.keep]
    if a.limit:  # balanced subset: first `limit` accepted + first `limit` rejected papers (file order is random)
        acc = [r for r in rows if "ccept" in r.get("decision", "")][:a.limit]
        rows = acc + [r for r in rows if "ccept" not in r.get("decision", "")][:a.limit]
    tag = f"{a.provider}:{a.model}:{a.cfg}"
    out_dir = os.path.join(HERE, "out_api" + a.out_suffix)
    out_path = os.path.join(out_dir, f"reviews_{a.provider}_{a.model.replace('/', '-')}_{a.cfg}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    prev = read_jsonl(out_path) if os.path.exists(out_path) else []
    done = {(r["paper_id"], r["version"], r["sample"]) for r in prev if not r.get("error")}  # errored calls are retried
    spent = sum(r.get("usd", 0) for r in prev)
    jobs = [(r, k) for r in rows for k in range(a.n) if (r["paper_id"], r["version"], k) not in done]
    worst = (30_000 * a.price_in + MAX_OUT * a.price_out) / 1e6  # upper bound for one request
    print(f"{tag}: {len(jobs)} requests to go, ${spent:.2f} already spent, cap ${a.max_usd:.2f}")
    call = make_caller(a.provider, a.model)

    def one(job):
        r, k = job
        t0 = time.time()
        try:
            text, tin, tout, stop = call(prefix + FORM + r["text"])
        except Exception as e:  # recorded, not retried here (the SDKs already retry transient errors)
            return {"paper_id": r["paper_id"], "version": r["version"], "model": tag, "sample": k,
                    "review": None, "raw": "", "error": repr(e)[:500], "usd": 0.0}
        usd = (tin * a.price_in + tout * a.price_out) / 1e6
        return {"paper_id": r["paper_id"], "version": r["version"], "model": tag, "sample": k,
                "review": extract_json(text), "raw": text, "stop": str(stop), "in_tok": tin, "out_tok": tout,
                "usd": usd, "sec": round(time.time() - t0, 1)}

    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as pool:
        for i in range(0, len(jobs), a.workers):
            if spent + worst * a.workers > a.max_usd:
                print(f"stopping: next batch could exceed the ${a.max_usd:.2f} cap (spent ${spent:.2f})")
                break
            batch = list(pool.map(one, jobs[i:i + a.workers]))
            if all(r.get("error") for r in batch) and i == 0:
                print("first batch failed entirely (check the key / model id):", batch[0]["error"][:300])
                break
            for res in batch:
                spent += res["usd"]
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                f.flush()
                print(f"  {res['paper_id']} {res['version']} s{res['sample']} parsed={res['review'] is not None}"
                      f" ${spent:.2f}" + (f" ERROR {res['error'][:120]}" if res.get("error") else ""))
    print(f"done: ${spent:.2f} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, choices=["anthropic", "openai", "gemini"])
    ap.add_argument("--model")
    ap.add_argument("--cfg", default="verify", choices=["base", "verify", "verify+rule"])
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--price-in", type=float, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, help="$ per 1M output (incl. reasoning) tokens")
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--versions", default=os.path.join(HERE, "versions_k0c.jsonl"))
    ap.add_argument("--keep", default="NF", help="versions to review: NF for K1 / K0c files, O for benchmark splits")
    ap.add_argument("--limit", type=int, default=0, help="per-decision cap for benchmark splits (0 = all rows)")
    ap.add_argument("--out-suffix", default="", help="output directory suffix, e.g. _k1 or _bench -> out_api_k1/")
    ap.add_argument("--list-models", action="store_true")
    a = ap.parse_args()
    if not a.list_models and (a.model is None or a.price_in is None or a.price_out is None):
        ap.error("--model, --price-in and --price-out are required (look prices up on the provider's pricing page)")
    main(a)
