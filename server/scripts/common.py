"""Shared helpers for K0: vLLM wrapper, JSON extraction, IO."""
import json
import os
import re

ROOT = os.environ.get("K0_ROOT", "/diskE/jiaqizhang/LLM_REVIEW_AUDIT_K0")
MODELS = {
    "qwen3-8b": "/diskF/jiaqizhang/stepkv/models/Qwen3-8B",  # shared with StepKV: read-only
    "qwen3-32b-awq": f"{ROOT}/models/Qwen3-32B-AWQ",
    "deepreviewer-7b": f"{ROOT}/models/DeepReviewer-7B",  # gated licence: research use, not official review
    "deepreviewer-14b": f"{ROOT}/models/DeepReviewer-14B",
    "openreviewer-8b": f"{ROOT}/models/Llama-OpenReviewer-8B",  # Llama 3.1 licence
    "cyclereviewer-8b": f"{ROOT}/models/CycleReviewer-ML-Llama-3.1-8B",  # gated licence: research use, not official review
    # base models of the fine-tuned reviewers (controls for "does review fine-tuning change detection"); /diskE is full
    "llama31-8b-instruct": "/diskF/jiaqizhang/models/Llama-3.1-8B-Instruct",  # base of OpenReviewer / CycleReviewer
    "qwen25-7b-instruct": "/diskF/jiaqizhang/models/Qwen2.5-7B-Instruct",  # base of DeepReviewer-7B
    "r1-llama-8b": "/diskF/jiaqizhang/models/DeepSeek-R1-Distill-Llama-8B",  # existing local copy, read-only
}
MAX_PAPER_CHARS = 90_000  # ~22k tokens; keeps prompt + 3k output under 32k


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_json(text):
    """Return the last parseable {...} or [...] block in text, else None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if m:
        text = m.group(1)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        end = text.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


class LLM:
    """Thin vLLM chat wrapper. Thinking disabled (K0 measures the default fast path)."""

    def __init__(self, model_key, gpu_mem=float(os.environ.get("K0_GPU_MEM", 0.38)), max_len=32768):
        # GPUs are shared with StepKV jobs whose memory fluctuates: small fixed budget,
        # eager mode (CUDA-graph capture OOMs when neighbours grow).
        from vllm import LLM as _VLLM
        self.llm = _VLLM(model=MODELS[model_key], max_model_len=max_len,
                         gpu_memory_utilization=gpu_mem, enable_prefix_caching=True,
                         enforce_eager=True, trust_remote_code=True)
        self.tok = self.llm.get_tokenizer()

    def chat(self, prompts, temperature=0.7, n=1, max_tokens=3000, seed=0, system=None, think=False):
        from vllm import SamplingParams
        sys_msg = [{"role": "system", "content": system}] if system else []
        texts = [self.tok.apply_chat_template(
            sys_msg + [{"role": "user", "content": p}], tokenize=False,
            add_generation_prompt=True, enable_thinking=think) for p in prompts]
        sp = SamplingParams(temperature=temperature, top_p=0.95, n=n,
                            max_tokens=max_tokens, seed=seed)
        outs = self.llm.generate(texts, sp)
        return [[c.text for c in o.outputs] for o in outs]

    def chat_prefill(self, prompts, prefill, temperature=0.7, n=1, max_tokens=3000, seed=0, system=None):
        """Like chat, but the assistant turn starts with `prefill` (the template's end-of-turn tokens are cut, as
        CycleReviewer's model card does with [:-4]); returned texts include the prefill."""
        from vllm import SamplingParams
        sys_msg = [{"role": "system", "content": system}] if system else []
        texts = []
        for p in prompts:
            # the template strips whitespace around message content, so render a marker and splice the prefill in
            t = self.tok.apply_chat_template(sys_msg + [{"role": "user", "content": p},
                                                        {"role": "assistant", "content": "PREFILL_MARKER"}], tokenize=False)
            texts.append(t[:t.rindex("PREFILL_MARKER")] + prefill)
        sp = SamplingParams(temperature=temperature, top_p=0.95, n=n, max_tokens=max_tokens, seed=seed)
        outs = self.llm.generate(texts, sp)
        return [[prefill + c.text for c in o.outputs] for o in outs]
