#!/usr/bin/env python3
"""
eval_models.py — compare local Ollama models on the tasks Arwanos actually does.

Why: the default model is the single biggest lever on answer quality, and the
weak spots (loose JSON, vague accountability questions, Arabic) are model-bound.
This scores each candidate on Arwanos-shaped tasks so you can pick with evidence,
then switch by setting "model_name" in config.json.

Usage:
    python eval_models.py                      # compares the DEFAULT set below
    python eval_models.py llama3:8b qwen2.5:7b # compare specific models
"""
import sys
import time
import json
import re

import ollama

DEFAULT_MODELS = ["llama3:8b-instruct-q4_K_M", "qwen2.5:7b"]
NUM_CTX = 8192


def _gen(model: str, prompt: str, num_predict: int = 400):
    t0 = time.perf_counter()
    res = ollama.generate(
        model=model, prompt=prompt,
        options={"num_ctx": NUM_CTX, "num_predict": num_predict, "temperature": 0.2},
        keep_alive="10m",
    )
    dt = time.perf_counter() - t0
    text = res.response if not isinstance(res, dict) else res.get("response", "")
    n_tok = 0
    try:
        n_tok = int(getattr(res, "eval_count", 0) or (res.get("eval_count", 0) if isinstance(res, dict) else 0))
    except Exception:
        pass
    return (text or "").strip(), dt, n_tok


# ── Task 1: strict JSON array (monitor question generation) ─────────────────
def task_json(model):
    prompt = (
        "Output ONLY a JSON array of exactly 3 objects, each {\"id\":\"qN\",\"text\":\"...\","
        "\"category\":\"emotional_awareness\"}. No prose, no markdown.\n"
        "Generate 3 short self-reflection questions about exam stress."
    )
    out, dt, _ = _gen(model, prompt, 400)
    m = re.search(r"\[.*\]", out, re.DOTALL)
    valid = False
    try:
        arr = json.loads(m.group()) if m else None
        valid = isinstance(arr, list) and len(arr) == 3 and all(
            isinstance(x, dict) and "text" in x for x in arr)
    except Exception:
        valid = False
    # "clean" = started straight with '[' (no "Here is the JSON:" preamble).
    # The monitor tolerates preamble via regex, but clean output is the real goal.
    clean = out.strip().startswith("[")
    return {"pass": valid and clean, "valid_json": valid, "time": dt, "sample": out[:120]}


# ── Task 2: accountability follow-up phrasing ───────────────────────────────
def task_accountability(model):
    prompt = (
        "Last session GMM committed to this ACTION STEP: "
        "\"Message Sara tonight and say one honest sentence about the exam stress.\"\n"
        "Write ONE short question (max 25 words) that plainly asks whether GMM actually "
        "did it and what got in the way. Output only the question."
    )
    out, dt, _ = _gen(model, prompt, 120)
    low = out.lower()
    # a good accountability check asks whether it was done (2nd OR 3rd person) + names Sara
    did = bool(re.search(r"\bdid (you|gmm|they|he)\b", low)) or \
        any(w in low for w in ("were you able", "have you", "manage to", "were you"))
    refs = "sara" in low
    obstacle = any(w in low for w in ("what", "if not", "stop", "get in the way",
                                      "prevent", "hold you back", "reason"))
    return {"pass": did and refs, "asks_obstacle": obstacle, "time": dt, "sample": out[:140]}


# ── Task 3: concise factual answer (normal chat) ────────────────────────────
def task_concise(model):
    prompt = ("In no more than 2 sentences, explain what a race condition is. "
              "Be direct, no preamble.")
    out, dt, _ = _gen(model, prompt, 160)
    sents = len(re.findall(r"[.!?]+", out))
    concise = sents <= 3 and len(out) < 400
    on_topic = "race" in out.lower() or "concurren" in out.lower() or "thread" in out.lower()
    return {"pass": concise and on_topic, "time": dt, "sample": out[:140]}


# ── Task 4: Arabic (bilingual — /ap and RTL) ────────────────────────────────
def task_arabic(model):
    prompt = "أجب بجملة واحدة فقط بالعربية: ما هي فائدة النوم الجيد للصحة؟"
    out, dt, _ = _gen(model, prompt, 160)
    ar_chars = len(re.findall(r"[؀-ۿ]", out))
    total = len(re.sub(r"\s", "", out)) or 1
    mostly_ar = ar_chars / total > 0.6
    # penalise CJK/latin contamination (a known llama3 Arabic failure)
    contamination = bool(re.search(r"[一-鿿]", out))
    return {"pass": mostly_ar and not contamination, "arabic_ratio": round(ar_chars / total, 2),
            "time": dt, "sample": out[:80]}


TASKS = [
    ("JSON format", task_json),
    ("Accountability Q", task_accountability),
    ("Concise answer", task_concise),
    ("Arabic", task_arabic),
]


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"\nComparing: {', '.join(models)}   (num_ctx={NUM_CTX})\n" + "=" * 66)
    scores = {m: 0 for m in models}
    times = {m: 0.0 for m in models}
    for name, fn in TASKS:
        print(f"\n### {name}")
        for m in models:
            try:
                r = fn(m)
            except Exception as e:
                print(f"  {m:32} ERROR: {e}")
                continue
            scores[m] += 1 if r.get("pass") else 0
            times[m] += r.get("time", 0)
            flag = "✅" if r.get("pass") else "❌"
            extra = " ".join(f"{k}={v}" for k, v in r.items()
                             if k not in ("pass", "time", "sample"))
            print(f"  {m:32} {flag}  {r.get('time',0):4.1f}s  {extra}")
            print(f"       └─ {r.get('sample','')!r}")
    print("\n" + "=" * 66 + "\nRESULT")
    for m in models:
        print(f"  {m:32} score {scores[m]}/{len(TASKS)}  ·  {times[m]:4.1f}s total")
    best = max(models, key=lambda m: (scores[m], -times[m]))
    print(f"\n  → Recommended: {best}")
    print(f"    Switch by setting \"model_name\": \"{best}\" in config.json\n")


if __name__ == "__main__":
    main()
