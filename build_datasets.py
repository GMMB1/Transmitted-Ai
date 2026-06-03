#!/usr/bin/env python3
"""
build_datasets.py — Download & index the 3 public psychology datasets
used by the Arwanos Mental State Monitor ML pipeline.

Datasets downloaded (all public, no API key required):
  1. CounselChat       — nbertagnolli/counsel-chat      (Hugging Face)
  2. Mental Health     — Amod/mental_health_counseling_conversations
  3. ESConv            — thu-coai/esconv                (ACL 2021)

Output:
  data/psych_datasets_index.json   — 7,500+ entries, keyword-indexed

Usage:
  python build_datasets.py
"""

from __future__ import annotations
import json
import re
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
OUTPUT   = DATA_DIR / "psych_datasets_index.json"


# ── helpers ───────────────────────────────────────────────────────────────────

def _keywords(text: str) -> list[str]:
    """Extract meaningful keywords from text (stop-word filtered, lowercase)."""
    STOP = {
        "i","me","my","you","your","we","our","a","an","the","is","are","was",
        "be","been","being","have","has","had","do","did","does","will","would",
        "can","could","should","may","might","it","its","this","that","these",
        "those","and","or","but","if","in","on","at","to","of","for","with",
        "as","by","from","about","so","what","how","when","where","why","just",
        "not","no","yes","very","also","more","than","then","there","their",
        "they","them","he","she","him","her","his","hers","was","were","am",
    }
    words = re.findall(r"\b[a-z]{3,}\b", (text or "").lower())
    seen: dict[str, int] = {}
    for w in words:
        if w not in STOP:
            seen[w] = seen.get(w, 0) + 1
    return [w for w, _ in sorted(seen.items(), key=lambda x: -x[1])[:20]]


def _safe_str(val) -> str:
    return str(val or "").strip()


# ── dataset builders ──────────────────────────────────────────────────────────

def _build_counsel_chat(hf_data) -> list[dict]:
    """
    CounselChat — nbertagnolli/counsel-chat
    Fields used: questionTitle, questionText, answerText, topic
    """
    entries = []
    for row in hf_data:
        ctx  = (_safe_str(row.get("questionTitle")) + "\n" +
                _safe_str(row.get("questionText"))).strip()
        resp = _safe_str(row.get("answerText"))
        if not ctx or not resp:
            continue
        entries.append({
            "source":   "counsel_chat",
            "topic":    _safe_str(row.get("topic", "general")),
            "context":  ctx[:1200],
            "response": resp[:1200],
            "keywords": _keywords(ctx + " " + resp),
        })
    return entries


def _build_mental_health(hf_data) -> list[dict]:
    """
    Mental Health Counseling Conversations — Amod/mental_health_counseling_conversations
    Fields used: Context, Response
    """
    entries = []
    for row in hf_data:
        ctx  = _safe_str(row.get("Context"))
        resp = _safe_str(row.get("Response"))
        if not ctx or not resp:
            continue
        entries.append({
            "source":   "mental_health_counseling",
            "topic":    "mental_health",
            "context":  ctx[:1200],
            "response": resp[:1200],
            "keywords": _keywords(ctx + " " + resp),
        })
    return entries


def _build_esconv(hf_data) -> list[dict]:
    """
    ESConv (Emotional Support Conversations) — thu-coai/esconv
    Each entry is a multi-turn dialogue; we extract the first supporter response.
    Fields used: dialog (list of turns with 'role' and 'content')
    """
    entries = []
    for row in hf_data:
        dialog = row.get("dialog") or row.get("conversation") or []
        if not dialog:
            # flat format: seeker / supporter fields
            ctx  = _safe_str(row.get("seeker_post") or row.get("context"))
            resp = _safe_str(row.get("response") or row.get("supporter_post"))
            if ctx and resp:
                entries.append({
                    "source":   "esconv",
                    "topic":    _safe_str(row.get("emotion_type", "emotional_support")),
                    "context":  ctx[:1200],
                    "response": resp[:1200],
                    "keywords": _keywords(ctx + " " + resp),
                })
            continue

        # multi-turn format
        seeker_lines   = []
        supporter_resp = ""
        for turn in dialog:
            role    = _safe_str(turn.get("role") or turn.get("speaker", ""))
            content = _safe_str(turn.get("content") or turn.get("text", ""))
            if not content:
                continue
            if "seeker" in role.lower() or "user" in role.lower():
                seeker_lines.append(content)
            elif "supporter" in role.lower() or "assistant" in role.lower():
                if not supporter_resp:
                    supporter_resp = content

        ctx = " ".join(seeker_lines[:3])
        if ctx and supporter_resp:
            entries.append({
                "source":   "esconv",
                "topic":    _safe_str(row.get("emotion_type", "emotional_support")),
                "context":  ctx[:1200],
                "response": supporter_resp[:1200],
                "keywords": _keywords(ctx + " " + supporter_resp),
            })
    return entries


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not installed.")
        print("Run:  pip install datasets")
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)
    all_entries: list[dict] = []

    # ── 1. CounselChat ────────────────────────────────────────────────────────
    print("Downloading CounselChat (nbertagnolli/counsel-chat)…", flush=True)
    try:
        ds = load_dataset("nbertagnolli/counsel-chat", split="train",
                          trust_remote_code=True)
        entries = _build_counsel_chat(ds)
        all_entries.extend(entries)
        print(f"  ✓ {len(entries)} entries")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # ── 2. Mental Health Counseling ───────────────────────────────────────────
    print("Downloading Mental Health Counseling Conversations…", flush=True)
    try:
        ds = load_dataset("Amod/mental_health_counseling_conversations",
                          split="train", trust_remote_code=True)
        entries = _build_mental_health(ds)
        all_entries.extend(entries)
        print(f"  ✓ {len(entries)} entries")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    # ── 3. ESConv ─────────────────────────────────────────────────────────────
    print("Downloading ESConv (thu-coai/esconv)…", flush=True)
    try:
        ds = load_dataset("thu-coai/esconv", split="train",
                          trust_remote_code=True)
        entries = _build_esconv(ds)
        all_entries.extend(entries)
        print(f"  ✓ {len(entries)} entries")
    except Exception as e:
        print(f"  ✗ Failed: {e}")

    if not all_entries:
        print("\nERROR: No entries were built. Check your internet connection.")
        sys.exit(1)

    # ── Write index ───────────────────────────────────────────────────────────
    OUTPUT.write_text(json.dumps(all_entries, ensure_ascii=False, indent=2),
                      encoding="utf-8")

    print(f"\n✅ Done — {len(all_entries):,} total entries")
    print(f"   Saved → {OUTPUT}")
    print("\nBreakdown by source:")
    from collections import Counter
    for src, count in Counter(e["source"] for e in all_entries).items():
        print(f"   {src:<35} {count:>5}")


if __name__ == "__main__":
    main()
