#!/usr/bin/env python3
"""
Arwanos Maintenance Tool
Run: python3 maintenance.py
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent / "data"
BACKUP_DIR = Path(__file__).parent / "data" / ".trash"

# ── helpers ───────────────────────────────────────────────────────────────────

def size_str(path: Path) -> str:
    b = path.stat().st_size
    for unit in ("B", "KB", "MB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} GB"

def ask(prompt: str) -> bool:
    return input(f"{prompt} [y/N] ").strip().lower() == "y"

def trash(path: Path):
    """Move file to .trash/ instead of deleting permanently."""
    BACKUP_DIR.mkdir(exist_ok=True)
    dest = BACKUP_DIR / f"{path.name}.{int(datetime.now().timestamp())}"
    shutil.move(str(path), dest)
    print(f"  → trashed {path.name}  ({size_str(dest)})")

# ── task 1: lo_memory deduplication ──────────────────────────────────────────

NOISE_PREFIXES = [
    "the user is currently", "user is currently", "the user is likely",
    "the user is a student", "user is a student", "life has not been",
    "the user is interested in", "user is familiar with",
    "the user is familiar", "the user is reading", "user is reading",
    "the user is studying", "user is studying", "the user is preparing",
    "user is preparing", "user is considering", "the user is considering",
    "the user is thinking", "user is thinking", "the user is unsure",
    "user is unsure", "the user may be", "user has a friendly",
    "user is agreeable", "user is pleased", "user is excited",
    "the user has a strong desire", "user is emotionally",
    "the user is emotionally", "user is seeking emotional",
    "the user is seeking", "user is feeling down and",
    "user didn't know the ai", "the user has a strong",
    "the user is in a romantic",
]

def clean_lo_memory():
    path = DATA_DIR / "lo_memory.json"
    if not path.exists():
        print("  lo_memory.json not found, skipping.")
        return

    facts = json.loads(path.read_text(encoding="utf-8"))
    before = len(facts)

    kept = []
    seen = set()
    for f in facts:
        text = (f.get("fact") or "").strip().lower()
        key = text[:50]

        skip = key in seen or any(text.startswith(p) for p in NOISE_PREFIXES)
        if not skip:
            seen.add(key)
            kept.append(f)

    removed = before - len(kept)
    if removed == 0:
        print(f"  lo_memory.json — {before} facts, nothing to clean.")
        return

    print(f"  lo_memory.json — {before} facts → {len(kept)} after removing {removed} duplicates/noise")
    if ask("  Apply?"):
        path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  Saved.")

# ── task 2: lovely_conversations trim ─────────────────────────────────────────

def trim_lovely_conversations(keep_last: int = 400):
    path = DATA_DIR / "lovely_conversations.json"
    if not path.exists():
        print("  lovely_conversations.json not found, skipping.")
        return

    convos = json.loads(path.read_text(encoding="utf-8"))
    before = len(convos)
    if before <= keep_last:
        print(f"  lovely_conversations.json — {before} records, nothing to trim.")
        return

    trimmed = convos[-keep_last:]
    print(f"  lovely_conversations.json — {before} records → keeping last {keep_last} ({before - keep_last} removed)")
    if ask("  Apply?"):
        path.write_text(json.dumps(trimmed, ensure_ascii=False, indent=2), encoding="utf-8")
        print("  Saved.")

# ── task 3: find and trash stale files ────────────────────────────────────────

def is_session_export(path: Path) -> bool:
    """Check if a JSON file is a one-off exported session dump."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(d, dict) and d.get("format") == "arwanos-session-v1":
            return True
    except Exception:
        pass
    return False

# Known cache files the app used to write but no longer reads
STALE_CACHE_FILES = {
    "query_cache.json",
    "rag_answer_cache.json",
    "query_staging.json",
}

# Backup extensions to trash
BACKUP_EXTENSIONS = {".bak", ".bak2", ".backup", ".old"}

def find_garbage():
    candidates = []

    for f in DATA_DIR.iterdir():
        if not f.is_file() or f.suffix not in (".json", *BACKUP_EXTENSIONS):
            continue
        if f.parent.name == ".trash":
            continue

        reason = None

        if f.suffix in BACKUP_EXTENSIONS:
            reason = "backup/old file"
        elif f.name in STALE_CACHE_FILES:
            reason = "stale cache (app no longer reads)"
        elif f.name.startswith("daily-productivity-backup"):
            reason = "old productivity backup"
        elif is_session_export(f):
            reason = "one-off session export"

        if reason:
            candidates.append((f, reason))

    return candidates

def clean_garbage():
    candidates = find_garbage()
    if not candidates:
        print("  No garbage files found.")
        return

    print(f"  Found {len(candidates)} file(s) to trash:\n")
    for f, reason in candidates:
        print(f"  {f.name:<45} {size_str(f):<10} — {reason}")

    print()
    if ask("  Trash all of these?"):
        for f, _ in candidates:
            trash(f)
    else:
        for f, reason in candidates:
            if ask(f"  Trash {f.name}?"):
                trash(f)

# ── task 4: show .trash contents ─────────────────────────────────────────────

def show_trash():
    if not BACKUP_DIR.exists() or not any(BACKUP_DIR.iterdir()):
        print("  .trash is empty.")
        return

    files = sorted(BACKUP_DIR.iterdir())
    total = sum(f.stat().st_size for f in files) / 1024
    print(f"  {len(files)} file(s) in .trash  ({total:.1f} KB total)")
    for f in files:
        print(f"  {f.name}")

    print()
    if ask("  Permanently delete everything in .trash?"):
        shutil.rmtree(BACKUP_DIR)
        print("  .trash cleared.")

# ── main menu ─────────────────────────────────────────────────────────────────

MENU = [
    ("Clean lo_memory.json (deduplicate / remove noise)", clean_lo_memory),
    ("Trim lovely_conversations.json (keep last 400)", trim_lovely_conversations),
    ("Find and trash garbage files (backups, stale caches, session exports)", clean_garbage),
    ("Show / empty .trash", show_trash),
    ("Run all", None),
    ("Quit", None),
]

def main():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("   Arwanos Maintenance Tool")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    for i, (label, _) in enumerate(MENU, 1):
        print(f"  {i}. {label}")

    print()
    choice = input("Choose [1-6]: ").strip()

    if choice == "5":
        print("\n── Clean lo_memory ──────────────────")
        clean_lo_memory()
        print("\n── Trim lovely_conversations ────────")
        trim_lovely_conversations()
        print("\n── Find garbage files ───────────────")
        clean_garbage()
    elif choice == "6":
        print("Bye.")
        return
    elif choice.isdigit() and 1 <= int(choice) <= 4:
        print()
        _, fn = MENU[int(choice) - 1]
        fn()
    else:
        print("Invalid choice.")
        return

    print("\nDone.\n")

if __name__ == "__main__":
    main()
