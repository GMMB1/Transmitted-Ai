#!/usr/bin/env python3
"""
vault_to_json.py  —  BugBounty vault → data/hunt_session.json

READS  (never modified): /home/gmm/Documents/BugBounty/
WRITES (project only)  : /home/gmm/Project/Transmitted-Ai/data/hunt_session.json

Run once to build the initial index:
    python3 vault_to_json.py

Re-run any time you add or update notes — only changed files are re-processed,
unchanged notes are kept exactly as they are (incremental update).

Run with --full to force a complete rebuild from scratch:
    python3 vault_to_json.py --full
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

VAULT_ROOT  = Path("/home/gmm/Documents/BugBounty")
OUTPUT_FILE = Path(__file__).resolve().parent / "data" / "hunt_session.json"

# Directories inside the vault to skip entirely (read-skip, never touched)
SKIP_DIRS = {".obsidian", "Assets", ".trash", ".git"}


# ── YAML frontmatter parser (zero third-party deps) ──────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """
    Split YAML frontmatter from markdown body.
    Returns (meta_dict, body_text).
    Handles both --- and +++ delimiters. Gracefully skips parse errors.
    """
    meta: dict = {}
    body = text

    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        m = re.match(r"^\+\+\+\s*\n(.*?)\n\+\+\+\s*\n", text, re.DOTALL)

    if m:
        raw_yaml = m.group(1)
        body = text[m.end():]
        in_tags = False
        for line in raw_yaml.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                in_tags = False
                continue

            if ":" in stripped and not stripped.startswith("-"):
                in_tags = False
                key, _, val = stripped.partition(":")
                key = key.strip().lower()
                val = val.strip()

                if key == "tags":
                    if "[" in val or "," in val:
                        val = val.strip("[]")
                        meta["tags"] = [
                            t.strip().strip('"').strip("'")
                            for t in re.split(r"[,\s]+", val)
                            if t.strip()
                        ]
                    elif val:
                        meta["tags"] = [val.strip('"').strip("'")]
                    else:
                        meta["tags"] = []
                        in_tags = True
                elif key in ("status", "type", "target", "title", "date", "created"):
                    meta[key] = val.strip('"').strip("'")

            elif stripped.startswith("- ") and in_tags:
                tag = stripped[2:].strip().strip('"').strip("'")
                if tag:
                    meta.setdefault("tags", []).append(tag)

    return meta, body.strip()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _top_folder(path: Path) -> str:
    """Return the immediate sub-folder of VAULT_ROOT (e.g. '04 - Vulnerabilities')."""
    try:
        parts = path.relative_to(VAULT_ROOT).parts
        return parts[0] if len(parts) > 1 else ""
    except ValueError:
        return ""


def _note_from_file(md_path: Path) -> dict | None:
    """Parse one .md file and return a note dict, or None if empty/unreadable."""
    try:
        raw = md_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"  WARN: could not read {md_path}: {e}", file=sys.stderr)
        return None

    meta, body = _parse_frontmatter(raw)
    if not body:
        return None

    rel    = md_path.relative_to(VAULT_ROOT)
    folder = _top_folder(md_path)
    tags   = list(meta.get("tags", []))

    # Synthetic folder tag so keyword search can match folder name
    folder_slug = re.sub(r"^\d+\s*-\s*", "", folder).strip().lower().replace(" ", "/")
    if folder_slug:
        synthetic = f"folder/{folder_slug}"
        if synthetic not in tags:
            tags.append(synthetic)

    return {
        "role":    "note",
        "content": body,
        "source":  str(rel),
        "tags":    tags,
        "folder":  folder,
        "title":   meta.get("title", md_path.stem),
        "status":  meta.get("status", ""),
        "_mtime":  md_path.stat().st_mtime,     # used for incremental tracking
    }


# ── Main converter ────────────────────────────────────────────────────────────

def convert(force_full: bool = False) -> None:
    if not VAULT_ROOT.exists():
        print(f"ERROR: vault not found at {VAULT_ROOT}", file=sys.stderr)
        sys.exit(1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Load existing index (for incremental mode) ────────────────────────
    existing:    dict[str, dict] = {}   # source → note dict
    if not force_full and OUTPUT_FILE.exists():
        try:
            old_notes = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            existing  = {n["source"]: n for n in old_notes if isinstance(n, dict)}
            print(f"  Incremental mode — {len(existing)} notes already indexed.")
        except Exception:
            print("  Could not read existing index — doing full rebuild.")
            existing = {}

    # ── Walk vault (read-only) ────────────────────────────────────────────
    md_files = sorted(VAULT_ROOT.rglob("*.md"))

    added, updated, unchanged, skipped = 0, 0, 0, 0
    final_notes: dict[str, dict] = dict(existing)   # start with everything we had

    for md_path in md_files:
        if any(part in SKIP_DIRS for part in md_path.parts):
            skipped += 1
            continue

        rel_str = str(md_path.relative_to(VAULT_ROOT))

        # Incremental check: skip if mtime unchanged
        if not force_full and rel_str in existing:
            old_mtime = existing[rel_str].get("_mtime", 0)
            try:
                cur_mtime = md_path.stat().st_mtime
            except OSError:
                cur_mtime = 0
            if cur_mtime == old_mtime:
                unchanged += 1
                continue

        note = _note_from_file(md_path)
        if note is None:
            skipped += 1
            continue

        if rel_str in existing:
            updated += 1
        else:
            added += 1

        final_notes[rel_str] = note

    # Remove notes whose source file no longer exists
    all_rel = {str(p.relative_to(VAULT_ROOT)) for p in md_files
               if not any(part in SKIP_DIRS for part in p.parts)}
    removed = [k for k in list(final_notes) if k not in all_rel]
    for k in removed:
        del final_notes[k]

    if not final_notes:
        print("No notes found — nothing written.", file=sys.stderr)
        sys.exit(1)

    # ── Write output to project folder only ──────────────────────────────
    turns = list(final_notes.values())
    OUTPUT_FILE.write_text(
        json.dumps(turns, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    size_kb     = OUTPUT_FILE.stat().st_size // 1024
    total_chars = sum(len(n.get("content", "")) for n in turns)

    print(f"\n{'Full rebuild' if force_full else 'Incremental update'} complete")
    print(f"  ✅  Total notes : {len(turns)}")
    print(f"  ➕  Added       : {added}")
    print(f"  🔄  Updated     : {updated}")
    print(f"  ⏭  Unchanged  : {unchanged}")
    print(f"  🗑  Removed     : {len(removed)}")
    print(f"  ⚠  Skipped     : {skipped}  (empty / excluded dirs)")
    print(f"  📦  Content     : {total_chars:,} chars  ({total_chars // 1024} KB)")
    print(f"  💾  Saved to    : {OUTPUT_FILE}  ({size_kb} KB)")

    # Folder breakdown
    folder_counts = Counter(n.get("folder", "") for n in turns)
    print("\nFolder breakdown:")
    for folder, count in sorted(folder_counts.items()):
        label = re.sub(r"^\d+\s*-\s*", "", folder) if folder else "(root)"
        print(f"  {count:3d}  {label}")

    if removed:
        print(f"\nRemoved {len(removed)} stale note(s):")
        for r in removed:
            print(f"  - {r}")


if __name__ == "__main__":
    force = "--full" in sys.argv
    if force:
        print("--full flag detected: rebuilding from scratch.")
    convert(force_full=force)
