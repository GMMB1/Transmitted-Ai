"""
Arwanos startup banner — a compact daily briefing (terminal ANSI + Tk chat).

Replaced the old ASCII robot: every line here is REAL data — greeting, model,
journal, check-ins, habits — so the first thing on screen is your actual day,
not decoration pretending to be telemetry.
"""
from __future__ import annotations
import json
import datetime
import webbrowser
from pathlib import Path


# ── data gathering (all guarded — a missing file never breaks startup) ──────

def _days_ago_label(iso_date: str) -> str:
    try:
        d = datetime.date.fromisoformat((iso_date or "")[:10])
        n = (datetime.date.today() - d).days
        if n <= 0:
            return "today"
        if n == 1:
            return "yesterday"
        return f"{n} days ago"
    except Exception:
        return "a while ago"


def _gather_daily_context(app=None) -> dict:
    ctx = {
        "username": "friend", "model": "local model", "num_ctx": None,
        "journal_count": None, "last_entry": None,
        "sessions": None, "avg_delta": None,
        "habits_unlogged": None, "habit_titles": [],
        "last_action_step": None,
    }
    root = Path(__file__).resolve().parent

    try:
        cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
        ctx["username"] = cfg.get("username") or ctx["username"]
        ctx["model"]    = cfg.get("model_name") or ctx["model"]
        ctx["num_ctx"]  = (cfg.get("ollama_settings") or {}).get("num_ctx")
    except Exception:
        pass

    data_dir = root / "data"
    try:
        if app is not None and isinstance(getattr(app, "paths", None), dict):
            data_dir = Path(app.paths.get("data_dir") or data_dir)
    except Exception:
        pass

    try:
        raw = json.loads((data_dir / "psychoanalytical.json").read_text(encoding="utf-8"))
        entries = raw if isinstance(raw, list) else raw.get("entries", [])
        ctx["journal_count"] = len(entries)
        if entries:
            ctx["last_entry"] = max((e.get("date") or "") for e in entries)
    except Exception:
        pass

    try:
        sess = json.loads((data_dir / "monitor_sessions.json").read_text(encoding="utf-8"))
        comp = [s for s in sess if s.get("status") == "completed"]
        ctx["sessions"] = len(comp)
        deltas = [s["mood_end"] - s["mood_start"] for s in comp
                  if isinstance(s.get("mood_end"), (int, float))
                  and isinstance(s.get("mood_start"), (int, float))]
        if deltas:
            ctx["avg_delta"] = sum(deltas) / len(deltas)
        # the action step from the most recent completed session (accountability)
        if comp:
            _step = (comp[-1].get("action_step") or "").strip()
            if _step:
                ctx["last_action_step"] = _step
    except Exception:
        pass

    try:
        habits = json.loads((data_dir / "habits.json").read_text(encoding="utf-8"))
        today = datetime.date.today().isoformat()
        ctx["habit_titles"] = [h.get("title") or "?" for h in habits]
        ctx["habits_unlogged"] = [
            h.get("title") or "?" for h in habits
            if not any(l.get("date") == today for l in (h.get("dailyLogs") or []))
        ]
    except Exception:
        pass

    return ctx


# ── briefing assembly — [(text, tag)] segments so Tk can color each part ────

_RULE = "  " + "─" * 68 + "\n"


def _build_briefing(ctx: dict) -> list:
    now = datetime.datetime.now()
    hour = now.hour
    greet = ("Good morning" if hour < 12 else
             "Good afternoon" if hour < 18 else "Good evening")
    date_line = now.strftime("%A, %d %B %Y")

    seg: list = []
    seg.append(("\n   ⟁  A R W A N O S", "banner_brand"))
    seg.append(("  v10\n", "banner_dim"))
    # motto lives inside the rule line — single tag/font, so it always aligns
    _motto = "local · private · yours"
    seg.append(("  " + "─" * (68 - len(_motto) - 6) + f"  {_motto}  ──\n", "banner_dim"))
    seg.append((f"\n   {greet}, {ctx['username']} — {date_line}\n\n", "banner_body"))

    model_line = f"   ●  {ctx['model']}"
    if ctx.get("num_ctx"):
        model_line += f"  ·  ctx {ctx['num_ctx']}"
    model_line += "  ·  running on your machine\n\n"
    seg.append((model_line, "banner_ok"))

    # Journal
    seg.append(("   Journal      ", "banner_label"))
    if ctx.get("journal_count"):
        seg.append((f"{ctx['journal_count']} entries · last written "
                    f"{_days_ago_label(ctx.get('last_entry'))}\n", "banner_body"))
    else:
        seg.append(("no entries yet — your story starts today\n", "banner_dim"))

    # Check-ins
    seg.append(("   Check-ins    ", "banner_label"))
    if ctx.get("sessions"):
        line = f"{ctx['sessions']} sessions"
        if ctx.get("avg_delta") is not None:
            line += f" · mood shift {ctx['avg_delta']:+.1f} on average"
        seg.append((line + "\n", "banner_body"))
    else:
        seg.append(("none yet — the Monitor is waiting in the Web UI\n", "banner_dim"))

    # Habits
    if ctx.get("habit_titles"):
        seg.append(("   Habits       ", "banner_label"))
        unlogged = ctx.get("habits_unlogged") or []
        if unlogged:
            seg.append((f"{', '.join(ctx['habit_titles'])} · ", "banner_body"))
            seg.append((f"{len(unlogged)} not logged today\n", "banner_warn"))
        else:
            seg.append((f"{', '.join(ctx['habit_titles'])} · all logged today ✓\n", "banner_ok"))

    # Accountability — last session's committed action step
    if ctx.get("last_action_step"):
        step = ctx["last_action_step"]
        step = step if len(step) <= 62 else step[:59] + "…"
        seg.append(("   Your step    ", "banner_label"))
        seg.append((f"“{step}” — revisit in today's check-in\n", "banner_accent"))

    seg.append(("\n   /help", "banner_accent"))
    seg.append((" all commands    ", "banner_hint"))
    seg.append(("/lovelyq", "banner_accent"))
    seg.append((" ask your journal    ", "banner_hint"))
    seg.append(("/deep", "banner_accent"))
    seg.append((" live research\n", "banner_hint"))
    seg.append((_RULE, "banner_dim"))
    return seg


# ── Tk tag palette (matches the V10 design tokens) ──────────────────────────

_TAGS = {
    "banner_brand":  {"foreground": "#06b6d4", "font": ("DejaVu Sans Mono", 14, "bold")},
    "banner_body":   {"foreground": "#e2e8f0", "font": ("DejaVu Sans Mono", 13)},
    "banner_label":  {"foreground": "#c4b5fd", "font": ("DejaVu Sans Mono", 13, "bold")},
    "banner_accent": {"foreground": "#7dd3fc", "font": ("DejaVu Sans Mono", 13, "bold")},
    "banner_hint":   {"foreground": "#94a3b8", "font": ("DejaVu Sans Mono", 12)},
    "banner_dim":    {"foreground": "#475569", "font": ("DejaVu Sans Mono", 12)},
    "banner_ok":     {"foreground": "#34d399", "font": ("DejaVu Sans Mono", 13)},
    "banner_warn":   {"foreground": "#fbbf24", "font": ("DejaVu Sans Mono", 13)},
}

_LINKS = {  # display text → target
    "github.com/GMMB1":        "https://github.com/GMMB1",
    "hbeoptcenhvc.com":        "https://hbeoptcenhvc.com/",
    "ko-fi.com/ghostman77506": "https://ko-fi.com/ghostman77506",
}


def _insert_link(ch, display: str, href: str) -> None:
    """Insert one clickable link with hover color + hand cursor."""
    tag = f"banner_link_{display.replace('/', '_').replace('.', '_')}"
    ch.insert("end", display, ("banner_hint", tag))
    ch.tag_config(tag, foreground="#00CFFF", underline=True)
    raw = getattr(ch, "_textbox", ch)

    def _enter(e, _t=tag, _w=raw):
        ch.tag_config(_t, foreground="#FFD700")
        try:
            _w.config(cursor="hand2")
        except Exception:
            pass

    def _leave(e, _t=tag, _w=raw):
        ch.tag_config(_t, foreground="#00CFFF")
        try:
            _w.config(cursor="")
        except Exception:
            pass

    ch.tag_bind(tag, "<Enter>", _enter)
    ch.tag_bind(tag, "<Leave>", _leave)
    ch.tag_bind(tag, "<Button-1>", lambda e, _u=href: webbrowser.open(_u))


def print_startup_banner(app=None) -> None:
    """
    Print the daily briefing to the terminal and, if *app* is given,
    insert it into the Tk chat widget with colored tags + clickable links.
    """
    ctx = _gather_daily_context(app)
    seg = _build_briefing(ctx)

    # ── terminal (ANSI) version ──────────────────────────────────────────
    _ANSI = {
        "banner_brand":  "\033[1;96m", "banner_body": "\033[97m",
        "banner_label":  "\033[38;5;183m", "banner_accent": "\033[96m",
        "banner_hint":   "\033[38;5;246m", "banner_dim": "\033[38;5;240m",
        "banner_ok":     "\033[92m", "banner_warn": "\033[93m",
    }
    try:
        art = "".join(f"{_ANSI.get(tag, '')}{text}\033[0m" for text, tag in seg)
        footer = "   " + "  ·  ".join([ctx["username"]] + list(_LINKS)) + "\n"
        print(art + f"\033[38;5;240m{footer}\033[0m")
    except Exception:
        print("\n⟁  ARWANOS v10 — ready. Type /help to begin.\n")

    if app is None:
        return

    # ── Tk chat version ──────────────────────────────────────────────────
    try:
        ch = getattr(app, "chat_history", None)
        if ch is None:
            return

        raw = getattr(ch, "_textbox", ch)
        for tag, cfg in _TAGS.items():
            try:
                raw.tag_configure(tag, **cfg)
            except Exception:
                pass

        for text, tag in seg:
            ch.insert("end", text, tag)

        # footer: author + clickable links
        ch.insert("end", f"   {ctx['username']}", "banner_hint")
        for display, href in _LINKS.items():
            ch.insert("end", "  ·  ", "banner_dim")
            _insert_link(ch, display, href)
        ch.insert("end", "\n\n")
        ch.see("end")
    except Exception:
        try:
            app._append_conversation("system", "⟁  ARWANOS v10 — ready. Type /help to begin.")
        except Exception:
            pass


def apply_ctk_theme() -> None:
    """Apply dark theme to customtkinter. Silent no-op if ctk is not installed."""
    try:
        import customtkinter as ctk
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
    except Exception:
        pass
