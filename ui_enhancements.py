# ui_enhancements.py — Rich Styling for Rona Chat Display

from typing import Any

try:
    import customtkinter as ctk
except Exception:
    ctk = None


# ─────────────────────────────────────────────────────────────────
# V10 DESIGN TOKENS (desktop) — mirrors renderer/js/themes.js so the
# native app and the Web UI share one visual identity. Every widget
# reads from here; no hardcoded chrome colors at call sites.
# ─────────────────────────────────────────────────────────────────
V10_TOKENS = {
    "accent":        "#7c3aed",   # primary purple
    "accent_hover":  "#6d28d9",
    "accent2":       "#06b6d4",   # brand cyan
    "bg_base":       "#0a0a0f",   # window
    "bg_surface":    "#12121a",   # panels, inputs, bars
    "bg_elevated":   "#1a1a28",   # hover / chips
    "bg_chat":       "#0b0b12",   # chat canvas (a hair deeper than surface)
    "border":        "#26263a",   # resting border
    "border_subtle": "#1c1c2a",   # hairlines / dividers
    "text":          "#e2e8f0",
    "text_muted":    "#94a3b8",
    "text_faint":    "#64748b",
    "assistant":     "#38bdf8",   # readable sky — replaces low-contrast #1384ad
    "ok":            "#34d399",
    "warn":          "#fbbf24",
    "err":           "#f87171",
}

# ─────────────────────────────────────────────────────────────────
# THEMES — named palettes the user can switch live (settings picker).
# Each overrides the 13 "chrome" tokens; ok/warn/err stay constant so
# status colors keep their meaning (green ready / amber busy / red error).
# All are dark: the syntax-highlight palette (_C) is tuned for dark
# backgrounds, so light themes would make code blocks unreadable.
# Inspired by renderer/js/themes.js, retuned for the desktop chrome.
# ─────────────────────────────────────────────────────────────────
THEMES = {
    "Midnight": {   # default — purple / cyan (identical to base V10_TOKENS)
        "accent": "#7c3aed", "accent_hover": "#6d28d9", "accent2": "#06b6d4",
        "bg_base": "#0a0a0f", "bg_surface": "#12121a", "bg_elevated": "#1a1a28",
        "bg_chat": "#0b0b12", "border": "#26263a", "border_subtle": "#1c1c2a",
        "text": "#e2e8f0", "text_muted": "#94a3b8", "text_faint": "#64748b",
        "assistant": "#38bdf8",
    },
    "Ocean": {      # cyan / blue on deep navy
        "accent": "#22d3ee", "accent_hover": "#0891b2", "accent2": "#3b82f6",
        "bg_base": "#04121f", "bg_surface": "#0a1a2f", "bg_elevated": "#0d2542",
        "bg_chat": "#05101c", "border": "#1e3a52", "border_subtle": "#142838",
        "text": "#dbeafe", "text_muted": "#93b4cf", "text_faint": "#5b7a94",
        "assistant": "#38bdf8",
    },
    "Rose": {       # pink / magenta on wine
        "accent": "#f472b6", "accent_hover": "#db2777", "accent2": "#fb7185",
        "bg_base": "#170710", "bg_surface": "#241019", "bg_elevated": "#2e1622",
        "bg_chat": "#12060d", "border": "#3d1f2e", "border_subtle": "#2a1620",
        "text": "#fce7ef", "text_muted": "#d1a3b6", "text_faint": "#9b6b7f",
        "assistant": "#f9a8d4",
    },
    "Amethyst": {   # violet / indigo
        "accent": "#a855f7", "accent_hover": "#7c3aed", "accent2": "#818cf8",
        "bg_base": "#120a24", "bg_surface": "#1a1230", "bg_elevated": "#241a40",
        "bg_chat": "#0e0820", "border": "#34285a", "border_subtle": "#241a40",
        "text": "#ede9fe", "text_muted": "#b4a7d6", "text_faint": "#7c6f9e",
        "assistant": "#c4b5fd",
    },
    "Emerald": {    # green / teal on forest
        "accent": "#10b981", "accent_hover": "#059669", "accent2": "#2dd4bf",
        "bg_base": "#04140e", "bg_surface": "#0a1f17", "bg_elevated": "#102a20",
        "bg_chat": "#03100b", "border": "#1c3d30", "border_subtle": "#14291f",
        "text": "#d1fae5", "text_muted": "#86b8a3", "text_faint": "#5a8271",
        "assistant": "#34d399",
    },
    "Ember": {      # orange / red on charcoal-brown
        "accent": "#f97316", "accent_hover": "#ea580c", "accent2": "#f43f5e",
        "bg_base": "#180a04", "bg_surface": "#24130a", "bg_elevated": "#301a0e",
        "bg_chat": "#120703", "border": "#3f2416", "border_subtle": "#2a1810",
        "text": "#ffedd5", "text_muted": "#d1a683", "text_faint": "#997555",
        "assistant": "#fb923c",
    },
    "Carbon": {     # monochrome slate — minimalist, no color cast
        "accent": "#7c8ba1", "accent_hover": "#5a6b81", "accent2": "#94a3b8",
        "bg_base": "#0d0d10", "bg_surface": "#16161b", "bg_elevated": "#20202a",
        "bg_chat": "#0a0a0d", "border": "#2c2c36", "border_subtle": "#1e1e26",
        "text": "#e2e8f0", "text_muted": "#9aa3b0", "text_faint": "#697080",
        "assistant": "#cbd5e1",
    },
}

# legacy hardcoded chrome colors that aren't tokens but should still retint
# so the whole UI moves together (e.g. the lavender secondary-button text).
THEME_LEGACY_REMAP_KEYS = {"#c4b5fd": "assistant"}

# ─────────────────────────────────────────────────────────────────
# COLOR PALETTE  (Arwanos v10 — dark purple/cyan)
# ─────────────────────────────────────────────────────────────────
_C = {
    # syntax tokens
    "keyword":   "#ff79c6",   # pink
    "string":    "#f1fa8c",   # yellow
    "comment":   "#6272a4",   # slate-blue
    "number":    "#bd93f9",   # purple
    "operator":  "#50fa7b",   # green
    "builtin":   "#67e8f9",   # cyan
    "decorator": "#ffb86c",   # orange
    "variable":  "#f1f5f9",   # near-white
    "error":     "#ff5555",   # red

    # code block chrome
    "codebg":    "#0d0d18",   # very dark bg
    "code_lang": "#7c6b9a",   # muted purple label

    # inline code
    "inline_code": "#67e8f9", # cyan

    # math / logic
    "math_sym":  "#ffb86c",
    "logic_and": "#50fa7b",
    "logic_or":  "#ff79c6",
    "logic_not": "#ff5555",
    "logic_imp": "#ffb86c",
    "logic_bic": "#67e8f9",
    "logic_xor": "#c4b5fd",
    "logic_qty": "#f1fa8c",
    "math_eq":   "#f1f5f9",

    # markdown
    "bold_text":   "#ffffff",
    "italic_text": "#a5b4fc",
    "heading":     "#c4b5fd",

    # misc
    "copy_btn":  "#2d2d4a",
    "separator": "#2d2d4a",
}

_MONO_FONT   = ("DejaVu Sans Mono", 13)
_MONO_BOLD   = ("DejaVu Sans Mono", 13, "bold")
_SANS_FONT   = ("DejaVu Sans", 17)


def _raw(widget):
    """Return the underlying tk.Text if widget is a CTkTextbox, else widget itself."""
    return getattr(widget, "_textbox", widget)


def apply_chat_styling(app: Any, zoom_delta: int = 0) -> None:
    """
    Style fonts/colors for app.chat_history and configure all display tags.
    zoom_delta shifts all sizes relative to the 20pt base (e.g. 4 → 24pt).
    Safe to call even if CTkFont is unavailable.
    """
    if not hasattr(app, "chat_history") or app.chat_history is None:
        return

    _base = 20 + zoom_delta
    _asst = 21 + zoom_delta
    _user = 20 + zoom_delta

    # ── fonts ──────────────────────────────────────────────────────────────
    try:
        app.ui_font = ctk.CTkFont(family="Noto Naskh Arabic", size=_base, weight="normal")
        app.ui_font_assistant = ctk.CTkFont(
            family="Noto Naskh Arabic", size=_asst, weight="normal"
        )
        app.ui_font_user = ctk.CTkFont(
            family="Noto Naskh Arabic", size=_user, weight="medium"
        )
    except Exception:
        try:
            import tkinter.font as tkfont
            app.ui_font           = tkfont.Font(family="Noto Naskh Arabic", size=_base)
            app.ui_font_assistant = tkfont.Font(family="Noto Naskh Arabic", size=_asst)
            app.ui_font_user      = tkfont.Font(family="Noto Naskh Arabic", size=_user, weight="bold")
        except Exception:
            app.ui_font = app.ui_font_assistant = app.ui_font_user = None

    # ── textbox base styling ───────────────────────────────────────────────
    try:
        app.chat_history.configure(font=("DejaVu Sans", _base))
        app.chat_history.configure(
            fg_color=(V10_TOKENS["bg_chat"], V10_TOKENS["bg_chat"]),
            text_color=V10_TOKENS["text"],
            corner_radius=12,
            wrap="word",
            border_color=V10_TOKENS["border_subtle"],
            border_width=1,
        )
        # breathing room inside the canvas — text off the border reads calmer
        _raw(app.chat_history).configure(padx=16, pady=12)
    except Exception:
        pass

    # ── standard role tags ─────────────────────────────────────────────────
    try:
        ch = app.chat_history
        ch.tag_config("assistant", foreground=V10_TOKENS["assistant"])
        ch.tag_config("user",      foreground=V10_TOKENS["text"])
        ch.tag_config("system",    foreground=V10_TOKENS["warn"])
        ch.tag_config("terminal",  foreground="#86efac")   # green

        if getattr(app, "ui_font_assistant", None):
            ch.tag_config("assistant", font=app.ui_font_assistant,
                          foreground=V10_TOKENS["assistant"])
        if getattr(app, "ui_font_user", None):
            ch.tag_config("user",      font=app.ui_font_user)
        if getattr(app, "ui_font", None):
            ch.tag_config("system",    font=app.ui_font)
            ch.tag_config("terminal",  font=app.ui_font)
            ch.tag_config("comment",   font=app.ui_font)

        ch.tag_config("rtl", justify="right")

        # vertical rhythm between turns — the single biggest readability win
        txt = _raw(ch)
        for role_tag in ("assistant", "user", "system", "terminal"):
            txt.tag_configure(role_tag, spacing1=4, spacing3=8)
    except Exception:
        pass

    # ── rich display tags on the underlying tk.Text ────────────────────────
    _setup_rich_tags(app, zoom_delta=zoom_delta)


def _setup_rich_tags(app: Any, zoom_delta: int = 0) -> None:
    """
    Configure all rich display tags (syntax highlight, math, logic, markdown)
    on the underlying tk.Text widget so they survive tag_config precedence.
    """
    try:
        txt = _raw(app.chat_history)
        tc  = txt.tag_configure          # shorthand

        _mono_pt = max(10, 13 + zoom_delta)
        _sans_pt = max(10, 17 + zoom_delta)
        _mono    = ("DejaVu Sans Mono", _mono_pt)
        _mono_b  = ("DejaVu Sans Mono", _mono_pt, "bold")
        _sans    = ("DejaVu Sans",      _sans_pt)

        # ── code block chrome ──────────────────────────────────────────────
        tc("codeblock",    foreground="#f8f8f2", font=_mono,
           lmargin1=20, lmargin2=20, spacing1=2, spacing3=2)
        tc("code_lang",    foreground=_C["code_lang"], font=_mono_b)
        tc("code_divider", foreground=_C["separator"])

        # ── syntax-highlight tokens ────────────────────────────────────────
        tc("tok_keyword",   foreground=_C["keyword"],   font=_mono_b)
        tc("tok_string",    foreground=_C["string"],    font=_mono)
        tc("tok_comment",   foreground=_C["comment"],   font=_mono)
        tc("tok_number",    foreground=_C["number"],    font=_mono)
        tc("tok_operator",  foreground=_C["operator"],  font=_mono)
        tc("tok_builtin",   foreground=_C["builtin"],   font=_mono_b)
        tc("tok_decorator", foreground=_C["decorator"], font=_mono)
        tc("tok_variable",  foreground=_C["variable"],  font=_mono)

        # ── inline code ────────────────────────────────────────────────────
        tc("inlinecode",   foreground=_C["inline_code"], font=_mono)

        # ── math symbols ───────────────────────────────────────────────────
        tc("math_sym",  foreground=_C["math_sym"],  font=_sans)
        tc("math_eq",   foreground=_C["math_eq"],   font=_sans)

        # ── logic symbols ──────────────────────────────────────────────────
        tc("logic_and", foreground=_C["logic_and"], font=_sans)
        tc("logic_or",  foreground=_C["logic_or"],  font=_sans)
        tc("logic_not", foreground=_C["logic_not"], font=_sans)
        tc("logic_imp", foreground=_C["logic_imp"], font=_sans)
        tc("logic_bic", foreground=_C["logic_bic"], font=_sans)
        tc("logic_xor", foreground=_C["logic_xor"], font=_sans)
        tc("logic_qty", foreground=_C["logic_qty"], font=_sans)

        # ── markdown decorators ────────────────────────────────────────────
        _body_pt = 20 + zoom_delta
        tc("bold_text",   foreground=_C["bold_text"],   font=("DejaVu Sans", _body_pt, "bold"))
        tc("italic_text", foreground=_C["italic_text"],
           font=("DejaVu Sans", _body_pt, "italic"))
        tc("heading",     foreground=_C["heading"],
           font=("DejaVu Sans", _body_pt + 2, "bold"), spacing1=6, spacing3=4)

        # ── separator ─────────────────────────────────────────────────────
        tc("separator",   foreground=_C["separator"])

        # ── note / comment (application-level) ────────────────────────────
        tc("note",        foreground="#FF8C00", font=("Helvetica", max(10, 12 + zoom_delta), "italic"))

        # ensure rich tags win over base role tags when combined
        for rich_tag in (
            "tok_keyword", "tok_string", "tok_comment", "tok_number",
            "tok_operator", "tok_builtin", "tok_decorator", "tok_variable",
            "inlinecode", "math_sym", "math_eq", "logic_and", "logic_or",
            "logic_not", "logic_imp", "logic_bic", "logic_xor", "logic_qty",
            "bold_text", "italic_text", "heading", "codeblock",
        ):
            try:
                txt.tag_raise(rich_tag)
            except Exception:
                pass

    except Exception as e:
        import logging
        logging.debug(f"[rich tags] setup skipped: {e}")


def add_top_controls(app: Any) -> None:
    """
    Adds a 'Clear Chat' button to app.web_controls (beside your Lovely button).
    Safe if controls already exist (won't crash).
    """
    if not hasattr(app, "web_controls") or app.web_controls is None:
        return
    if not hasattr(app, "_cmd_clear"):
        def _inline_clear():
            try:
                if hasattr(app, "chat_history") and app.chat_history:
                    app.chat_history.delete("1.0", "end")
                app.conversation_history = []
                if hasattr(app, "_reply_assistant"):
                    app._reply_assistant("Chat cleared.")
            except Exception:
                pass
        clear_cmd = _inline_clear
    else:
        clear_cmd = lambda: app._cmd_clear("")

    try:
        btn = ctk.CTkButton(
            app.web_controls, text="Clear Chat", command=clear_cmd, width=120,
            height=36,
            fg_color=V10_TOKENS["bg_surface"],
            hover_color=V10_TOKENS["bg_elevated"],
            text_color="#c4b5fd",
            border_color=V10_TOKENS["border"], border_width=1,
            corner_radius=8,
            font=("DejaVu Sans", 13),
        )
        btn.pack(side="left", padx=6)
    except Exception:
        pass
