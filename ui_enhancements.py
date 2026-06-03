# ui_enhancements.py — Rich Styling for Rona Chat Display

from typing import Any

try:
    import customtkinter as ctk
except Exception:
    ctk = None


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
            fg_color=("#09090f", "#09090f"),
            text_color="#e2e8f0",
            corner_radius=12,
            wrap="word",
            border_color="#1e1e2e",
            border_width=1,
        )
    except Exception:
        pass

    # ── standard role tags ─────────────────────────────────────────────────
    try:
        ch = app.chat_history
        ch.tag_config("assistant", foreground="#7dd3fc")   # sky blue
        ch.tag_config("user",      foreground="#e2e8f0")   # near-white
        ch.tag_config("system",    foreground="#fbbf24")   # amber
        ch.tag_config("terminal",  foreground="#86efac")   # green

        if getattr(app, "ui_font_assistant", None):
            ch.tag_config("assistant", font=app.ui_font_assistant,
                          foreground="#1384ad")
        if getattr(app, "ui_font_user", None):
            ch.tag_config("user",      font=app.ui_font_user)
        if getattr(app, "ui_font", None):
            ch.tag_config("system",    font=app.ui_font)
            ch.tag_config("terminal",  font=app.ui_font)
            ch.tag_config("comment",   font=app.ui_font)

        ch.tag_config("rtl", justify="right")
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
            fg_color="#1a0a3c", hover_color="#2d1069",
            text_color="#c4b5fd",
            border_color="#4c1d95", border_width=1,
            corner_radius=8,
            font=("DejaVu Sans", 13),
        )
        btn.pack(side="left", padx=6)
    except Exception:
        pass
