# ui_enhancements.py — Rich Styling for Rona Chat Display

from typing import Any

try:
    import customtkinter as ctk
except Exception:
    ctk = None


# ─────────────────────────────────────────────────────────────────
# COLOR PALETTE  (Dracula-inspired, dark-mode friendly)
# ─────────────────────────────────────────────────────────────────
_C = {
    # syntax tokens
    "keyword":   "#ff79c6",   # pink
    "string":    "#f1fa8c",   # yellow
    "comment":   "#6272a4",   # slate-blue
    "number":    "#bd93f9",   # purple
    "operator":  "#50fa7b",   # green
    "builtin":   "#8be9fd",   # cyan
    "decorator": "#ffb86c",   # orange
    "variable":  "#f8f8f2",   # near-white
    "error":     "#ff5555",   # red

    # code block chrome
    "codebg":    "#1e1e2e",   # very dark bg colour (foreground tint only)
    "code_lang": "#6272a4",   # language label

    # inline code
    "inline_code": "#50fa7b", # green

    # math / logic
    "math_sym":  "#ffb86c",   # orange  →  ∑ ∫ ∞ √
    "logic_and": "#50fa7b",   # green   →  ∧
    "logic_or":  "#ff79c6",   # pink    →  ∨
    "logic_not": "#ff5555",   # red     →  ¬
    "logic_imp": "#ffb86c",   # orange  →  → ⇒
    "logic_bic": "#8be9fd",   # cyan    →  ↔ ⟺
    "logic_xor": "#bd93f9",   # purple  →  ⊕
    "logic_qty": "#f1fa8c",   # yellow  →  ∀ ∃
    "math_eq":   "#f8f8f2",   # white   →  = ≠ ≤ ≥ < >

    # markdown
    "bold_text":   "#ffffff",
    "italic_text": "#c6dbff",
    "heading":     "#ffb86c",

    # misc
    "copy_btn":  "#44475a",
    "separator": "#44475a",
}

_MONO_FONT   = ("DejaVu Sans Mono", 13)
_MONO_BOLD   = ("DejaVu Sans Mono", 13, "bold")
_SANS_FONT   = ("DejaVu Sans", 17)


def _raw(widget):
    """Return the underlying tk.Text if widget is a CTkTextbox, else widget itself."""
    return getattr(widget, "_textbox", widget)


def apply_chat_styling(app: Any) -> None:
    """
    Style fonts/colors for app.chat_history and configure all display tags.
    Safe to call even if CTkFont is unavailable.
    """
    if not hasattr(app, "chat_history") or app.chat_history is None:
        return

    # ── fonts ──────────────────────────────────────────────────────────────
    try:
        app.ui_font = ctk.CTkFont(family="Noto Naskh Arabic", size=20, weight="normal")
        app.ui_font_assistant = ctk.CTkFont(
            family="Noto Naskh Arabic", size=21, weight="normal"
        )
        app.ui_font_user = ctk.CTkFont(
            family="Noto Naskh Arabic", size=20, weight="medium"
        )
    except Exception:
        try:
            import tkinter.font as tkfont
            app.ui_font           = tkfont.Font(family="Noto Naskh Arabic", size=20)
            app.ui_font_assistant = tkfont.Font(family="Noto Naskh Arabic", size=25)
            app.ui_font_user      = tkfont.Font(family="Noto Naskh Arabic", size=30, weight="bold")
        except Exception:
            app.ui_font = app.ui_font_assistant = app.ui_font_user = None

    # ── textbox base styling ───────────────────────────────────────────────
    try:
        app.chat_history.configure(font=("DejaVu Sans", 18))
        app.chat_history.configure(
            fg_color=("#121212", "#121212"),
            text_color="#EDEFF4",
            corner_radius=10,
            wrap="word",
        )
    except Exception:
        pass

    # ── standard role tags ─────────────────────────────────────────────────
    try:
        ch = app.chat_history
        ch.tag_config("assistant", foreground="#1384ad")
        ch.tag_config("user",      foreground="#FFFFFF")
        ch.tag_config("system",    foreground="#FFCC66")
        ch.tag_config("terminal",  foreground="#F3F99D")

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
    _setup_rich_tags(app)


def _setup_rich_tags(app: Any) -> None:
    """
    Configure all rich display tags (syntax highlight, math, logic, markdown)
    on the underlying tk.Text widget so they survive tag_config precedence.
    """
    try:
        txt = _raw(app.chat_history)
        tc  = txt.tag_configure          # shorthand

        # ── code block chrome ──────────────────────────────────────────────
        tc("codeblock",    foreground="#f8f8f2", font=_MONO_FONT,
           lmargin1=20, lmargin2=20, spacing1=2, spacing3=2)
        tc("code_lang",    foreground=_C["code_lang"], font=_MONO_BOLD)
        tc("code_divider", foreground=_C["separator"])

        # ── syntax-highlight tokens ────────────────────────────────────────
        tc("tok_keyword",   foreground=_C["keyword"],   font=_MONO_BOLD)
        tc("tok_string",    foreground=_C["string"],    font=_MONO_FONT)
        tc("tok_comment",   foreground=_C["comment"],   font=_MONO_FONT)
        tc("tok_number",    foreground=_C["number"],    font=_MONO_FONT)
        tc("tok_operator",  foreground=_C["operator"],  font=_MONO_FONT)
        tc("tok_builtin",   foreground=_C["builtin"],   font=_MONO_BOLD)
        tc("tok_decorator", foreground=_C["decorator"], font=_MONO_FONT)
        tc("tok_variable",  foreground=_C["variable"],  font=_MONO_FONT)

        # ── inline code ────────────────────────────────────────────────────
        tc("inlinecode",   foreground=_C["inline_code"], font=_MONO_FONT)

        # ── math symbols ───────────────────────────────────────────────────
        tc("math_sym",  foreground=_C["math_sym"],  font=_SANS_FONT)
        tc("math_eq",   foreground=_C["math_eq"],   font=_SANS_FONT)

        # ── logic symbols ──────────────────────────────────────────────────
        tc("logic_and", foreground=_C["logic_and"], font=_SANS_FONT)
        tc("logic_or",  foreground=_C["logic_or"],  font=_SANS_FONT)
        tc("logic_not", foreground=_C["logic_not"], font=_SANS_FONT)
        tc("logic_imp", foreground=_C["logic_imp"], font=_SANS_FONT)
        tc("logic_bic", foreground=_C["logic_bic"], font=_SANS_FONT)
        tc("logic_xor", foreground=_C["logic_xor"], font=_SANS_FONT)
        tc("logic_qty", foreground=_C["logic_qty"], font=_SANS_FONT)

        # ── markdown decorators ────────────────────────────────────────────
        tc("bold_text",   foreground=_C["bold_text"],   font=("DejaVu Sans", 18, "bold"))
        tc("italic_text", foreground=_C["italic_text"],
           font=("DejaVu Sans", 18, "italic"))
        tc("heading",     foreground=_C["heading"],
           font=("DejaVu Sans", 20, "bold"), spacing1=6, spacing3=4)

        # ── separator ─────────────────────────────────────────────────────
        tc("separator",   foreground=_C["separator"])

        # ── note / comment (application-level) ────────────────────────────
        tc("note",        foreground="#FF8C00", font=("Helvetica", 12, "italic"))

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
            fg_color="#0b3770", hover_color="#0f4a9e",
        )
        btn.pack(side="left", padx=6)
    except Exception:
        pass
