# ui_enhancements.py

from typing import Any

try:
    import customtkinter as ctk
except Exception:
    ctk = None


def apply_chat_styling(app: Any) -> None:
    """
    Style fonts/colors for app.chat_history and add tag fonts.
    Safe to call even if CTkFont is unavailable.
    """
    if not hasattr(app, "chat_history") or app.chat_history is None:
        return

    # --- fonts ---
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

            app.ui_font = tkfont.Font(family="Noto Naskh Arabic", size=20)
            app.ui_font_assistant = tkfont.Font(family="Noto Naskh Arabic", size=25)
            app.ui_font_user = tkfont.Font(
                family="Noto Naskh Arabic", size=30, weight="bold"
            )
        except Exception:
            app.ui_font = app.ui_font_assistant = app.ui_font_user = None

    # -------- apply textbox styling --------
    try:
        app.chat_history.configure(font=("DejaVu Sans", 20))  # ensure insertable
        app.chat_history.configure(
            fg_color=("#121212", "#121212"),
            text_color="#EDEFF4",
            corner_radius=10,
            wrap="word",
        )
    except Exception:
        pass

    try:
        app.chat_history.tag_config("assistant", foreground="#C6DBFF")
        app.chat_history.tag_config("user", foreground="#FFFFFF")
        app.chat_history.tag_config("system", foreground="#FFCC66")
        app.chat_history.tag_config("terminal", foreground="#F3F99D")

        if getattr(app, "ui_font_assistant", None):
            app.chat_history.tag_config("assistant", font=app.ui_font_assistant)
        if getattr(app, "ui_font_user", None):
            app.chat_history.tag_config("user", font=app.ui_font_user)
        if getattr(app, "ui_font", None):
            app.chat_history.tag_config("system", font=app.ui_font)
            app.chat_history.tag_config("terminal", font=app.ui_font)

        # Right-align RTL lines
        app.chat_history.tag_config("rtl", justify="right")
    except Exception:
        pass


def add_top_controls(app: Any) -> None:
    """
    Adds a 'Clear Chat' button to app.web_controls (beside your Lovely button).
    Safe if controls already exist (won’t crash).
    """
    if not hasattr(app, "web_controls") or app.web_controls is None:
        return
    if not hasattr(app, "_cmd_clear"):
        # fallback inline clear if your _cmd_clear doesn’t exist
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
            app.web_controls, text="Clear Chat", command=clear_cmd, width=120
        )
        btn.pack(side="left", padx=6)
    except Exception:
        # don’t let UI crash if CTkButton fails
        pass
