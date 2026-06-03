"""
Arwanos startup banner — terminal ANSI art + Tk chat insertion.
"""
from __future__ import annotations
import webbrowser


def print_startup_banner(app=None) -> None:
    """
    Print the cyberpunk Arwanos banner to the terminal and, if *app* is given,
    insert it into the Tk chat widget with clickable URL tags.
    """
    banner = r"""
  ┌─────────────────────────────────────────────────────────────┐
  │  ◈  A R W A N O S  ◈   AI Agent  ·  v10  ·  Online        │
  └─────────────────────────────────────────────────────────────┘

        ╔══════════════════════════════════════════════╗
        ║     ·  · ─ ─ ╸ ╺ ─ ─ ·  ·                  ║
        ║     ███████████████████████                  ║
        ║    █ ╔═══════════════════╗ █                 ║
        ║    █ ║  ╭───────────╮   ║ █                 ║
        ║    █ ║  │  ◉     ◉  │   ║ █   ← eyes [ON]  ║
        ║    █ ║  │   ╲   ╱   │   ║ █                 ║
        ║    █ ║  │    ─────  │   ║ █                 ║
        ║    █ ║  ╰───────────╯   ║ █                 ║
        ║    █ ╚═══════════════════╝ █                 ║
        ║     ███████████████████████                  ║
        ║        │  ╠══════╣  │                        ║
        ║       ═╪══╬══════╬══╪═                       ║
        ║        │  ╠══════╣  │                        ║
        ║     ·  · ─ ─ ╸ ╺ ─ ─ ·  ·                  ║
        ╚══════════════════════════════════════════════╝

  ◈──────────────────── SYSTEMS ONLINE ─────────────────────◈
  │  ⬡ NEURAL   CORE  : ████████████████ ACTIVE            │
  │  ⬡ COGNITION LINK : ████████████░░░░ SYNCED            │
  │  ⬡ LANGUAGE MODEL : ████████████████ READY             │
  │  ⬡ AGENT PROTOCOL : ██████████░░░░░░ STANDBY           │
  ◈─────────────────────────────────────────────────────────◈

  ◦ Type /help to see all commands.

  ◦ Author   : GMM
  ◦ GitHub   : https://github.com/GMMB1
  ◦ Website  : https://hbeoptcenhvc.com/
  ◦ Support  : https://ko-fi.com/ghostman77506
"""

    # ANSI-colored terminal version
    C  = "\033[96m"
    TC = "\033[38;5;51m"
    CY = "\033[38;5;123m"
    Y  = "\033[93m"
    W  = "\033[97m"
    G  = "\033[92m"
    DM = "\033[38;5;33m"
    GR = "\033[38;5;240m"
    D  = "\033[2m"
    N  = "\033[0m"
    BD = "\033[1m"

    art = f"""
{TC}{BD}  ┌─────────────────────────────────────────────────────────────┐{N}
{TC}{BD}  │  {CY}◈{TC}  A R W A N O S  {CY}◈{TC}   AI Agent  ·  v10  ·  Online        │{N}
{TC}{BD}  └─────────────────────────────────────────────────────────────┘{N}

{DM}        ╔══════════════════════════════════════════════╗{N}
{DM}        ║{GR}     ·  · ─ ─ ╸ ╺ ─ ─ ·  ·                  {DM}║{N}
{DM}        ║{W}     ███████████████████████                  {DM}║{N}
{DM}        ║{W}    █ {DM}╔═══════════════════╗{W} █                 {DM}║{N}
{DM}        ║{W}    █ {DM}║  {W}╭───────────╮   {DM}║{W} █                 {DM}║{N}
{DM}        ║{W}    █ {DM}║  {W}│  {TC}{BD}◉{N}{W}     {TC}{BD}◉{N}{W}  │   {DM}║{W} █   {GR}← eyes [ON]{W}  {DM}║{N}
{DM}        ║{W}    █ {DM}║  {W}│   {GR}╲   ╱{W}   │   {DM}║{W} █                 {DM}║{N}
{DM}        ║{W}    █ {DM}║  {W}│    {C}─────{W}  │   {DM}║{W} █                 {DM}║{N}
{DM}        ║{W}    █ {DM}║  {W}╰───────────╯   {DM}║{W} █                 {DM}║{N}
{DM}        ║{W}    █ {DM}╚═══════════════════╝{W} █                 {DM}║{N}
{DM}        ║{W}     ███████████████████████                  {DM}║{N}
{DM}        ║{C}        │  ╠══════╣  │{DM}                        ║{N}
{DM}        ║{C}       ═╪══╬══════╬══╪═{DM}                       ║{N}
{DM}        ║{C}        │  ╠══════╣  │{DM}                        ║{N}
{DM}        ║{GR}     ·  · ─ ─ ╸ ╺ ─ ─ ·  ·                  {DM}║{N}
{DM}        ╚══════════════════════════════════════════════╝{N}

{TC}{BD}  ◈{N}{TC}──────────────────── SYSTEMS ONLINE ─────────────────────{TC}{BD}◈{N}
{DM}  │{N}  {C}⬡ NEURAL   CORE  : {G}████████████████{C} ACTIVE            {DM}│{N}
{DM}  │{N}  {C}⬡ COGNITION LINK : {G}████████████{GR}░░░░{C} SYNCED            {DM}│{N}
{DM}  │{N}  {C}⬡ LANGUAGE MODEL : {G}████████████████{C} READY             {DM}│{N}
{DM}  │{N}  {C}⬡ AGENT PROTOCOL : {Y}██████████{GR}░░░░░░{C} STANDBY           {DM}│{N}
{TC}{BD}  ◈{N}{TC}─────────────────────────────────────────────────────────{TC}{BD}◈{N}

{GR}  ◦{N} {Y}Type /help to see all commands.{N}

{GR}  ◦ Author   :{N} {W}GMM{N}
{GR}  ◦ GitHub   :{N} {C}https://github.com/GMMB1{N}
{GR}  ◦ Website  :{N} {C}https://hbeoptcenhvc.com/{N}
{GR}  ◦ Support  :{N} {C}https://ko-fi.com/ghostman77506{N}
"""

    try:
        print(art)
    except Exception:
        print("\n◈  ARWANOS v10 — AI Agent Online. Type /help to begin.\n")

    if app is None:
        return

    try:
        ch = getattr(app, "chat_history", None)
        if ch is None:
            return

        _url_map = {
            "https://github.com/GMMB1":        "https://github.com/GMMB1",
            "https://hbeoptcenhvc.com/":        "https://hbeoptcenhvc.com/",
            "https://ko-fi.com/ghostman77506":  "https://ko-fi.com/ghostman77506",
        }

        remaining = banner
        for url_text, url_href in _url_map.items():
            before, sep, remaining = remaining.partition(url_text)
            if sep:
                ch.insert("end", before, "codeblock")
                tag_name = f"banner_link_{url_text.replace('/', '_').replace(':', '')}"
                ch.insert("end", url_text, ("codeblock", tag_name))
                ch.tag_config(tag_name, foreground="#00CFFF", underline=True)
                _raw_ch = getattr(ch, "_textbox", ch)

                def _on_enter(e, _t=tag_name, _w=_raw_ch):
                    ch.tag_config(_t, foreground="#FFD700")
                    try:
                        _w.config(cursor="hand2")
                    except Exception:
                        pass

                def _on_leave(e, _t=tag_name, _w=_raw_ch):
                    ch.tag_config(_t, foreground="#00CFFF")
                    try:
                        _w.config(cursor="")
                    except Exception:
                        pass

                def _on_click(e, _u=url_href):
                    webbrowser.open(_u)

                ch.tag_bind(tag_name, "<Enter>", _on_enter)
                ch.tag_bind(tag_name, "<Leave>", _on_leave)
                ch.tag_bind(tag_name, "<Button-1>", _on_click)
            else:
                remaining = url_text + remaining

        if remaining:
            ch.insert("end", remaining, "codeblock")
        ch.insert("end", "\n")
        ch.see("end")
    except Exception:
        try:
            app._append_conversation("system", banner)
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
