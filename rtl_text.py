# rtl_text.py

from typing import Optional

try:
    import arabic_reshaper
    from bidi.algorithm import get_display as _bidi_get_display
except Exception:
    arabic_reshaper = None
    _bidi_get_display = None


def has_arabic(text: str) -> bool:
    if not text:
        return False
    # Require at least 15% of non-whitespace chars to be Arabic.
    # A single stray Arabic Unicode char in an English model response
    # was triggering full BiDi reversal of the entire output.
    arabic_count = 0
    total_chars = 0
    for ch in text:
        if not ch.isspace():
            total_chars += 1
            cp = ord(ch)
            if (
                0x0600 <= cp <= 0x06FF  # Arabic
                or 0x0750 <= cp <= 0x077F  # Arabic Supplement
                or 0x08A0 <= cp <= 0x08FF  # Arabic Extended-A
                or 0xFB50 <= cp <= 0xFDFF  # Arabic Presentation Forms-A
                or 0xFE70 <= cp <= 0xFEFF  # Arabic Presentation Forms-B
            ):
                arabic_count += 1
    if total_chars == 0:
        return False
    return (arabic_count / total_chars) >= 0.15


def shape_for_tk(text: str) -> str:
    """
    Shape Arabic text for correct display in Tk widgets.

    Tk's Text widget does NOT apply Unicode bidi natively, so text is always
    rendered left-to-right.  We therefore:
      1. arabic_reshaper  → converts base Arabic chars to connected presentation
                            forms (proper letter shapes).
      2. get_display()    → reorders the characters to visual (LTR) order so
                            that Tk's LTR rendering produces the correct RTL
                            appearance.

    NOTE: Both steps are required.  Without reshaper the letter forms are wrong.
    Without get_display the word order appears reversed (LTR instead of RTL).
    """
    t = text or ""
    if not t or not has_arabic(t):
        return t
    if arabic_reshaper is None or _bidi_get_display is None:
        return t
    try:
        reshaped = arabic_reshaper.reshape(t)
        return _bidi_get_display(reshaped)
    except Exception:
        return t
