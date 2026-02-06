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
    # Cover main Arabic blocks (not just \u0600-\u06FF)
    for ch in text:
        cp = ord(ch)
        if (
            0x0600 <= cp <= 0x06FF  # Arabic
            or 0x0750 <= cp <= 0x077F  # Arabic Supplement
            or 0x08A0 <= cp <= 0x08FF  # Arabic Extended-A
            or 0xFB50 <= cp <= 0xFDFF  # Arabic Presentation Forms-A
            or 0xFE70 <= cp <= 0xFEFF  # Arabic Presentation Forms-B
        ):
            return True
    return False


def shape_for_tk(text: str) -> str:
    """
    Shape + bidi for Arabic text to render correctly in Tk widgets.
    Adds RLE (U+202B) ... PDF (U+202C) only when Arabic is detected.
    """
    t = text or ""
    if not t or not has_arabic(t):
        return t
    if arabic_reshaper is None or _bidi_get_display is None:
        return t
    try:
        reshaped = arabic_reshaper.reshape(t)
        visual = _bidi_get_display(reshaped)
        # Avoid double-wrapping if already has RTL markers
        if "\u202b" in visual or "\u202c" in visual:
            return visual
        return "\u202b" + visual + "\u202c"
    except Exception:
        return t
