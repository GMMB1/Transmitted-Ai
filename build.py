"""
build.py — Build Rona_v9_4.py into a Windows .exe using PyInstaller.

Usage:
    python build.py              # normal build
    python build.py --onefile    # single .exe (slower startup, easier to share)

Output: dist/Rona/Rona.exe  (or dist/Rona.exe with --onefile)
"""

import subprocess
import sys
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths (all resolved relative to this script so it works on any machine)
# ─────────────────────────────────────────────────────────────────────────────
HERE     = Path(__file__).resolve().parent
SCRIPT   = HERE / "Rona_v9_4.py"
ICON     = HERE / "rona_icon.ico"
DIST_DIR = HERE / "dist"
BUILD_DIR = HERE / "build"

# ─────────────────────────────────────────────────────────────────────────────
# Data files/folders to bundle inside the exe
# Format: ("source_path", "destination_folder_inside_bundle")
# ─────────────────────────────────────────────────────────────────────────────
SEP = ";" if sys.platform.startswith("win") else ":"   # PyInstaller separator

datas = [
    (str(HERE / "assets"),               "assets"),
    (str(HERE / "renderer"),             "renderer"),
    (str(HERE / "config.py"),            "."),
    (str(HERE / "rtl_text.py"),          "."),
    (str(HERE / "ui_enhancements.py"),   "."),
    (str(HERE / "config.example.json"),  "."),
    (str(HERE / "rona_icon.ico"),        "."),
    (str(HERE / "rona_avatar.png"),      "."),
]

# ─────────────────────────────────────────────────────────────────────────────
# Hidden imports that PyInstaller might miss
# ─────────────────────────────────────────────────────────────────────────────
hidden_imports = [
    "customtkinter",
    "customtkinter.windows",
    "customtkinter.windows.widgets",
    "flask",
    "flask_cors",
    "aiohttp",
    "requests",
    "PIL",
    "PIL.Image",
    "PIL.ImageTk",
    "PIL.ImageSequence",
    "tkinter",
    "tkinter.font",
    "tkinter.messagebox",
    "tkinter.filedialog",
    "ollama",
    "psutil",
    "rtl_text",
    "ui_enhancements",
    "config",
    "packaging.version",        # customtkinter dependency
    "packaging.specifiers",
    "colorama",
]

# ─────────────────────────────────────────────────────────────────────────────
# Build command
# ─────────────────────────────────────────────────────────────────────────────
def build(onefile: bool = False):
    # Make sure dist/build dirs are clean to avoid stale file issues
    for d in (DIST_DIR, BUILD_DIR):
        if d.exists():
            print(f"[build] Cleaning {d} ...")
            shutil.rmtree(d)

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name",        "Rona",
        "--icon",        str(ICON),
        "--noconfirm",
        "--clean",
        "--log-level",   "WARN",
    ]

    # One-directory (default, faster startup) vs one-file
    if onefile:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")

    # Bundle data files
    for src, dst in datas:
        p = Path(src)
        if p.exists():
            cmd += ["--add-data", f"{src}{SEP}{dst}"]
        else:
            print(f"[build] WARNING: skipping missing path: {src}")

    # Hidden imports
    for imp in hidden_imports:
        cmd += ["--hidden-import", imp]

    # Collect whole packages (needed for customtkinter themes/images)
    for pkg in ("customtkinter", "PIL"):
        cmd += ["--collect-all", pkg]

    # Exclude heavy unused packages to keep size down
    for exc in ("matplotlib", "numpy", "scipy", "cv2", "torch"):
        cmd += ["--exclude-module", exc]

    cmd.append(str(SCRIPT))

    print("\n[build] Running PyInstaller...\n")
    print("  " + " ".join(cmd) + "\n")

    result = subprocess.run(cmd, cwd=str(HERE))

    if result.returncode == 0:
        if onefile:
            out = DIST_DIR / "Rona.exe"
        else:
            out = DIST_DIR / "Rona" / "Rona.exe"
        print(f"\n✅ Build successful!\n   Output: {out}\n")
    else:
        print("\n❌ Build failed. Check errors above.\n")
        sys.exit(result.returncode)


if __name__ == "__main__":
    onefile_mode = "--onefile" in sys.argv
    build(onefile=onefile_mode)
