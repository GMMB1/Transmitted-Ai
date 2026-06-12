"""
Build Arwanos executable with PyInstaller.

Usage:
    python build.py             # folder build — faster startup (recommended)
    python build.py --onefile   # single executable — easier to share

Output: dist/Arwanos/  (or dist/Arwanos.exe with --onefile on Windows)
Ollama must still be installed and running separately on the target machine.
"""
import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEP = ";" if os.name == "nt" else ":"  # --add-data separator differs per OS

def main() -> int:
    onefile = "--onefile" in sys.argv

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",
        "--windowed",
        "--name", "Arwanos",
        "--add-data", f"assets{SEP}assets",
        "--add-data", f"renderer{SEP}renderer",
        "--add-data", f"config.json{SEP}.",
    ]

    icon = HERE / "assets" / "Arwanos_icon.ico"
    if icon.exists():
        cmd += ["--icon", str(icon)]

    cmd.append("--onefile" if onefile else "--onedir")
    cmd.append("Arwanos_v10.py")

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode == 0:
        out = HERE / "dist" / ("Arwanos.exe" if onefile and os.name == "nt" else "Arwanos")
        print(f"\n✅ Build complete → {out}")
    else:
        print("\n❌ Build failed — check the PyInstaller output above.")
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
