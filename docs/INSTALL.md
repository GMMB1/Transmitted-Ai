# Installation & Troubleshooting

> Back to [README](../README.md)

## Get the code

```bash
git clone https://github.com/GMMB1/Transmitted-Ai.git
cd Transmitted-Ai
```

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running — [ollama.ai](https://ollama.ai)
- A local model (e.g. `llama3:8b` or `llama3.1`)
- Supported platforms: Linux (primary, tested), Windows, macOS
- **Voice input and call mode currently require Linux** — recording uses ALSA `arecord`

```bash
ollama pull llama3:8b-instruct-q4_K_M
```

> This must match `model_name` in `config.json`. Pulling a different tag (for example
> plain `llama3:8b`) and running with the default config fails with "model not found".
> To use another model, pull it and update `model_name` to the exact same tag.

Verify Ollama is running:
```bash
ollama list       # lists your downloaded models
ollama serve      # start manually if not already running
```

> **GPU acceleration — important:**  
> By default Ollama may not pick up your GPU. To make sure it runs on your NVIDIA (or other) GPU, start it like this:
> ```bash
> CUDA_VISIBLE_DEVICES=0 ollama serve > /dev/null 2>&1 &
> ```
> `CUDA_VISIBLE_DEVICES=0` tells the driver to use your first GPU. Change to `1`, `2`, etc. for a different card.  
> Omit `> /dev/null 2>&1 &` if you want to see the Ollama logs in the terminal.  
> On Windows, set the environment variable before starting: `set CUDA_VISIBLE_DEVICES=0` then `ollama serve`.

---

## System-Level Packages

> These are **OS-level** dependencies — install with your system package manager, **not** pip.  
> Some features degrade gracefully if missing.

### Linux — Debian / Ubuntu / Mint

```bash
sudo apt install python3-tk
sudo apt install pulseaudio-utils    # audio fallback (paplay)
sudo apt install alsa-utils           # alternative audio (aplay)
sudo apt install libfribidi0          # Arabic/RTL shaping
sudo apt install default-jre          # only if using language_tool_python
```

### Linux — Fedora / RHEL / CentOS

```bash
sudo dnf install python3-tkinter pulseaudio-utils java-latest-openjdk
```

### macOS

```bash
brew install python-tk
brew install --cask temurin    # Java, only for language_tool_python
# Audio: built-in afplay, no extra steps needed
```

### Windows

```
✔ Tkinter  — included in the standard python.org installer
✔ Audio    — winsound is built-in
✔ Java     — https://adoptium.net (only for language_tool_python)

⚠ Tick "Add Python to PATH" during installation.
⚠ Run terminal as Administrator if pip gives permission errors.
```

> **Audio player priority (all platforms):**  
> `pygame` → `winsound` (Windows) → `playsound` → `paplay` → `aplay` → `afplay`

---

## Install Python Dependencies

```bash
# Create virtual environment
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows CMD
.venv\Scripts\activate.bat

# Activate — Windows PowerShell
.venv\Scripts\Activate.ps1
```

Then install:

```bash
pip install -r requirements.txt
```

### Optional extras

| Package | What it enables |
|---|---|
| `pygame` | Cross-platform audio (dragon sounds) |
| `playsound` | Lightweight audio fallback |
| `spacy` | NLP / language detection |
| `nltk` | Tokenization |
| `pdfplumber` | Import PDF files into sessions |
| `language_tool_python` | Grammar check (needs Java 8+) |

---

## Configuration

Open `config.json` in the root folder and edit it directly:

```json
{
  "model_name": "llama3:8b-instruct-q4_K_M",
  "ollama_settings": {
    "temperature": 0.2
  },
  "demo_mode": false,
  "username": "GMM"
}
```

**`username`** — your name or handle. Arwanos uses it throughout prompts and the web UI to personalize responses.

**`demo_mode`** — when set to `true`, Arwanos loads from `data_test/` instead of `data/`, keeping demo runs isolated from your real journal and sessions. Leave it `false` for normal use.

**To change the model** — replace the `model_name` value with any model you have pulled in Ollama:

```json
"model_name": "llama3.1:8b"
"model_name": "mistral:7b"
"model_name": "gemma2:9b"
"model_name": "qwen2.5:7b"
"model_name": "deepseek-r1:8b"
```

Just run `ollama list` to see all models available on your machine, pick one, paste the name in, save the file, and restart Arwanos.

```bash
ollama list          # see what you have
ollama pull mistral  # pull a new one if needed
```

**To change the temperature** — controls how creative vs focused the responses are:

| Value | Behavior |
|---|---|
| `0.1` | Very focused, deterministic |
| `0.2` | Default — balanced *(recommended)* |
| `0.5` | More creative, varied responses |
| `0.8` | Very creative, less predictable |

---

## Sound Toggle

The dragon animation sound is **muted by default**.  
Find this near the top of `utils.py`:

```python
ARWANOS_SOUND_ENABLED: int = 1   # 0 = play | 1 = mute
```

| Value | Meaning |
|---|---|
| `0` | Sound ON — dragon roar plays on startup and on the dragon button |
| `1` | Sound OFF — completely silent *(default)* |

---

## Linux Desktop Launcher (optional)

`arwanos_launcher.sh` is a shell launcher for Linux desktop environments. Before starting Arwanos it:

- Detects your NVIDIA GPU and VRAM
- Checks whether Ollama is running on GPU or CPU
- Starts Ollama automatically if it is offline (with `CUDA_VISIBLE_DEVICES=0`)
- Shows a `zenity` dialog with GPU status and the current model before launch
- Writes a timestamped startup log to `logs/arwanos_startup.log`

To use it, make it executable once:

```bash
chmod +x arwanos_launcher.sh
./arwanos_launcher.sh
```

> **Note:** The script expects your virtual environment at `.venv/`. If you named yours differently, edit the `source .venv/bin/activate` line near the bottom of the script.  
> `zenity` must be installed (`sudo apt install zenity` on Debian/Ubuntu).

---

## Run Arwanos

```bash
python Arwanos_v10.py
```

Arwanos opens in desktop mode.  
Access the web UI by clicking **Open Webui** inside the app, or directly:

```
http://127.0.0.1:5005/renderer
```

---

## Build Arwanos.exe (Windows)

```bash
pip install pyinstaller

# Folder build — faster startup (recommended)
python build.py

# Single .exe — easier to share
python build.py --onefile
```

Output: `dist/Arwanos/Arwanos.exe`  
The icon (`Arwanos_icon.ico`) and all assets are bundled automatically.

> Ollama must still be installed and running separately on the target machine.

---


---

# Troubleshooting

## `ModuleNotFoundError` (any module)

Always use a virtual environment:

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Arwanos_v10.py
```

### Windows (CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python Arwanos_v10.py
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Arwanos_v10.py
```

---

## Ollama not found / model not responding

```bash
ollama serve
ollama list
ollama pull llama3:8b-instruct-q4_K_M
```

---

