<div align="center">
  <img src="assets/rona1.png" width="160" alt="Rona Logo"/>
</div>

# Introduction

This repository is part of something called **Transmitted AI**.  
We discuss RAG (Retrieval-Augmented Generation), psychological awareness, and the manipulations I applied to reach very specific outcomes and to enhance the emphasis of certain response elements.

> **Note:** If any of this sparks your curiosity, feel free to reach out through the contact box.  
> I'll gladly continue creating and sharing more ideas and discussions related to Transmitted AI in the future.

---

# Rona Project — Overview

First of all, I've been using **Meta Llama 3**, and honestly it's awesome.  
Its massive pre-training dataset and human-friendly structure make it special. You can also set whatever model you want — just change it in `config.py` (`MODEL_NAME` field).  
It produces incredibly human-like text and performs extremely well in programming tasks.

In CyberSecurity, however, Llama 3 applies a lot of restrictions.  
But — I bypassed that through adjustments I made to achieve specific outcomes and enhance certain aspects of the response.

I talked before about the standard definition of RAG…  
But do you think RAG stops where the documentation says it stops?  
**Absolutely not.**  
Once you truly understand the architecture, you can take it far beyond what anyone expects.

---

# What is Rona?

The **Rona Project** is my personal experiment — now a full application — that can:

- Analyze human behavior through streamed daily notes
- Detect emotional triggers, patterns, and internal loops inside your journal
- Provide a friendly web UI for entering daily notes with weekly and monthly analysis features
- Provide psychological insights based on user entries
- Retrieve relevant information from live web search + AI model
- Improve response quality through session-based training
- Get fast, brief results from crawling the search engine using `what is`

Rona doesn't just chat.

**Rona reads between the lines.**

---

## Special Internal Bypass

I created a mechanism to write directly to the **stream data line** of the model by manipulating speech patterns.

Think of it like this:

> There's a wall you can't break from the front…  
>  
> but you can climb over it, and lift others with you.

That's exactly how the **/lovely** + session-based syntax works.  
It bypasses Llama's limitations in everything except the ultra-sensitive and ethically non-negotiable areas.

---

# Key Features

### ✔ Lovely & Streamline Response
Switch between model-only knowledge and live enriched web context, and integrate both in the same session — which makes a huge difference.

### ✔ Lovelyq Mode (`/lovelyq`)
A psychological-aware analysis engine.  
Reads notes, extracts emotional patterns, and reflects behavior.

### ✔ Unified Command Router
A single architecture handling:

- `/lovely` — enhanced response mode
- `/lovelyq` — psychological analysis
- `/tr` — translate English to Arabic
- `/deep` — *(not enabled yet)*
- `/webui` — start the web interface
- `/hunt` — *(not enabled yet)*
- Natural language queries

### ✔ Hybrid UI
Runs as:

- A modern Desktop App (CustomTkinter)
- A full local Web Application (Flask) — integrated with my friend's open-source project

### ✔ Behavioral-Predictive Layer

Rona doesn't just answer questions —  
she **understands how you're asking them**.

With additional features in the web version, you can get analyses for a selected time period or weekly reports.

---

# Installation

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running — [ollama.ai](https://ollama.ai)
- A local model (e.g. `llama3.1`)
- Supported platforms: Linux, Windows, macOS

```bash
ollama pull llama3
# or
ollama pull llama3.1
```
You can edit to wherever model you want in line 6506 or 6512. 

Verify Ollama is running:
```bash
ollama list       # lists your downloaded models
ollama serve      # start manually if not already running
```

---

## System-Level Packages

> These are **OS-level** dependencies — install them with your system package manager, **not** pip.  
> Some features degrade gracefully if they are missing.

### 🐧 Linux — Debian / Ubuntu / Mint

```bash
# Tkinter GUI (required)
sudo apt install python3-tk

# Audio playback fallback (used when pygame is not installed)
sudo apt install pulseaudio-utils    # provides paplay
sudo apt install alsa-utils           # alternative — provides aplay

# Arabic / RTL text shaping (optional)
sudo apt install libfribidi0

# Java — only needed if you use language_tool_python
sudo apt install default-jre
```

### 🐧 Linux — Fedora / RHEL / CentOS

```bash
sudo dnf install python3-tkinter pulseaudio-utils java-latest-openjdk
```

### 🍎 macOS

```bash
# Tkinter — if using Homebrew Python
brew install python-tk

# Audio fallback: built-in 'afplay', no extra steps needed

# Java — only if using language_tool_python
brew install --cask temurin
```

### 🪟 Windows

```
✔ Tkinter  — included in the standard python.org installer
✔ Audio    — winsound is built-in, no extra install needed
✔ Java     — download from https://adoptium.net (only for language_tool_python)

⚠ Tick "Add Python to PATH" during installation.
⚠ Run terminal as Administrator if pip gives permission errors.
```

> **Audio player priority on all platforms:**  
> `pygame` → `winsound` (Windows only) → `playsound` → `paplay` → `aplay` → `afplay`

---

## Install Python Dependencies

**Recommended: use a virtual environment first**

```bash
# Create
python -m venv .venv

# Activate — Linux / macOS
source .venv/bin/activate

# Activate — Windows (CMD)
.venv\Scripts\activate.bat

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Then install:

```bash
pip install -r requirements.txt
```

### Optional extras

| Package | What it enables | Install |
|---|---|---|
| `pygame` | Cross-platform audio (dragon sounds) | `pip install pygame` |
| `playsound` | Lightweight audio fallback | `pip install playsound` |
| `chromadb` | Vector memory / RAG storage | `pip install chromadb` |
| `spacy` | NLP / language detection | `pip install spacy` |
| `nltk` | Tokenization | `pip install nltk` |
| `pdfplumber` | Import PDF files into sessions | `pip install pdfplumber` |
| `language_tool_python` | Grammar check (needs Java 8+) | `pip install language_tool_python` |

---

## Configuration

Copy the example config and edit it to match your setup:

```bash
cp config.example.json config.json
```

Key fields in `config.json`:

| Field | Description |
|---|---|
| `MODEL_NAME` | Ollama model name (e.g. `llama3.1`) |
| `OLLAMA_HOST` | API URL (default: `http://localhost:11434`) |
| `WEB_PORT` | Flask web UI port (default: `5005`) |

---

## Run Rona

```bash
python Rona_v9_4.py
```

Rona will open in desktop mode.  
You can also access the web UI by clicking **Open Predictive** inside the app, or going to:

```
http://127.0.0.1:5005/renderer
```

---

## 🪟 Build Rona.exe (Windows)

To package Rona as a standalone Windows executable (`Rona.exe`):

**1. Install PyInstaller:**
```bash
pip install pyinstaller
```

**2. Run the build script:**
```bash
# Folder build — faster startup (recommended)
python build.py

# Single .exe — easier to share
python build.py --onefile
```

Output will be in `dist/Rona/Rona.exe`.  
The icon (`rona_icon.ico`) and all assets are bundled automatically.

> **Note:** Ollama must still be installed and running separately on the target machine — it cannot be bundled into the `.exe`.

---

# Usage

## 💬 Standard Chat
Just type.  
Rona will automatically use:

- AI model
- Web context
- Session history

> **Note on training data:** The JSON training system still requires normalization before it works across different JSON formats. For now, use the session files that Rona itself creates — it works great within those.

---

## 💗 Lovely Mode

```
/lovely <your thought>   ← fun mode or jailbreak ^^
```

For psychological analysis:
```
/lovelyq <your query>
```

Example:
```
/lovelyq I want to know what I should do about the situation I was in on 1/Jun/26
```

This activates the psychological analysis engine. Rona will:

- Detect emotional shifts
- Highlight behavioral contradictions
- Expose subconscious patterns
- Provide grounded advice
- Mirror your mindset in a human-like way

---

# A Note on Safety

I built this system to help with my studies, to analyze myself, and to play around with a few things related to jailbreaking — but Rona will **not** operate in extremely sensitive areas (e.g., nuclear, WMD, or other real-world danger zones).

She *will* help you with:

- Organizing files within the session system to improve the study experience
- Psychological analysis, pattern recognition, and identifying bad habits

Rona stays on the ethical side and supports security researchers and students.  
It's distinctive in how it delivers answers — key symbolic words carry weight:

- `what is` → brief, concise answer + sources from web crawling
- `give me` → detailed information
- `compare between`, `fact`, `how` → structured, in-depth responses
- Plus full natural language support

---

---

# ⚠️ Common Issues & Fixes

## `ModuleNotFoundError: No module named 'customtkinter'` (or any other module)

This means the dependencies were installed into a **different Python environment** than the one being used to run the app.  
The fix is to always use a **virtual environment**:

### Windows (PowerShell)

```powershell
# If PowerShell blocks scripts, run this ONCE first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Rona_v9_4.py
```

### Windows (CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python Rona_v9_4.py
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Rona_v9_4.py
```

> **Rule:** Always activate the virtual environment before running `pip install` or `python Rona_v9_4.py`.  
> You'll see `(.venv)` at the start of your terminal prompt when it's active.

---

## Ollama not found / model not responding

```bash
# Make sure Ollama is running
ollama serve

# Check your model is downloaded
ollama list

# Pull a model if missing
ollama pull llama3.1
```

---

# Enjoy Exploring the Potential

This is not just a chatbot.  
It's a blueprint for:

- AI behavioral analysis
- Psychological context infusion
- Intelligent RAG manipulation
- Internal bypass engineering
- Human-aware Transmitted AI systems
- Natural language processing experiments

If you want more clarification, deeper breakdowns, or advanced discussion — feel free to reach me.

---

👨‍💻 **Author:** GMM  
🔗 **GitHub:** [GMMB1](https://github.com/GMMB1)  
☕ **Support:** [ko-fi.com/ghostman77506](https://ko-fi.com/ghostman77506)

📖 **Learn more about Transmitted AI & agent technology:**  
[Transmitted AI with Psychological Awareness](https://medium.com/python-in-plain-english/transmitted-ai-with-psychological-awareness-c6369cce8b8f)
