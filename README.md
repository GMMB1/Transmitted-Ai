<div align="center">
  <img src="Arwanos_icon.png" width="160" alt="Arwanos Logo"/>
</div>

# Arwanos — Transmitted AI Personal Assistant

**Arwanos** is a locally-running, privacy-first AI assistant built in Python.  
It runs entirely on your machine using **Ollama** as the inference backend — no data ever leaves your device, no cloud API is called for core reasoning.

This project is part of **Transmitted AI** — a framework that goes beyond standard RAG and chatbot patterns by infusing psychological awareness, adaptive resource management, and behavioral analysis into a local AI system.

> If any of this sparks your curiosity, feel free to reach out.

---

# What is Arwanos?

Arwanos is a personal experiment turned full application. It can:

- Analyze human behavior through streamed daily notes
- Detect emotional triggers, patterns, and internal loops inside your journal
- Provide a friendly web UI for entering daily notes with weekly and monthly analysis features
- Provide psychological insights based on user entries
- Retrieve relevant information from live web search + AI model
- Improve response quality through session-based RAG
- Route queries intelligently — simple questions answered instantly, complex ones automatically trigger web search

Arwanos doesn't just chat.

**Arwanos reads between the lines.**

---

## ARM — Adaptive Resource Management

Every query is **scored before the LLM is called**. This is what makes Arwanos different from a standard chatbot wrapper.

### Complexity Scoring (0–3 per dimension)

| Dimension | What it measures |
|---|---|
| `search_need` | Does the query need live/external data? |
| `context_need` | Does it need session history? |
| `response_length` | How long should the answer be? |
| `reasoning_steps` | How complex is the reasoning? |

### Resource Budget (auto-assigned)

| Level | Web Search | Context Items | Max Tokens |
|---|---|---|---|
| 0 — simple | ✗ | 0 | 640 |
| 1 — moderate | ✗ | 3 | 960 |
| 2 — detailed | ✓ 3 results | 8 | 1280 |
| 3 — complex | ✓ 7 results | 15 | 1600 |

A simple factual question uses almost no resources. A complex research question automatically triggers web search and longer output — without the user configuring anything.

---

## Special Internal Bypass

I created a mechanism to write directly to the **stream data line** of the model by manipulating speech patterns.

Think of it like this:

> There's a wall you can't break from the front…  
>  
> but you can climb over it, and lift others with you.

That's exactly how the **/lo** + session-based syntax works.  
It bypasses Llama's default restrictions in everything except the ultra-sensitive and ethically non-negotiable areas.

---

# Key Features

### Lo & Streamline Response
Switch between model-only knowledge and live enriched web context, and integrate both in the same session.

### Analyze Mode (`/analyze`)
A psychologically-aware analysis engine.  
Reads your private journal, extracts emotional patterns, detects behavioral loops, and reflects them back with intent detection across 6 analytical lenses:

| Query type | Analytical lens |
|---|---|
| "pattern", "trigger", "habit" | Behavioral Patterns |
| "feel", "mood", "anxious" | Emotional Arc |
| "avoid", "procrastinate" | Avoidance Patterns |
| "contradict", "promise" | Contradictions |
| "progress", "improve", "grow" | Growth Tracking |
| "friend", "family", "alone" | Social / Relationship |

### Deep Search (`/deep`)
Forces a live web search regardless of ARM complexity score — use this when you know you need fresh, sourced information. Returns numbered results with titles, URLs, and a synthesized answer.

### Session RAG (`/rag`)
Searches an imported JSON session using a keyword inverted index built at import time. Lookup is near-instant even for 200+ turn sessions — O(query keywords), not O(session size).

### Unified Command Router

| Command | Function |
|---|---|
| `/lo <text>` | Enhanced emotional response mode |
| `/analyze <query>` | Psychological journal analysis |
| `/deep <question>` | Forced live web search |
| `/rag <question>` | Search imported session |
| `/ap <query>` | Arabic processing pipeline — full AR→EN→pipeline→AR sandwich |
| `/tr <text>` | Translate English ↔ Arabic |
| `/webui start` | Launch local web interface |

### Hybrid UI
Runs as a modern Desktop App (CustomTkinter) or a full local Web Application (Flask). All data stays on `127.0.0.1` — nothing is sent to external servers.

### Bilingual — Arabic / English
Arabic text is detected via a 15% character threshold and rendered with full BiDi/RTL support using `python-bidi`. Stray Arabic characters in English text do not trigger reversal.

---

# Installation

## Prerequisites

- **Python 3.10+**
- **Ollama** installed and running — [ollama.ai](https://ollama.ai)
- A local model (e.g. `llama3:8b` or `llama3.1`)
- Supported platforms: Linux, Windows, macOS

```bash
ollama pull llama3:8b
# or
ollama pull llama3.1
```

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
| `chromadb` | Vector memory / RAG storage |
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
  }
}
```

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
Find this near the top of `Arwanos_v9_8.py`:

```python
ARWANOS_SOUND_ENABLED = 1   # 0 = play | 1 = mute
```

| Value | Meaning |
|---|---|
| `0` | Sound ON — dragon roar plays on startup and on the dragon button |
| `1` | Sound OFF — completely silent *(default)* |

---

## Run Arwanos

```bash
python Arwanos_v9_8.py
```

Arwanos opens in desktop mode.  
Access the web UI by clicking **Open Predictive** inside the app, or directly:

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

# Usage

## Standard Chat

Just type. Arwanos will automatically:
- Score the query with ARM
- Decide whether to search the web
- Pull session context if relevant
- Return a response scaled to the query's complexity

> `what is` → brief answer + web sources  
> `give me` → detailed information  
> `compare`, `how to`, `fact` → structured, in-depth response

---

## Psychological Analysis

```
/analyze <your query>
```

Example:
```
/analyze what patterns do I keep repeating when I'm stressed?
```

Arwanos will:
- Read your `psychoanalytical.json` journal
- Extract entries relevant to your question
- Detect emotional shifts, behavioral contradictions, and subconscious patterns
- Respond like a close friend who has read your diary

For conversational mode (talks *with* you):
```
/lo <your thought>
```

---

## Arabic Processing Mode (`/ap`)

`/ap` is a full **AR → EN → pipeline → AR** translation sandwich.  
Write your query in Arabic — Arwanos translates it to English using `qwen2.5:7b`, runs it through the chosen pipeline (`llama3:8b`), then translates the answer back to Arabic.

```
/ap <query>
```

You can also combine it with any other pipeline using a flag:

| Command | What it runs |
|---|---|
| `/ap <query>` | Normal RAG pipeline |
| `/ap -lo <query>` | Lo companion mode |
| `/ap -analyze <query>` | Journal analysis |
| `/ap -rag <query>` | RAG session search |
| `/ap -deep <query>` | Deep live web search |


Arwanos translates the Arabic to English, does a live web search, synthesizes the results, then returns the answer in Arabic.

> **Requires:** `qwen2.5:7b` pulled in Ollama alongside your main model.
> ```bash
> ollama pull qwen2.5:7b
> ```

---

# A Note on Safety

Arwanos will **not** operate in extremely sensitive areas (nuclear, WMD, or other real-world danger zones).

It supports:
- Security researchers and students
- Psychological analysis and self-reflection
- Study session organization and RAG-based learning

---

# What Makes Arwanos Different

| Feature | Arwanos | Typical chatbot |
|---|---|---|
| Privacy | 100% local, no cloud | Cloud API, data sent to servers |
| Web search | ARM-adaptive — only when needed | Always on or always off |
| Psychoanalysis | Reads private journal, intent-aware | No personal data |
| Session RAG | Keyword-indexed, near-instant lookup | Full re-scan every query |
| Bilingual | Native Arabic BiDi rendering | Basic Unicode |
| Resource control | Per-query ARM budget | Fixed context / token limit |

---

# Common Issues

## `ModuleNotFoundError` (any module)

Always use a virtual environment:

### Windows (PowerShell)

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python Arwanos_v9_8.py
```

### Windows (CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
python Arwanos_v9_8.py
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python Arwanos_v9_8.py
```

---

## Ollama not found / model not responding

```bash
ollama serve
ollama list
ollama pull llama3.1
```

---

# Enjoy Exploring the Potential

This is not just a chatbot. It's a blueprint for:

- AI behavioral analysis
- Psychological context infusion
- Intelligent RAG manipulation
- Internal bypass engineering
- Human-aware Transmitted AI systems
- Natural language processing experiments

If you want more clarification, deeper breakdowns, or advanced discussion — reach out.

---

**Author:** GMM  
**GitHub:** [GMMB1](https://github.com/GMMB1)  
**Support:** [ko-fi.com/ghostman77506](https://ko-fi.com/ghostman77506)

**Learn more about Transmitted AI:**  
[Transmitted AI with Psychological Awareness](https://medium.com/python-in-plain-english/transmitted-ai-with-psychological-awareness-c6369cce8b8f)
