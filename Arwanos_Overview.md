# Arwanos — AI Personal Assistant
## Technical Overview & Feature Reference

---

## What Is Arwanos?

Arwanos is a **locally-running, privacy-first AI assistant** built in Python.  
It runs entirely on your own machine using **Ollama** as the inference backend — no data ever leaves your device, no cloud API is called for core reasoning.

**Model:** `llama3:8b-instruct-q4_K_M` (quantized, GPU-accelerated)  
**Interface:** Native desktop GUI (CustomTkinter) + optional local Web UI (Flask)  
**Languages:** English and Arabic (full BiDi/RTL rendering support)

---

## Core Architecture

```
User Input
    │
    ▼
Command Parser  ──►  Slash Command Handler
    │                      │
    │ (normal text)         ▼
    ▼              Specific pipeline
ARM Scorer                 (see below)
    │
    ▼
ComplexityProfile  →  ResourceBudget
    │                  (context items, web search on/off,
    │                   response token limit)
    ▼
generate_response()
    │
    ├── Web Search (DDG)  [only if ARM says needed]
    ├── Session RAG       [only if session imported]
    ├── Conversation History (last 4 turns, 200–400 chars each)
    │
    ▼
_call_llm_with_context()
    │
    ▼
SimpleOllama → Ollama HTTP API → LLM
    │
    ▼
BiDi/RTL post-process → Display
```

---

## ARM — Adaptive Resource Management

Every normal query is **scored before the LLM is called**.

### Step 1 — ComplexityProfile
Arwanos scores the query across 4 dimensions (0–3 each):

| Dimension | What it measures | Example |
|---|---|---|
| `search_need` | Does the query need live/external data? | "latest news about X" = 3 |
| `context_need` | Does it need session history? | "explain that again" = 2 |
| `response_length` | How long should the answer be? | "list all..." = 3 |
| `reasoning_steps` | How complex is the reasoning? | "compare A and B" = 2 |

### Step 2 — ResourceBudget
The profile is converted into hard resource limits:

| Profile Level | Web Search | Context Items | Snippet Length | Max Tokens |
|---|---|---|---|---|
| 0 (simple) | ✗ | 2 | 200 chars | 640 |
| 1 (moderate) | ✗ | 3 | 300 chars | 960 |
| 2 (detailed) | ✓ (3 results) | 4 | 350 chars | 1280 |
| 3 (complex) | ✓ (7 results) | 5 | 400 chars | 1600 |

**Effect:** A simple factual question uses almost no resources. A complex research question automatically triggers web search and longer output — without the user doing anything.

---

## Input Modes

### 1. Normal Conversation
**Trigger:** Any text without a `/` prefix  
**Pipeline:**
```
User text → ARM score → [optional DDG search] → [optional session context] →
4-turn history → LLM → response
```
**What makes it smart:** ARM decides dynamically whether to search the web. Simple questions are answered instantly from the model's knowledge. Questions about current events, specific documentation, or recent data automatically pull live results.

---

### 2. `/deep <question>`
**Trigger:** `/deep` command  
**Pipeline:**
```
User text → FORCE DDG web search (skip ARM) → synthesize results → LLM → response
```
**Difference from normal:** ARM might decide a query doesn't need web search. `/deep` overrides that — it always searches, regardless of complexity score. Use this when you know you need fresh, sourced information.

**Output format:** Numbered source list with titles, URLs, and synthesized answer.

---

### 3. `/rag <question>`
**Trigger:** `/rag` command (requires an imported session)  
**Pipeline:**
```
User text → keyword inverted index lookup (O(keywords) not O(turns)) →
top-3 scored excerpts → LLM with citation → response [1][2][3]
```
**What it does:** Searches a previously imported JSON session file using a keyword index built at import time. The index maps every meaningful word to the turn numbers where it appears. Lookup is near-instant even for 200+ turn sessions.

**Difference from normal:** No web search, no live LLM reasoning from scratch. The answer comes directly from content YOU imported — your own notes, a study session, a document.

**Import flow:**
```
File picker → JSON parse → build inverted keyword index →
store _imported_session_len → display "N keywords across M turns"
```

---

### 4. `/lovelyq <question>`
**Trigger:** `/lovelyq` command  
**Pipeline:**
```
Read ALL psychoanalytical.json entries
    │
    ▼
_detect_intent(question)
[behavioral / emotional / avoidance / contradiction / growth / social]
    │
    ▼
_build_focused_context(entries, question)
→ Python extracts only sentences matching question keywords
→ Matching entries: relevant sentences (up to 5 per entry, 250 chars each)
→ Other entries: compact 90-char summary line
    │
    ▼
_pattern_summary(all_entries)
→ mood trend, activity counts, emotion counts, gap days
    │
    ▼
Single LLM call (8192 ctx, 1200 tokens)
with rich psychoanalytical system prompt
    │
    ▼
Save Q&A to lovely_conversations.json
```

**Difference from all other modes:** This mode is NOT about knowledge — it is about YOU. It reads your private psychological journal (`psychoanalytical.json`), extracts the entries most relevant to your question, and responds as a close friend who has read your diary.

**Intent detection examples:**

| Question contains | Analytical lens activated |
|---|---|
| "pattern", "trigger", "habit", "behavior" | Behavioral Patterns & Triggers |
| "feel", "mood", "anxious", "sad", "emotion" | Emotional Arc Analysis |
| "avoid", "procrastinate", "delay", "skip" | Avoidance Patterns |
| "contradict", "promise", "said I would" | Contradictions (stated vs actual) |
| "progress", "improve", "grow", "better" | Growth Tracking |
| "friend", "family", "alone", "relationship" | Social/Relationship Patterns |

**Custom instructions:** `/lovelyq set <your format>` — saves a permanent instruction file that gets injected into every `/lovelyq` call, letting you control the exact output structure.

---

### 5. `/lovely <text>`
**Trigger:** `/lo` or `/lovely` command  
**Data source:** `lovely_conversations.json` (conversational history)  
**Difference from /lovelyq:** `/lovely` is the emotional conversation mode — it talks WITH you like a close friend. `/lovelyq` is the analytical mode — it reads your journal and reports back what it finds.

---

### 6. Reply-to-Selection (fast path)
**Trigger:** Highlight text in the chat → click "Reply" → enter question  
**Pipeline:**
```
Selected text (≤600 chars) + question → DIRECT LLM call
(bypasses ARM, web search, history scan entirely)
→ response in ~10 seconds
```
**Why it exists:** When you want to ask about something specific already on screen, running the full pipeline is wasteful. The selected excerpt IS the context — no search needed.

---

### 7. `/tr <text>`
**Trigger:** `/tr` command  
**Function:** Detects source language → translates to English (if Arabic) or Arabic (if English) → explains key terms.

---

### 8. `/webui`
**Trigger:** `/webui start`  
**Function:** Starts a local Flask web server. Provides a web interface for journal browsing, date-range analysis, and weekly reports. All data stays local — the server binds to `127.0.0.1` only.

---

### 9. `/ap <query>`
**Trigger:** `/ap` command  
**Pipeline:**
```
Arabic query
    │
    ▼
AR → EN  (qwen2.5:7b, num_ctx: 4096)
    │
    ▼
Selected pipeline (flag-controlled)
    │
    ▼
EN → AR  (qwen2.5:7b)
    │
    ▼
Arabic response → Display
```
**Available flags:**

| Flag | Pipeline used |
|---|---|
| *(none)* | Normal RAG |
| `-lo` | Lovely companion |
| `-lovelyq` | Journal analysis |
| `-rag` | RAG session search |
| `-deep` | Deep live web search |

**What it does:** A full translation sandwich. The user writes in Arabic, Arwanos translates to English, runs the chosen pipeline, then translates the answer back to Arabic. Requires `qwen2.5:7b` pulled in Ollama.

---

### 10. `/dev <arg>`
**Trigger:** `/dev` command  
**Function:** Internal developer inspector — provides in-app source introspection without file reads. Uses an AST index for zero-cost list operations and chunked reads for slice operations. Intended for development and debugging.

---

## Data Layer

| File | Used by | Content |
|---|---|---|
| `psychoanalytical.json` | `/lovelyq` | Private journal entries (date, title, details, mood score) |
| `lovely_conversations.json` | `/lovely`, `/lovelyq` | Q&A history from lovely/lovelyq sessions |
| `data/lovelyq_custom_prompt.txt` | `/lovelyq` | User-defined response format instructions |
| `config.json` | All | Model, GPU, UI, ARM limits |
| Imported session `.json` | `/rag` | Any JSON session file (study notes, documents) |

---

## Switching to Test Data (`data_test/`)

The repo includes a `data_test/` folder with **synthetic demo data** — fictional entries designed to showcase every feature without exposing real personal data.

### What's inside `data_test/`

| File | Content |
|---|---|
| `psychoanalytical.json` | 25 demo journal entries (Nov 2025 – Mar 2026), covering behavioral patterns, trigger mapping, habit chains, avoidance, and social comparison cycles |
| `lovely_conversations.json` | 10 demo `/lovely` conversation turns showing companion tone |
| `demo_session.json` | 7-turn AI/ML conversation — import this to demo `/rag` |
| `lovelyq_custom_prompt.txt` | Sample custom instruction format for `/lovelyq set` |
| `Notes_Data/general_notes.json` | 4 technical notes about Arwanos architecture |

### Method 1 — Edit the path in code (permanent switch)

There are two `_find_repo_paths()` functions in `Arwanos_v9_8.py`. Change `"data"` to `"data_test"` in both:

**First one (desktop app / LovelyAnalyzer) — around line 2880:**
```python
# Before
"data": root / "data",

# After
"data": root / "data_test",
```

**Second one (Flask / Web UI) — around line 4492:**
```python
# Before
data_dir = (
    here / "data"
).resolve()

# After
data_dir = (
    here / "data_test"
).resolve()
```

Revert both lines to `"data"` when done testing.

### Method 2 — Copy files into `data/` (quick, no code change)

```bash
# Back up your real data first
cp data/psychoanalytical.json data/psychoanalytical.json.bak
cp data/lovely_conversations.json data/lovely_conversations.json.bak

# Copy test data in
cp data_test/psychoanalytical.json data/
cp data_test/lovely_conversations.json data/
cp data_test/lovelyq_custom_prompt.txt data/

# Restore when done
cp data/psychoanalytical.json.bak data/psychoanalytical.json
cp data/lovely_conversations.json.bak data/lovely_conversations.json
```

### Demo queries to try after switching

Once pointing at `data_test/`, use these to verify all pipelines work:

```
# Journal analysis
/lovelyq what are my recurring behavioral patterns?
/lovelyq what triggers my bad habits?
/lovelyq where am I growing?
/lovelyq what contradictions do you see in my behavior?

# RAG — import demo_session.json first via session import UI
/rag what is RAG?
/rag how does ChromaDB work?
/rag what are common RAG pitfalls?
```

---

## Unique Technical Features

### BiDi / RTL Support
Arabic text is detected using a **15% character threshold** (at least 15% of non-whitespace chars must be Arabic). When true, the text is processed through the `python-bidi` library and wrapped in RTL Unicode markers before display. English text with stray Arabic characters does NOT trigger reversal.

### Session Import + Keyword Index
When a JSON session is imported:
1. An **inverted keyword index** `{word → [turn_indices]}` is built in Python
2. Every subsequent new turn is **incrementally added** to the index
3. `/rag` searches this index in O(query_keywords) — constant time regardless of session size
4. `_imported_session_len` marks where imported turns end, so the LLM's conversation history excludes them (prevents hallucination from old context)

### ARM Bypass Protection
The `_route_user_input_async` function previously bypassed ARM entirely. It now delegates to `generate_response()` which always goes through the full ARM pipeline. Web search only fires when the query actually needs it.

---

## What Makes Arwanos Different

| Feature | Arwanos | Typical chatbot |
|---|---|---|
| Privacy | 100% local, no cloud | Cloud API, data sent to servers |
| Web search | ARM-adaptive (only when needed) | Always on or always off |
| Psychoanalysis | Reads private journal, intent-aware | No personal data |
| Session RAG | Keyword-indexed O(1) lookup | Full re-scan every query |
| Bilingual | Native Arabic BiDi rendering | Basic Unicode |
| Resource control | Per-query ARM budget | Fixed context/token limit |
| Persona | Named character with emotional depth | Generic assistant |

---

## System Requirements

- Python 3.11+
- Ollama (local inference server)
- CUDA-capable GPU (configured for 25 layers GPU offload)
- ~6GB VRAM for `llama3:8b-instruct-q4_K_M`

---

*Arwanos v9.8 — Built by GMM*

