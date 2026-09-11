# Mental State Monitor — ML-Powered Self-Analysis

> Back to [README](../README.md)

The **Mental State Monitor** is the most advanced feature in Arwanos v10.  
It transforms the application from a journaling assistant into an intelligent psychological monitoring system that learns from your history, adapts to your patterns, and asks increasingly precise questions over time.

Access it by clicking **🧭 Monitor** in the web UI action bar — it opens as a dedicated page in the same browser.

---

## How It Works — The Full ML Pipeline

### Phase 1 — Tuning (on every new session)

When you click **Start Session**, Arwanos does not just generate random questions.  
It first performs a full tuning pass over **all your data**:

- Reads **every journal entry** you have ever written (not just recent ones)
- Reads the **complete habits history** — total success/failure counts, failure dates, patterns
- Reads the **analyzed.json** deep conversation transcript (up to 5,000 characters)
- Reads **all past monitor session insights** — the distilled analysis from previous check-ins
- Reads **past journal cross-reference inferences** — what past sessions revealed when answers were compared against journal history

The LLM synthesizes all of this into a **psychological profile**:

```json
{
  "dominant_themes": ["..."],
  "recurring_patterns": ["..."],
  "habit_struggles": ["..."],
  "mood_trend": "...",
  "unresolved_tensions": ["..."],
  "unexplored_areas": ["..."],
  "strengths": ["..."]
}
```

This profile becomes the foundation for every question generated.

---

### Phase 2 — Question Generation (3 training sources)

Questions are generated using **three simultaneous training sources** — all cross-referenced before any question is written:

| Source | What it contains |
|---|---|
| **Source A** — `analyzed.json` | Deep conversation transcript revealing your core patterns |
| **Source B** — Monitor session history | Every answer you gave in past check-ins + AI insights from those sessions |
| **Source C** — Professional psychology datasets | 7,557 real examples from licensed therapists and counselors |

Source C is not generic — it is **keyword-searched** using your profile themes and inferences, so only the most relevant professional examples are pulled in.

---

### Phase 3 — Insight Pipeline (after you answer)

After answering the session questions, a **3-phase analysis pipeline** runs:

**Phase 1 — Journal Cross-Reference**  
Your answers are compared against **all your journal entries**.  
The LLM finds:
- `confirmed_patterns` — what your answers confirm that journals already showed
- `contradictions` — where what you said today conflicts with what journals show
- `new_revelations` — insights only visible by comparing both sources together
- `progression` — what has moved forward since older entries
- `key_inference` — one sentence capturing the most important finding

**Phase 2 — Dataset Enrichment**  
The inferences from Phase 1 (not the raw answers) become the keyword search signal against the 7,557 professional examples — producing far more precise therapeutic references than searching on answer text alone.

**Phase 3 — Final Insights**  
The LLM generates 4 focused paragraphs using all three phases combined:
1. What today's answers confirm or contradict in the journal history
2. The most significant new revelation from cross-referencing
3. What skipped questions signal and how they connect to confirmed patterns
4. One specific, actionable step for tomorrow grounded in the key inference

---

## Anti-Duplication System

Questions never repeat across sessions. A three-layer programmatic check runs after every generation:

| Layer | What it catches |
|---|---|
| **Character n-gram similarity** (≥ 0.22) | Morphological variants — `strategies` / `strategy`, `emotional` / `emotionally` |
| **Named entity overlap** (≥ 2 shared) | Same real-world subject — person names, exam codes, named habits |
| **Key noun overlap** (≥ 3 shared) | Same topic domain even with completely different wording |

If any generated question is flagged, a **targeted second LLM call** regenerates only that question with explicit instructions to explore different territory. The prompt also includes a per-category **covered-topics map** — extracted keywords from every past question grouped by category — so the LLM knows exactly which ground is already exhausted.

---

## Angle Rotation — Questions Deepen Over Time

Each category has 5 dimensions that rotate with session count:

| Session | `emotional_awareness` explores... | `avoidance_detection` explores... |
|---|---|---|
| #1 | what the emotion actually is | what exactly is being avoided |
| #2 | where it originates | the real cost of continued avoidance |
| #3 | how it drives decisions | what you fear finding if you look |
| #4 | when it first appeared | the smallest step toward it |
| #5 | what consistently triggers it | when this avoidance pattern started |
| #6+ | cycle repeats — with full history to reference | ... |

Depth also scales with session count: early sessions establish baseline awareness, mid-range sessions probe gaps between what you say and what journals show, deep sessions challenge resistance to change directly.

---

## Psychology Datasets

Three public datasets are embedded and locally indexed:

| Dataset | Hugging Face ID | Entries |
|---|---|---|
| **CounselChat** | `nbertagnolli/counsel-chat` | 2,749 |
| **Mental Health Counseling Conversations** | `Amod/mental_health_counseling_conversations` | 3,508 |
| **ESConv** (ACL 2021) | `thu-coai/esconv` | 1,300 |

**Total: 7,557 professional examples** — keyword-searched at session start and at insight generation, matched to your specific patterns, not used as generic templates.

These datasets are **not included in the repository** (they belong in your local `data/` folder which is gitignored for privacy). Use the setup script below to download and index them automatically.

---

## Dataset Setup — One Command

> **This step is required for the Mental State Monitor to work.**  
> Without `data/psych_datasets_index.json`, the ML pipeline has no professional reference material.

### Step 1 — Install the datasets library

```bash
pip install datasets
```

> Already included if you ran `pip install -r requirements.txt` — the `datasets` package from Hugging Face is listed there.

### Step 2 — Run the build script

```bash
python build_datasets.py
```

That's it. The script will:

1. Download **CounselChat** from `nbertagnolli/counsel-chat`
2. Download **Mental Health Counseling Conversations** from `Amod/mental_health_counseling_conversations`
3. Download **ESConv** from `thu-coai/esconv`
4. Process all three into a unified keyword-indexed format
5. Save the result as `data/psych_datasets_index.json`

Expected output:

```
Downloading CounselChat (nbertagnolli/counsel-chat)…
  ✓ 2749 entries
Downloading Mental Health Counseling Conversations…
  ✓ 3508 entries
Downloading ESConv (thu-coai/esconv)…
  ✓ 1300 entries

✅ Done — 7,557 total entries
   Saved → /your/path/Arwanos-v10/data/psych_datasets_index.json

Breakdown by source:
   counsel_chat                          2749
   mental_health_counseling              3508
   esconv                                1300
```

> **No account or API key required** — all three datasets are publicly available on Hugging Face.  
> The download requires ~50 MB of disk space and a one-time internet connection.  
> After the script completes, the Monitor works **100% offline**.

### What the index looks like

Each entry in `psych_datasets_index.json` follows this format:

```json
{
  "source": "counsel_chat",
  "topic": "depression",
  "context": "The client's question or situation...",
  "response": "The therapist's or counselor's response...",
  "keywords": ["anxiety", "depression", "patterns", "avoidance", "..."]
}
```

The Monitor searches this index using your psychological profile keywords — it never uses these as generic templates. Every example pulled is matched specifically to your current patterns and session inferences.

---

## How the Web UI Combines Everything

When you click **🧭 Monitor** in the web UI sidebar:

```
Your journal + habits history
         ↓
   LLM builds your psychological profile
         ↓
   Profile keywords → search psych_datasets_index.json
         ↓
   3 most relevant professional examples per theme (Source C)
         ↓
   LLM generates questions using: your data + past sessions + professional examples
         ↓
   You answer in the browser
         ↓
   Answers cross-referenced against your full journal history
         ↓
   Dataset searched again using inference keywords (not raw answers)
         ↓
   Final insights: 4 paragraphs + one actionable step
         ↓
   Saved to data/monitor_sessions.json → fed into next session's tuning
```

The datasets power **Source C** in this pipeline — they provide the professional therapeutic vocabulary and response patterns that the LLM uses to frame questions and enrich insights. Your personal data never leaves your machine.

All datasets are stored locally in `data/` — no external API calls after setup.

---

## Progress Tracking

The monitor tracks cumulative engagement across sessions:

- **Topic progress bars** — how often each psychological category was engaged vs skipped
- **Average mood shift** — tracks whether sessions correlate with mood improvement over time
- **Journal inferences feed-forward** — each session's cross-reference findings are stored and fed back into the next session's tuning phase, making the system progressively more precise
- **Session history** — full Q&A + AI insights for every completed session, viewable in the browser

---

