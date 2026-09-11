# Roadmap

Where Arwanos is heading, and where help would make the biggest difference. Nothing here has a date attached — priorities move based on [what users tell us](https://github.com/GMMB1/Transmitted-Ai/issues/new?template=feedback.yml).

Items marked **good first issue** are well-scoped and don't require understanding the whole codebase.

---

## 1. Make it easy to try

The biggest barrier right now is getting from "this looks interesting" to "it's running".

- [x] Fictional demo dataset that pre-fills Demo mode on first use
- [ ] **One-command setup script** — detect Python and Ollama, pull the model, create the virtual environment
- [ ] **Cross-platform voice** — recording uses ALSA `arecord`, so voice is Linux-only. Moving to a portable audio library would bring it to Windows and macOS
- [ ] **Remove Linux- and CUDA-specific paths** from the launcher so it works across distributions and GPU setups
- [ ] **Pre-built releases** with versioned changelogs
- [ ] Demo weekly and monthly reports — these live in browser storage and aren't covered by the demo dataset yet
- [ ] **Verified install reports** for Windows and macOS — **good first issue**

## 2. Make it easy to contribute

- [ ] **Automated tests** — there are none yet. Several pieces are pure functions and ideal starting points — **good first issue**:
  - the reference-resolution classifier that decides whether a message answers the previous question
  - the prompt budget that trims history to fit the context window
  - the recall noise filter and relative-date labels
- [ ] **Continuous integration** — at minimum, an import check and JavaScript syntax check on every pull request
- [ ] **Split `Arwanos_v10.py`** — at roughly 16,000 lines, one file is the main obstacle for new contributors. Natural boundaries: companion memory, journal analysis, voice, web API, and UI

## 3. Make it smarter

- [ ] **Evaluate conversation quality** — memory and retrieval are measured at the data level, but how replies actually read in conversation hasn't been systematically evaluated
- [ ] **Broader model support** — prompts are tuned for Llama 3 8B, which struggles to follow many rules at once. Validate alternatives such as Qwen 2.5 and adapt prompts per model
- [ ] **Configurable recall thresholds** — the correct similarity cutoff depends on the embedding model, and a mismatch silently disables memory
- [ ] Surface *why* a memory was recalled, so users can see and correct what Arwanos believes about them

## 4. Make it safer

- [ ] **Optional encryption at rest** for journal, memory, and conversation data
- [ ] **Memory review and deletion** — let users see every stored fact about them and remove any of it
- [ ] Privacy audit of every path that can write personal content to disk — logs, caches, and exports

---

## Recently completed

- Long-term companion memory: fact extraction, semantic recall, time awareness, and archiving instead of deleting old conversations
- Prompt budgeting that keeps the system prompt intact on long conversations
- Voice mode selector and noise-floor calibration
- Separate storage for Demo mode so personal reports never appear in it

---

Want to take something on? Comment on the related issue — or open one — so work isn't duplicated. See the [contributing guide](CONTRIBUTING.md).
