# Contributing to Arwanos

Thanks for your interest. Arwanos is young, and contributions of every size matter — a typo fix, a bug report, a tested install on a platform we haven't tried, or a new feature.

## The one rule: never commit personal data

Arwanos exists to handle the most private things people write. Treat that seriously in every contribution.

- **Never commit** anything from `data/`, `data_test/`, `chroma_db/` or `logs/`. They're gitignored — don't force-add them.
- **Never paste real journal entries, conversations, or memory facts** into issues, pull requests, commit messages, code comments, or tests. Invent examples instead.
- **Take screenshots in Demo mode only** (Settings → Data Folder → Demo).
- If you find a way personal data could leak — into a log, a prompt dump, the web UI, or git — report it privately and treat it as a security issue, not a normal bug.

## Getting set up

Follow the [installation guide](docs/INSTALL.md), then switch to **Demo mode** so you're developing against sample data rather than your own.

## Finding your way around

Most of the application lives in `Arwanos_v10.py`, a single large file. Splitting it up is on the [roadmap](ROADMAP.md). Until then, these are the places you'll most likely need:

| Area | Where to look |
|---|---|
| Companion mode (`/lo`) | `LovelyAnalyzer.analyze_no` |
| Long-term memory — fact extraction | `LovelyAnalyzer._extract_and_save_lo_memory` |
| Long-term memory — semantic recall | `LovelyAnalyzer._lo_rag_recall` |
| Journal analysis (`/analyze`) | `ArwanosApp._cmd_lovelyq` |
| Every LLM call and prompt assembly | `ArwanosApp._call_llm_with_context` |
| Voice | `ArwanosApp._voice_listen_turn` |
| Local web API | `ArwanosApp.start_web_ui` |
| Settings dialog | `ArwanosApp._show_settings_dialog` |
| Web app front end | `renderer/` |
| Startup briefing | `banner.py` |

Search the file for the name — it's faster than scrolling.

A useful thing to know: `_call_llm_with_context` is shared by every command. Changes there affect `/rag`, `/analyze`, `/deep` and the Mental State Monitor, not just the feature you're working on. Conversation-specific behaviour is gated behind its `chat_mode` flag.

## Before opening a pull request

There's no automated test suite yet — adding one is on the roadmap and would be a very welcome contribution. Until then, please check:

```bash
python -c "import Arwanos_v10"               # the module still imports
node --check renderer/js/<file-you-changed>.js # front-end syntax, if you touched JS
```

Then run the app in Demo mode and exercise the feature you changed. In your pull request, say what you tested and on which platform — "tested on Ubuntu 24.04, `/lo` and `/analyze`" is exactly the right level of detail.

## Pull requests

1. Fork the repository and create a branch from `main`.
2. Keep each pull request focused on one change.
3. Explain **why**, not only what. If you fixed a bug, describe what was going wrong.
4. Link the issue it addresses, if there is one.

## Good first contributions

- **Try the install on your platform** and [report how it went](https://github.com/GMMB1/Transmitted-Ai/issues/new?template=feedback.yml). Windows and macOS reports are especially valuable.
- **Improve the docs** wherever you got stuck.
- **Pick up a roadmap item** marked as a good first issue.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

## Questions

Open an issue — there are no silly questions, and a question often reveals where the documentation needs work.
