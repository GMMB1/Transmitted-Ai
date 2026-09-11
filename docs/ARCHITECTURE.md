# ARM — Adaptive Resource Management

> Back to [README](../README.md)

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


## Command reference



| Command | Function |
|---|---|
| `/lo <text>` | Enhanced emotional response mode |
| `/analyze <query>` | Psychological journal analysis |
| `/deep <question>` | Forced live web search |
| `/rag <question>` | Search imported session |
| `/vo /lo <text>` | Companion mode + text-to-speech voice output |
| `/webui start` | Launch local web interface |
