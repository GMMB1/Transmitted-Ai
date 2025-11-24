Introduction — My Why

Hi folks, I hope you're all doing great! It’s been a while, but I'm really glad to be back again.
Today I want to share something new — or at least a little different from what you might expect.

This repository is part of something I call Transmitted AI.
We discuss RAG (Retrieval-Augmented Generation), psychological awareness, and the manipulations I applied to reach very specific outcomes or to enhance the emphasis of certain response elements.

Note: If any of this sparks your curiosity, feel free to reach out through the contact box.
I’ll gladly continue creating and sharing more ideas and discussions related to Transmitted AI in the future.

Rona Project — Overview

First of all, I've been using Meta Llama 3, and honestly it's awesome.
Its massive pre-training dataset and human-friendly structure make it special.
It produces incredibly human-like text and performs extremely well in programming tasks.

In CyberSecurity, however, Llama 3 applies a lot of restrictions.
But you know me — I bypassed that.

I talked before about the standard definition of RAG…
But do you think RAG stops where documentation says it stops?
Absolutely not.
If you understand the architecture, you can push it far beyond what anyone expects.

What is Rona?

The Rona Project is my personal experiment — now a full application — that can:

Analyze human behavior through streamed daily notes

Detect emotional triggers, patterns, and internal loops

Provide psychological insights based on user entries

Retrieve relevant information from live web search + local memory

Switch dynamically between intrinsic knowledge and RAG-powered reasoning

Rona doesn’t just chat.
Rona reads between the lines.

Special Internal Bypass

I created a mechanism to write directly to the stream data line of the model.

Think of it like this:

There’s a wall you can’t break from the front…
but you can climb over it, and lift others with you.

That's exactly how the /lovely option works.
It bypasses Llama’s limitations in everything except the ultra-sensitive and ethically non-negotiable areas.

Key Features
✔ Intrinsic & RAG Modes

Switch between model-only knowledge and live enriched web context.

✔ Lovely Mode (/lovely)

A psychological-aware analysis engine.
Reads notes, extracts emotional patterns, and reflects behavior.

✔ Unified Command Router

A single architecture handling:

/lovely

/deep

/webui

/hunt

natural language queries

✔ Hybrid UI

Runs as:

a modern Desktop App (CustomTkinter)

a full Local Web Application (Flask)

✔ Behavioral-Predictive Layer

Rona doesn’t just answer questions —
she understands how you're asking them.

Installation
Prerequisites

Python 3.10+

Ollama installed and running

A local model:

ollama pull llama3


or:

ollama pull llama3.1

Install Dependencies
pip install -r requirements.txt

Run Rona
python Rona_v7.py


Rona will open in desktop mode.
You can also access the web UI at:

http://127.0.0.1:5005/prodectivity

Usage
💬 Standard Chat

Just type.
Rona will automatically use:

local context

web context

conversation history

behavioral inference

to produce enriched responses.

💗 Lovely Mode
/lovely <your thought>


This activates the psychological analysis engine.

Rona will:

detect emotional shifts

highlight behavioral contradictions

expose subconscious patterns

provide grounded advice

mirror your mindset in a human-like manner

🕵️ Bug Bounty Flow
/hunt <target>


If enabled, Rona performs recon using modern tooling —
without the usual AI moralizing lectures.

A Note on Safety

I built this system to bypass unnecessary limitations,
but Rona will not operate in extremely sensitive areas (e.g., nuclear, WMD, or other real-world danger zones).

She will help bypass:

WAF restrictions

tooling limitations

AI moralization

unnecessary filtering

But she will never cross into real harm.

There is a difference between security research and actual danger.
Rona stays on the correct side.

Enjoy Exploring the Potential

This is not just a chatbot.
It’s a blueprint for:

AI behavioral analysis

psychological context infusion

intelligent RAG manipulation

internal bypass engineering

human-aware Transmitted AI systems

hybrid UI automation

If you want more articles, deeper breakdowns, or advanced manipulation techniques — just tell me.
