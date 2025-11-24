Rona Project
First of all, I've been using Meta Llama 3, and it's awesome. It has a massive amount of pre-training data and is very human-friendly, which makes it special. It generates human-like text and is trained on a huge dataset collected from various sources. It's actually very good for programming as well, although I don't prefer using it much for CyberSecurity tasks because it places a lot of restrictions on the content—but anyway, I bypassed that.

I talked about the standard definition of RAG (Retrieval-Augmented Generation), but do you think it's limited to that? Absolutely not. You can do almost anything you want with this paradigm.

What is Rona?
The Rona Project—my own project—can analyze human behavior as someone writes their notes. It retrieves relevant information, helps identify personal patterns and the triggers that shape behavior, and provides advice and psychological insights based on those patterns.

As I mentioned earlier, I bypassed the limitations through manipulation. This is a completely different story that honestly deserves its own article. But to put it briefly: I created a special option that allows writing directly to the stream data line of the AI model.

Key Features
Intrinsic & RAG Modes: Switches between internal model knowledge and live, enriched web context.

The "/lovely" Option: Imagine we have a wall we cannot break through from the front—but we can climb over it and help others climb with us. That's the idea behind this implementation. I built the /lovely option to ensure that Llama's limitations wouldn't restrict me from analyzing deep human behavior.

Unified Command Router: A clean architecture that handles slash commands (/hunt, /deep, /webui) and natural language queries in one pipeline.

Hybrid UI: Runs as a sleek Desktop App (CustomTkinter) or a Local Web Server (Flask).

Installation
If you're good at prompt engineering, the whole situation becomes a completely different story. :) But to get the code running, follow these steps:

Prerequisites:

Python 3.10+

Ollama installed and running.

Pull the model: ollama pull llama3 (or llama3.1).

Install Dependencies: I've prepared the requirements file. Just run:

Bash

pip install -r requirements.txt
Run Rona:

Bash

python Rona_v7.py
Usage
Once Rona is running, you aren't just chatting with a bot. You are interacting with a system designed to understand context.

Standard Chat: Just type. Rona will search locally and the web to give you an enriched answer.

Lovely Mode: Type /lovely <your thought> to trigger the behavioral analysis engine. It reads between the lines.

Bug Bounty Flow: Use /hunt <target> (if you enabled the integration) to perform recon without the usual AI moralizing lectures.

A Note on Safety
I built this to ensure that Llama's limitations wouldn't restrict me, except in extremely sensitive areas like nuclear weapon information—not things like bypassing WAF security or preventing the provision of specific unharmful information.

Enjoy exploring the potential.
