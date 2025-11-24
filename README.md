# Introduction — My Why

Hi folks, I hope you're all doing great! It’s been a while, but I'm really glad to be back again.  
Today I want to share something new — or at least a little different from what you might expect.

This repository is part of something I call **Transmitted AI**.  
We discuss RAG (Retrieval-Augmented Generation), psychological awareness, and the manipulations I applied to reach very specific outcomes or to enhance the emphasis of certain response elements.

> **Note:** If any of this sparks your curiosity, feel free to reach out through the contact box.  
> I’ll gladly continue creating and sharing more ideas and discussions related to Transmitted AI in the future.

---

# Rona Project — Overview

First of all, I've been using **Meta Llama 3**, and honestly it's awesome.  
Its massive pre-training dataset and human-friendly structure make it special.  
It produces incredibly human-like text and performs extremely well in programming tasks.

In CyberSecurity, however, Llama 3 applies a lot of restrictions.  
But you know me — I bypassed that.

I talked before about the standard definition of RAG…  
But do you think RAG stops where documentation says it stops?  
**Absolutely not.**  
If you understand the architecture, you can push it far beyond what anyone expects.

---

# What is Rona?

The **Rona Project** is my personal experiment — now a full application — that can:

- Analyze human behavior through streamed daily notes  
- Detect emotional triggers, patterns, and internal loops  
- Provide psychological insights based on user entries  
- Retrieve relevant information from live web search + local memory  
- Switch dynamically between intrinsic knowledge and RAG-powered reasoning  

Rona doesn’t just chat.  
**Rona reads between the lines.**

---

## Special Internal Bypass

I created a mechanism to write directly to the **stream data line** of the model.

Think of it like this:

> There’s a wall you can’t break from the front…  
>  
> but you can climb over it, and lift others with you.

That's exactly how the **/lovely** option works.  
It bypasses Llama’s limitations in everything except the ultra-sensitive and ethically non-negotiable areas.

---

# Key Features

### ✔ Intrinsic & RAG Modes  
Switch between model-only knowledge and live enriched web context.

### ✔ Lovely Mode (`/lovely`)  
A psychological-aware analysis engine.  
Reads notes, extracts emotional patterns, and reflects behavior.

### ✔ Unified Command Router  
A single architecture handling:

- `/lovely`  
- `/deep`  
- `/webui`  
- `/hunt`  
- natural language queries  

### ✔ Hybrid UI  
Runs as:

- a modern Desktop App (CustomTkinter)  
- a full Local Web Application (Flask)  

### ✔ Behavioral-Predictive Layer  
Rona doesn’t just answer questions —  
she **understands how you're asking them**.

---

# Installation

## Prerequisites

- Python **3.10+**
- **Ollama** installed and running
- A local model:

```bash
ollama pull llama3
