ARWANOS — TEST / DEMO DATA
==========================

This folder contains synthetic demonstration data for showing Arwanos
to reviewers, competition judges, or the public.

ALL DATA IS FICTIONAL — no real personal information is included.


FILES
-----

psychoanalytical.json
  25 demo journal entries spanning Nov 2025 – Mar 2026.
  Designed to showcase /lovelyq analysis:
    - Behavioral patterns (all-or-nothing cycles)
    - Trigger mapping (phone + boredom + night)
    - Habit chains (gym as anchor habit)
    - Recovery patterns ("I always come back")
    - Avoidance after criticism
    - Social comparison spirals
  Covers enough data for rich pattern detection.

lovely_conversations.json
  10 demo conversation turns in /lovely mode.
  Shows the companion's tone: honest, direct, emotionally intelligent,
  asking follow-up questions, not giving generic advice.

demo_session.json
  7-turn conversation about AI/ML topics (RAG, LLMs, embeddings).
  Import via the session import feature to demo /rag command.

lovelyq_custom_prompt.txt
  Sample custom instruction format for /lovelyq.
  Shows the /lovelyq set feature — user-defined response format.

Notes_Data/general_notes.json
  4 technical notes about Arwanos architecture and features.
  Useful for demos of the notes search functionality.


HOW TO USE FOR DEMO
-------------------

1. Point the app to this folder instead of data/:
   - Change data_dir in the path resolver to point to data_test/

2. Or copy the files into data/ temporarily for a demo run.

3. The psychoanalytical.json is ready for /lovelyq queries like:
   - "what are my recurring behavioral patterns?"
   - "what triggers my bad habits?"
   - "where am I growing?"
   - "what contradictions do you see in my behavior?"

4. Import demo_session.json via the session import UI, then use:
   - /rag what is RAG?
   - /rag how does ChromaDB work?
   - /rag what are common RAG pitfalls?
