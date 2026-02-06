"""
Dependency manifest for Rona.

This is a simple, readable list of the Python packages used by the project.
If you prefer a standard pip file, you can mirror these into requirements.txt.
"""

# Core runtime dependencies (needed for the main app UI + basic features)
REQUIRED = [
    "customtkinter",
    "requests",
    "aiohttp",
    "pillow",
    "flask",
    "flask-cors",
]

# Optional features (install if you want RAG, local DB, or extra tooling)
OPTIONAL = [
    "ollama",          # Python client for Ollama
    "chromadb",        # Chroma vector DB
    "langchain",
    "langchain-ollama",
    "psutil",
    "pdfplumber",
    "spacy",
    "nltk",
]

# Notes:
# - tkinter ships with Python on most systems and is not a pip package.
# - If you don't need the web UI, you can skip flask/flask-cors.
# - If you don't use RAG or embeddings, you can skip OPTIONAL.
