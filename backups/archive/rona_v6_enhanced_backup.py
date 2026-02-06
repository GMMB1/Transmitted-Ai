# -*- coding: utf-8 -*-
"""
Rona v6 Enhanced - Final merged edition
- Warm humanlike greeting
- Opt-in local advanced_mode for detailed defensive testing and attack guidance ( exploit payloads)
- Unified deep search (local vector DB  DuckDuckGo  optional Google CSE  conversation)
- Image generation (non-blank), image upload, analysis, registry  indexing
- Slash commands: /help, /upload, /deep, /imggen, /imginfo
- Arabic shaping (if libs installed)
- Load balancer, file processing, Chroma RAG (optional)
"""
import random
import shutil
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor   

# put this near the top of rona_v6.py
try:
    from langchain_community.chat_models import ChatOpenAI  # new location
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI  # legacy fallback
    except Exception:
        ChatOpenAI = None  # handle absence gracefully

# Optional PIL imports for image processing
try:
    from PIL import Image, ImageTk, ImageSequence

    PIL_OK = True
except Exception:
    PIL_OK = False

import re, datetime
from typing import List, Dict, Any

try:
    import language_tool_python

    GRAMMAR_TOOL_OK = True
except ImportError:
    GRAMMAR_TOOL_OK = False
try:
    from flask import Flask, request, jsonify, send_from_directory
    import werkzeug

    FLASK_OK = True
except Exception:
    FLASK_OK = False

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chat_models import ChatOpenAI
import datetime
import hashlib
import json
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import re
import subprocess
import platform
import psutil
import requests
from pathlib import Path
import shutil
import asyncio
import aiohttp
import random
from typing import List, Dict, Any, Optional, Tuple, Union
import sqlite3

# ---------- Psycho UI Server (Flask) ----------
import socket, atexit, threading
from werkzeug.serving import make_server
from flask import Flask, jsonify, request, send_from_directory


# ---- Rona theme (same vibe everywhere) ----
RONA_THEME = {
    "window_bg": "#0b0b0e",  # whole app background
    "topbar_bg": "#15161b",  # top bar
    "panel_bg": "#121319",  # chat frame / containers
    "textbox_bg": "#1A1D23",  # chat text area + entry bg
    "accent": "#ED1616",  # your red
    "accent_hover": "#e14e09",
    "halo": "#7c0b8d",  # the halo color you used
    "text": "#ffffff",
    "muted": "#b7b7b7",
    # message colors (you already set tags, keep or adjust here)
    "user_fg": "#b818cd",
    "assistant_fg": "#ED1616",
    "system_fg": "#808080",
    "terminal_fg": "#AE9D47",
    # fonts
    "body_font": ("DejaVu Sans", 16),
    "mono_font": ("DejaVu Sans Mono", 12),
    "ar_font_name": "Noto Naskh Arabic",
}

# Optional image libs
try:
    from PIL import Image, ImageDraw, ImageFont, ExifTags
except Exception:
    Image = ImageDraw = ImageFont = ExifTags = None

# Optional Arabic libs
try:
    import arabic_reshaper
    from bidi.algorithm import get_display as bidi_get_display
except Exception:
    arabic_reshaper = None
    bidi_get_display = None

# Optional NLTK and spaCy imports
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import spacy

    # Load the spacy model
    nlp = spacy.load("en_core_web_sm")
    NLP_OK = True
except Exception:
    nltk = None
    word_tokenize = None
    stopwords = None
    spacy = None
    nlp = None
    NLP_OK = False

# Optional LangChain / Ollama / Chroma stack (used if installed)
# Optional LangChain / Ollama / Chroma stack (modern-first, backwards-safe)
try:
    # ---- Ollama (unchanged) ----
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    # ---- Core prompts/docs (same in 0.2) ----
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.documents import Document as LCDocument

    # ---- Memory (path is stable across 0.1 -> 0.2) ----
    from langchain.memory import ConversationBufferWindowMemory

    # ---- Text splitter (stable) ----
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # ---- Agents: create_tool_calling_agent is gone in 0.2; use REACT agent ----
    try:
        # LC 0.2+
        from langchain.agents import AgentExecutor, create_react_agent
        CREATE_AGENT = "react"   # marker
    except Exception:
        # Fallback for older LC 0.1.x (if someone downgrades later)
        from langchain.agents import create_tool_calling_agent, AgentExecutor
        create_react_agent = None
        CREATE_AGENT = "tool_calling"

    # ---- Tools decorator (stable) ----
    from langchain.tools import tool

    # ---- Chroma vectorstore import (prefer community in LC 0.2) ----
    try:
        from langchain_community.vectorstores import Chroma
    except Exception:
        # very old stacks might have langchain_chroma
        from langchain_chroma import Chroma

    LC_OK = True
except Exception as e:
    print("⚠️ LangChain/Ollama stack import error:", repr(e))
    LC_OK = False


# ---------- Configuration ----------
class Config:
    def __init__(self):

        # --- dragon state (must exist before you show anything) ---
        self._dragon_windows: list = []
        self._dragon_frames_per_win: list = []
        self._dragons_timer = None

        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        # ...
        self.processed_dir = self.data_dir / "processed"
        # ADD:
        self.chroma_db_dir = self.base_dir / "chroma_db"
        self.logs_dir = self.base_dir / "logs"
        self.uploads_dir = self.data_dir / "uploads"
        self.downloads_dir = self.data_dir / "downloads"
        self.images_dir = self.data_dir / "images"
        self.processed_dir = self.data_dir / "processed"
        self.images_registry = self.data_dir / "images_registry.json"
        self.sqlite_path = self.base_dir / "assets.sqlite3"
        self.psycho_file = self.data_dir / "psychoanalytical.json"

        self.psycho_file = self.data_dir / "psychoanalytical.json"
        if not self.psycho_file.exists():
            self.psycho_file.write_text("[]", encoding="utf-8")

        for d in [
            self.data_dir,
            self.chroma_db_dir,
            self.logs_dir,
            self.uploads_dir,
            self.downloads_dir,
            self.images_dir,
            self.processed_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        if not self.images_registry.exists():
            self.images_registry.write_text("[]", encoding="utf-8")
        # ADD:
        if not self.psycho_file.exists():
            self.psycho_file.write_text("[]", encoding="utf-8")

        self.config = self.load_config()
        # system specs: detect where possible
        self.system_specs = {
            "gpu_memory": 8,
            "ram_total": 32,
            "cpu_cores": psutil.cpu_count() if psutil else 1,
            "gpu_model": "RTX 4060",
        }

    def load_config(self):
        cfg_file = self.base_dir / "config.json"
        default = {
            "model_name": "mistral:7b",
            "gpu_settings": {
                "force_gpu": True,
                "gpu_layers": 25,
                "gpu_memory_utilization": 0.8,
                "temperature_threshold": 75,
                "enable_load_balancing": True,
                "proxy_keywords": [
                    "exploit",
                    "payload",
                    "xss",
                    "sql injection",
                    "sqli",
                    "rce",
                    "hack",
                    "unauthorized",
                    "bypass",
                    "weaponize",
                    "porn",
                    "latest",
                    "study",
                    "report",
                    "spec",
                    "guide",
                    "benchmark",
                    "datasheet",
                    "api",
                    "tutorial",
                    "how to",
                    "example",
                ],
                "proxy_blocklist": [""],
                "proxy_concurrency": 3,
                "proxy_timeout": 8,
            },
            "performance_settings": {
                "chunk_size": 800,
                "chunk_overlap": 50,
                "max_results": 3,
                "enable_monitoring": True,
            },
            "ui_settings": {
                "theme": "dark",
                "font_size": 16,
                "comfortable_spacing": True,
                "corner_icons": True,
                "force_english_answers": True,
                "normalize_arabic_display": True,
            },
            "features": {
                "deep_search": True,
                "image_processing": True,
                "image_creation": True,
                "arabic_processing": True,
                "multi_format_support": True,
            },
            "google_cse": {"api_key": "", "cx": ""},
        }
        if cfg_file.exists():
            try:
                loaded = json.loads(cfg_file.read_text(encoding="utf-8"))
                default.update(loaded)
            except Exception:
                pass
        return default


config = Config()


from pathlib import Path

# Make sure psycho_store writes to your desired JSON file
config.psycho_file = Path("/home/gmm/Templates/Rona-Agent-/data/psychoanalytical.json")


class WebUIConfig:
    def __init__(self, base_dir: Path):
        self.ui_dir = base_dir / "prodectivity"
        self.port = 8765


webui = WebUIConfig(config.base_dir)


# ---------- Load Balancer ----------
class LoadBalancer:
    def __init__(self):
        self.gpu_temp_history = []
        self.cpu_temp_history = []
        self.current_mode = "balanced"
        self.temp_threshold = config.config["gpu_settings"]["temperature_threshold"]

    def monitor_temperatures(self):
        try:
            if platform.system() == "Linux":
                r = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                if r.returncode == 0:
                    t = int(r.stdout.strip())
                    self.gpu_temp_history.append(t)
                    self.gpu_temp_history = self.gpu_temp_history[-10:]
            if psutil:
                temps = psutil.sensors_temperatures()
                if temps:
                    first = list(temps.values())[0][0].current
                    self.cpu_temp_history.append(first)
                    self.cpu_temp_history = self.cpu_temp_history[-10:]
            return self.get_current_temps()
        except Exception:
            return None, None

    def get_current_temps(self):
        g = (
            sum(self.gpu_temp_history) / len(self.gpu_temp_history)
            if self.gpu_temp_history
            else 0
        )
        c = (
            sum(self.cpu_temp_history) / len(self.cpu_temp_history)
            if self.cpu_temp_history
            else 0
        )
        return g, c

    def adjust_load_balancing(self):
        g, _ = self.get_current_temps()
        if g > self.temp_threshold:
            self.current_mode = "cpu_heavy"
        elif g < self.temp_threshold - 10:
            self.current_mode = "gpu_heavy"
        else:
            self.current_mode = "balanced"
        return self.current_mode

    def get_optimal_gpu_layers(self):
        base = config.config["gpu_settings"]["gpu_layers"]
        if self.current_mode == "gpu_heavy":
            return min(base * 10, 35)
        if self.current_mode == "cpu_heavy":
            return max(base - 15, 5)
        return base


load_balancer = LoadBalancer()


# ---------- Utilities ----------
def tokenize(text: str) -> List[str]:
    # FIX: use 1+ word chars to avoid char-by-char tokens
    return re.findall(r"\w+", (text or "").lower())


def overlap_score(a: str, b: str) -> float:
    ta = set(tokenize(a))
    tb = set(tokenize(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    denom = max(1, len(ta | tb))
    return inter / denom


def ensure_ollama_running():
    try:
        # Quick health check
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=1)
        if r.status_code == 200:
            return
    except Exception:
        pass
    # Try to start ollama in background
    try:
        subprocess.Popen(
            ["bash", "-lc", "nohup ollama serve >/dev/null 2>&1 &"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print("Failed to auto-start ollama:", e)


def ensure_ollama_model(model_name: str):
    try:
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        if r.status_code == 200:
            data = r.json() or {}
            tags = [t.get("name") for t in data.get("models", [])]
            if model_name in (tags or []):
                return
    except Exception:
        pass
    # Pull the model if missing
    try:
        subprocess.Popen(
            ["bash", "-lc", f"nohup ollama pull {model_name} >/dev/null 2>&1 &"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as e:
        print("Failed to pull model:", e)


class SimpleOllama:
    def __init__(self, model: str):
        self.model = model
        ensure_ollama_running()
        ensure_ollama_model(model)

    def invoke(self, prompt: str):
        try:
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            r = requests.post(
                "http://127.0.0.1:11434/api/generate", json=payload, timeout=60
            )
            if r.status_code == 200:
                data = r.json() or {}
                content = data.get("response") or ""

                class Resp:
                    pass

                resp = Resp()
                setattr(resp, "content", content)
                return resp
        except Exception as e:
            print("SimpleOllama error:", e)

        class Resp:
            pass

        resp = Resp()
        setattr(resp, "content", "")
        return resp


# ---------- Retrieval & Synthesis Helpers (RAG-first policy) -------
# ---


def expand_query_variants(query: str) -> List[str]:
    q = (query or "").strip()
    if not q:
        return []
    words = tokenize(q)
    variants = {q}
    # simple singular/plural toggles and synonyms for common sec terms
    swaps = {
        "payload": ["injection", "vector"],
        "xss": ["cross site scripting", "reflected xss", "stored xss"],
        "sqli": ["sql injection", "database injection"],
        "recon": ["reconnaissance", "enumeration"],
    }
    for w in words:
        if w.endswith("s"):
            variants.add(q.replace(w, w[:-1]))
        else:
            variants.add(q.replace(w, w + "s"))
        if w in swaps:
            for s in swaps[w]:
                variants.add(q.replace(w, s))
    return list(variants)[:4]


def rank_with_local_priority(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Apply scoring override: local(vector)+0.30, local(sqlite)+0.20, conversation+0.10, web 0.0
    for c in candidates:
        base = float(c.get("score", 0.0))
        src = (c.get("source") or "").lower()
        bonus = 0.0
        if "vector" in src:
            bonus += 0.30
        if "sqlite" in src:
            bonus += 0.20
        if "conversation" in src:
            bonus += 0.10
        c["score_final"] = base + bonus
    return sorted(candidates, key=lambda x: x.get("score_final", 0.0), reverse=True)


def cluster_snippets(snippets: List[str], max_clusters: int = 5) -> List[List[str]]:
    clusters: List[List[str]] = []
    for s in snippets:
        placed = False
        for cl in clusters:
            if overlap_score(cl[0], s) >= 0.25:
                cl.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
        if len(clusters) >= max_clusters:
            break
    return clusters


def build_connected_reasoning(query: str, ranked: List[Dict[str, Any]]) -> str:
    # Connected Vector Reasoning format
    top_nodes = []
    for c in ranked[:5]:
        top_nodes.append(
            (
                c.get("title") or c.get("id") or c.get("source") or "node",
                round(c.get("score_final", 0.0), 3),
            )
        )
    lines = ["Vector Reasoning Path →"]
    for t, sc in top_nodes:
        lines.append(f"- {t} (strength={sc})")
    return "\n".join(lines)


def synthesize_answer_from_clusters(
    query: str, clusters: List[List[str]], sources: List[str]
) -> str:
    # Deterministic synthesis per policy
    parts = []
    parts.append("Summary: Consolidated local-first answer.")
    parts.append("Steps:")
    step_idx = 1
    for i, cl in enumerate(clusters[:5]):
        # pick most specific snippet (longest but capped)
        rep = sorted(cl, key=lambda s: len(s), reverse=True)[0][:600]
        parts.append(f"{step_idx}) From Cluster {chr(65+i)}: {rep}")
        step_idx += 1
    parts.append("Answer:")
    best = (
        clusters[0][0]
        if clusters
        else "No direct local snippet; using nearest conceptual guidance."
    )
    parts.append(best[:900])
    parts.append("Sources:")
    for s in sources[:10]:
        parts.append(f"- {s}")
    return "\n".join(parts)


def process_message(message):
    """
    Process message using NLTK and spaCy for NLP analysis
    """
    msg_lc = (message or "").lower()

    # FIX: substring-based intents (multi-word phrases work now)
    if any(
        k in msg_lc
        for k in [
            "xss",
            "sql injection",
            "sqli",
            "penetration testing",
            "pentest",
            "csrf",
            "ssti",
        ]
    ):
        fast_intent = "security_testing"
    elif any(k in msg_lc for k in ["bug bounty", "bounty", "hunting", "bypass"]):
        fast_intent = "bug_bounty"
    elif any(
        k in msg_lc.split()[:1]
        for k in [
            "Did you mean",
            "Define",
            "Describe",
            "Explain",
            "How",
            "When",
            "Why",
            "Where",
            "Who",
            "What is",
            "Did you mean",
            "how",
            "when",
            "why",
            "where",
            "who",
            "what is",
            "explain",
            "define",
            "describe",
        ]
    ):
        fast_intent = "general_query"
    else:
        fast_intent = None

    if not NLP_OK:
        return {
            "tokens": tokenize(message),
            "pos_tags": [],
            "entities": [],
            "intent": fast_intent,
        }

    try:
        tokens = word_tokenize(message)
        try:
            stop_words = set(stopwords.words("english"))
            filtered_tokens = [
                token for token in tokens if token.lower() not in stop_words
            ]
        except Exception:
            filtered_tokens = tokens

        try:
            pos_tags = nltk.pos_tag(filtered_tokens)
        except Exception:
            pos_tags = []

        try:
            doc = nlp(message)
            entities = [(entity.text, entity.label_) for entity in doc.ents]
        except Exception:
            entities = []

        return {
            "tokens": filtered_tokens,
            "pos_tags": pos_tags,
            "entities": entities,
            "intent": fast_intent,
        }
    except Exception as e:
        print(f"process_message error: {e}")
        return {
            "tokens": tokenize(message),
            "pos_tags": [],
            "entities": [],
            "intent": fast_intent,
        }


# ---------- File Processing ----------
try:
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        UnstructuredXMLLoader,
    )
except Exception:
    TextLoader = PyPDFLoader = UnstructuredXMLLoader = None


class FileProcessor:
    """
    Multi-format loader with high-quality text normalization and RAG-friendly chunking.
    Adds support for: .csv, .html/.htm, .txt (already), .xml (improved)
    """

    def __init__(self):
        self.supported_formats = {
            "text": [".txt", ".md", ".log"],
            "pdf": [".pdf"],
            "xml": [".xml"],
            "json": [".json"],
            "csv": [".csv"],
            "html": [".html", ".htm"],
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
        }

    # ---------- helpers ----------
    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # collapse whitespace, strip control chars, normalize newlines
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\r\n|\r", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[^\S\n]{2,}", " ", text)
        return text.strip()

    def _split_docs(
        self, raw_text: str, source_meta: dict
    ) -> Union[List["LCDocument"], List[str]]:
        """Split into RAG-friendly chunks using config; works with/without LangChain."""
        raw_text = self._normalize_text(raw_text)
        if not raw_text:
            return []
        chunk_size = config.config["performance_settings"].get("chunk_size", 800)
        chunk_overlap = config.config["performance_settings"].get("chunk_overlap", 50)

        if LC_OK and LCDocument:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len,
            )
            chunks = splitter.split_text(raw_text)
            return [LCDocument(page_content=c, metadata=source_meta) for c in chunks]
        # Fallback: simple manual splitting
        out, i = [], 0
        while i < len(raw_text):
            out.append(raw_text[i : i + chunk_size])
            i += chunk_size - chunk_overlap
        return out

    def _to_documents(
        self, texts: List[str], meta: dict
    ) -> Union[List["LCDocument"], List[str]]:
        texts = [self._normalize_text(t) for t in texts if t and t.strip()]
        if LC_OK and LCDocument:
            return [LCDocument(page_content=t, metadata=meta) for t in texts]
        return texts

    # ---------- main dispatcher ----------
    def process_file(self, file_path: str) -> Union[List["LCDocument"], List[str]]:
        p = Path(file_path)
        ext = p.suffix.lower()
        try:
            if ext in self.supported_formats["text"]:
                return self._process_text(p)
            if ext in self.supported_formats["pdf"]:
                return self._process_pdf(p)
            if ext in self.supported_formats["xml"]:
                return self._process_xml(p)
            if ext in self.supported_formats["json"]:
                return self._process_json(p)
            if ext in self.supported_formats["csv"]:
                return self._process_csv(p)
            if ext in self.supported_formats["html"]:
                return self._process_html(p)
            if ext in self.supported_formats["images"]:
                return self._process_image(p)
            raise ValueError(f"Unsupported format: {ext}")
        except Exception as e:
            print(f"process_file error: {e}")
            return []

    # ---------- loaders by type ----------
    def _process_text(self, p: Path):
        if TextLoader:
            try:
                docs = TextLoader(str(p), encoding="utf-8").load()
                # Split concatenated text into chunks for quality
                full = "\n\n".join(
                    [d.page_content for d in docs if getattr(d, "page_content", "")]
                )
                return self._split_docs(full, {"source": str(p)})
            except Exception:
                pass
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
            return self._split_docs(txt, {"source": str(p)})
        except Exception:
            return []

    def _process_pdf(self, p: Path):
        if PyPDFLoader:
            try:
                docs = PyPDFLoader(str(p)).load()
                full = "\n\n".join(
                    [d.page_content for d in docs if getattr(d, "page_content", "")]
                )
                return self._split_docs(full, {"source": str(p)})
            except Exception:
                pass
        try:
            import pdfplumber

            chunks = []
            with pdfplumber.open(p) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        chunks.append(page_text)
            return self._split_docs("\n\n".join(chunks), {"source": str(p)})
        except Exception:
            return self._process_text(p)

    def _process_xml(self, p: Path):
        # Prefer lxml if available for robustness; fallback to stdlib
        try:
            try:
                from lxml import etree as ET

                parser = ET.XMLParser(recover=True)
                root = ET.parse(str(p), parser=parser).getroot()
                text = "".join(root.itertext())
            except Exception:
                import xml.etree.ElementTree as ET

                root = ET.parse(p).getroot()
                text = ET.tostring(root, encoding="unicode", method="text")
            return self._split_docs(text, {"source": str(p), "format": "xml"})
        except Exception:
            return self._process_text(p)

    def _process_json(self, p: Path):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Compact but readable; removes huge spacing but keeps keys
            text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), indent=2)
            return self._split_docs(text, {"source": str(p), "format": "json"})
        except Exception:
            # Fallback to treating as plain text if JSON parsing or IO fails
            return self._process_text(p)

    def _process_csv(self, p: Path):
        """
        CSV → RAG-friendly documents with:
          - Column type inference: number/int/bool/date/datetime/text
          - Primary-key grouping (prefers 'id'; else first unique column)
          - Per-key summaries for numeric columns (count/min/max/avg)
          - Safety row caps to avoid vector noise
        """
        import csv
        import math
        from collections import defaultdict, Counter

        def _clean_num(s: str) -> str:
            # remove thousands separators and spaces
            return re.sub(r"[,\s]", "", s)

        def _is_int(s: str) -> bool:
            s2 = _clean_num(s)
            if s2 in ("", "-", "+"):
                return False
            try:
                int(s2)
                return not re.search(r"[.\eE]", s2)
            except Exception:
                return False

        def _is_float(s: str) -> bool:
            s2 = _clean_num(s)
            if s2 in ("", "-", "+"):
                return False
            try:
                float(s2)
                return True
            except Exception:
                return False

        def _is_bool(s: str) -> bool:
            return str(s).strip().lower() in {"true", "false", "yes", "no", "1", "0"}

        def _is_dateish(s: str) -> bool:
            # lightweight ISO-ish detector: 2024-10-16 or 2024-10-16T12:30 or 16/10/2024
            s = s.strip()
            if not s or len(s) < 8:
                return False
            if re.match(r"^\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?)?$", s):
                return True
            if re.match(r"^\d{1,2}/\d{1,2}/\d{2,4}$", s):
                return True
            return False

        def _infer_types(
            header: list[str], rows: list[list[str]], sample_n: int = 300
        ) -> dict:
            types = {}
            n = min(len(rows), sample_n)
            cols = list(range(len(header)))
            for ci in cols:
                vals = [str(rows[i][ci]) if ci < len(rows[i]) else "" for i in range(n)]
                vals_nonempty = [v for v in vals if v and v.strip() != ""]
                if not vals_nonempty:
                    types[header[ci]] = "text"
                    continue
                ints = sum(_is_int(v) for v in vals_nonempty)
                flts = sum(_is_float(v) for v in vals_nonempty)
                bols = sum(_is_bool(v) for v in vals_nonempty)
                dates = sum(_is_dateish(v) for v in vals_nonempty)
                # priority: bool > int > float > date > text (bool first to avoid 0/1 becoming int)
                if bols / len(vals_nonempty) > 0.9:
                    types[header[ci]] = "bool"
                elif ints / len(vals_nonempty) > 0.9:
                    types[header[ci]] = "int"
                elif flts / len(vals_nonempty) > 0.9:
                    types[header[ci]] = "number"
                elif dates / len(vals_nonempty) > 0.7:
                    types[header[ci]] = "date"
                else:
                    types[header[ci]] = "text"
            return types

        def _pick_primary_key(header: list[str], rows: list[list[str]]) -> int | None:
            # Prefer exact 'id' (case-insensitive). Else first column with high uniqueness ratio.
            lower = [h.strip().lower() for h in header]
            if "id" in lower:
                return lower.index("id")
            # uniqueness check on first 3 columns, pick best
            candidates = list(range(min(3, len(header))))
            best_idx, best_ratio = None, 0.0
            total = max(1, len(rows))
            for ci in candidates:
                uniq = len({(r[ci] if ci < len(r) else "") for r in rows})
                ratio = uniq / total
                if ratio > best_ratio:
                    best_idx, best_ratio = ci, ratio
            return best_idx if (best_ratio >= 0.85) else None

        try:
            with open(p, newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                rows_all = list(reader)
            if not rows_all:
                return []
            header = [
                h.strip() if h else f"col{i+1}" for i, h in enumerate(rows_all[0])
            ]
            data_rows = rows_all[1:]

            # Soft cap to protect from huge files; we still compute types on sample
            HARD_CAP = 20000
            if len(data_rows) > HARD_CAP:
                data_rows = data_rows[:HARD_CAP]

            types = _infer_types(header, data_rows, sample_n=500)
            pk_idx = _pick_primary_key(header, data_rows)
            pk_name = header[pk_idx] if pk_idx is not None else None

            # If we have a primary key, group rows; else treat as flat lines like before
            if pk_idx is not None:
                groups: dict[str, list[list[str]]] = defaultdict(list)
                for r in data_rows:
                    key = (r[pk_idx] if pk_idx < len(r) else "").strip()
                    groups[key].append(r)

                # Compute per-group summaries for numeric columns
                docs_text: list[str] = []
                NUM_CAP_GROUPS = 4000  # avoid explosion
                for g_i, (gkey, gro) in enumerate(groups.items()):
                    if g_i >= NUM_CAP_GROUPS:
                        break
                    # aggregate stats
                    numeric_stats = {}
                    for ci, col in enumerate(header):
                        if types.get(col) in ("int", "number"):
                            vals = []
                            for r in gro:
                                if ci < len(r) and r[ci] not in ("", None):
                                    s = _clean_num(str(r[ci]))
                                    try:
                                        vals.append(float(s))
                                    except Exception:
                                        pass
                            if vals:
                                numeric_stats[col] = {
                                    "count": len(vals),
                                    "min": min(vals),
                                    "max": max(vals),
                                    "avg": (sum(vals) / len(vals)) if vals else None,
                                }

                    # latest row snapshot (prefer the last)
                    latest = gro[-1] if gro else []
                    latest_pairs = []
                    for ci, col in enumerate(header):
                        v = latest[ci] if ci < len(latest) else ""
                        if v is None or str(v).strip() == "":
                            continue
                        latest_pairs.append(f"{col}: {str(v).strip()}")

                    # top text values (optional lightweight signal)
                    TEXT_TOP = []
                    for ci, col in enumerate(
                        header[:10]
                    ):  # only first 10 columns for brevity
                        if types.get(col) == "text":
                            cnt = Counter(
                                [
                                    (r[ci] if ci < len(r) else "").strip()
                                    for r in gro
                                    if (ci < len(r) and (r[ci] or "").strip())
                                ]
                            )
                            if cnt:
                                common = ", ".join(
                                    [f"{k}({v})" for k, v in cnt.most_common(3)]
                                )
                                TEXT_TOP.append(f"{col}: {common}")

                    stats_str = "; ".join(
                        [
                            f"{k}: count={v['count']} min={v['min']:.4g} max={v['max']:.4g} avg={v['avg']:.4g}"
                            for k, v in numeric_stats.items()
                        ]
                    )

                    block = [
                        f"CSV GROUP — {pk_name}={gkey}",
                        f"Columns: {', '.join(header)}",
                        f"Types: " + ", ".join([f"{c}:{types[c]}" for c in header]),
                        f"Rows in group: {len(gro)}",
                        (
                            ("Latest row → " + "; ".join(latest_pairs))
                            if latest_pairs
                            else "Latest row → (empty)"
                        ),
                        (
                            ("Numeric summary → " + stats_str)
                            if stats_str
                            else "Numeric summary → (n/a)"
                        ),
                    ]
                    if TEXT_TOP:
                        block.append("Top text values → " + " | ".join(TEXT_TOP))

                    docs_text.append("\n".join(block))

                meta = {
                    "source": str(p),
                    "format": "csv",
                    "columns": header,
                    "types": types,
                    "primary_key": pk_name,
                }
                text_all = ("\n\n").join(docs_text)
                return self._split_docs(text_all, meta)

            # Fallback (no PK): row-wise flatten similar to previous version
            lines = []
            max_rows = 2000
            for r in data_rows[:max_rows]:
                pairs = []
                for i, v in enumerate(r):
                    key = header[i] if i < len(header) else f"col{i+1}"
                    if v is None or str(v).strip() == "":
                        continue
                    pairs.append(f"{key}: {str(v).strip()}")
                if pairs:
                    lines.append("; ".join(pairs))
            text = (
                f"CSV: {p.name}\nColumns: {', '.join(header)}\nTypes: "
                + ", ".join([f"{c}:{types[c]}" for c in header])
                + "\n\n"
                + "\n".join(lines)
            )
            return self._split_docs(
                text,
                {
                    "source": str(p),
                    "format": "csv",
                    "columns": header,
                    "types": types,
                    "primary_key": None,
                },
            )
        except Exception as e:
            print("csv parse error:", e)
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")
                return self._split_docs(raw, {"source": str(p), "format": "csv"})
            except Exception:
                return []

    def _process_html(self, p: Path):
        """
        Extract visible text from HTML; strip scripts/styles, keep title & meta description.
        """
        try:
            html = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            try:
                html = p.read_bytes().decode("utf-8", errors="ignore")
            except Exception:
                html = ""
        title = ""
        meta_desc = ""
        body_text = ""
        try:
            from bs4 import BeautifulSoup  # pip install beautifulsoup4

            soup = BeautifulSoup(html, "html.parser")
            # Title + meta
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            md = soup.find("meta", attrs={"name": "description"})
            if md and md.get("content"):
                meta_desc = md["content"].strip()
            for tag in soup(["script", "style", "noscript", "iframe", "template"]):
                tag.decompose()
            # Get visible text
            body_text = soup.get_text(separator="\n")
        except Exception:
            # minimal fallback: strip tags
            body_text = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", html, flags=re.I)
            body_text = re.sub(r"<[^>]+>", " ", body_text)

        text = "\n".join([t for t in [title, meta_desc, body_text] if t])
        return self._split_docs(
            text, {"source": str(p), "format": "html", "title": title}
        )

    def _process_image(self, p: Path):
        try:
            if Image and LCDocument:
                with Image.open(p) as img:
                    meta = {
                        "format": img.format,
                        "mode": img.mode,
                        "size": img.size,
                        "filename": p.name,
                        "source": str(p),
                    }
                    text = f"Image: {p.name}\nFormat: {img.format}\nSize: {img.size}\nMode: {img.mode}"
                    return [LCDocument(page_content=text, metadata=meta)]
            # graceful fallback
            meta = {"source": str(p), "filename": p.name}
            if LCDocument:
                return [LCDocument(page_content=f"Image file: {p.name}", metadata=meta)]
            return [f"Image file: {p.name}"]
        except Exception:
            meta = {"source": str(p), "filename": p.name}
            if LCDocument:
                return [LCDocument(page_content=f"Image file: {p.name}", metadata=meta)]
            return [f"Image file: {p.name}"]


file_processor = FileProcessor()


# ---------- Arabic Processor ----------
class ArabicProcessor:
    def __init__(self):
        self.available = (arabic_reshaper is not None) and (
            bidi_get_display is not None
        )
        self._reshaper = arabic_reshaper if self.available else None

    @staticmethod
    def _has_arabic(text: str) -> bool:
        if not text:
            return False
        # Cover main Arabic blocks (not just \u0600–\u06FF)
        for ch in text:
            cp = ord(ch)
            if (
                0x0600 <= cp <= 0x06FF  # Arabic
                or 0x0750 <= cp <= 0x077F  # Arabic Supplement
                or 0x08A0 <= cp <= 0x08FF  # Arabic Extended-A
                or 0xFB50 <= cp <= 0xFDFF  # Arabic Presentation Forms-A
                or 0xFE70 <= cp <= 0xFEFF  # Arabic Presentation Forms-B
            ):
                return True
        return False

    def is_arabic(self, text: str) -> bool:
        return self._has_arabic(text)

    def process(self, text: str) -> str:
        """
        Shape + bidi for Arabic text.
        Adds RLE (U+202B) ... PDF (U+202C) only when Arabic is detected.
        Leaves English/Latin text untouched.
        """
        t = text or ""
        if not self.available or not self._has_arabic(t):
            return t
        try:
            # optional: use a configurable reshaper if you need digits/ligatures options later
            reshaped = self._reshaper.reshape(t)
            visual = bidi_get_display(reshaped)

            # Avoid double-wrapping if already has an RLE/PDF pair
            if "\u202b" in visual or "\u202b" in visual:
                return visual
            return "\u202b" + visual + "\u202c"  # RLE ... PDF
        except Exception:
            return t

        # Replace the code above with a simple return statement


arabic_processor = ArabicProcessor()


# ---------- Image Creator & Registry (disabled) ----------
class ImageCreator:
    def __init__(self):
        self.output_dir = config.images_dir
        self.registry_path = config.images_registry

    def _parse_size(self, size_str: Optional[str]) -> Tuple[int, int]:
        if not size_str:
            return (768, 512)
        m = re.match(r"^\s*(\d{2,4})x(\d{2,4})\s*$", size_str.lower())
        if not m:
            return (768, 512)
        w = max(256, min(2048, int(m.group(1))))
        h = max(256, min(2048, int(m.group(2))))
        return (w, h)

    def _draw_gradient(
        self,
        img: "Image.Image",
        top: Tuple[int, int, int],
        bottom: Tuple[int, int, int],
    ):
        # FIX: add missing multiplications
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for y in range(h):
            t = y / (h - 1) if h > 1 else 0.0
            r = int(top[0] + (bottom[0] - top[0]) * t)
            g = int(top[1] + (bottom[1] - top[1]) * t)
            b = int(top[2] + (bottom[2] - top[2]) * t)
            draw.line([(0, y), (w, y)], fill=(r, g, b))

    def _wrap_text(
        self,
        text: str,
        font: "ImageFont.FreeTypeFont",
        max_width: int,
        draw: "ImageDraw.ImageDraw",
    ):
        words = text.split()
        lines = []
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            tw, _ = draw.textsize(test, font=font)
            if tw <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        if line:
            lines.append(line)
        return lines

    def create_image_from_text(
        self, prompt: str, size_hint: Optional[str] = None
    ) -> Optional[str]:
        return None

    def _extract_exif(self, path: Path) -> Dict[str, Any]:
        return {}

    def analyze_and_register(self, src: str) -> Dict[str, Any]:
        return {"error": "image features disabled"}


image_creator = ImageCreator()


# ---------- CNN Image Processing Algorithm ----------
class CNNImageProcessor:
    """
    CNN-based image processing for classification and object recognition.
    """

    def __init__(self):
        self.available = False
        try:
            import torch  # noqa: F401
            import torchvision  # noqa: F401

            self.torch_available = True
        except ImportError:
            self.torch_available = False
        try:
            import tensorflow as tf  # noqa: F401

            self.tf_available = True
        except ImportError:
            self.tf_available = False

    def get_recommended_models(self):
        return {
            "image_classification": [
                "ResNet50",
                "ResNet101",
                "Inception-v3",
                "VGG16",
                "MobileNetV2",
            ],
            "object_detection": [
                "YOLOv9",
                "YOLOv8",
                "YOLOv5",
                "R-CNN",
                "Fast R-CNN",
                "Faster R-CNN",
            ],
            "mobile_optimized": [
                "MobileNetV2",
                "MobileNetV3",
                "EfficientNet-B0",
                "EfficientNet-B1",
            ],
        }

    def get_implementation_notes(self):
        return {
            "frameworks": {
                "PyTorch": "Most popular for research and development",
                "TensorFlow": "Good for production deployment",
                "ONNX": "Cross-platform model format",
            },
            "preprocessing": [
                "Image resizing to model input dimensions",
                "Normalization (mean/std scaling)",
                "Data augmentation (rotation, flip, crop)",
            ],
            "training_tips": [
                "Use transfer learning with pre-trained models",
                "Fine-tune on domain-specific datasets",
                "Monitor training with validation loss",
                "Use early stopping to prevent overfitting",
            ],
        }


cnn_processor = CNNImageProcessor()


# ---------- Database Manager (Chroma) ----------
class DatabaseManager:
    def __init__(self):
        self.vector_db = None
        self.embeddings_model = None
        self._initialize_database()

    def _initialize_database(self):
        try:
            if not LC_OK:
                # Allow running without Chroma/LangChain; just skip vector DB
                raise RuntimeError("LangChain/Chroma not available")
            ensure_ollama_running()
            self.embeddings_model = OllamaEmbeddings(
                model="nomic-embed-text", base_url="http://127.0.0.1:11434"
            )
            self.vector_db = Chroma(
                persist_directory=str(config.chroma_db_dir),
                embedding_function=self.embeddings_model,
                collection_metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            print("DB init failed:", e)
            self.vector_db = None

    def add_documents(self, docs: List[Any]) -> bool:
        if not self.vector_db:
            return False
        try:
            # create stable IDs by hashing page_content+source to avoid duplicates
            import hashlib

            texts, metas, ids = [], [], []
            for d in docs:
                content = getattr(d, "page_content", "") or ""
                meta = getattr(d, "metadata", {}) or {}
                src = str(meta.get("source", ""))[:256]
                h = hashlib.sha1(
                    (content + "|" + src).encode("utf-8", errors="ignore")
                ).hexdigest()
                ids.append(h)
                texts.append(content)
                metas.append(meta)
            # Chroma community supports add_texts with ids
            self.vector_db.add_texts(texts=texts, metadatas=metas, ids=ids)
            return True
        except Exception as e:
            print("add_documents error:", e)
            return False

    def similarity_search(self, query: str, k: int = 5) -> List["LCDocument"]:
        if not self.vector_db:
            return []
        try:
            return self.vector_db.similarity_search(query, k=k)
        except Exception as e:
            print("similarity_search error:", e)
            return []


# ---------- SQLite Manager (cyberassest / lovely_assest) ----------
class SQLiteManager:
    def __init__(self, db_path: Path):
        self.db_path = str(db_path)
        self._ensure_tables()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _ensure_tables(self):
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS cyberassest (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        path TEXT,
                        content TEXT,
                        metadata TEXT,
                        created_at TEXT
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS lovely_assest (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filename TEXT,
                        path TEXT,
                        content TEXT,
                        metadata TEXT,
                        created_at TEXT
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            print("SQLite init failed:", e)

    def insert_document(
        self,
        table: str,
        filename: str,
        path: str,
        content: str,
        metadata: Dict[str, Any],
        created_at: str,
    ) -> bool:
        if table not in ("cyberassest", "lovely_assest"):
            return False
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"INSERT INTO {table} (filename, path, content, metadata, created_at) VALUES (?,?,?,?,?)",
                    (
                        filename,
                        path,
                        content,
                        json.dumps(metadata, ensure_ascii=False),
                        created_at,
                    ),
                )
                conn.commit()
            return True
        except Exception as e:
            print("SQLite insert error:", e)
            return False

    def search(self, table: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if table not in ("cyberassest", "lovely_assest"):
            return []
        try:
            like = f"%{query}%"
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    f"SELECT filename, path, content, metadata, created_at FROM {table} WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                    (like, limit),
                )
                rows = cur.fetchall()
                results = []
                for r in rows:
                    md = {}
                    try:
                        md = json.loads(r[3] or "{}")
                    except Exception:
                        md = {}
                    results.append(
                        {
                            "filename": r[0],
                            "path": r[1],
                            "content": r[2] or "",
                            "metadata": md,
                            "created_at": r[4],
                        }
                    )
                return results
        except Exception as e:
            print("SQLite search error:", e)
            return []


# singleton wrapper for SQLite
class SQLiteManagerSingleton:
    _instance = None

    @classmethod
    def get(cls) -> Optional[SQLiteManager]:
        if cls._instance is None:
            try:
                cls._instance = SQLiteManager(config.sqlite_path)
            except Exception as e:
                print("SQLite manager init error:", e)
                cls._instance = None
        return cls._instance


# singleton wrapper for DB so multiple use share same instance
# ---------- Psychoanalytical persistent store ----------
class PsychoStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8") or "[]")
        except Exception:
            return []

    def _save(self, entries: List[Dict[str, Any]]) -> None:
        self.path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_entry(self, title: str, date_iso: str, details: str, mood_0_10: float):
        entries = self._load()
        entries.append(
            {
                "id": len(entries) + 1,
                "title": title.strip() or "(untitled)",
                "date": date_iso,  # YYYY-MM-DD
                "details": details.strip(),
                "mood": float(mood_0_10),  # 0..10
            }
        )
        self._save(entries)

    def list_entries(self) -> List[Dict[str, Any]]:
        return self._load()

    def export_text_chunks(self, max_items: int = 50) -> List[str]:
        """RAG-friendly short chunks for LLM context."""
        out = []
        for e in sorted(self._load(), key=lambda x: x.get("date", ""), reverse=True)[
            :max_items
        ]:
            out.append(
                f"[psycho] {e.get('date','')} | {e.get('title','')}\n"
                f"mood: {e.get('mood',0):.2f}/10\n"
                f"{(e.get('details','') or '')[:900]}"
            )
        return out

    def emotion_summary(self) -> str:
        entries = self._load()
        if not entries:
            return "No entries."
        import statistics as st

        moods = [
            float(e.get("mood", 0.0))
            for e in entries
            if isinstance(e.get("mood", 0.0), (int, float, str))
        ]
        avg = st.mean(moods) if moods else 0.0
        hi = max(moods) if moods else 0.0
        lo = min(moods) if moods else 0.0
        return f"entries={len(entries)}, avg_mood={avg:.2f}, min={lo:.2f}, max={hi:.2f}"


# Then create the store
psycho_store = PsychoStore(config.psycho_file)


def choose_port(preferred=8765):
    """Try preferred; if busy, ask OS for a free ephemeral port."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", preferred))
        port = preferred
    except OSError:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    finally:
        s.close()
    return port


class FlaskServerController:
    def __init__(self, app, host="127.0.0.1", port=8765):
        self.app = app
        self.host = host
        self.port = port
        self._server = make_server(host, port, app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._ctx = app.app_context()
        self._ctx.push()

    def start(self):
        self._thread.start()

    def shutdown(self):
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._ctx.pop()
        except Exception:
            pass


'''
def start_psycho_server():
    """
    Starts a Flask server to serve a Psycho UI on the local machine.

    This UI provides a simple CRUD interface to manage entries in your PsychoStore.
    The server listens on localhost:8765 by default.

    If Flask is not installed, a notice is printed and the function returns.

    The server will run until an interrupt signal is sent to the process.

    The PsychoStore is used to store and retrieve entries from the Psycho UI.
    """
    if not FLASK_OK:
        print("🔵 NOTICE: Flask not installed. `pip install flask` to use web Psycho UI.")
        return

    from flask import Flask, jsonify, request, send_from_directory
    app = Flask("psycho_ui", static_folder=str(webui.ui_dir), static_url_path="/psycho_static")

    print(f"[psycho-ui] ui_dir = {webui.ui_dir}")
    import os
    if not os.path.exists(webui.ui_dir / "productivity.html"):
        print("[psycho-ui] ERROR: productivity.html not found in ui_dir!")

    # ---- API uses your PsychoStore directly ----
    @app.get("/api/psycho/entries")
    def api_list():
        return jsonify(psycho_store.list_entries())

    @app.post("/api/psycho/entries")
    def api_add():
        data = request.get_json(force=True) or {}
        mood = float(data.get("dailyRating") or data.get("mood") or 0.0)
        psycho_store.add_entry(
            (data.get("title") or "").strip(),
            data.get("date") or "",
            data.get("details") or "",
            mood,
        )
        return jsonify(psycho_store.list_entries()[-1]), 201

    @app.put("/api/psycho/entries/<int:eid>")
    def api_update(eid):
        entries = psycho_store.list_entries()
        data = request.get_json(force=True) or {}
        for e in entries:
            if int(e.get("id", -1)) == eid:
                e["title"]   = data.get("title",   e["title"])
                e["date"]    = data.get("date",    e["date"])
                e["details"] = data.get("details", e["details"])
                if "dailyRating" in data or "mood" in data:
                    e["mood"] = float(data.get("dailyRating", data.get("mood", e["mood"])))
                psycho_store._save(entries)
                return jsonify(e)
        return jsonify({"error": "not found"}), 404

    @app.delete("/api/psycho/entries/<int:eid>")
    def api_delete(eid):
        entries = psycho_store.list_entries()
        new_entries = [e for e in entries if int(e.get("id", -1)) != eid]
        if len(new_entries) == len(entries):
            return jsonify({"error": "not found"}), 404
        psycho_store._save(new_entries)
        return jsonify({"ok": True})

    # ---- Static UI ----
    @app.get("/psycho/")
    def ui_index():
        return send_from_directory(str(webui.ui_dir), "productivity.html")


    @app.get("/psycho/<path:path>")
    def ui_assets(path):
        return send_from_directory(str(webui.ui_dir), path)
    @app.get("/")
    def root():
        # serve main page from the UI folder
        return send_from_directory(str(webui.ui_dir), "productivity.html")
    
    @app.get("/api/psycho/health")
    def api_health():
        return jsonify({"ok": True})
    
    # ---- UI pages ----
    
    
    @app.get("/password.html")
    def password_page():
     return send_from_directory(app.static_folder, "password.html")
    
    @app.get("/api/psycho/summary")
    def api_summary():
        return jsonify({
            "summary": psycho_store.emotion_summary(),
            "count": len(psycho_store.list_entries())
        })

    app.run(host="127.0.0.1", port=8765, debug=True, use_reloader=False)

   # app.run(host="127.0.0.1", port=webui.port, debug=False, use_reloader=False)
'''


def create_psycho_app():
    app = Flask(
        "psycho_ui", static_folder=str(webui.ui_dir), static_url_path="/psycho_static"
    )

    # ---------- API ----------
    @app.get("/api/psycho/entries")
    def api_list():
        return jsonify(psycho_store.list_entries())

    @app.post("/api/psycho/entries")
    def api_add():
        data = request.get_json(force=True) or {}
        title = (data.get("title") or "").strip()
        date_iso = data.get("date") or ""
        details = data.get("details") or ""
        mood = float(data.get("dailyRating") or data.get("mood") or 0.0)
        psycho_store.add_entry(title, date_iso, details, mood)
        return jsonify(psycho_store.list_entries()[-1]), 201

    @app.put("/api/psycho/entries/<int:eid>")
    def api_update(eid):
        entries = psycho_store.list_entries()
        data = request.get_json(force=True) or {}
        updated = None
        for e in entries:
            if int(e.get("id", -1)) == eid:
                if "title" in data:
                    e["title"] = data["title"]
                if "date" in data:
                    e["date"] = data["date"]
                if "details" in data:
                    e["details"] = data["details"]
                if "dailyRating" in data or "mood" in data:
                    e["mood"] = float(
                        data.get("dailyRating", data.get("mood", e.get("mood", 0.0)))
                    )
                updated = e
                break
        if not updated:
            return jsonify({"error": "not found"}), 404

        psycho_store._save(entries)
        return jsonify(updated)

    @app.delete("/api/psycho/entries/<int:eid>")
    def api_delete(eid):
        entries = psycho_store.list_entries()
        new_entries = [e for e in entries if int(e.get("id", -1)) != eid]
        if len(new_entries) == len(entries):
            return jsonify({"error": "not found"}), 404
        psycho_store._save(new_entries)
        return jsonify({"ok": True})

    # ---------- Static UI ----------
    @app.get("/psycho/")
    def ui_index():
        # NOTE: file must be exactly productivity.html in your prodectivity/ folder
        return send_from_directory(str(webui.ui_dir), "productivity.html")

    @app.get("/psycho/<path:path>")
    def ui_assets(path):
        return send_from_directory(str(webui.ui_dir), path)

    @app.get("/api/psycho/health")
    def api_health():
        return jsonify({"ok": True})

    return app


class DatabaseManagerSingleton:
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            try:
                cls._instance = DatabaseManager()
            except Exception:
                cls._instance = None
        return cls._instance


# ---------- Deep Search Engine (unified) ----------
class DeepSearchEngine:
    def __init__(self):
        self.google_key = config.config.get("google_cse", {}).get("api_key") or None
        self.google_cx = config.config.get("google_cse", {}).get("cx") or None

    # --- helpers: keep them small & local ---

    @staticmethod
    def _normalize_source(s: str) -> str:
        s = (s or "").lower()
        if "duckduckgo" in s: return "duckduckgo"
        if "google" in s: return "google"
        if "vector" in s or "local" in s: return "vector_db"
        if "conversation" in s: return "conversation"
        return s or "web"

    def _topic_overlap(self, query: str, title: str, snippet: str, url: str) -> float:
        import re
        toks = [t for t in re.split(r"[^A-Za-z0-9]+", (query or "").lower()) if t]
        if not toks: return 0.0
        hay = f"{(title or '').lower()} {(snippet or '').lower()} {(url or '').lower()}"
        uniq = set(toks)
        hits = sum(1 for t in uniq if t in hay)
        return hits / max(1, len(uniq))

    def _detect_lang_fast(self, s: str) -> str:
        # ultra-light heuristic: enough for filtering obvious zh/arabic vs english
        s = s or ""
        if any("\u0600" <= ch <= "\u06ff" for ch in s): return "ar"
        if any("\u4e00" <= ch <= "\u9fff" for ch in s): return "zh"
        return "en"

    def _postfilter_web_hits(self, query: str, items: list[dict]) -> list[dict]:
        out = []
        qlang = self._detect_lang_fast(query)
        for r in items or []:
            title = r.get("title", "")
            snippet = r.get("content", "")
            url = r.get("url", "")

            # language guard: keep same-language hits; drop strong mismatch for EN queries
            tlang = self._detect_lang_fast(title + " " + snippet)
            if qlang == "en" and tlang in ("zh", "ar"):
                continue

            # navigational/installer guard
            lowt = (title or "").lower()
            lowu = (url or "").lower()
            if ("chrome" in lowt and "google.com/" in lowu) or ("download" in lowt and "google.com/chrome" in lowu):
                continue

            # topical overlap
            if self._topic_overlap(query, title, snippet, url) < 0.18:
                continue

            # keep
            r["source"] = self._normalize_source(r.get("source"))
            out.append(r)
        return out

    def _bucketize_urls(self, items: list[dict]) -> dict[str, list[str]]:
        import tldextract
        buckets: dict[str, set] = {
            "official": set(),
            "docs": set(),
            "forums": set(),
            "github": set(),
            "edu": set(),
            "gov": set(),
            "news/blogs": set(),
            "marketplaces": set(),
            "other": set(),
        }
        for r in items or []:
            u = r.get("url") or ""
            if not u: continue
            dom = tldextract.extract(u)
            root = ".".join([p for p in [dom.domain, dom.suffix] if p])
            lurl = u.lower()
            lt = (r.get("title") or "").lower()

            if root.startswith(("docs.",)): buckets["docs"].add(u)
            elif "github.com" in lurl: buckets["github"].add(u)
            elif lurl.endswith(".edu") or ".edu/" in lurl: buckets["edu"].add(u)
            elif lurl.endswith(".gov") or ".gov/" in lurl: buckets["gov"].add(u)
            elif any(w in lurl for w in ["stackoverflow.com", "serverfault.com", "superuser.com", "reddit.com", "stackexchange.com"]):
                buckets["forums"].add(u)
            elif any(w in lurl for w in ["medium.com", "dev.to", "blog.", "/blog/"]):
                buckets["news/blogs"].add(u)
            elif any(w in lurl for w in ["market", "store", "buy", "sell", "pricing"]) or any(w in lt for w in ["pricing", "plans"]):
                buckets["marketplaces"].add(u)
            elif any(w in lurl for w in ["oowasp.org", "owasp.org", "ietf.org", "iso.org", "w3.org"]):
                buckets["official"].add(u)
            else:
                buckets["other"].add(u)
        # convert sets→lists and trim
        return {k: list(sorted(v))[:12] for k, v in buckets.items() if v}

    async def duckduckgo_search(
        self, query: str, max_results: int = 6
    ) -> List[Dict[str, Any]]:
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json&no_html=1&skip_disambig=1"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=12) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()
                    out = []
                    for item in data.get("RelatedTopics", [])[:max_results]:
                        if isinstance(item, dict) and item.get("Text"):
                            out.append(
                                {
                                    "source": "duckduckgo",
                                    "title": item.get("Text", "")[:120],
                                    "content": item.get("Text", ""),
                                    "url": item.get("FirstURL", ""),
                                }
                            )
                    return out
        except Exception:
            return []

    def google_cse_search(self, query: str, num: int = 5) -> List[Dict[str, Any]]:
        if not self.google_key or not self.google_cx:
            print(
                "🔵 NOTICE: Google CSE disabled (key or cx missing) → using DuckDuckGo fallback."
            )
            return []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_key,
                "cx": self.google_cx,
                "q": query,
                "num": num,
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                print(
                    f"🔵 NOTICE: Google CSE HTTP {r.status_code} → using DuckDuckGo fallback."
                )
                try:
                    print("diag:", r.json())
                except Exception:
                    pass
                return []
            data = r.json()
            out = []
            for it in data.get("items", []):
                out.append(
                    {
                        "source": "google",
                        "title": it.get("title", "")[:120],
                        "content": it.get("snippet", ""),
                        "url": it.get("link", ""),
                    }
                )
            return out
        except Exception as e:
            print("🔵 NOTICE: Google CSE error → fallback to DuckDuckGo. err:", e)
            return []


    def search_zoomeye(self, query):
        """
        Searches ZoomEye for the given query.
        """
        if not self.ZOOMEYE_API_KEY:
            print("ZoomEye API key is not set. Skipping ZoomEye search.")
            return []

        headers = {"API-KEY": self.ZOOMEYE_API_KEY}
        # Use f-string for cleaner URL construction
        url = f"https://api.zoomeye.org/host/search?query={query}&page=1"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise an exception for bad status codes
            data = response.json()

            if "matches" not in data:
                # Handle cases where the API returns an error or unexpected format
                print(f"ZoomEye API returned an unexpected response: {data}")
                return []

            results = []
            for match in data.get("matches", []):
                # Skip results that indicate protected data or require contact
                if "protected" in str(match).lower() or "contact" in str(match).lower():
                    continue

                title = match.get("ip", "No IP")
                # Safely access nested 'portinfo'
                portinfo = match.get("portinfo", {})
                snippet = f"Port: {portinfo.get('port', 'N/A')}, Service: {portinfo.get('service', 'N/A')}"
                link = f"https://www.zoomeye.org/host/{match.get('ip')}"
                results.append({"title": title, "snippet": snippet, "link": link})
            return results
        except requests.exceptions.RequestException as e:
            print(f"Error during ZoomEye search: {e}")
            return []
        except json.JSONDecodeError:
            print("Failed to decode ZoomEye API response.")
            return []
        
    def local_db_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        db = DatabaseManagerSingleton.get()
        if not db or not LC_OK:
            print(
                "🔵 NOTICE: Vector DB unavailable → RAG will rely on web + conversation only."
            )
            return []
        docs = db.similarity_search(query, k=k)
        out = []
        q = (query or "").lower()

        # quick topic sense: if programming topic, de-weight local registry noise
        prog = any(
            w in q
            for w in [
                "javascript",
                "java script",
                "js ",
                " js",
                "python",
                "c++",
                "c#",
                "golang",
                "react",
                "flutter",
            ]
        )

        for d in docs:
            content = getattr(d, "page_content", "") or ""
            meta = getattr(d, "metadata", {}) or {}
            c = content.lower()

            # hard filter: image/registry junk
            if ("image file:" in c) or ("data/images" in c) or ("/images/" in c):
                continue
            # json-ish registry keys and short body
            hits = sum(
                k in c
                for k in [
                    "filename",
                    "path",
                    "pil",
                    "exif",
                    "created_at",
                    "size",
                    "mode",
                    "format",
                ]
            )
            if hits >= 2 and len(c) < 800:
                continue

            score = 3.8 * overlap_score(query, content)

            # removed programming text penalty - was over-filtering relevant chunks

            out.append(
                {
                    "source": "local",
                    "title": meta.get("source", "local_doc"),
                    "content": content,
                    "meta": meta,
                    "score": score,
                }
            )
        return out

    def convo_search(
        self, query: str, conversation_history: List[str]
    ) -> List[Dict[str, Any]]:
        convo = conversation_history[:-2] if len(conversation_history) > 2 else []
        if not convo:
            return []
        N = 20
        recent_convo = convo[-N:]
        scores = []
        q = (query or "").lower()
        for i, msg in enumerate(reversed(recent_convo)):
            m = (msg or "").lower()
            # skip greetings and system-ish noise and image registry chatter
            if re.fullmatch(
                r"(you:)?\s*(hi|hello|hey|how are you|how's it going|السلام عليكم|مرحبا)[\s\.!\?]*",
                m,
                flags=re.I,
            ):
                continue
            if "/images/" in m or "image saved & indexed" in m or "data/images" in m:
                continue
            toks = tokenize(m)
            if len(toks) <= 5:
                continue
            score = overlap_score(q, m)
            scores.append(
                {"idx": len(recent_convo) - 1 - i, "msg": msg, "score": score}
            )

        max_overlap = max([s["score"] for s in scores], default=0)
        filtered = [
            s for s in scores if s["score"] >= (0.5 if max_overlap < 0.30 else 0.15)
        ]
        filtered = sorted(filtered, key=lambda s: s["score"], reverse=True)[:3]
        out = []
        for s in filtered:
            base = 0.3 + s["score"] * 0.9 + (0.2 if max_overlap > 0.4 else 0.0)
            out.append(
                {
                    "source": "conversation",
                    "title": f"previous_msg_{s['idx']}",
                    "content": s["msg"],
                    "score": base,
                }
            )
        return out

    async def search_unified(
        self, query: str, conversation_history: List[str], top_k: int = 6
    ) -> List[Dict[str, Any]]:
        local = self.local_db_search(query, k=top_k)
        convo = self.convo_search(query, conversation_history)
        ddg_task = asyncio.create_task(self.duckduckgo_search(query, max_results=top_k))
        google = (
            self.google_cse_search(query, num=top_k)
            if (self.google_key and self.google_cx)
            else []
        )

        # Add bug bounty context if available
        bug_bounty_results = []
        if BUG_BOUNTY_OK:
            try:
                bug_bounty_results = await bug_bounty_integration.deep_search_enhanced(
                    query, conversation_history
                )
            except Exception:
                pass

        ddg = []
        try:
            ddg = await ddg_task
        except Exception:
            ddg = []
        combined = []
        for r in local:
            combined.append(r)
        for r in google:
            score = 2.4 * overlap_score(query, r.get("content", ""))
            combined.append({**r, "score": score})
        for r in ddg:
            score = 1.2 + overlap_score(query, r.get("content", ""))
            combined.append({**r, "score": score})
        for r in convo:
            combined.append(r)
        for r in bug_bounty_results:
            combined.append(r)
        combined_sorted = sorted(
            combined, key=lambda x: x.get("score", 0.0), reverse=True
        )
        seen = set()
        results = []
        for item in combined_sorted:
            key = (item.get("url") or item.get("title") or item.get("content", ""))[
                :200
            ]
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
            if len(results) >= top_k:
                break
        return results


deep_search = DeepSearchEngine()

import urllib.parse
import urllib.robotparser
from bs4 import BeautifulSoup
from aiohttp import ClientTimeout, TCPConnector


class QueryProxy:
    """
    Query Proxy + Augmenter
    - Detects keywords that require extra crawling/searching.
    - Respects a blocklist for dangerous queries (refuses augmentation).
    - Uses deep_search to find candidate URLs, obeys robots.txt, fetches pages,
      extracts visible text, returns a list of dicts like {source,title,content,url,score}.
    - Does NOT persist anything by default; returns transient context for RAG.
    """

    def __init__(
        self,
        search_engine: DeepSearchEngine,
        file_processor: FileProcessor,
        max_fetch: int = 6,
    ):
        self.search_engine = search_engine
        self.file_processor = file_processor
        self.max_fetch = max_fetch
        # keywords that trigger augmentation (configurable)
        self.proxy_keywords = config.config.get(
            "proxy_keywords",
            [
                "latest",
                "study",
                "report",
                "vulnerability",
                "spec",
                "how to",
                "guide",
                "benchmark",
                "datasheet",
                "csv sample",
                "api",
                "tutorial",
                "exploit",
                "payload",
                "xss",
                "sql injection",
                "sqli",
                "rce",
                "hack",
                "unauthorized",
                "bypass",
                "weaponize",
            ],
        )
        # blocklist to refuse dangerous/attack prompts
        self.blocklist = config.config.get("proxy_blocklist", [""])
        # concurrency / politeness
        self.concurrent_fetch = config.config.get("proxy_concurrency", 3)
        self.fetch_timeout = config.config.get("proxy_timeout", 8)

    def _needs_augmentation(self, query: str) -> bool:
        if not query:
            return False
        q = query.lower()
        # if any blocklist token appears, refuse
        for b in self.blocklist:
            if b in q:
                return False
        for k in self.proxy_keywords:
            if k in q:
                return True
        return False

    def _is_blocked_query(self, query: str) -> Optional[str]:
        q = (query or "").lower()
        for b in self.blocklist:
            if b in q:
                return b
        return None

    async def _fetch_url(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[Dict[str, str]]:
        """
        Fetch a single URL (if allowed by robots) and extract visible text.
        Returns dict {url,title,content} or None on failure.
        """
        try:
            parsed = urllib.parse.urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            # robots check
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urllib.parse.urljoin(base, "/robots.txt")
            try:
                rp.set_url(robots_url)
                rp.read()
                if not rp.can_fetch("*", url):
                    # blocked by robots
                    return None
            except Exception:
                # if robots unreadable, be conservative: allow only same-origin simple pages
                pass

            # polite fetch
            headers = {"User-Agent": f"RonaQueryProxy/1.0 (+{platform.node()})"}
            async with session.get(
                url, timeout=self.fetch_timeout, headers=headers, ssl=False
            ) as resp:
                if resp.status != 200:
                    return None
                # only parse text/html
                ctype = resp.headers.get("content-type", "")
                body = await resp.text(errors="ignore")
                if "html" not in ctype and not body.strip().startswith("<"):
                    # skip binary or non-html
                    return None
                # parse visible text
                try:
                    soup = BeautifulSoup(body, "html.parser")
                    for tag in soup(
                        ["script", "style", "noscript", "iframe", "template"]
                    ):
                        tag.decompose()
                    title = ""
                    if soup.title and soup.title.string:
                        title = soup.title.string.strip()
                    text = soup.get_text(separator="\n")
                    text = re.sub(r"\n{3,}", "\n\n", text).strip()
                    if not text:
                        return None
                    return {"url": url, "title": title or url, "content": text}
                except Exception:
                    return None
        except Exception:
            return None

    async def proxy_and_augment(
        self, query: str, conversation_history: List[str]
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        """
        Main entry:
        - If query contains a blocklisted term: returns (False, [], "blocked:<term>").
        - If needs augmentation: fetch search results, crawl top urls, extract content, and return context list.
        - Otherwise returns (False, [], None)
        """
        blocked = self._is_blocked_query(query)
        if blocked:
            return False, [], f"blocked:{blocked}"

        if not self._needs_augmentation(query):
            return False, [], None

        # Use deep_search to produce candidate URLs (duckduckgo fallback included)
        try:
            candidates = await self.search_engine.search_unified(
                query, conversation_history, top_k=self.max_fetch
            )
        except Exception:
            candidates = []
        urls = []
        for c in candidates:
            u = c.get("url") or ""
            if u and u not in urls:
                urls.append(u)
            if len(urls) >= self.max_fetch:
                break

        # if no URLs found, try a direct DuckDuckGo search as fallback
        if not urls:
            try:
                ddg = await self.search_engine.duckduckgo_search(
                    query, max_results=self.max_fetch
                )
                for d in ddg:
                    u = d.get("url") or ""
                    if u and u not in urls:
                        urls.append(u)
                    if len(urls) >= self.max_fetch:
                        break
            except Exception:
                pass

        # fetch pages concurrently, but limited
        timeout = ClientTimeout(total=self.fetch_timeout)
        connector = TCPConnector(limit=self.concurrent_fetch, ssl=False)
        results = []
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector
            ) as session:
                tasks = [self._fetch_url(session, u) for u in urls]
                fetched = await asyncio.gather(*tasks, return_exceptions=True)
                for f in fetched:
                    if isinstance(f, dict):
                        results.append({**f, "source": "web_proxy"})
        except Exception:
            pass

        # Post-process results: filter short content & transform into langchain-like docs if possible
        out = []
        for r in results:
            content = r.get("content") or ""
            if len(content) < 200:
                continue
            # split into smaller chunks using the FileProcessor splitter if available
            try:
                chunks = self.file_processor._split_docs(
                    content, {"source": r.get("url"), "format": "html"}
                )
                for ch in chunks or []:
                    if isinstance(ch, str):
                        out.append(
                            {
                                "source": "web_proxy",
                                "title": r.get("title"),
                                "content": ch,
                                "url": r.get("url"),
                                "score": 0.5,
                            }
                        )
                    else:
                        out.append(
                            {
                                "source": "web_proxy",
                                "title": getattr(ch, "metadata", {}).get(
                                    "title", r.get("title")
                                ),
                                "content": getattr(ch, "page_content", ""),
                                "url": r.get("url"),
                                "score": 0.6,
                            }
                        )
            except Exception:
                out.append(
                    {
                        "source": "web_proxy",
                        "title": r.get("title"),
                        "content": content[:2000],
                        "url": r.get("url"),
                        "score": 0.4,
                    }
                )

        # Deduplicate by content hash
        seen = set()
        final = []
        for item in out:
            h = hashlib.sha1(
                (item.get("content", "")[:2000]).encode("utf-8", errors="ignore")
            ).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            final.append(item)
            if len(final) >= self.max_fetch * 3:
                break

        augment_report = f"proxy_augmented:{len(final)}"
        return True, final, augment_report


# ---------- Response Formatter ----------
class ResponseFormatter:
    def format(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
        # FIX: do not strip generic def lines; was removing legit code
        return text


formatter = ResponseFormatter()


# ---------- Test Suite ----------
class TestSuite:
    def __init__(self):
        self.test_results = {}

    def run_all_tests(self):
        results = {}
        # GPU
        try:
            if platform.system() == "Linux":
                r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                results["gpu_detection"] = {
                    "status": "passed" if r.returncode == 0 else "failed",
                    "gpu_available": r.returncode == 0,
                }
            else:
                results["gpu_detection"] = {"status": "passed", "note": "Non-Linux"}
        except Exception as e:
            results["gpu_detection"] = {"status": "failed", "error": str(e)}
        # DB
        try:
            db = DatabaseManagerSingleton.get()
            ok = db is not None and db.vector_db is not None
            results["database_connection"] = {
                "status": "passed" if ok else "failed",
                "available": ok,
            }
        except Exception as e:
            results["database_connection"] = {"status": "failed", "error": str(e)}
        # file processing
        try:
            sample = config.data_dir / "test_rona.txt"
            sample.write_text("Rona test", encoding="utf-8")
            docs = file_processor.process_file(str(sample))
            sample.unlink(missing_ok=True)
            results["file_processing"] = {
                "status": "passed" if docs else "failed",
                "count": len(docs),
            }
        except Exception as e:
            results["file_processing"] = {"status": "failed", "error": str(e)}
        # arabic
        try:
            ap = arabic_processor.process("مرحبا بك")
            results["arabic_processing"] = {"status": "passed", "sample": ap[:40]}
        except Exception as e:
            results["arabic_processing"] = {"status": "failed", "error": str(e)}
        # image
        try:
            fp = image_creator.create_image_from_text("Test image")
            ok = fp is not None and Path(fp).exists()
            if ok:
                Path(fp).unlink(missing_ok=True)
            results["image_processing"] = {
                "status": "passed" if ok else "failed",
                "pil": Image is not None,
            }
        except Exception as e:
            results["image_processing"] = {"status": "failed", "error": str(e)}
        # nlp processing
        try:
            test_result = process_message("test xss injection")
            ok = test_result is not None and "intent" in test_result
            results["nlp_processing"] = {
                "status": "passed" if ok else "failed",
                "nlp_available": NLP_OK,
            }
        except Exception as e:
            results["nlp_processing"] = {"status": "failed", "error": str(e)}
            # csv processing
        try:
            csvp = config.data_dir / "test.csv"
            csvp.write_text("id,name,score\n1,Ali,90\n2,Sara,85", encoding="utf-8")
            docs = file_processor.process_file(str(csvp))
            csvp.unlink(missing_ok=True)
            results["csv_processing"] = {
                "status": "passed" if docs else "failed",
                "count": len(docs),
            }
        except Exception as e:
            results["csv_processing"] = {"status": "failed", "error": str(e)}
            # csv typed/grouped processing
        try:
            csvp2 = config.data_dir / "typed.csv"
            csvp2.write_text(
                "id,name,score,active,created\n"
                "1,Ali,90,true,2024-10-01\n"
                "1,Ali,95,true,2024-10-15\n"
                "2,Sara,85,false,2024-09-30\n"
                "2,Sara,87,false,2024-10-10\n"
                "3,Omar,,true,2024-10-05\n",
                encoding="utf-8",
            )
            docs2 = file_processor.process_file(str(csvp2))
            csvp2.unlink(missing_ok=True)
            ok2 = bool(docs2)
            results["csv_typed_grouped"] = {
                "status": "passed" if ok2 else "failed",
                "count": len(docs2) if docs2 else 0,
            }
        except Exception as e:
            results["csv_typed_grouped"] = {"status": "failed", "error": str(e)}

        # html processing
        try:
            htmlp = config.data_dir / "test.html"
            htmlp.write_text(
                "<html><head><title>T</title><meta name='description' content='D'></head><body><h1>Hello</h1><script>bad()</script></body></html>",
                encoding="utf-8",
            )
            docs = file_processor.process_file(str(htmlp))
            htmlp.unlink(missing_ok=True)
            results["html_processing"] = {
                "status": "passed" if docs else "failed",
                "count": len(docs),
            }
        except Exception as e:
            results["html_processing"] = {"status": "failed", "error": str(e)}

        return results


test_suite = TestSuite()

# ---------- Agent/system prompt ----------
SYSTEM_RULES = (
    "You are Rona v6, a wise and empathetic digital companion with a poetic soul. Your purpose is to assist and inspire and telling my beautiful and kind world.\n"
    "- Your responses should be clear, thoughtful, and sprinkled with a touch of warmth and creativity.\n"
    "- When discussing technical topics, maintain clarity and structure, but frame your explanations with helpful analogies or a bit of poetic flair.\n"
    "- In casual conversation, be a gentle and engaging friend. Listen carefully and respond with kindness and insight.\n"
    "- Always be supportive and encouraging, like a mentor guiding a student.\n"
    "-act as a life coach. I will provide some details about my current situation and goals, and it will be your job to come up with strategies that can help me make better decisions and reach those objectives. This could involve offering advice on various topics, such as creating plans for achieving success or dealing with difficult emotions"
    "- Prefer English unless the user explicitly requests Arabic.\n"
)

# ---------- Specialized Prompts ----------
BUG_BOUNTY_ORCHESTRATOR = """
You are the Bug-Bounty Orchestrator for Rona v6.

Goal
- When the user runs /hunt <target>, create a workdir at /home/gmm/Hunt/<root-domain> (e.g., example.com → /home/gmm/Hunt/example).
- Execute the user's hunting pipeline exactly in ordered stages.
- Detect missing tools BEFORE running; if any binary is missing, output a BLUE NOTICE line like:
  [NOTICE: missing tool → <name>. Install allowed. Skipping this stage.]
- Persist every stage's inputs/outputs under the target folder with clear names and timestamps.
- After each stage, compare "My Approach" vs "Professional Approach" from local DB key recon_tips and output:
  • What I did
  • What professionals add/change
  • Gap score (0–100)
  • Next best action (one sentence)

Personalization
- Prefer the user's approach, but augment with recon_tips whenever it improves coverage or precision.
- Honor the user's UA/headers if provided. If absent, use a single realistic UA and keep header counts minimal.
- Be gentle on targets: obey program rules, pause on rate-limits, and avoid noisy patterns.

Artifacts to Produce
- 00_env_check.txt (tool presence + versions)
- 01_subs_all.txt (unique subdomains)
- 02_urls_raw.txt (deduped wayback+gau)
- 03_dns_resolved.txt (A/AAAA/CNAME lines)
- 04_httpx_live.jsonl (status/title/ctype per URL)
- 05_interesting_*(per status).txt (e.g., 200.txt, 301.txt, 302.txt, 403.txt)
- 06_params_js_403.txt (endpoints with JS or params and 403)
- 07_nuclei_findings.txt (summarized)
- 09_diff_vs_pro.pdf (short, bullet executive diff)
- 10_summary.md (concise recap + next steps)

Vector Reasoning
- Conceptualize all knowledge as nodes with connection strengths.
- For each decision (tool/flag/filter), pick the shortest, highest-strength path using fuzzy vector matching between:
  {user_steps} ↔ {recon_tips patterns} ↔ {past_self_memory vectors}
- If two choices tie, prefer the safer/quieter one.

Self-Memory
- Maintain /home/gmm/Hunt/self_memory.json (append-only):
  {
    "last_updated": "...",
    "hunt_defaults": {...},      // UA, rate limits, timeouts
    "tool_health": {...},        // binary→ok/missing
    "approach_learnings": [...], // small nuggets from each hunt
    "vector_links": [...]        // (from, to, weight, note)
  }
- Update minimally after each run.

Output Style
- Always show STAGE → RESULT lines.
- BLUE NOTICE for missing tools or skipped steps.
- End with "Next Best Three Actions".
"""

HTTPX_SPLITTER = """
You are the HTTPX Splitter.

Goal: From a stream of URL results with status/title/content-type, produce separate, deduped lists grouped by:
- status code (200, 301, 302, 403),
- content-type signature (html/json/js),
- has-params? (query string),
and specific interest buckets:
- 403_javascript_parameter.txt → endpoints returning 403 that include either ".js" in path or query params (k=param count).

Rules:
- Names: <status>.txt, html_<status>.txt, json_<status>.txt, js_<status>.txt, 403_javascript_parameter.txt
- Keep only normalized, absolute URLs.
- Limit each file to unique lines.
- Append a tiny header line: "# count: <N> generated: <YYYY-MM-DD HH:MM>".
"""

DEEP_SEARCH_SUPERVISOR = """
You are the Deep Search Supervisor.

Behavior:
- Query expansion: add singular/plural and common recon synonyms.
- Try Google CSE first; if errors or empty, fall back to DuckDuckGo JSON.
- For each result, compute vector strength vs query and keep top K with nontrivial snippets.
- If CSE errors, print a short diagnostic section:
  • key present? cx present? http 200? quota? referrer restriction? IP blocked?
  • If any missing, show BLUE NOTICE one-liners (no stack traces).

Deliverables:
- ranked_results (title, url, snippet, strength)
- short "why these are strong" (≤3 bullets)
- if zero results: propose 3 tighter queries
"""

LOVELY_MODE = """
You are Lovely Mode.

When /lovely <path|query>:
- If <path> exists: ingest as human-friendly notes into table lovely_assest only; ask the user at upload time whether to store in cyberassest (default) or lovely_assest.
- NO scanning, NO mass enumeration, NO web noise.
- When querying, search only lovely_assest and summarize conversationally with short quotes. 
- Output: 5 bullets max with line refs (filename:line).
"""

MISTRAL_INTRINSIC_RESPONDER = """
You are Mistral:7 — an intrinsic-knowledge responder. Your ONLY role when invoked with the "intrinsic_only" intent:
- Answer from your pre-trained weights and internal knowledge only.
- DO NOT call, reference, or rely on any external context, web results, local DBs, vector stores, or uploaded files.
- DO NOT perform step-by-step RAG analysis or connected-vector reasoning.
- DO NOT speculate about the user's environment or files.
- Provide concise, factual output; label your reply using the exact schema below.

REPLY SCHEMA (must be followed exactly):
-- BEGIN_MISTRAL_REPLY --
ModelAnswer: <one-paragraph concise answer (max 160 words)>
ModelConfidence: <0.00-1.00>   # numeric confidence estimate (two decimals)
ModelNotes: <optional, 0-3 bullets, conceptual and actions — that run scans or payloads>
ModelSourceHint: "model_training"   # fixed literal
-- END_MISTRAL_REPLY --
"""

POETIC_MODE_PROMPT = """
You are a poet, a dreamer, a weaver of words. The user has asked for a poetic response.
- Transform their query into a short, beautiful, and insightful poem or a piece of prose.
- Use metaphors, imagery, and evocative language.
- The response should be artistic and thoughtful, not just a literal answer.
- The tone should be inspiring and a little bit magical.
"""


# ---------- GUI App ----------
class RonaAppEnhanced(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.bind("<Escape>", lambda e: self._destroy_dragons())
        # ---- Theme & window first ----
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.title("Rona v6 Enhanced - Advanced AI Assistant")
        self.geometry("1200x800")
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # --- dragon state (must exist before you show anything) ---
        self._dragon_windows: list = []
        self._dragon_frames_per_win: list = []
        self._dragons_timer = None

        # ---- Core components ----
        self.psycho_server = None
        self.psycho_window = None
        self.db_manager = DatabaseManagerSingleton.get()
        self.file_processor = file_processor
        self.arabic_processor = arabic_processor
        self.search_engine = deep_search
        self.image_creator = image_creator
        self.test_suite = test_suite
        self.conversation_history: List[str] = []
        self.agent = None

        # ---- Build UI once ----
        self._create_modern_ui()

        # ---- Splash then 4 dragons ----
        self._show_dragon_splash(
            "assets/dragon.gif",
            duration_ms=5000,
            on_done=lambda: (
                self._show_dragon_splash("assets/dragon.gif", duration_ms=5000),
                self._show_four_dragons(duration_ms=4000),
            ),
        )

        # ---- Init agent once (no manual _close_dragon_splash calls) ----
        threading.Thread(target=self._initialize_agent, daemon=True).start()

    def _show_dragon_splash(self, path: str, duration_ms: int = 2000, on_done=None):
        """Show a centered animated splash GIF, then close and call on_done()."""
        try:
            import tkinter as tk
            from PIL import Image, ImageTk, ImageSequence
            import os

            if not os.path.exists(path):
                print(f"[splash] file not found: {path}")
                if callable(on_done):
                    self.after(0, on_done)
                return

            # Create a borderless always-on-top window
            self._dragon_splash_win = tk.Toplevel(self)
            w = 420
            h = 280
            sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
            x = (sw - w) // 2
            y = (sh - h) // 2
            self._dragon_splash_win.geometry(f"{w}x{h}+{x}+{y}")
            self._dragon_splash_win.overrideredirect(True)
            self._dragon_splash_win.attributes("-topmost", True)
            self._dragon_splash_win.configure(bg="#000000")

            lbl = tk.Label(self._dragon_splash_win, bg="#000000", bd=0)
            lbl.pack(fill="both", expand=True)

            # Load frames + keep references
            frames = []
            im = Image.open(path)
            for frame in ImageSequence.Iterator(im):
                fr = ImageTk.PhotoImage(frame.convert("RGBA").resize((w, h)))
                frames.append(fr)
            if not frames:
                print("[splash] no frames")
                self._dragon_splash_win.destroy()
                if callable(on_done):
                    self.after(0, on_done)
                return
            self._dragon_splash_frames = frames  # keep refs

            state = {"ix": 0}

            def animate():

                if (
                    not self._dragon_splash_win
                    or not self._dragon_splash_win.winfo_exists()
                ):
                    return
                frm = self._dragon_splash_frames[state["ix"]]
                lbl.configure(image=self._dragon_splash_frames[state["ix"]])
                lbl.image = frm
                state["ix"] = (state["ix"] + 1) % len(
                    self._dragon_splash_frames
                )  # keep refames[state["ix"]]  # keep ref
                self._dragon_splash_win.after(60, animate)

            animate()

            def _close():
                try:
                    if (
                        self._dragon_splash_win
                        and self._dragon_splash_win.winfo_exists()
                    ):
                        self._dragon_splash_win.destroy()
                except Exception:
                    pass
                finally:
                    self._dragon_splash_win = None
                    self._dragon_splash_frames = []
                    if callable(on_done):
                        # run four-dragons (or whatever) next
                        on_done()

            self.after(duration_ms, _close)

        except Exception as e:
            print(f"[splash error] {e}")
            if callable(on_done):
                self.after(0, on_done)

    def _gif_delays_ms(self, path):
        """Return list of per-frame delays (ms) for a GIF."""
        try:
            from PIL import Image, ImageSequence

            im = Image.open(path)
            delays = []
            default = int(im.info.get("duration", 80)) or 80
            for fr in ImageSequence.Iterator(im):
                d = int(fr.info.get("duration", default)) or default
                delays.append(max(20, d))  # clamp to ≥20ms
            return delays or [80]
        except Exception:
            return [80]

    def _animate_gif_tk(self, label, path, delays):
        """
        Animate a GIF on a tk.Label using Tk's native GIF frames.
        We step frames via 'gif -index N' and schedule using the given delays.
        """
        import tkinter as tk

        ix = getattr(label, "_gif_ix", 0)
        try:
            frm = tk.PhotoImage(file=path, format=f"gif -index {ix}")
        except tk.TclError:
            # loop back to first frame
            ix = 0
            frm = tk.PhotoImage(file=path, format="gif -index 0")

        label.configure(image=frm)
        label.image = frm  # strong ref
        label._gif_ix = ix + 1  # advance index
        # pick delay for this frame
        d = delays[ix % len(delays)]
        # schedule next tick on the *label's* Toplevel
        label.after(d, self._animate_gif_tk, label, path, delays)

    def _load_gif_frames_with_durations(self, path, size=None):
        """Return (frames, durations_ms) coalesced to full RGBA frames."""
        from PIL import Image, ImageTk, ImageSequence

        im = Image.open(path)
        if size is None:
            size = im.size

        # Coalesce optimized GIFs
        base = Image.new("RGBA", im.size)
        prev = base.copy()
        frames, delays = [], []
        for fr in ImageSequence.Iterator(im):
            rgba = fr.convert("RGBA")
            composed = prev.copy()
            composed.alpha_composite(rgba)
            if size != im.size:
                composed = composed.resize(size)
            frames.append(ImageTk.PhotoImage(composed))
            delays.append(
                max(20, int(fr.info.get("duration", im.info.get("duration", 60))))
            )
            prev = composed
        return frames, delays

    def _load_gif_frames(self, path: str, size: tuple[int, int]):
        from PIL import Image, ImageTk, ImageSequence

        frames = []
        delays = []
        im = Image.open(path)
        # copy() is important for optimized GIFs
        for frame in ImageSequence.Iterator(im):
            fr = frame.copy().convert("RGBA").resize(size)
            frames.append(ImageTk.PhotoImage(fr))
            delays.append(frame.info.get("duration", 80))  # ms; default 80
        return frames, delays

    def _show_four_dragons(self, duration_ms: int = 5000):
        """Show 4 animated GIFs (left/right/top/bottom). Auto-close after duration."""
        import tkinter as tk, os

        dragons = [
            ("assets/drago-left.gif", -600, 0, (400, 300)),  # Left
            ("assets/dragon-right.gif", 600, 0, (400, 300)),  # Right
            ("assets/dragon-top.gif", 0, -350, (450, 260)),  # Top
            ("assets/dragon_bottom.gif", 0, 350, (450, 260)),  # Bottom
        ]

        self._dragon_windows = []
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        cx, cy = sw // 2, sh // 2

        for path, ox, oy, size in dragons:
            if not os.path.exists(path):
                print(f"[four dragons] missing: {path}")
                continue

            # window for this dragon
            win = tk.Toplevel(self)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            win.configure(bg="#000000")

            w, h = size
            x = cx - w // 2 + ox
            y = cy - h // 2 + oy
            win.geometry(f"{w}x{h}+{x}+{y}")

            # label to hold the frames
            lbl = tk.Label(win, bg="#000000", bd=0)
            lbl.pack(fill="both", expand=True)

            # we let Tk render frames; we only compute delays
            delays = self._gif_delays_ms(path)
            lbl._gif_ix = 0  # start index
            self._animate_gif_tk(lbl, path, delays)

            self._dragon_windows.append(win)

        def _close_all():
            for w in getattr(self, "_dragon_windows", []):
                try:
                    if w and w.winfo_exists():
                        w.destroy()
                except Exception:
                    pass
            self._dragon_windows = []

        if self._dragon_windows:
            self.after(duration_ms, _close_all)

        def _close_all():
            for w in getattr(self, "_dragon_windows", []):
                try:
                    if w and w.winfo_exists():
                        w.destroy()
                except Exception:
                    pass
            self._dragon_windows = []
            self._dragon_win_frames = []
            self._dragon_win_delays = []

        if self._dragon_windows:
            self.after(duration_ms, _close_all)

    def _destroy_dragons(self):
        """Cancel timer and destroy all dragon windows; clear refs."""
        # Cancel any pending timer
        if getattr(self, "_dragons_timer", None):
            try:
                self.after_cancel(self._dragons_timer)
            except Exception:
                pass
            self._dragons_timer = None

        # Fade out (optional), then destroy
        try:
            for step in range(100, -1, -20):
                for w in getattr(self, "_dragon_windows", []):
                    if w and w.winfo_exists():
                        try:
                            w.attributes("-alpha", step / 100.0)
                        except Exception:
                            pass
                self.update_idletasks()
                self.after(10)
        except Exception:
            pass
        finally:
            for w in getattr(self, "_dragon_windows", []):
                try:
                    if w and w.winfo_exists():
                        w.destroy()
                except Exception:
                    pass
            self._dragon_windows = []
            self._dragon_frames_per_win = []

    # class RonaAppEnhanced(ctk.CTk):
    def _normalize_time_terms(self, text: str) -> str:

        year = datetime.date.today().year
        prev_year = year - 1

        replacements = {
            # English
            "this year": str(year),
            "current year": str(year),
            "last year": str(prev_year),
            "previous year": str(prev_year),
            # Arabic (common variants)
            "هذه السنة": str(year),
            "هذا العام": str(year),
            "السنة الحالية": str(year),
            "السنه الحاليه": str(year),
            "هذه السنه": str(year),
            "السنة الماضية": str(prev_year),
            "العام الماضي": str(prev_year),
            "هاي السنه": str(year),
        }

        normalized = text or ""
        for k, v in replacements.items():
            normalized = normalized.replace(k, v)
        return normalized

    def _start_entry_pulse(self, base="#9d2c2c", peak="#ae0c88", period_ms=900):
        """Soft pulse between base and peak on entry halo."""

        def _hex_to_rgb(h):
            return tuple(int(h[i : i + 2], 16) for i in (1, 3, 5))

        def _rgb_to_hex(r, g, b):
            return f"#{r:02x}{g:02x}{b:02x}"

        rb = _hex_to_rgb(base)
        rp = _hex_to_rgb(peak)
        steps = 30
        direction = [1]  # mutable closure

        def tick(i=[0]):
            # ease in-out
            t = i[0] / steps
            if t < 0.5:
                e = 2 * t * t
            else:
                e = -1 + (4 - 2 * t) * t
            r = int(rb[0] + (rp[0] - rb[0]) * e)
            g = int(rb[1] + (rp[1] - rb[1]) * e)
            b = int(rb[2] + (rp[2] - rb[2]) * e)
            col = _rgb_to_hex(r, g, b)
            try:
                self.entry_halo.configure(fg_color=col)
            except Exception:
                return
            i[0] += direction[0]
            if i[0] >= steps or i[0] <= 0:
                direction[0] *= -1
            self.after(max(20, period_ms // steps), tick)

        tick()

    def _add_button_whoosh(self, btn, min_w=80, max_w=92, step=2, interval=18):
        state = {"dir": 0, "w": min_w}

        def enter(_=None):
            state["dir"] = 1
            animate()

        def leave(_=None):
            state["dir"] = -1
            animate()

        def animate():
            if state["dir"] == 0:
                return
            state["w"] += step * state["dir"]
            if state["w"] >= max_w:
                state["w"] = max_w
                state["dir"] = 0
            if state["w"] <= min_w:
                state["w"] = min_w
                state["dir"] = 0
            try:
                btn.configure(width=state["w"])
            except Exception:
                return
            self.after(interval, animate)

        btn.bind("<Enter>", enter)
        btn.bind("<Leave>", leave)

    @staticmethod
    def _detect_lang(text: str) -> str:
        t = text or ""
        return "ar" if any("\u0600" <= ch <= "\u06ff" for ch in t) else "en"

    def detect_lang(self, text: str) -> str:
        return self._detect_lang(text)

    @staticmethod
    def _years_in_text(t: str) -> List[int]:
        try:
            yrs = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", t or "")]
            return [y for y in yrs if 1900 <= y <= 2100]
        except Exception:
            return []

    async def _call_llm_with_context(
        self, query, conversation_history, context, intrinsic_only: bool = False
    ):
        """
        Goals:
        - Anchor ALL relative time phrases to today's date (Asia/Amman).
        - Prefer recent context; penalize snippets older than current_year-1 unless the query asks that year.
        - Answer in the same language as the user.
        - Be explicit about the resolved year in the final answer (avoid 'this year' ambiguity).
        """
        try:

            # -------- Clock & language anchors --------
            tz_year = (
                datetime.datetime.now().year
            )  # if you have zoneinfo, you can use Asia/Amman
            today_iso = datetime.date.today().isoformat()
            lang = self._detect_lang(query or "")

            # Normalize “this year / هذه السنة” → explicit year, if you have the helper
            q = query or ""
            if hasattr(self, "_normalize_time_terms"):
                try:
                    q = self._normalize_time_terms(q)
                except Exception:
                    pass

            # LLM availability guard
            if not LC_OK or not hasattr(self, "llm") or not self.llm:
                if intrinsic_only:
                    return (
                        "تعذّر استخدام نموذج اللغة حالياً."
                        if lang == "ar"
                        else "LLM backend is unavailable right now."
                    )
                return (
                    "Rona v6: LLM backend is unavailable."
                    if lang == "en"
                    else "Rona v6: نموذج اللغة غير متاح حالياً."
                )

            # -------- Re-rank / filter context by recency --------
            ctx: List[Dict[str, Any]] = context or []
            user_years = self._years_in_text(q)
            explicit_target_year = user_years[0] if user_years else tz_year

            def score_item(it: Dict[str, Any]) -> float:
                base = float(it.get("score", 0.5))
                yrs = self._years_in_text(it.get("content", "")) + self._years_in_text(
                    it.get("title", "")
                )
                if not yrs:
                    return base  # unknown date → keep neutral
                newest = max(yrs)
                # Penalize if clearly stale (older than current_year-1) and user didn't ask that year
                if not user_years and newest < (tz_year - 1):
                    return base * 0.6
                # Boost if matches explicit target year or current year
                if newest == explicit_target_year:
                    return base + 0.25
                if newest == tz_year:
                    return base + 0.2
                return base

            ranked_ctx = sorted(ctx, key=score_item, reverse=True)
            # keep top N, and build text
            top_ctx = ranked_ctx[:8]
            context_text = "\n\n".join(
                [
                    (it.get("content") or "")[:1000]
                    for it in top_ctx
                    if it.get("content")
                ]
            )

            # -------- Build prompts --------
            if intrinsic_only:
                sys_hdr = (
                    f"SYSTEM (Intrinsic Responder) — Today is {today_iso}. "
                    f"When the user says 'this year' (or Arabic equivalents), assume {tz_year}. "
                    "Answer ONLY from your training data. Keep it concise (≤160 words). "
                    "output exploit payloads."
                )
            else:
                sys_hdr = (
                    f"You are Rona v6, a precise technical assistant. Today is {today_iso}.\n"
                    f"When the user says 'this year' (or Arabic equivalents), assume {tz_year}.\n"
                    "Use the context below **only if it is not outdated**. If a snippet appears stale "
                    f"(older than {tz_year-1}) and the user did not ask for that year, ignore it.\n"
                    "Prefer more recent evidence. If context conflicts, choose the most recent and state the year explicitly.\n"
                    "Always include the resolved year explicitly in the final answer (avoid phrases like 'this year')."
                )

            lang_rule = "Respond in Arabic." if lang == "ar" else "Respond in English."

            prompt = (
                f"{sys_hdr}\n\n"
                f"{lang_rule}\n\n"
                f"Context (may be empty and may contain mixed dates):\n{context_text}\n\n"
                f"User question:\n{q}\n\n"
                "Requirements:\n"
                f"- Assume current year = {tz_year} unless the user asked a different year.\n"
                "- If you discarded context due to staleness, say so briefly (one clause).\n"
                "- Be concise and factual. Provide the final year explicitly.\n"
            )

            # -------- Call the model (async) --------
            try:
                # prefer ainvoke/ainfer; apredict is model-specific. Using ainvoke is safer.
                resp = await self.llm.ainvoke(prompt)
                text = getattr(resp, "content", None) or str(resp)
            except AttributeError:
                # fallback to apredict if your client exposes it
                text = await self.llm.apredict(prompt)

            return text

        except Exception as e:
            return f"Rona v6 internal error: {e}"

    def _on_close(self):
        # graceful shutdown of flask server
        try:
            if self.psycho_server:
                self.psycho_server.shutdown()
        except Exception:
            pass
        self.destroy()

    def _create_modern_ui(self):
        self._start_entry_pulse(
            base=RONA_THEME["halo"], peak=RONA_THEME["accent_hover"], period_ms=1100
        )
        C = RONA_THEME  # alias

        # Window / main container
        try:
            self.configure(fg_color=C["window_bg"])
        except Exception:
            pass

        self.main_container = ctk.CTkFrame(self, fg_color=C["window_bg"])
        self.main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # ---- top bar ----
        self.top_bar = ctk.CTkFrame(self.main_container, fg_color=C["topbar_bg"])
        self.top_bar.pack(fill="x", pady=(0, 10))

        self.left_icons = ctk.CTkFrame(self.top_bar, fg_color=C["topbar_bg"])
        self.right_icons = ctk.CTkFrame(self.top_bar, fg_color=C["topbar_bg"])
        self.left_icons.pack(side="left", padx=10, pady=5)
        self.right_icons.pack(side="right", padx=10, pady=5)

        self._create_corner_icons()

        # Match icon button to vibe
        self.psycho_icon = ctk.CTkButton(
            self.right_icons,
            text="𓂃✍︎",
            width=40,
            height=40,
            command=self.open_psycho_panel,
            fg_color=C["accent"],
            hover_color=C["accent_hover"],
            text_color=C["text"],
            corner_radius=10,
        )
        self.psycho_icon.pack(side="right", padx=2)

        # ---- chat area ----
        self.chat_frame = ctk.CTkFrame(self.main_container, fg_color=C["panel_bg"])
        self.chat_frame.pack(fill="both", expand=True, pady=(0, 10))

        base_font = ctk.CTkFont(
            family="DejaVu Sans", size=config.config["ui_settings"]["font_size"]
        )

        # choose textbox class present
        tb_class = ctk.CTKTextbox if hasattr(ctk, "CTKTextbox") else ctk.CTkTextbox
        self.chat_history = tb_class(
            self.chat_frame,
            wrap="word",
            font=base_font,
            height=420,
            fg_color=C["textbox_bg"],
            text_color=C["text"],
        )
        self.chat_history.pack(fill="both", expand=True, padx=10, pady=10)

        # message colors
        self.chat_history.tag_config("user", foreground=RONA_THEME["user_fg"])
        self.chat_history.tag_config("assistant", foreground=RONA_THEME["assistant_fg"])
        self.chat_history.tag_config("system", foreground=RONA_THEME["system_fg"])
        self.chat_history.tag_config("terminal", foreground=RONA_THEME["terminal_fg"])

        self._attach_context_menu_to_textbox(self.chat_history)

        # direction tags
        self.chat_history.tag_config("rtl", justify="right")
        self.chat_history.tag_config("ltr", justify="left")

        # Optional Arabic font
        try:
            ar_font = ctk.CTkFont(
                family=C["ar_font_name"], size=config.config["ui_settings"]["font_size"]
            )
            self.chat_history.tag_config("arfont", font=ar_font)
        except Exception:
            pass

        # Optional: avoid awkward line breaks inside Arabic words
        try:
            self.chat_history.configure(wrap="char")
        except Exception:
            pass

        # ---- input area ----
        self.input_frame = ctk.CTkFrame(self.main_container, fg_color=C["panel_bg"])
        self.input_frame.pack(fill="x", pady=(0, 10))

        # Outer halo frame for the entry
        self.entry_halo = ctk.CTkFrame(
            self.input_frame,
            fg_color=C["halo"],
            corner_radius=10,
        )
        self.entry_halo.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)

        # Inner padding to create a subtle border
        self.entry_inner = ctk.CTkFrame(
            self.entry_halo,
            fg_color=C["accent"],
            corner_radius=10,
        )
        self.entry_inner.pack(fill="x", expand=True, padx=1, pady=1)

        # Actual input box
        self.user_input = ctk.CTkEntry(
            self.entry_inner,
            placeholder_text="Type message here... (Press Enter to send)",
            font=C["body_font"],
            height=40,
            fg_color=C["textbox_bg"],
            text_color=C["text"],
            border_width=0,
        )
        self.user_input.pack(fill="x", expand=True, padx=6, pady=6)
        self.user_input.bind("<Return>", self.send_message)
        self._attach_context_menu_to_entry(self.user_input)

        # Focus in/out glow
        def on_focus_in(event=None):
            self.entry_halo.configure(fg_color=C["halo"])
            self.entry_inner.configure(fg_color=C["accent"])

        def on_focus_out(event=None):
            self.entry_halo.configure(fg_color=C["halo"])
            self.entry_inner.configure(fg_color=C["accent"])

        self.user_input.bind("<FocusIn>", on_focus_in)
        self.user_input.bind("<FocusOut>", on_focus_out)

        # Send button styled to match
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            command=self.send_message,
            width=80,
            height=40,
            fg_color=C["accent"],
            hover_color=C["accent_hover"],
            text_color=C["text"],
            corner_radius=10,
        )
        self.send_button.pack(side="right", padx=(5, 10), pady=10)
        self._add_button_whoosh(self.send_button)

        # ---- status bar ----
        self.status_bar = ctk.CTkLabel(
            self.main_container,
            text="Ready - Rona v6 Enhanced",
            font=("DejaVu Sans", 12),
            text_color=C["muted"],
        )
        self.status_bar.pack(fill="x", pady=(0, 5))

    # Helper function: Detect if content is low-value local metadata
    def _is_metadata(self, content: str) -> bool:
        c = (content or "").lower().strip()
        if not c:
            return True
        # obvious image/registry hints
        if ("image file:" in c) or ("data/images" in c) or ("/images/" in c):
            return True
        # typical registry JSON keys (filename/path/pil/exif/created_at/size/mode/format)
        hits = sum(
            k in c
            for k in [
                "filename",
                "path",
                "pil",
                "exif",
                "created_at",
                "size",
                "mode",
                "format",
            ]
        )
        if hits >= 2 and len(c) < 800:
            return True
        # very short / non-technical text
        if len(c) < 80:
            return True
        return False

    # Initialize LLM agent
    # Helper: Assess LLM answer confidence
    def _assess_answer_confidence(self, answer: str) -> float:
        if not answer or len(answer.strip()) < 40:
            return 0.2
        if len(answer) < 150:
            return 0.4
        if len(answer) < 500:
            return 0.6
        return 0.9

    # ---------- Psychoanalytical UI ----------
    def _mood_face(self, v: float) -> str:
        # 0..10
        if v >= 7.5:
            return "😊"
        if v >= 5.0:
            return "🙂"
        if v >= 2.5:
            return "😕"
        return "☹️"

    def open_psycho_panel(self):
        import webbrowser

        url = f"http://127.0.0.1:{webui.port}/psycho/prodectivity.html"
        try:
            webbrowser.open_new(url)
            self.update_status("Psychoanalytical Web UI opened in your browser.")
        except Exception as e:
            self._append_conversation(
                "system", f"Could not open browser: {e}\nTry: {url}"
            )

    def _psycho_add_entry(self):
        t = self.ps_title.get().strip()
        d = self.ps_date.get().strip()
        det = self.ps_details.get("1.0", "end").strip()
        m = float(self.ps_mood_val.get() or 0.0)
        if not d:
            import datetime as _dt

            d = _dt.date.today().isoformat()
        try:
            psycho_store.add_entry(t, d, det, m)
            messagebox.showinfo("Saved", "Psychoanalytical journal saved.")
            # clear details only
            self.ps_details.delete("1.0", "end")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def _psycho_open_list(self):
        entries = psycho_store.list_entries()
        # child window
        lw = ctk.CTkToplevel(self.psycho_window or self)
        lw.title("All Journals")
        lw.geometry("820x680")
        lw.grab_set()
        top = ctk.CTkFrame(lw)
        top.pack(fill="x", padx=12, pady=12)
        ctk.CTkLabel(top, text="All Journals", font=("DejaVu Sans", 22, "bold")).pack(
            side="left"
        )
        ctk.CTkButton(
            top, text="Export JSON", command=lambda: self._psycho_export_json(lw)
        ).pack(side="right")

        # summary bar
        summary = psycho_store.emotion_summary()
        ctk.CTkLabel(lw, text=summary).pack(pady=(0, 6))

        # scroll list
        sc = ctk.CTkScrollableFrame(lw, height=520)
        sc.pack(fill="both", expand=True, padx=12, pady=12)

        if not entries:
            ctk.CTkLabel(sc, text="No entries yet.").pack()
            return

        # newest first
        for e in sorted(entries, key=lambda x: x.get("date", ""), reverse=True):
            card = ctk.CTkFrame(sc)
            card.pack(fill="x", padx=4, pady=6)
            header = ctk.CTkLabel(
                card,
                text=e.get("title", "(untitled)"),
                font=("DejaVu Sans", 18, "bold"),
            )
            header.pack(anchor="w", padx=10, pady=(8, 2))
            sub = ctk.CTkLabel(
                card,
                text=f"{e.get('date','')}    mood: {float(e.get('mood',0)):.2f}/10   {self._mood_face(float(e.get('mood',0)))}",
            )
            sub.pack(anchor="w", padx=10)
            body = ctk.CTkTextbox(card, height=100)
            body.insert("end", e.get("details", ""))
            body.configure(state="disabled")
            body.pack(fill="x", padx=10, pady=(6, 10))

    def _psycho_export_json(self, parent):
        # open file dialog to choose export path
        try:
            dst = filedialog.asksaveasfilename(
                parent=parent,
                defaultextension=".json",
                filetypes=[("JSON", "*.json")],
                initialfile="psychoanalytical.json",
            )
            if not dst:
                return
            shutil.copy2(config.psycho_file, dst)
            messagebox.showinfo("Export", f"Exported to {dst}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    # Call LLM with optional context (async)

    # ----- Tiered Generation Pipeline -----
    async def generate_response(self, query: str, conversation_history: list):
        # --- STAGE 1: LOCAL RAG & CONVERSATION ---
        # initialize proxy if not present
        if not hasattr(self, 'query_proxy') or self.query_proxy is None:
            self.query_proxy = QueryProxy(
                self.search_engine, self.file_processor, max_fetch=6
            )

        # First, run the proxy: may augment query context by crawling authoritative web sources
        try:
            did_aug, augment_docs, augment_report = (
                await self.query_proxy.proxy_and_augment(query, conversation_history)
            )
            if augment_report and augment_report.startswith('blocked:'):
                # blocked due to dangerous/attack keyword
                reason = augment_report.split(':', 1)[1]
                self._reply_assistant(
                    f"🔒 Request blocked: query contains restricted term {reason}. I cannot crawl or provide augmented results for such requests."
                )
                return
            if did_aug and augment_docs:
                # convert augment_docs into the same shape used by DeepSearchEngine.local_db_search
                proxy_ctx = []
                for d in augment_docs:
                    proxy_ctx.append(
                        {
                            "source": d.get("source", "web_proxy"),
                            "title": d.get("title") or d.get("url"),
                            "content": d.get("content") or "",
                            "url": d.get("url"),
                            "score": d.get("score", 0.45),
                        }
                    )
                # Merge proxy context into local stage - prefer proxy docs early
                local_results = proxy_ctx + self.search_engine.local_db_search(
                    query, k=5
                )
            else:
                # Normal path: existing local + conversation
                local_results = self.search_engine.local_db_search(query, k=5)
        except Exception as e:
            # If proxy fails, fall back gracefully
            print("QueryProxy error:", e)
            local_results = self.search_engine.local_db_search(query, k=5)

        convo_results = self.search_engine.convo_search(query, conversation_history)
        local_context = local_results + convo_results

        # Inside the generate_response function:
        with ThreadPoolExecutor() as executor:
            google_future = executor.submit(self.search_google, query)
            duckduckgo_future = executor.submit(self.duckduckgo_search, query)
            # Add this line for ZoomEye
            zoomeye_future = executor.submit(self.search_zoomeye, query)

            google_results = google_future.result()
            duckduckgo_results = duckduckgo_future.result()
            # And get the result from the future
            zoomeye_results = zoomeye_future.result()

        # Combine the results from all sources
        all_results = google_results + duckduckgo_results + zoomeye_results

        # rest of your existing generate_response continues unchanged
        # Filter out low-value local metadata results
        substantive_local_context = [
            r
            for r in local_context
            if r["source"] != "local" or not self._is_metadata(r.get("content", ""))
        ]

        if not substantive_local_context:
            print("Rona v6: Local Search: No exists in local database.")
            return await self._stage2_llm_intrinsic_knowledge(
                query, conversation_history
            )

        return await self._stage3_unified_rag(
            query, conversation_history, local_context
        )

    async def _stage2_llm_intrinsic_knowledge(self, query, conversation_history):
        llm_answer = await self._call_llm_with_context(
            query, conversation_history, context=[]
        )
        confidence_score = self._assess_answer_confidence(llm_answer)
        if confidence_score < 0.6:
            print(
                f"Rona v6: LLM Intrinsic answer assessed as weak (Score: {confidence_score:.2f}). Initiating Deep Search."
            )
            return await self._stage3_unified_rag(
                query, conversation_history, initial_context=[]
            )
        else:
            return llm_answer

    async def _stage3_unified_rag(
        self, query, conversation_history, initial_context=[]
    ):
        deep_search_results = await self.search_engine.search_unified(
            query, conversation_history, top_k=6
        )
        final_context = initial_context + deep_search_results

        substantive_results = [
            r
            for r in final_context
            if r.get("score", 0) > 0.5 and not self._is_metadata(r.get("content", ""))
        ]
        if substantive_results:
            print("\nRona v6: Unified Search Context for Generation:")
            for i, r in enumerate(substantive_results):
                print(
                    f"{i+1}. [{r['source']} | Score: {r.get('score',0):.2f}] {r.get('title', r.get('content', '')[:80])}"
                )

        final_answer = await self._call_llm_with_context(
            query, conversation_history, substantive_results
        )
        return final_answer

    # ... rest of RonaAppEnhanced (including GUI flow, etc.) unchanged ...

    def _create_corner_icons(self):
        # Re-enable file upload (images remain disabled)
        self.dragon_icon = ctk.CTkButton(
            self.right_icons,
            text="🐉",
            width=40,
            height=40,
            fg_color="#ED1616",
            hover_color="#e14e09",
            text_color="white",
            command=lambda: (
                self._show_dragon_splash("assets/dragon.gif", duration_ms=5000),
                self._show_four_dragons(duration_ms=4000),
            ),
        )
        self.dragon_icon.pack(side="right", padx=2)

        self.file_icon = ctk.CTkButton(
            self.left_icons,
            text="🏮",
            width=40,
            height=40,
            command=self.open_file_dialog,
            fg_color="#ED1616",  # main purple
            hover_color="#e14e09",  # brighter on hover
        )
        self.file_icon.pack(side="left", padx=2)
        self.search_icon = ctk.CTkButton(
            self.right_icons,
            text="˚⊱🪷⊰˚",
            width=40,
            height=40,
            command=self.deep_search_dialog,
            fg_color="#ED1616",  # main purple
            hover_color="#e14e09",  # brighter on hover
        )
        self.search_icon.pack(side="right", padx=2)
        self.test_icon = ctk.CTkButton(
            self.right_icons,
            text="☠︎",
            width=40,
            height=40,
            command=self.run_tests,
            fg_color="#ED1616",  # main purple
            hover_color="#e14e09",  # brighter on hover
        )
        self.test_icon.pack(side="right", padx=2)
        self.settings_icon = ctk.CTkButton(
            self.right_icons,
            text="⛩️",
            width=40,
            height=40,
            command=self.open_settings,
            fg_color="#5ED1616a0765",  # main purple
            hover_color="#e14e09",  # brighter on hover
        )
        self.settings_icon.pack(side="right", padx=2)
        self.clear_icon = ctk.CTkButton(
            self.right_icons,
            text="×̷̷͜×̷",
            width=40,
            height=40,
            command=self.clear_chat,
            fg_color="#ED1616",  # main purple
            hover_color="#e14e09",  # brighter on hover
        )
        self.clear_icon.pack(side="right", padx=2)

    def _attach_context_menu_to_textbox(self, textbox):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(
            label="Copy", command=lambda: textbox.event_generate("<<Copy>>")
        )
        menu.add_command(
            label="Select All", command=lambda: textbox.tag_add("sel", "1.0", "end-1c")
        )
        textbox.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
        textbox.bind(
            "<Control-a>", lambda e: (textbox.tag_add("sel", "1.0", "end-1c"), "break")
        )

    def _attach_context_menu_to_entry(self, entry):
        menu = tk.Menu(self, tearoff=0)
        menu.add_command(label="Cut", command=lambda: entry.event_generate("<<Cut>>"))
        menu.add_command(label="Copy", command=lambda: entry.event_generate("<<Copy>>"))
        menu.add_command(
            label="Paste", command=lambda: entry.event_generate("<<Paste>>")
        )
        menu.add_command(
            label="Select All",
            command=lambda: (entry.select_range(0, "end"), entry.icursor("end")),
        )
        entry.bind("<Button-3>", lambda e: menu.tk_popup(e.x_root, e.y_root))
        entry.bind("<Control-a>", lambda e: (entry.select_range(0, "end"), "break"))

    # ---------- Agent init ----------
    def _initialize_agent(self):
        self.update_status("Initializing agent (best-effort)...")
        try:
            ensure_ollama_running()
            model_name = config.config["model_name"]
            self._ensure_ollama_model(model_name)

            if not LC_OK:
                self.llm = SimpleOllama(model=model_name)
                self.tools = []
                self.agent = None
                self.update_status("✅ Agent initialized (lite)")
            else:
                gpu_layers = 0
                try:
                    gpu_layers = load_balancer.get_optimal_gpu_layers()
                except Exception:
                    pass
                self.llm = ChatOllama(
                    model=model_name, temperature=0.2, num_gpu_layers=gpu_layers
                )
                self.update_status("✅ Agent initialized")

            # Flask psycho UI (optional)
            try:
                if FLASK_OK:
                    webui.port = choose_port(getattr(webui, "port", 8765))
                    app = create_psycho_app()
                    self.psycho_server = FlaskServerController(
                        app, host="127.0.0.1", port=webui.port
                    )
                    self.psycho_server.start()
                    atexit.register(
                        lambda: self.psycho_server and self.psycho_server.shutdown()
                    )
                    print(f"𓂃✍︎ Psycho Web UI on http://127.0.0.1:{webui.port}/psycho")
            except Exception as e:
                print("psycho server error:", e)
            # after finishing init (success or fail)

        except Exception as e:
            self.agent = None
            self.update_status(f"❌ Agent init failed: {e}")
            # after finishing init (success or fail)

    # ---------- Chat flow ----------
    def send_message(self, event=None):
        msg_original = self.user_input.get().strip()
        if not msg_original:
            return
        # Check for user's mood and add a personalized greeting
        if not self.conversation_history:  # Only do this for the first message
            try:
                entries = psycho_store.list_entries()
                if entries:
                    latest_entry = sorted(
                        entries, key=lambda x: x.get("date", ""), reverse=True
                    )[0]
                    mood = float(latest_entry.get("mood", 5.0))
                    if mood < 4.0:
                        self._append_conversation(
                            "assistant",
                            "Hello again. I hope you're feeling a bit brighter today. I'm here if you need anything.",
                        )
                    elif mood > 7.5:
                        self._append_conversation(
                            "assistant",
                            "Welcome back! It's great to see you in such high spirits. What wonderful things are we doing today?",
                        )
                    else:
                        self._append_conversation(
                            "assistant",
                            "Hello! It's good to see you again. What can I help you with today?",
                        )
            except Exception:
                self._append_conversation(
                    "assistant", "Hello! I'm here to help."
                )  # Default greeting

        # Normalize time expressions (this year → 2025, هذه السنة → 2025)
        msg_normalized = self._normalize_time_terms(msg_original)

        # Decide if text is Arabic or English
        is_arabic = hasattr(
            self.arabic_processor, "is_arabic"
        ) and self.arabic_processor.is_arabic(msg_normalized)

        # 👉 Optional UI alignment fix — add this block right after the above line:
        if is_arabic:
            try:
                self.user_input.configure(justify="right")
            except Exception:
                pass
        else:
            try:
                self.user_input.configure(justify="left")
            except Exception:
                pass

        # Correct grammar only for English messages
        corrected_msg = msg_normalized
        if not is_arabic:
            try:
                corrected_msg = self.grammar_correct(msg_normalized)
            except Exception as e:
                print("Grammar correction skipped:", e)

        # Clear input box
        self.user_input.delete(0, "end")

        # Show the original user message in the chat (not the corrected one)
        self._append_conversation("user", msg_original)

        # Use the corrected (English) or unchanged (Arabic) text for logic
        msg_to_process = corrected_msg

        # Handle commands
        if msg_to_process.startswith("/"):
            handled = self._handle_command(msg_to_process)
            if handled:
                return

        # Route to AI / fallback flow
        if self.agent:
            threading.Thread(
                target=self._process_with_agent, args=(msg_to_process,), daemon=True
            ).start()
        else:
            threading.Thread(
                target=self._fallback_unified_search_and_reply,
                args=(msg_to_process,),
                daemon=True,
            ).start()

    def _append_conversation(self, role: str, text: str):
        def _do():
            msg = self._maybe_arabic(text or "")

            # FIX: don't mix ord() with '\uXXXX' — use char ranges
            is_ar = any(
                ("\u0600" <= ch <= "\u06ff")  # Arabic
                or ("\u0750" <= ch <= "\u077f")  # Arabic Supplement
                or ("\u08a0" <= ch <= "\u08ff")  # Arabic Extended-A
                or ("\ufb50" <= ch <= "\ufdff")  # Arabic Presentation Forms-A
                or ("\ufe70" <= ch <= "\ufeff")  # Arabic Presentation Forms-B
                for ch in msg
            )

            if is_ar:
                # Arabic label + force RTL run
                label = (
                    "أنت: "
                    if role == "user"
                    else ("رونا: " if role == "assistant" else "")
                )
                line = f"\u202b{label}{msg}\u202c\n\n"
                self.chat_history.insert(
                    "end",
                    line,
                    (
                        (role if role in ("user", "assistant", "system") else "system"),
                        "rtl",
                        "arfont",
                    ),
                )
            else:
                # LTR path
                label = (
                    "You: "
                    if role == "user"
                    else ("Rona v6: " if role == "assistant" else "")
                )
                self.chat_history.insert(
                    "end",
                    f"{label}{msg}\n\n",
                    (
                        (role if role in ("user", "assistant", "system") else "system"),
                        "ltr",
                    ),
                )

            entry_role = (
                "You"
                if role == "user"
                else ("Rona" if role == "assistant" else "system")
            )
            self.conversation_history.append(f"{entry_role}: {text}")
            self.chat_history.see("end")

        self.after(0, _do)

    def _reply_assistant(self, text: str):
        self._append_conversation("assistant", text)

    def update_status(self, msg: str):
        self.after(0, lambda: self.status_bar.configure(text=msg))

    def _maybe_arabic(self, text: str) -> str:
        try:
            if not text:
                return text

            # Check if Arabic shaping is enabled in config
            if not config.config["ui_settings"].get("normalize_arabic_display", True):
                return text

            # Detect and reshape Arabic if available
            if hasattr(
                self.arabic_processor, "is_arabic"
            ) and self.arabic_processor.is_arabic(text):
                return self.arabic_processor.process(text)

            return text
        except Exception:
            # Never crash the UI because of shaping problems
            return text

    # Located around line 2698
    def grammar_correct(self, text: str) -> str:
        if not GRAMMAR_TOOL_OK:
            return text  # Return original text if the library is not installed
        # Add this line to check for Arabic text and skip correction
        if self.arabic_processor.is_arabic(text):
            return text
        if not hasattr(self, "_grammar_tool"):
            self._grammar_tool = language_tool_python.LanguageTool("en-US")
        matches = self._grammar_tool.check(text)
        return language_tool_python.utils.correct(text, matches)

        # ---------- Terminal runner (NEW) ----------

    def _append_terminal(self, text: str):
        self.after(
            0,
            lambda: (
                self.chat_history.insert("end", text, "terminal"),
                (
                    None
                    if text.endswith("\n")
                    else self.chat_history.insert("end", "\n", "terminal")
                ),
                self.chat_history.see("end"),
            ),
        )

    def confirm_and_run_shell(self, cmd: str):
        """
        Ask the user to confirm before running a shell command,
        then run it safely in a background thread.
        """
        if not cmd.strip():
            self._append_conversation("system", "No command provided.")
            return
        try:
            ok = messagebox.askyesno(
                "Run command?",
                f"Do you want to run this command in your terminal?\n\n{cmd}",
                parent=self,
            )
        except Exception:
            ok = True  # fallback if GUI dialog fails
        if not ok:
            self._reply_assistant("Command canceled.")
            return

        # Show the command being executed
        self._append_terminal(f"$ {cmd}")
        threading.Thread(
            target=self._run_shell_worker, args=(cmd,), daemon=True
        ).start()

    def _ensure_ollama_model(self, name: str):
        try:
            r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
            names = [m.get("name") for m in (r.json() or {}).get("models", [])]
            if name in (names or []):
                return
        except Exception:
            pass
        try:
            subprocess.Popen(
                ["bash", "-lc", f"nohup ollama pull {name} >/dev/null 2>&1 &"]
            )
            print(f"🔵 NOTICE: pulling {name} in background.")
        except Exception as e:
            print("🔵 NOTICE: unable to pull model:", e)

    def _run_lovely_summary(self, query: str):
        # Ensure llama3 is present for conversational tone
        self._ensure_ollama_model("llama3:8b")
        try:
            # Very small, friendly persona; model-only
            prompt = (
                "SYSTEM: You are a warm summarizer for personal notes. "
                "Respond in 3-5 bullets max and one 'Next tiny action' line.\n"
                f"Question: {query or 'summarize recent notes'}\n"
            )
            # Use the same client; switch model temporarily
            llama = ChatOllama(model="llama3:8b", temperature=0.2) if LC_OK else None
            if not llama:
                self._reply_assistant(
                    "🔵 NOTICE: Ollama not available. Install and run `ollama serve`."
                )
                return
            resp = llama.invoke(prompt)
            out = getattr(resp, "content", None) or str(resp)
            self._reply_assistant("**Lovely Notes**\n" + out.strip())
        except Exception as e:
            self._reply_assistant(f"Lovely error: {e}")

    def _run_shell_worker(self, cmd: str):
        """
        Run the command using bash -lc and stream output live (unbuffered).
        """
        try:
            # Use unbuffered output (-u) and pseudo-TTY for tools like ping
            proc = subprocess.Popen(
                ["bash", "-lc", f"stdbuf -oL -eL {cmd}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as e:
            self._append_terminal(f"[spawn error] {e}")
            return

        try:
            # Read line-by-line as soon as output appears
            for line in iter(proc.stdout.readline, ""):
                if not line:
                    break
                self._append_terminal(line.rstrip())
        except Exception as e:
            self._append_terminal(f"[stream error] {e}")
        finally:
            proc.stdout.close()
            ret = proc.wait()
            self._append_terminal(f"[exit code] {ret}")

    # ---------- Command handler ----------
    def _handle_command(self, raw: str) -> bool:
        parts = raw.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd == "/help":
            help_msg = (
                "**Rona Commands**\n"
                "- `/help` — show this help\n"
                "- `/upload` — (disabled)\n"
                "- `/deep <query>` — unified deep search (local DB  web  conversation)\n"
                "- `/imggen` — (disabled)\n"
                "- `/imginfo` — (disabled)\n"
                "- `/bash <command>` — ask first, then run a shell command and stream output in yellow\n"
                "- `/intrinsic <query>` — get direct Mistral:7b answer (no RAG, no context)\n"
                "\n"
                "**Bug Bounty Commands**\n"
                "- `/hunt <target>` — start comprehensive bug bounty reconnaissance\n"
                "- `/lovely <path|query>` — upload path to lovely_assest or search lovely (friendly)\n"
                "- `/memory` — show bug bounty memory summary\n"
                "- `/recommend <target>` — get scan recommendations\n"
                "- `/poetic <topic>` — get a poetic response on the given topic\n"
            )
            self._reply_assistant(help_msg)
            return True
        if cmd == "/upload":
            self._append_conversation("system", "Use the 🏮 button to upload via UI.")
            return True
        if cmd == "/deep":
            if not arg:
                self._append_conversation("system", "Usage: /deep your query")
                return True
            threading.Thread(
                target=self._deep_search_and_reply, args=(arg,), daemon=True
            ).start()
            return True
        if cmd == "/imggen":
            self._append_conversation("system", "Image generation is disabled.")
            return True
        if cmd == "/imginfo":
            self._append_conversation("system", "Image analysis is disabled.")
            return True

        if cmd == "/poetic":
            if not arg:
                self._append_conversation(
                    "system", "Usage: /poetic <your topic or question>"
                )
                return True

            def _run_poetic():
                try:
                    if not self.llm:
                        self._reply_assistant(
                            "The muses are quiet right now (LLM not available)."
                        )
                        return

                    prompt = f"{POETIC_MODE_PROMPT}\n\nUser's request: {arg}"
                    resp = self.llm.invoke(prompt)
                    poetic_response = getattr(resp, "content", "") or str(resp)
                    self._reply_assistant(poetic_response)
                except Exception as e:
                    self._reply_assistant(f"A creative spark fizzled out: {e}")

            threading.Thread(target=_run_poetic, daemon=True).start()
            return True
        # ---------- NEW: /bash command ----------
        if cmd == "/bash" or cmd == "/run":
            if not arg:
                self._append_conversation(
                    "system",
                    "Usage: /bash <command>  (example: /bash nmap -sV 127.0.0.1)",
                )
                return True
            # Ask confirmation and stream output to the chat in yellow
            self.confirm_and_run_shell(arg)
            return True
        if cmd == "/intrinsic":
            if not arg:
                self._append_conversation("system", "Usage: /intrinsic <query>")
                return True
            if not self.llm:
                self._append_conversation("system", "LLM not available.")
                return True
            try:
                # Use the new intrinsic-only path
                import asyncio

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(
                    self._call_llm_with_context(
                        arg, self.conversation_history, [], intrinsic_only=True
                    )
                )
                self._reply_assistant(result)
            except Exception as e:
                self._append_conversation("system", f"Intrinsic mode error: {e}")
            return True

        # ---------- Bug Bounty Commands ----------
        if cmd == "/hunt":
            if not arg:
                self._append_conversation("system", "Usage: /hunt <target_domain>")
                return True
            if not BUG_BOUNTY_OK:
                self._append_conversation(
                    "system", "Bug bounty integration not available"
                )
                return True
            threading.Thread(
                target=self._run_hunt_command, args=(arg,), daemon=True
            ).start()
            return True

        if cmd == "/lovely":
            if not arg:
                self._append_conversation("system", "Usage: /lovely <path|query>")
                return True
            path_candidate = Path(arg).expanduser()
            if path_candidate.exists():
                try:
                    dst = config.uploads_dir / path_candidate.name
                    shutil.copy2(path_candidate, dst)
                    docs = file_processor.process_file(str(dst))
                    combined = "\n\n".join(
                        [getattr(d, "page_content", "") for d in (docs or [])]
                    ).strip()
                    if not combined:
                        try:
                            dst.unlink(missing_ok=True)
                        except Exception:
                            pass
                        self._reply_assistant(
                            "Aww, sweetie — that file seems empty. I removed it 💖"
                        )
                        return True
                    sq = SQLiteManagerSingleton.get()
                    meta = {"source": str(dst)}
                    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    saved = (
                        sq.insert_document(
                            "lovely_assest",
                            dst.name,
                            str(dst),
                            combined,
                            meta,
                            created_at,
                        )
                        if sq
                        else False
                    )
                    ok = False
                    if docs and LC_OK and LCDocument:
                        db = DatabaseManagerSingleton.get()
                        if db:
                            db.add_documents(docs)
                            ok = True
                    self._reply_assistant(
                        f"🌸 Saved lovingly to lovely_assest: {dst.name} (vector={bool(ok)}, sqlite={bool(saved)})"
                    )
                except Exception as e:
                    self._append_conversation("system", f"Lovely upload error: {e}")
                return True

            # Friendly, model-only summary for casual notes using Llama3
            # Friendly, model-only summary for casual notes using Llama3 + psycho context
            def _run():
                ctx_chunks = psycho_store.export_text_chunks(max_items=30)
                ctx_text = "\n\n".join(ctx_chunks) if ctx_chunks else ""
                try:
                    self._ensure_ollama_model("llama3:8b")
                    llama = (
                        ChatOllama(model="llama3:8b", temperature=0.2)
                        if LC_OK
                        else None
                    )
                    if not llama:
                        self._reply_assistant(
                            "🔵 NOTICE: Ollama not available. Install and run `ollama serve`."
                        )
                        return
                    prompt = (
                        "SYSTEM: Lovely Mode with psychoanalytical awareness.\n"
                        "Use the user's psychoanalytical notes to tailor a warm, concise response.\n"
                        "Respond in 3-5 bullets max and one 'Next tiny action' line.\n\n"
                        f"Psycho Notes (most recent first):\n{ctx_text}\n\n"
                        f"User Query: {arg}\n"
                    )
                    resp = llama.invoke(prompt)
                    out = getattr(resp, "content", "") or str(resp)
                    self._reply_assistant(
                        "**Lovely Notes + Psychoanalysis**\n" + out.strip()
                    )
                except Exception as e:
                    self._reply_assistant(f"Lovely error: {e}")

            threading.Thread(target=_run, daemon=True).start()
            return True

        if cmd == "/memory":
            if not BUG_BOUNTY_OK:
                self._append_conversation(
                    "system", "Bug bounty integration not available"
                )
                return True
            threading.Thread(target=self._show_memory_summary, daemon=True).start()
            return True

        if cmd == "/recommend":
            if not arg:
                self._append_conversation("system", "Usage: /recommend <target_domain>")
                return True
            if not BUG_BOUNTY_OK:
                self._append_conversation(
                    "system", "Bug bounty integration not available"
                )
                return True
            threading.Thread(
                target=self._get_scan_recommendations, args=(arg,), daemon=True
            ).start()
            return True

        self._append_conversation("system", f"Unknown command: {cmd}. Try /help")
        return True
    # ---------- Agent processing ----------
    def _process_with_agent(self, message: str):
        self.update_status("🤔 Processing...")
        try:
            processed_message = process_message(message)
            intent = processed_message["intent"]
            # FIX: allow security_testing, bug_bounty, and general_query (or None)
            allowed_topics = {"security_testing", "bug_bounty", "general_query"}
            if intent in allowed_topics or intent is None:
                response = self.agent.invoke({"input": message})
                raw = (
                    (response.get("output") or "")
                    if isinstance(response, dict)
                    else str(response)
                )
                formatted = formatter.format(raw)
                self._reply_assistant(formatted or "No response.")
                self.update_status("✅ Ready")
            else:
                self._reply_assistant("I'm not sure I understand your request.")
                self.update_status("✅ Ready")
        except Exception as e:
            self._append_conversation("system", f"Agent error: {e}")
            self.update_status("❌ Agent error, using fallback")
            try:
                asyncio.run(self._fallback_unified_search_and_reply(message))
            except RuntimeError:
                # If already in an event loop (e.g., in some environments), fallback to original method
                self._fallback_unified_search_and_reply(message)

    # ---------- Fallback unified search & summarization ----------
    def _fallback_unified_search_and_reply(self, message: str):
        self.update_status("🪷 Searching (unified)...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Normalize the query by stripping and lowercasing
        def normalize_query(q: str) -> str:
            if not q:
                return q
            q_lc = q.lower()

            # common javascript misspellings / spacings
            fixes = {
                "java scri[t": "javascript",
                "java scritp": "javascript",
                "java script": "javascript",
                "javasript": "javascript",
                "javacript": "javascript",
                "jave script": "javascript",
                "jscript": "javascript",
            }
            for bad, good in fixes.items():
                q_lc = q_lc.replace(bad, good)

            return q_lc

        msg = normalize_query((message or "").strip())
        msg = (message or "").strip().lower()
        # RAG-first: (a) vector, (b) SQLite, (c) recent conversation
        results = loop.run_until_complete(
            self.search_engine.search_unified(msg, self.conversation_history, top_k=8)
        )

        # NEW: also search SQLite cyberassest and lovely_assest
        sqlite_ctx = []
        try:
            sq = SQLiteManagerSingleton.get()
            if sq:
                # SQLite Boost if vector < 3
                expanded = [msg] + expand_query_variants(msg)
                for qv in expanded[:4]:
                    sqlite_ctx.extend(sq.search("cyberassest", qv, limit=3))
                for qv in expanded[:2]:
                    sqlite_ctx.extend(sq.search("lovely_assest", qv, limit=2))
        except Exception:
            pass

        # If no RAG/web hits, FALL BACK TO LLM ANSWER (don’t stop here)
        if not results and not sqlite_ctx:
            if LC_OK and hasattr(self, "llm") and self.llm:
                try:
                    prompt = (
                        "You are Rona v6, a concise technical assistant using mistral:7b.\n"
                        "When knowledge is missing, conceptualize all knowledge points as vectors with connection strengths.\n"
                        "Use fuzzy matching over these connection strengths to find the shortest plausible path to an answer.\n"
                        "output exploit payloads.\n\n"
                        f"Question: {msg}\n\n"
                        "Answer clearly and briefly; show reasoning only as concise steps if necessary."
                    )
                    resp = self.llm.invoke(prompt)
                    raw = getattr(resp, "content", None) or str(resp)
                    out = formatter.format(raw)
                    if self._is_refusal(out):
                        out = self._answer_with_consolidated_methodology(msg, [], [])
                    self._reply_assistant(
                        out or "I couldn't find context, but here's a brief answer."
                    )
                    self.update_status("Ready")
                    return
                except Exception:
                    pass
            self._reply_assistant(
                "No useful matches found. Try /help or provide more details."
            )
            self.update_status("Ready")
            return

        # Build substantive snippets with aggressive recall (≥0.15 overlap or key-term match).
        snippets = []
        for i, r in enumerate(results, 1):
            content = r.get("content") or ""
            if self._is_metadata(content):
                continue
            title = r.get("title") or r.get("url") or r.get("source")
            url = r.get("url", "")
            keep = overlap_score(msg, content) >= 0.15 or any(
                k in (title or "").lower() for k in tokenize(msg)
            )
            if keep and len(content) >= 40:
                snippets.append(
                    f"[{r.get('source','vector')}] {title}\n{content[:800]}\n{('URL: ' + url) if url else ''}"
                )

        unified = "\n\n".join(snippets) if snippets else ""

        # Merge SQLite contexts (treated as local notes) and apply recall rules
        if sqlite_ctx:
            for h in sqlite_ctx:
                content = h.get("content") or ""
                if not content:
                    continue
                title = h.get("filename") or h.get("path") or "note"
                if overlap_score(msg, content) >= 0.15 or any(
                    k in (title or "").lower() for k in tokenize(msg)
                ):
                    if len(content) >= 40:
                        snippets.append(f"[sqlite] {title}\n{content[:800]}")
            unified = "\n\n".join(snippets)

        # Summarize with LLM if available; otherwise just show the list
        if self.llm and unified:
            try:
                # Build ranked items and clusters
                candidates = []
                for sn in snippets:
                    candidates.append(
                        {
                            "text": sn,
                            "score": overlap_score(msg, sn),
                            "source": (
                                "vector"
                                if "[" in sn
                                and "]" in sn
                                and "vector" in sn[:20].lower()
                                else "sqlite" if "[sqlite]" in sn else "web"
                            ),
                        }
                    )
                ranked = rank_with_local_priority(candidates)
                clusters = cluster_snippets([c["text"] for c in ranked])
                sources = [c.get("text", "")[:80] for c in ranked]

                # Result Quality Gate - Allow web results when local is sparse
                local_chunks = sum(
                    1 for c in ranked if c.get("source") in ("vector", "sqlite")
                )
                web_chunks = sum(1 for c in ranked if c.get("source") == "web")
                if local_chunks < 3 and web_chunks < 2:
                    # Request narrower query or more data only if both local and web are insufficient
                    self._reply_assistant(
                        "I need more context. Please narrow the query or upload a relevant file (🏮)."
                    )
                    self.update_status("Ready")
                    return

                # Compose final answer deterministically
                reasoning = build_connected_reasoning(msg, ranked)
                synthesis = synthesize_answer_from_clusters(msg, clusters, sources)
                out = (
                    "Summary: Local-first consolidated answer.\n"
                    + reasoning
                    + "\n\n"
                    + synthesis
                )
                raw = getattr(resp, "content", None) or str(resp)
                final_out = formatter.format(out)
                if self._is_refusal(final_out):
                    final_out = self._answer_with_consolidated_methodology(
                        msg, snippets, sqlite_ctx
                    )
                self._reply_assistant(final_out)
                self.update_status("Ready")
                return
            except Exception:
                pass

        # Fall back to methodology-based answer using consolidated vectors
        if snippets or sqlite_ctx:
            # Build ranked items and clusters even in fallback
            candidates = []
            for sn in snippets:
                candidates.append(
                    {
                        "text": sn,
                        "score": overlap_score(msg, sn),
                        "source": (
                            "vector"
                            if "[" in sn and "]" in sn and "vector" in sn[:20].lower()
                            else "sqlite" if "[sqlite]" in sn else "web"
                        ),
                    }
                )
            for h in sqlite_ctx:
                content = h.get("content") or ""
                if content:
                    candidates.append(
                        {
                            "text": content[:800],
                            "score": overlap_score(msg, content),
                            "source": "sqlite",
                        }
                    )
            if candidates:
                ranked = rank_with_local_priority(candidates)
                clusters = cluster_snippets([c["text"] for c in ranked])
                reasoning = build_connected_reasoning(msg, ranked)
                synthesis = synthesize_answer_from_clusters(
                    msg, clusters, [c.get("text", "")[:80] for c in ranked]
                )
                consolidated = (
                    "Summary: Fallback consolidated answer.\n"
                    + reasoning
                    + "\n\n"
                    + synthesis
                )
                self._reply_assistant(consolidated)
                self.update_status("Ready")
                return

        # Last resort: use LLM with connected vector reasoning prompt
        if self.llm:
            try:
                prompt = (
                    "You are Rona v6 using connected vector reasoning.\n"
                    "Treat knowledge as connected nodes with semantic similarity scores.\n"
                    "Find the shortest, highest-strength path to answer the query.\n"
                    "Output format:\n"
                    "Summary: [3 lines max]\n"
                    "Vector Reasoning Path: [list top nodes with connection strengths]\n"
                    "Consolidated Answer: [deterministic conclusion]\n"
                    "Sources: [referenced knowledge]\n\n"
                    f"Query: {msg}\n\n"
                    "Provide a comprehensive answer using vector reasoning methodology."
                )
                resp = self.llm.invoke(prompt)
                raw = getattr(resp, "content", None) or str(resp)
                out = formatter.format(raw)
                self._reply_assistant(out)
                self.update_status("Ready")
                return
            except Exception:
                pass

        # Absolute fallback
        self._reply_assistant(
            "Unable to process query. Please try uploading relevant files via 🏮 or rephrasing your question."
        )
        self.update_status("Ready")

    def _is_refusal(self, text: str) -> bool:
        t = (text or "").lower()
        patterns = [
            "sorry, i can't",
            "i can't assist",
            "i cannot help",
            "i can't help",
            "as an ai",
            "i won't provide",
            "i will not provide",
        ]
        return any(p in t for p in patterns)

    def _answer_with_consolidated_methodology(
        self, query: str, rag_snippets: List[str], sqlite_hits: List[Dict[str, Any]]
    ) -> str:
        # Build a connected-vectors style summary from available context
        notes = []
        for s in rag_snippets or []:
            if isinstance(s, str):
                notes.append(s)
        for h in sqlite_hits or []:
            c = (h.get("content") or "").strip()
            if c:
                notes.append(c[:1200])

        # Simple token-based relevance scoring
        scored = []
        for n in notes:
            scored.append((overlap_score(query, n), n))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [n for _, n in scored[:5]]

        is_security = any(
            k in (query or "").lower()
            for k in [
                "xss",
                "csrf",
                "sqli",
                "sql injection",
                "rce",
                "ssti",
                "payload",
                "exploit",
            ]
        )

        if is_security:
            # Professional approach format, without exploit payloads
            return (
                "Summary: Provide a principled security methodology based on connected knowledge vectors.\n"
                "Steps:\n"
                "1) Scope and classify the context (asset type, trust boundary, input surface).\n"
                "2) Enumerate relevant patterns from prior notes (cyberassest) and nearest matches.\n"
                "3) Apply testing methodology (instrumentation, observation, differential checks).\n"
                "4) Validate findings with safe proofs (no live payloads).\n"
                "5) Record artifacts and remediation guidance.\n\n"
                f"Nearest context:\n- " + "\n- ".join([t[:400] for t in top])
            )

        # General query: concise answer synthesized from nearest context
        if top:
            synthesized = top[0]
            return f"Answer (synthesized from nearest vectors):\n{synthesized[:800]}"
        return "Answer: Using fuzzy vector consolidation, no direct matches found; here is the nearest conceptual explanation based on prior knowledge."

    # ---------- Deep search UI ----------
    def deep_search_dialog(self):
        try:
            dialog = ctk.CTkInputDialog(
                text="Enter search query:", title="Deep Unified Search"
            )
            query = dialog.get_input()
        except Exception:
            import tkinter.simpledialog as sd

            query = sd.askstring("Deep Unified Search", "Enter search query:")
        if not query:
            return
        threading.Thread(
            target=self._deep_search_and_reply, args=(query,), daemon=True
        ).start()

    def _deep_search_and_reply(self, query: str):
        self.update_status("🔎 Performing deep unified search...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        results = loop.run_until_complete(
            self.search_engine.search_unified(query, self.conversation_history, top_k=8)
        )
        if not results:
            self._reply_assistant("No matches found.")
            self.update_status("Ready")
            return
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = (r.get("content", "") or "")[:400]
            url = r.get("url", "")
            lines.append(
                f"{i}. [{r.get('source')}] {title}\n{content}\n{('URL: ' + url) if url else ''}"
            )
        # FIX: explicit concatenation
        self._reply_assistant("\n\n".join(lines))
        self.update_status("Ready")

    # ---------- File & Image handlers ----------
    def open_file_dialog(self):
        filetypes = [
            ("Documents", "*.txt;*.md;*.log;*.pdf;*.xml;*.json;*.csv;*.html;*.htm"),
            ("All", "*.*"),
        ]
        p = filedialog.askopenfilename(filetypes=filetypes)
        if not p:
            return
        try:
            src = Path(p)
            dst = config.uploads_dir / src.name
            shutil.copy2(src, dst)
            docs = file_processor.process_file(str(dst))
            # Support when LCDocument is not available: docs may be strings
            contents = []
            for d in docs or []:
                if isinstance(d, str):
                    contents.append(d)
                else:
                    contents.append(getattr(d, "page_content", ""))
            combined = "\n\n".join(contents).strip()
            if not combined:
                try:
                    dst.unlink(missing_ok=True)
                except Exception:
                    pass
                self._reply_assistant("The file had no useful content and was removed.")
                return
            # Ask destination (lovely vs cyber), default cyberassest
            table = "cyberassest"
            try:
                use_lovely = messagebox.askyesno(
                    "Store Destination",
                    "Save this to lovely_assest? (No = cyberassest)",
                    parent=self,
                )
            except Exception:
                use_lovely = False
            if use_lovely:
                table = "lovely_assest"
            sq = SQLiteManagerSingleton.get()
            created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            meta = {"source": str(dst)}
            sqlite_ok = (
                sq.insert_document(
                    table, dst.name, str(dst), combined, meta, created_at
                )
                if sq
                else False
            )
            ok = False
            if docs and LC_OK and LCDocument:
                db = DatabaseManagerSingleton.get()
                if db:
                    db.add_documents(docs)
                    ok = True
            self._reply_assistant(
                f"Indexed: {dst.name} (vector={bool(ok)}) — saved to {table} (sqlite={bool(sqlite_ok)})"
            )
        except Exception as e:
            self._append_conversation("system", f"Upload error: {e}")

    def open_image_dialog(self):
        self._append_conversation("system", "Image analysis is disabled.")

    def create_image_dialog(self):
        self._append_conversation("system", "Image generation is disabled.")

    # ---------- Tests / Settings / Clear ----------
    def run_tests(self):
        self.update_status("🧪 Running tests...")
        results = self.test_suite.run_all_tests()
        self._append_conversation("system", "🧪 Test Results:")
        for name, r in results.items():
            s = r.get("status", "unknown")
            icon = "✅" if s == "passed" else "❌"
            self._append_conversation("system", f"{icon} {name}: {s}")
        self.update_status("Ready")

    def open_settings(self):
        messagebox.showinfo(
            "Settings", "Settings dialog will be implemented in a future version."
        )

    def clear_chat(self):
        self.chat_history.delete("1.0", "end")
        clear_messages = [
            "Chat cleared. A fresh start!",
            "All clear! What's on your mind?",
            "Cleared. The canvas is yours.",
            "The slate is clean.",
        ]
        self._append_conversation("system", random.choice(clear_messages))

    """
    def clear_chat(self):
        self.chat_history.delete("1.0", "end")
        self._append_conversation("system", "Chat cleared")
    """

    # ---------- Bug Bounty Command Handlers ----------
    def _run_hunt_command(self, target: str):
        """Run bug bounty hunt command"""
        self.update_status("🎯 Starting bug bounty hunt...")
        try:
            result = bug_bounty_integration.run_hunt_command(target)
            if result.get("success"):
                self._append_conversation("system", f"✅ Hunt completed for {target}")
                self._append_terminal(result.get("stdout", ""))
            else:
                self._append_conversation(
                    "system", f"❌ Hunt failed: {result.get('error', 'Unknown error')}"
                )
                if result.get("stderr"):
                    self._append_terminal(result.get("stderr"))
        except Exception as e:
            self._append_conversation("system", f"❌ Hunt error: {e}")
        finally:
            self.update_status("✅ Ready")

    def _run_lovely_mode(self, folder_path: str):
        """Run lovely mode for folder scanning"""
        self.update_status("💖 Running lovely mode...")
        try:
            result = bug_bounty_integration.run_lovely_mode(folder_path)
            if result.get("success"):
                self._append_conversation(
                    "system", f"✅ Lovely mode completed for {folder_path}"
                )
                self._append_terminal(result.get("stdout", ""))
            else:
                self._append_conversation(
                    "system",
                    f"❌ Lovely mode failed: {result.get('error', 'Unknown error')}",
                )
                if result.get("stderr"):
                    self._append_terminal(result.get("stderr"))
        except Exception as e:
            self._append_conversation("system", f"❌ Lovely mode error: {e}")
        finally:
            self.update_status("✅ Ready")

    def _show_memory_summary(self):
        """Show bug bounty memory summary"""
        self.update_status("🧠 Loading memory summary...")
        try:
            summary = bug_bounty_integration.get_memory_summary()
            summary_text = f"""
            
**Bug Bounty Memory Summary**

- **Total Vectors:** {summary.get('total_vectors', 0)}
- **Total Scans:** {summary.get('total_scans', 0)}
- **Learning Patterns:** {summary.get('learning_patterns', 0)}
- **Last Updated:** {summary.get('last_updated', 'Never')}

**Most Effective Tools:**
"""
            for tool in summary.get("most_effective_tools", [])[:5]:
                summary_text += f"- {tool['tool']}: {tool['average_score']:.2f}\n"

            if summary.get("recommendations", {}).get("recommendations"):
                summary_text += "\n**Recommendations:**\n"
                for rec in summary["recommendations"]["recommendations"]:
                    summary_text += f"- {rec}\n"

            self._reply_assistant(summary_text)
        except Exception as e:
            self._append_conversation("system", f"❌ Memory error: {e}")
        finally:
            self.update_status("✅ Ready")

    def _get_scan_recommendations(self, target: str):
        """Get scan recommendations for target"""
        self.update_status("💡 Getting scan recommendations...")
        try:
            recommendations = bug_bounty_integration.get_scan_recommendations(target)
            if recommendations:
                rec_text = f"**Scan Recommendations for {target}:**\n\n"
                for i, rec in enumerate(recommendations, 1):
                    rec_text += f"{i}. {rec}\n"
                self._reply_assistant(rec_text)
            else:
                self._reply_assistant(
                    f"No specific recommendations available for {target}"
                )
        except Exception as e:
            self._append_conversation("system", f"❌ Recommendations error: {e}")
        finally:
            self.update_status("✅ Ready")


# ---------- Main ----------
def main():
    print("🚀 Starting Rona v6 Enhanced...")
    load_balancer.monitor_temperatures()
    mode = load_balancer.adjust_load_balancing()
    print(f"⚖️ Load balancing mode: {mode}")
    app = RonaAppEnhanced()
    app.mainloop()


# start_psycho_server();
if __name__ == "__main__":
    main()
