# -*- coding: utf-8 -*-
"""`
Rona v6 Enhanced - Final merged edition
- Warm humanlike greeting
- Opt-in local advanced_mode for detailed defensive testing and attack guidance ( exploit payloads)
- Unified deep search (local vector DB  DuckDuckGo  optional Google CSE  conversation)
- Image generation (non-blank), image upload, analysis, registry  indexing
- Slash commands: /help, /upload, /deep, /imggen, /imginfo
- Arabic shaping (if libs installed)
- Load balancer, file processing, Chroma RAG (optional)
"""
from flask import Flask, jsonify, request, send_from_directory

import re, urllib.parse
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
import time
from langchain_ollama import OllamaEmbeddings
import urllib.parse
import re, datetime
from typing import List, Dict, Any
import tldextract
import logging
import inspect

import requests
from functools import partial

# --- Ollama chat model import guard ---
try:
    from langchain_ollama import ChatOllama

    OLLAMA_CHAT_OK = True
except Exception as e:
    print("⚠️ ChatOllama import error:", e)
    ChatOllama = None
    OLLAMA_CHAT_OK = False


def check_local_server(text, lang="en-US", host="http://127.0.0.1:8010"):
    resp = requests.post(
        f"{host}/v2/check", data={"text": text, "language": lang}, timeout=10
    )
    resp.raise_for_status()
    return resp.json()


DEBUG_SILENT_SELFTEST = False  # Set True to test DB manually

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# ---

DEFAULT_EMBED_CANDIDATES = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
]
BLOCKED_KEYWORDS = {"exploit"}

# Add near other imports
try:
    from duckduckgo_search import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    DDGS = None  # Placeholder


try:
    import language_tool_python

    GRAMMAR_TOOL_OK = True
except ImportError:
    GRAMMAR_TOOL_OK = False
    language_tool_python = None


try:
    from flask import Flask, request, jsonify, send_from_directory

    FLASK_OK = True
except Exception:
    FLASK_OK = False

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
import socket, threading
from werkzeug.serving import make_server
from flask import Flask, jsonify, request, send_from_directory

# --- LangChain / Chroma availability check ---
try:
    from langchain_chroma import Chroma
    from langchain_community.embeddings import OllamaEmbeddings

    LC_OK = True
except Exception as e:
    print("⚠️ LangChain/Chroma import error:", e)
    LC_OK = False
    ChatOllama = None


DEBUG_MODE = True  # toggle this easily

import logging

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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
# Optional NLTK and spaCy imports
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import spacy

    nlp = spacy.load("en_core_web_sm")
    NLP_OK = True
except (ImportError, OSError):
    nltk = None
    word_tokenize = None
    stopwords = None
    spacy = None
    nlp = None
    NLP_OK = False

# Optional LangChain / Ollama / Chroma stack (used if installed)
# ---- Critical imports needed ONLY for vector DB ----
LC_VECTOR_OK = False
LC_VECTOR_ERR = None
try:
    from langchain_ollama import OllamaEmbeddings
    import chromadb  # ensures backend lib present

    LC_VECTOR_OK = True
except Exception as e:
    LC_VECTOR_ERR = e
    print("❌ Critical import error (Chroma/Ollama):", repr(e))
    Chroma = None
    OllamaEmbeddings = None

# (Optional) everything else – DO NOT affect LC_VECTOR_OK
try:
    from langchain_community.document_loaders import (
        TextLoader,
        PyPDFLoader,
        UnstructuredXMLLoader,
    )
except Exception as e:
    print("⚠️ Loader import error:", repr(e))
    TextLoader = PyPDFLoader = UnstructuredXMLLoader = None

try:
    from langchain_community.memory import ConversationBufferWindowMemory
    from langchain_community.chat_message_histories import ChatMessageHistory
except Exception as e:
    print("⚠️ Memory import error:", repr(e))
    ConversationBufferWindowMemory = None
    ChatMessageHistory = None

try:
    from langchain.agents import AgentExecutor  # ok in 0.2

    try:
        from langchain.agents import create_tool_calling_agent  # may be missing
    except Exception:
        create_tool_calling_agent = None
        print(
            "⚠️ 'create_tool_calling_agent' not available in this LC build (OK to skip)."
        )
    from langchain.tools import tool
except Exception as e:
    print("⚠️ Agents/tools import error:", repr(e))
    AgentExecutor = None
    create_tool_calling_agent = None
    tool = None


# ---- LangChain / vendors (robust import block) ----
try:
    # Core LC
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.documents import Document as LCDocument

    # Vendors (split packages)
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_chroma import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Agents & tools
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain.tools import tool

    try:
        from langchain_community.chat_message_histories import ChatMessageHistory
        from langchain_community.memory import ConversationBufferWindowMemory
    except Exception as e:
        print("⚠️ LangChain memory import error:", e)
        ChatMessageHistory = None
        ConversationBufferWindowMemory = None

    LC_VECTOR_OK = True
except Exception as e:
    LC_ERR = e
    print("LC/Chroma import error:", repr(e))
    # Optional: define soft shims so the app can keep running without agents/vector DB
    ChatOllama = None
    OllamaEmbeddings = None
    Chroma = None
    ChatPromptTemplate = None
    MessagesPlaceholder = None
    LCDocument = None
    RecursiveCharacterTextSplitter = None
    create_tool_calling_agent = None
    AgentExecutor = None
    tool = None
    ConversationBufferWindowMemory = None


# Define a dummy class for LCDocument if it's not imported
class _DummyLCDocument:
    pass
# Define a dummy class for LCDocument if it's not imported
class _DummyLCDocument:
    pass


LCDocument = _DummyLCDocument()

# Bug Bounty Integration
try:
    from bug_bounty_integration import bug_bounty_integration

    BUG_BOUNTY_OK = True
except Exception:
    BUG_BOUNTY_OK = False
    bug_bounty_integration = None
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
        # Define default config structure clearly
        default = {
            "model_name": "mistral:7b",
            "gpu_settings": {
                "force_gpu": True,
                "gpu_layers": 25,
                "gpu_memory_utilization": 0.8,
                "temperature_threshold": 75,
                "enable_load_balancing": True,
                "proxy_keywords": [  # Default keywords
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
                    "news",
                    "update",
                    "current",
                    "details",
                    "information",
                    "compare",
                    "review",
                ],
                "proxy_blocklist": ["porn"],  # Example blocklist
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
            "zoomeye": {
                "api_key": "",
                "enabled": True,
                "cache_ttl_sec": 3600,
                "cooldown_sec": 3600,
            },  # Added zoomeye section
            "ollama_settings": {  # Added ollama_settings
                "host": "127.0.0.1:11434",
                "keep_alive": "5m",
                "num_parallel": 1,
                "timeout": 60,
                "num_gpu": 1,  # Example, adjust as needed
            },
            # Add other settings sections if they were present in user's config
            "vector_db_settings": {"similarity_metric": "cosine", "max_results": 5},
            "file_processing": {"max_file_size_mb": 100},
            "arabic_settings": {"enable_reshaping": True, "enable_bidi": True},
            "search_settings": {"max_search_results": 7, "search_timeout": 15},
            "image_settings": {"max_image_size_mb": 20},
        }
        if cfg_file.exists():
            try:
                loaded = json.loads(cfg_file.read_text(encoding="utf-8"))

                # Deep update dictionary to preserve nested defaults
                def deep_update(d, u):
                    for k, v in u.items():
                        if isinstance(v, dict):
                            d[k] = deep_update(d.get(k, {}), v)
                        else:
                            d[k] = v
                    return d

                default = deep_update(default, loaded)
            except json.JSONDecodeError as e_json:
                print(
                    f"Warning: config.json is invalid JSON, using defaults. Error: {e_json}"
                )
            except Exception as e_cfg:
                print(
                    f"Warning: Error loading config.json, using defaults. Error: {e_cfg}"
                )
        return default


config = Config()


from pathlib import Path

# Make sure psycho_store writes to your desired JSON file
config.psycho_file = Path("/home/gmm/Templates/Rona-Agent-/data/psychoanalytical.json")


class WebUIConfig:
    def __init__(self, base_dir: Path, preferred: int = 8765):
        self.ui_dir = base_dir / "productivity"
        self.port = self._choose_free_port(preferred)

    def _choose_free_port(self, preferred: int) -> int:
        """
        Try preferred first; if busy, pick a random free one.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", preferred))
            port = preferred
        except OSError:
            # fallback to a truly free ephemeral port
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        return port


webui = WebUIConfig(config.base_dir)
print(f"[webui] assigned port: {webui.port}")


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

import unicodedata

# keep existing imports; add these light ones if not present
import re, unicodedata
from typing import List

_AR_NUMS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def _basic_clean(t: str) -> str:
    # unicode normalize, normalize Arabic digits, collapse whitespace, strip bidi controls
    t = unicodedata.normalize("NFKC", t or "")
    t = t.translate(_AR_NUMS)
    t = re.sub(r"[\u202A-\u202E]", "", t)  # strip bidi control chars
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _strip_noise(t: str) -> str:
    # remove urls, emojis/punct clutter, keep Arabic/English letters+digits+% .
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[@#]+", " ", t)
    t = re.sub(r"[^\w\s\u0600-\u06FF.%]", " ", t)  # allow Arabic block
    t = re.sub(r"\s+", " ", t).strip()
    return t


# Focused bilingual time/ops synonyms frequently asked in Rona context
_BILINGUAL_SWAPS = [
    ("today", "اليوم"),
    ("yesterday", "أمس"),
    ("production", "الإنتاج"),
    ("inventory", "المخزون"),
    ("energy", "الطاقة"),
    ("consumption", "الاستهلاك"),
    ("temperature", "الحرارة"),
    ("report", "تقرير"),
]

# Security / bug-bounty synonyms (kept from your intent, expanded carefully)
_SEC_SWAPS = {
    "xss": ["cross site scripting", "reflected xss", "stored xss"],
    "sqli": ["sql injection", "database injection"],
    "payload": ["injection", "vector"],
    "recon": ["reconnaissance", "enumeration"],
}


def _apply_swaps(q: str, swaps_map: dict | None = None) -> set[str]:
    out = {q}
    if not swaps_map:
        return out
    low = q.lower()
    # replace whole words only to avoid over-expansion
    for key, vals in swaps_map.items():
        if re.search(rf"\b{re.escape(key)}\b", low):
            for v in vals:
                out.add(re.sub(rf"\b{re.escape(key)}\b", v, q, flags=re.IGNORECASE))
    return out


def _arabic_english_pairs(q: str) -> set[str]:
    out = {q}
    low = q.lower()
    for en, ar in _BILINGUAL_SWAPS:
        if en in low:
            out.add(low.replace(en, ar))
        if ar in q:
            out.add(q.replace(ar, en))
    return out


def expand_query_variants(query: str) -> List[str]:
    """
    Expand a user query into normalized, de-noised, bilingual variants with
    focused synonym expansion (security + factory ops). Returns up to 8 variants,
    original first, shorter/cleaner ones prioritized to improve top-1 retrieval.
    """
    q = _basic_clean(query)
    if not q:
        return []

    # 1) denoise
    q_clean = _strip_noise(q)

    # 2) start set with original + clean
    variants = {q, q_clean}

    # 3) controlled plural/singular toggles, but only for content words (>=3 chars)
    words = q_clean.split()
    for w in words:
        if len(w) < 3:
            continue
        if re.match(r"^[A-Za-z\u0600-\u06FF0-9%]+$", w) is None:
            continue
        if w.endswith("s") and len(w[:-1]) >= 3:
            variants.add(q_clean.replace(w, w[:-1]))
        elif not w.endswith("s") and re.match(r"^[A-Za-z]+$", w):
            variants.add(q_clean.replace(w, w + "s"))

    # 4) bilingual swaps (Arabic/English)
    for v in list(variants):
        variants |= _arabic_english_pairs(v)

    # 5) security synonyms – only apply if the trigger tokens are present
    for v in list(variants):
        variants |= _apply_swaps(v, _SEC_SWAPS)

    # 6) de-dupe, rank: shorter & cleaner first, then lexicographic for stability
    ranked = sorted({_basic_clean(x) for x in variants if x}, key=lambda s: (len(s), s))

    # 7) cap and ensure original first if present
    if q in ranked:
        ranked.remove(q)
        ranked.insert(0, q)

    return ranked[:8]


from typing import List, Dict, Any


def rank_with_local_priority(
    candidates: List[Dict[str, Any]], k: int = 5
) -> List[Dict[str, Any]]:
    """
    Hybrid fusion ranking:
      • Uses Reciprocal Rank Fusion (RRF) to merge candidates from multiple sources
      • Keeps your local-first bias as additive bonus (vector/sqlite/conversation > web)
      • Returns top-k sorted descending by 'score_final'
    """
    if not candidates:
        return []

    # --- group by source type ---
    buckets = {"vector": [], "sqlite": [], "conversation": [], "web": [], "other": []}
    for c in candidates:
        src = (c.get("source") or "").lower()
        placed = False
        for key in buckets.keys():
            if key in src:
                buckets[key].append(c)
                placed = True
                break
        if not placed:
            buckets["other"].append(c)

    # --- sort each group by its raw score descending ---
    for arr in buckets.values():
        arr.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)

    # --- compute Reciprocal Rank Fusion (RRF) scores ---
    K = 50.0  # typical constant for RRF
    rrf_scores = {}
    for src, arr in buckets.items():
        for rank, item in enumerate(arr, start=1):
            rrf_scores[id(item)] = rrf_scores.get(id(item), 0.0) + 1.0 / (K + rank)

    # --- apply local-first bias as soft bonus ---
    def local_bonus(src: str) -> float:
        src = src.lower()
        if "vector" in src:
            return 0.30
        if "sqlite" in src:
            return 0.20
        if "conversation" in src:
            return 0.10
        if "web" in src:
            return 0.0
        return 0.05  # small default for unknown

    for c in candidates:
        base = float(c.get("score", 0.0))
        src = c.get("source", "")
        rrf = rrf_scores.get(id(c), 0.0)
        bonus = local_bonus(src)
        c["score_final"] = base + rrf + bonus

    # --- normalize to [0,1] for confidence gating downstream ---
    min_s = min(c["score_final"] for c in candidates)
    max_s = max(c["score_final"] for c in candidates)
    rng = max(1e-9, max_s - min_s)
    for c in candidates:
        c["score_final_norm"] = (c["score_final"] - min_s) / rng

    # --- return top-k sorted ---
    ranked = sorted(candidates, key=lambda x: x["score_final_norm"], reverse=True)
    return ranked[:k]


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


from typing import List, Dict, Any


# ---------- NEW: score-aware payload builder ----------
def build_connected_reasoning_payload(
    query: str, ranked: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert ranked evidence (from vector/sqlite/convo/web) into a uniform, score-aware list.
    Expected keys per candidate (best effort): 'content' or 'text', 'title', 'source', 'url',
    and either 'score_final_norm' (preferred) or 'score_final' or 'score'.
    Returns items sorted by descending 'score' (normalized to [0,1] when possible).
    """
    items: List[Dict[str, Any]] = []
    for c in ranked or []:
        text = c.get("content") or c.get("text") or ""
        title = c.get("title") or c.get("id") or c.get("source") or "node"
        source = c.get("source") or ""
        url = c.get("url") or (c.get("metadata", {}) or {}).get("source")
        # prefer normalized score; fallback gracefully
        sc = c.get("score_final_norm", c.get("score_final", c.get("score", 0.0)))
        try:
            sc = float(sc)
        except Exception:
            sc = 0.0
        items.append(
            {
                "text": text,
                "title": title,
                "source": source,
                "url": url,
                "score": max(0.0, min(1.0, sc)),
            }
        )
    # order by score desc and keep top few
    items.sort(key=lambda x: x["score"], reverse=True)
    return items[:8]


# ---------- NEW: score-aware synthesis payload ----------
def synthesize_answer_payload(
    query: str, context_items: List[Dict[str, Any]], sources: List[str]
) -> Dict[str, Any]:
    """
    Pick a top-1 'best' evidence using score, backfilled by length if scores tie.
    Also return up to 3 alternatives for optional UI expansion.
    """
    if not context_items:
        return {
            "query": query,
            "best": None,
            "alternatives": [],
            "sources": sources or [],
        }

    # choose best by score then by text length
    ranked = sorted(
        context_items,
        key=lambda x: (x.get("score", 0.0), len(x.get("text", ""))),
        reverse=True,
    )
    best = ranked[0]
    alts = ranked[1:4]

    return {
        "query": query,
        "best": best,
        "alternatives": alts,
        "sources": sources or [],
    }


# ---------- UPDATED: your original string formatter (now score-aware) ----------
def build_connected_reasoning(query: str, ranked: List[Dict[str, Any]]) -> str:
    """
    Text view of the reasoning path, now using normalized scores when available.
    """
    items = build_connected_reasoning_payload(query, ranked)
    lines = ["Vector Reasoning Path →"]
    for it in items[:5]:
        title = it.get("title") or "node"
        sc = round(float(it.get("score", 0.0)), 3)
        src = it.get("source") or ""
        lines.append(f"- {title} [{src}] (strength={sc})")
    return "\n".join(lines)


def synthesize_answer_from_clusters(
    query: str, clusters: List[List[str]], sources: List[str]
) -> str:
    """
    Back-compatible textual synthesis:
    - If clusters contain dict-like items with 'text'/'score', use them.
    - Otherwise behave like your old logic (longest snippet per cluster).
    """
    parts: List[str] = []
    parts.append("Summary: Consolidated local-first answer.")
    parts.append("Steps:")

    def _best_in_cluster(cl) -> str:
        # Support both list[str] and list[dict]
        try:
            # dict path
            best = sorted(
                cl,
                key=lambda x: (
                    float(x.get("score", 0.0)),
                    len(x.get("text", "")),
                ),
                reverse=True,
            )[0]
            return (best.get("text") or "")[:600]
        except Exception:
            # string path
            cl = [str(s) for s in cl]
            return sorted(cl, key=lambda s: len(s), reverse=True)[0][:600]

    # render up to 5 clusters
    for i, cl in enumerate(clusters[:5]):
        rep = _best_in_cluster(cl)
        parts.append(f"{i+1}) From Cluster {chr(65+i)}: {rep}")

    parts.append("Answer:")
    if clusters:
        # determine "best" across clusters with score if present
        try:
            # flatten and pick by score then length
            flat = []
            for cl in clusters:
                for x in cl:
                    if isinstance(x, dict):
                        flat.append(x)
                    else:
                        flat.append({"text": str(x), "score": 0.0})
            best = sorted(
                flat,
                key=lambda x: (float(x.get("score", 0.0)), len(x.get("text", ""))),
                reverse=True,
            )[0]
            parts.append((best.get("text") or "")[:900])
        except Exception:
            parts.append(str(clusters[0][0])[:900])
    else:
        parts.append("No direct local snippet; using nearest conceptual guidance.")

    parts.append("Sources:")
    for s in (sources or [])[:10]:
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
# --- Document Loaders (modern LangChain v0.2+) ---
try:
    from langchain_unstructured import (
        UnstructuredTextLoader as TextLoader,
        UnstructuredPDFLoader as PyPDFLoader,
        UnstructuredXMLLoader,
    )
except Exception as e:
    print("⚠️ Unified loader import error:", e)
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

    def _split_docs(
        self, raw_text: str, source_meta: dict
    ) -> Union[List[Any], List[str]]:  # Changed _DocumentType to Any
        """Split into RAG-friendly chunks using config; works with/without LangChain."""
        raw_text = self._normalize_text(raw_text)
        if not raw_text:
            return []
        chunk_size = config.config["performance_settings"].get("chunk_size", 800)
        chunk_overlap = config.config["performance_settings"].get("chunk_overlap", 50)

        if LC_VECTOR_OK and LCDocument:
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
            chunk = raw_text[i : i + chunk_size]
            out.append(chunk)
            i += chunk_size - chunk_overlap
        return out

    def _to_documents(
        self, texts: List[str], meta: dict
    ) -> Union[List[Any], List[str]]:  # Changed _DocumentType to Any
        texts = [self._normalize_text(t) for t in texts if t and t.strip()]
        if LC_VECTOR_OK and LCDocument:
            return [LCDocument(page_content=t, metadata=meta) for t in texts]
        return texts

    # ---------- main dispatcher ----------
    def process_file(
        self, file_path: str
    ) -> Union[List[Any], List[str]]:  # Changed List["LCDocument"] to List[Any]
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


# ArabicProcessor (Keep as is)
# ... (Class definition remains unchanged) ...
class ArabicProcessor:
    def __init__(self):
        self.available = arabic_reshaper is not None and bidi_get_display is not None
        self._reshaper = (
            arabic_reshaper.ArabicReshaper({}) if self.available else None
        )  # Basic config

    def is_arabic(self, text: str) -> bool:
        if not text or not self.available:
            return False
        return any("\u0600" <= char <= "\u06ff" for char in text)  # Simplified check

    def process(self, text: str) -> str:
        if not self.is_arabic(text):
            return text  # Use is_arabic check
        try:
            reshaped = self._reshaper.reshape(text)
            visual = bidi_get_display(reshaped)
            # Add RLE/PDF only if not already present
            if not visual.startswith("\u202b") and not visual.endswith("\u202c"):
                return "\u202b" + visual + "\u202c"
            return visual
        except Exception as e:
            print(f"Arabic processing error: {e}")
            return text  # Log error


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
        img: Any,
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
        font: Any,
        max_width: int,
        draw: Any,
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
# add near the top of the file if not already present
from typing import Any, List, Tuple


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
# add near the top of the file if not already present
from typing import Any, List, Tuple


try:
    from langchain_community.vectorstores import FAISS
    from langchain_ollama import OllamaEmbeddings  # same embeddings

    FAISS_OK = True
except Exception as e:
    print("⚠️ FAISS import error:", repr(e))
    FAISS_OK = False

class DatabaseManager:
    def __init__(self):
        self.vector_db = None
        self.embeddings_model = None
        # This part of the init was duplicated and moved from the original location
        # to ensure it's only initialized once and correctly.
        if hasattr(self, "_initialize_database"):
            self._initialize_database()
        elif hasattr(self, "initialize_database"):
            self.initialize_database()
        else:
            print("WARNING: No database initializer found; vector_db=None")

    def _probe_vector_stack():
        print("[PROBE] starting…")
        try:
            from langchain_ollama import OllamaEmbeddings

            emb = OllamaEmbeddings(
                model="nomic-embed-text", base_url="http://127.0.0.1:11434"
            )
            v = emb.embed_query("ok")
            print("[PROBE] embeddings ok, dim:", len(v))
        except Exception as e:
            print("[PROBE] embeddings FAIL:", repr(e))
            return False

        try:
            import os

            persist_dir = str(getattr(config, "chroma_db_dir", "./chroma_db"))
            os.makedirs(persist_dir, exist_ok=True)
            print("[PROBE] persist dir ok:", persist_dir)
        except Exception as e:
            print("[PROBE] persist dir FAIL:", repr(e))
            return False

        try:
            from langchain_chroma import Chroma

            db = Chroma(
                persist_directory=persist_dir,
                embedding_function=emb,
                collection_name="rona",  # explicit name helps
                collection_metadata={"hnsw:space": "cosine"},
            )
            _ = db.get(limit=1)
            print("[PROBE] chroma get() ok")
        except Exception as e:
            print("[PROBE] chroma FAIL:", repr(e))
            return False

        return True

    def _initialize_database(self):
        import os, traceback

        try:
            if not LC_VECTOR_OK:
                raise RuntimeError(f"Critical imports failed: {repr(LC_VECTOR_ERR)}")

            if not LC_OK:
                raise RuntimeError("LangChain/Chroma not available (imports failed)")

            base_url = "http://127.0.0.1:11434"
            embed_model = getattr(config, "embed_model", None)
            candidates = [embed_model] if embed_model else DEFAULT_EMBED_CANDIDATES

            last_err = None
            for name in candidates:
                if not name:
                    continue
                try:
                    self.embeddings_model = OllamaEmbeddings(
                        model=name, base_url=base_url
                    )
                    self.embeddings_model.embed_query("healthcheck")
                    print(f"[DB] ✅ Embeddings OK via '{name}'")
                    break
                except Exception as e:
                    print(f"[DB] Embedding '{name}' failed: {e}")
                    last_err = e
                    self.embeddings_model = None

            if not self.embeddings_model:
                raise RuntimeError(
                    f"Ollama embeddings failed for all candidates: {last_err}"
                )

            persist_dir = str(getattr(config, "chroma_db_dir", "./chroma_db"))
            os.makedirs(persist_dir, exist_ok=True)
            from langchain_chroma import Chroma

            self.vector_db = Chroma(
                persist_directory=persist_dir,
                embedding_function=self.embeddings_model,
                collection_name="rona",
                collection_metadata={"hnsw:space": "cosine"},
            )
            _ = self.vector_db.get(limit=1)
            print("[DB] ✅ Chroma initialized")

        except Exception as e:
            print("DB init failed (Chroma path):", repr(e))
            traceback.print_exc()
            self.vector_db = None

    # ---------- NEW: quick introspection ----------
    def backend_info(self) -> str:
        if self.vector_db is None:
            return "None"
        t = type(self.vector_db)
        return f"{t.__module__}.{t.__name__}"

    # ---------- NEW: scored retrieval ----------
    def add_documents(self, docs) -> bool:
        """
        Upsert a list of docs into Chroma.
        Accepts LangChain Documents OR plain dicts like:
        {"page_content": "...", "metadata": {...}}
        Creates stable IDs by hashing content|source to dedupe.
        """
        import hashlib

        if not getattr(self, "vector_db", None):
            print("[DB] add_documents: vector_db is None")
            return False

        texts, metas, ids = [], [], []

        for d in docs or []:
            # Accept LC Document or dict
            content = ""
            meta = {}
            if hasattr(d, "page_content"):
                content = getattr(d, "page_content", "") or ""
                meta = getattr(d, "metadata", {}) or {}
            else:
                content = (d or {}).get("page_content", "") or ""
                meta = (d or {}).get("metadata", {}) or {}

            src = str(meta.get("source", ""))[:256]
            # stable id: sha1(content|source)
            h = hashlib.sha1(
                (content + "|" + src).encode("utf-8", "ignore")
            ).hexdigest()
            ids.append(h)
            texts.append(content)
            metas.append(meta)

        if not texts:
            print("[DB] add_documents: nothing to add")
            return False

        try:
            self.vector_db.add_texts(texts=texts, metadatas=metas, ids=ids)
            return True
        except Exception as e:
            print("[DB] add_documents error:", repr(e))
            return False

    def similarity_search(self, query: str, k: int = 5):
        if not getattr(self, "vector_db", None):
            return []
        try:
            return self.vector_db.similarity_search(query, k=k) or []
        except Exception as e:
            print("similarity_search error:", e)
            return []

    def similarity_search_scored(self, query: str, k: int = 5):
        """
        Return standardized: [(doc_dict, score(0..1), 'vector_db'), ...]
        Works across different Chroma/LangChain versions.
        """
        if not getattr(self, "vector_db", None):
            return []
        # Try modern relevance scores
        try:
            items = self.vector_db.similarity_search_with_relevance_scores(query, k=k)
            out = []
            for doc, score in items:
                out.append(
                    (
                        {
                            "text": getattr(doc, "page_content", ""),
                            "metadata": getattr(doc, "metadata", {}),
                        },
                        float(score or 0.0),
                        "vector_db",
                    )
                )
            return out
        except Exception:
            pass
        # Try distance-based scores
        try:
            items = self.vector_db.similarity_search_with_score(query, k=k)
            out = []
            for doc, dist in items:
                sim = 1.0 / (1.0 + float(dist or 0.0))
                out.append(
                    (
                        {
                            "text": getattr(doc, "page_content", ""),
                            "metadata": getattr(doc, "metadata", {}),
                        },
                        sim,
                        "vector_db",
                    )
                )
            return out
        except Exception:
            pass
        # Fallback: plain search without scores
        docs = self.similarity_search(query, k=k)
        return [
            (
                {
                    "text": getattr(d, "page_content", ""),
                    "metadata": getattr(d, "metadata", {}),
                },
                0.5,
                "vector_db",
            )
            for d in (docs or [])
        ]


database_manager = DatabaseManager()


# singleton wrapper for DB so multiple use share same instance
class DatabaseManagerSingleton:
    _instance = None

    @classmethod
    def get(cls) -> Optional[DatabaseManager]:
        if cls._instance is None:
            try:
                cls._instance = DatabaseManager()
            except Exception as e:
                print("Database manager init error:", e)
                cls._instance = None
        return cls._instance


def sanity_check_vector_db():
    """
    Deep sanity check that LangChain/Chroma + Ollama embeddings + persistence all work.
    Prints precise failure points and returns True/False.
    Also inserts a tiny self-test doc, queries it, and then cleans it up.
    """
    import os, time, traceback, hashlib

    try:
        # 0) Basic import state
        try:
            LC_state = "OK" if LC_VECTOR_OK else "NOT OK"
        except NameError:
            LC_state = "UNKNOWN"
        print(f"[DB] LC_VECTOR_OK = {LC_state}")

        # 1) Get singleton + vector store
        db = DatabaseManagerSingleton.get()
        if not db:
            print("[DB] ❌ DatabaseManagerSingleton.get() returned None")
            return False
        if not getattr(db, "vector_db", None):
            print("[DB] ❌ vector_db is None (Chroma not initialized)")
            return False

        # 2) Embeddings health (Ollama reachable)
        emb = getattr(db, "embeddings_model", None)
        if hasattr(emb, "embed_query"):
            try:
                _ = emb.embed_query("healthcheck")
                print("[DB] ✅ Embeddings OK (embed_query works)")
            except Exception as e:
                print(f"[DB] ❌ Embeddings failed (Ollama not reachable?): {e}")
                return False
        else:
            print("[DB] ⚠️ embeddings_model has no embed_query; skipping check")

        # 3) Persist dir existence & writability
        persist_dir = str(getattr(config, "chroma_db_dir", "./chroma_db"))
        try:
            os.makedirs(persist_dir, exist_ok=True)
        except Exception as e:
            print(f"[DB] ❌ Cannot create persist dir '{persist_dir}': {e}")
            return False
        if not os.access(persist_dir, os.W_OK):
            print(f"[DB] ❌ Persist dir not writable: {persist_dir}")
            return False
        print(f"[DB] ✅ Persist dir OK: {persist_dir}")

        # 4) Chroma collection “open” check
        try:
            _ = db.vector_db.get(limit=1)
            print("[DB] ✅ Chroma collection opened")
        except Exception as e:
            print(f"[DB] ❌ Chroma .get() failed: {e}")
            traceback.print_exc()
            return False

        # 5) Insert a small self-test doc (with deterministic id your add_documents uses)
        marker = f"selftest-{int(time.time())}"
        content = f"[Rona selftest] {marker}. OWASP Top 10 are common web application security risks."
        meta = {"source": "__rona_selftest__.txt"}
        test_id = hashlib.sha1(
            (content + "|" + meta["source"]).encode("utf-8", "ignore")
        ).hexdigest()

        ok = db.add_documents([{"page_content": content, "metadata": meta}])
        print("[DB] add_documents:", ok)
        if not ok:
            print("[DB] ❌ add_documents failed")
            return False

        # 6) Query back (scored wrapper → fallback)
        try:
            hits = db.similarity_search_scored("OWASP Top 10", k=3)
        except Exception:
            try:
                docs = db.similarity_search("OWASP Top 10", k=3)
                hits = [
                    (
                        {
                            "text": getattr(d, "page_content", ""),
                            "metadata": getattr(d, "metadata", {}),
                        },
                        0.5,
                        "vector_db",
                    )
                    for d in (docs or [])
                ]
            except Exception as e:
                print(f"[DB] ❌ similarity_search failed: {e}")
                traceback.print_exc()
                return False

        if not hits:
            print("[DB] ❌ similarity_search returned 0 results")
            return False

        print(f"[DB] ✅ Vector DB sanity OK (hits={len(hits)})")

        # 7) Cleanup: remove selftest doc if the backend supports delete by id
        try:
            if hasattr(db.vector_db, "delete"):
                db.vector_db.delete(ids=[test_id])
                print("[DB] 🧹 Cleanup: removed selftest doc")
        except Exception:
            # non-fatal
            pass

        return True

    except Exception as e:
        print(f"[DB] ❌ Sanity check error: {e}")
        traceback.print_exc()
        return False


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
                # --- ADD THIS NEW TABLE ---
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS psycho_clues (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        note_hash TEXT UNIQUE NOT NULL,
                        clue TEXT NOT NULL,
                        created_at TEXT NOT NULL
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

    # --- ADD THESE NEW METHODS FOR CACHING ---
    def get_clue_by_hash(self, note_hash: str) -> Optional[str]:
        """Retrieves a cached clue from the database using its hash."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute(
                    "SELECT clue FROM psycho_clues WHERE note_hash = ?", (note_hash,)
                )
                row = cur.fetchone()
                return row[0] if row else None
        except Exception as e:
            print(f"SQLite get_clue_by_hash error: {e}")
            return None

    def add_clue(self, note_hash: str, clue: str) -> bool:
        """Adds a new clue to the cache."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                created_at = datetime.datetime.now().isoformat()
                cur.execute(
                    "INSERT INTO psycho_clues (note_hash, clue, created_at) VALUES (?, ?, ?)",
                    (note_hash, clue, created_at),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            # This can happen if two threads try to insert the same hash, which is fine.
            return True
        except Exception as e:
            print(f"SQLite add_clue error: {e}")
            return False

    def get_all_clues(self) -> List[str]:
        """Retrieves all cached clues."""
        try:
            with self._connect() as conn:
                cur = conn.cursor()
                cur.execute("SELECT clue FROM psycho_clues ORDER BY created_at DESC")
                rows = cur.fetchall()
                return [row[0] for row in rows]
        except Exception as e:
            print(f"SQLite get_all_clues error: {e}")
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




class FlaskServerController:
    def __init__(self, app, host="127.0.0.1", port=8765):
        from werkzeug.serving import make_server
        import threading

        self.app = app
        self.host = host
        self.port = port
        self._server = make_server(host, port, app)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._ctx = app.app_context()
        self._started = False

    def shutdown(self):
        try:
            if self._started:
                self._server.shutdown()
        except Exception:
            pass
        try:
            if self._thread.is_alive():
                self._thread.join(timeout=2)
        except Exception:
            pass
        try:
            self._ctx.pop()
        except Exception:
            pass


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
        details = (data.get("details") or "").strip()
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


# Add this function definition in the global scope, for example, near create_psycho_app
def start_psycho_ui_or_raise(app_instance, bind_host: str, preferred_port: int):
    """
    Starts the Flask server for the Psycho UI.
    """
    if not FLASK_OK:
        print("🔵 NOTICE: Flask not installed. `pip install flask` to use web Psycho UI.")
        return

    psycho_app = create_psycho_app()
    server_controller = FlaskServerController(psycho_app, host=bind_host, port=preferred_port)
    app_instance.psycho_server = server_controller # Store for shutdown
    server_controller._thread.start()
    server_controller._started = True
    print(f"[psycho-ui] Server started on http://{bind_host}:{preferred_port}/psycho/")
# +++++ CHANGE 1: ADDED extract_zoomeye_directive and _relevance +++++
ZOOM_DIRECTIVE_RE = re.compile(
    r"zoomeye:\{(?P<kw>[^:}]+)(?::(?P<start>\d+)-(?P<end>\d+))?\}", re.IGNORECASE
)


def extract_zoomeye_directive(query: str):
    """
    Returns (natural_question, keyword, start_page, end_page) or (query, None, None, None)
    """
    m = ZOOM_DIRECTIVE_RE.search(query or "")
    if not m:
        return query, None, None, None

    kw = (m.group("kw") or "").strip()
    sp = int(m.group("start")) if m.group("start") else 1
    ep = int(m.group("end")) if m.group("end") else sp

    # natural question = query with directive removed
    natural_q = (query[: m.start()] + query[m.end() :]).strip()
    # collapse double spaces
    natural_q = re.sub(r"\s{2,}", " ", natural_q)
    return natural_q, kw, max(1, sp), max(sp, ep)


import json, time, requests
from typing import List, Dict, Any
from collections import defaultdict  # Added import

import json, time, requests
from typing import List, Dict, Any


def _relevance(item_text: str, question: str) -> float:
    # uses your existing overlap_score if available; else simple fallback
    try:
        return overlap_score(question, item_text)
    except Exception:
        q = (question or "").lower()
        t = (item_text or "").lower()
        shared = sum(1 for w in set(q.split()) if w in t)
        return shared / (1 + len(set(q.split())))


# --- HUNT MODE GUARD (must be first lines in router) ---


class DeepSearchEngine:
    def __init__(self):
        self.google_key = config.config.get("google_cse", {}).get("api_key") or None
        self.google_cx = config.config.get("google_cse", {}).get("cx") or None

        self._su_cache = {}  # {query: (timestamp, results)}
        self._su_cache_ttl = (
            float(
                getattr(config, "config", {}).get("search_unified_cache_ttl_sec", 3.0)
            )
            if hasattr(config, "config")
            else 3.0
        )
        self._last_log = ("", 0.0)  # (last_message, timestamp)
        # Load ZoomEye API key from config, fallback to environment variable
        zcfg = config.config.get("zoomeye", {})
        self.zoomeye_api_key = (
            zcfg.get("api_key") or os.getenv("ZOOMEYE_API_KEY") or ""
        ).strip()
        self.zoomeye_enabled = bool(zcfg.get("enabled", True))

        # +++++ CHANGE A: ADDED INIT FIELDS +++++
        self._zoomeye_cache = {}  # dict[keyword] = (ts, results)
        self._zoomeye_cache_ttl = int(zcfg.get("cache_ttl_sec", 3600))
        self._zoomeye_cooldown_until = 0  # epoch seconds
        self._zoomeye_cooldown_sec = int(
            zcfg.get("cooldown_sec", 3600)
        )  # 1h; adjust as you like

        # Optional: sanity log (masked)

    def _mask(k):
        return (
            (k[:4] + "..." + k[-4:])
            if isinstance(k, str) and len(k) >= 10
            else ("***" if k else "")
        )

    # print(f"[ZoomEye] Loaded API key: {_mask(self.zoomeye_api_key)}")

    @staticmethod
    def _normalize_source(s: str) -> str:
        s = (s or "").lower()
        if "conversation" in s:
            return "conversation"
        return s or "web"

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

        for d in docs:
            content = getattr(d, "page_content", "") or ""
            meta = getattr(d, "metadata", {}) or {}

            # 🚫 Skip warmup/self-test docs
            if (meta.get("source") == "__warmup__") or (
                "__warmup__" in meta.get("source", "")
            ):
                continue
            if content.strip() == "__rona_warmup__":
                continue

            c = content.lower()
            # hard filter: image/registry junk
            if ("image file:" in c) or ("data/images" in c) or ("/images/" in c):
                continue
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
            out.append(
                {
                    "source": "vector_db",
                    "title": meta.get("source", "local_doc"),
                    "content": content,
                    "meta": meta,
                    "score": score,
                }
            )
        return out

    def _is_probably_english(self, s: str) -> bool:
        if not s:
            return False
        letters = [ch for ch in s if ch.isalpha()]
        if not letters:
            return True
        latin = sum(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in letters)
        return latin / max(1, len(letters)) >= float(
            config.config.get("web_lang_en_ratio", 0.85)
        )

    def _is_allowed_source(self, url: str, topic_hint: str = "") -> bool:
        host = (urllib.parse.urlparse(url).netloc or "").lower()
        allow = {
            "owasp.org",
            "cheatsheetseries.owasp.org",
            "portswigger.net",
            "cwe.mitre.org",
            "nvd.nist.gov",
            "mitre.org",
            "developer.mozilla.org",
        }
        allow |= set(map(str.lower, config.config.get("web_allowlist", [])))
        block = {"zhihu.com", "quora.com", "pinterest.com", "facebook.com"}
        block |= set(map(str.lower, config.config.get("web_blocklist", [])))
        if any(b in host for b in block):
            return False
        if topic_hint and "owasp" in topic_hint.lower():
            # Prefer canonical domains; still allow others (ranker can demote)
            return True
        return True

    # Inside the DeepSearchEngine class...
    # Replace the existing duckduckgo_search function in DeepSearchEngine

    # ---- DYNAMIC, PER-QUERY SCORING BRAIN ----

    def _detect_lang_fast(self, s: str) -> str:
        # ultra-light heuristic: enough for filtering obvious zh/arabic vs english
        s = s or ""
        if any("\u0600" <= ch <= "\u06ff" for ch in s):
            return "ar"
        if any("\u4e00" <= ch <= "\u9fff" for ch in s):
            return "zh"
        return "en"

    def _should_wide_range(self, query: str) -> bool:
        q = (query or "").lower().strip()
        cues = [
            "give me",
            "list",
            "websites",
            "sites",
            "vendors",
            "providers",
            "stores",
            "shops",
            "best",
            "top",
            "where to buy",
            "alternatives",
            "resources",
            "sources",
        ]
        return any(c in q for c in cues) or bool(
            re.search(r"\b(10|20|50|top\s*\d+)\b", q)
        )

    # ---------- Light NLP helpers ----------
    _STOP = set(
        """a an the of to for in on with by and or vs versus how what which where when
    buy price prices cheap cheapest best top list websites website site vendor vendors
    sell selling store stores shop shops online official docs doc documentation api
    tutorial guide review reviews compare comparison alternatives resources sources
    """.split()
    )

    def _tokenize_loose(self, s: str) -> list[str]:
        return [t for t in re.split(r"[^A-Za-z0-9]+", (s or "").lower()) if t]

    def _lemmatize_simple(self, w: str) -> str:
        # ultra-light lemmatizer for English verbs/nouns (no external deps)
        if w.endswith("ies") and len(w) > 3:
            return w[:-3] + "y"  # studies→study
        if w.endswith("es") and len(w) > 2:
            return w[:-2]  # adjusts→adjust (rough)
        if w.endswith("s") and len(w) > 2:
            return w[:-1]
        return w

    def _is_definitional_intent(self, q: str) -> bool:
        ql = (q or "").lower()
        # patterns like: what is/are/does X, define X, meaning of X, X meaning
        return bool(
            re.search(r"\b(define|definition|meaning|what\s+(is|are|does))\b", ql)
        ) or bool(re.match(r"^\s*(what|meaning|define)\b", ql))

    def _query_doctor(self, q: str) -> tuple[str, dict]:
        """
        Returns (rewritten_query, meta).
        If definitional & short: rewrite to 'define <headword>' and expose headword.
        """
        raw = (q or "").strip()
        toks = self._tokenize_loose(raw)
        meta = {"intent": "informational", "headword": None, "rewritten": False}

        if not toks:
            return raw, meta

        # detect language quickly; if Arabic, keep as-is but mark definitional by Arabic cues too

        lang = getattr(self, "_detect_lang_fast", lambda s: "en")(raw)
        is_def = self._is_definitional_intent(raw) or bool(
            re.search(r"\b(معنى|تعريف)\b", raw)
        )

        # very short queries (<= 5 tokens) that look definitional → rewrite
        if is_def or len(toks) <= 5:
            # Find a candidate headword (skip stop words)
            stop = {
                "what",
                "is",
                "are",
                "does",
                "the",
            stop = {
                "what", "is", "are", "does", "the", "a", "an", "of", "to", "in", "on", "for",
                "mean", "define", "meaning", "definition", "how", "when", "why", "where", "who"
            }
            headword_candidates = [t for t in toks if t not in stop]
            if headword_candidates:
                headword = self._lemmatize_simple(headword_candidates[0])
                meta["headword"] = headword
                meta["rewritten"] = True
                return f"define {headword}", meta
        return raw, meta

    def _host(self, url: str) -> str:
        return (urllib.parse.urlparse(url or "").netloc or "").lower()

    # ---------- Dynamic category scorer (per-result, per-query) ----------
    def _category_scores(self, query: str, r: Dict[str, Any]) -> Dict[str, float]:
        text = " ".join([q, title, snippet, url])

        def has(*keys):
            return any(k in text for k in keys)

        scores = {
            "Official/Organization": 0.0,
            "Documentation/Specs": 0.0,
            "Tutorials/How-to": 0.0,
            "Vendors/Marketplaces": 0.0,
            "Reviews/Comparisons": 0.0,
            "Community/Q&A": 0.0,
            "News/Media": 0.0,
            "General/Info": 0.0,
        }

        # Intent-ish features (all dynamic; no static allowlists)
        if has("official", "homepage", "foundation", "/about", ".org/"):
            scores["Official/Organization"] += 0.7
        if has(
            "docs",
            "documentation",
            "reference",
            "api",
            "rfc",
            "spec",
            "/docs",
            "/developer",
        ):
            scores["Documentation/Specs"] += 0.8
        if has(
            "how to",
            "tutorial",
            "guide",
            "step",
            "walkthrough",
            "quickstart",
            "example",
        ):
            scores["Tutorials/How-to"] += 0.8
        if has(
            "buy",
            "price",
            "pricing",
            "shop",
            "store",
            "marketplace",
            "sell",
            "for sale",
        ):
            scores["Vendors/Marketplaces"] += 0.85
        if has(
            "review",
            "reviews",
            "compare",
            "comparison",
            "vs ",
            "pros",
            "cons",
            "rating",
            "top",
        ):
            scores["Reviews/Comparisons"] += 0.75
        if has("forum", "community", "discuss", "stack", "q&a", "reddit", "discussion"):
            scores["Community/Q&A"] += 0.7
        if has("news", "press", "announces", "launches", "today", "2024", "2025"):
            scores["News/Media"] += 0.6

        # Light host+path heuristics (dynamic)
        path = urllib.parse.urlparse(url).path or "/"
        depth = max(0, len([p for p in path.split("/") if p]))
        if depth >= 3:  # often docs/tutorial detail
            scores["Documentation/Specs"] += 0.1
            scores["Tutorials/How-to"] += 0.05

        # If nothing strong matched, nudge to general info
        if max(scores.values() or [0]) < 0.25:
            scores["General/Info"] += 0.3

        return scores

    def _best_category(self, query: str, r: Dict[str, Any]) -> Tuple[str, float]:
        scores = self._category_scores(query, r)
        cat, sc = max(scores.items(), key=lambda kv: kv[1])
        return cat, sc

    # ---------- Build categorized map (label -> list of urls) ----------
    def categorize_web_results(
        self, query: str, results: List[Dict[str, Any]], max_per_cat: int = 6
    ) -> Dict[str, List[str]]:
        buckets: Dict[str, List[str]] = defaultdict(list)
        seen_urls = set()
        for r in results:
            url = r.get("url") or ""
            if not url or url in seen_urls:
                continue
            cat, sc = self._best_category(query, r)
            if sc < 0.25:
                cat = "General/Info"
            if len(buckets[cat]) < max_per_cat:
                buckets[cat].append(url)
                seen_urls.add(url)
        # Optional: add a topic prefix for readability
        topic = self._head_topic(query)
        return {(f"{topic} — {k}" if topic else k): v for k, v in buckets.items() if v}

    def _infer_intent(self, query: str) -> str:
        q = (query or "").strip().lower()
        if q.startswith("/hunt "):
            return "hunt"
        if "owasp" in q or "top 10" in q or "top10" in q:
            return "definition"
        if any(k in q for k in ["what is", "define", "definition"]):
            return "definition"
        if any(
            k in q for k in ["how to", "guide", "tutorial", "step", "fix", "mitigation"]
        ):
            return "tutorial"
        if any(k in q for k in ["news", "latest", "today", "2025", "2024"]):
            return "news"
        if any(k in q for k in ["reference", "docs", "api", "rfc", "spec"]):
            return "reference"
        return "informational"

    def _host_features(self, url: str) -> dict:
        """Reputation-ish signals without static lists."""
        host = (urllib.parse.urlparse(url).netloc or "").lower()
        tld = host.split(".")[-1] if host else ""
        parts = host.split(".")
        depth = len((urllib.parse.urlparse(url).path or "/").strip("/").split("/"))
        # cheap reputation priors
        tld_boost = {
            "org": 0.10,
            "edu": 0.18,
            "gov": 0.20,
            "mil": 0.20,
            "io": 0.04,
            "dev": 0.04,
            "com": 0.02,
            "net": 0.02,
        }.get(tld, 0.0)
        # community/forum heuristics (category-based, not static domain list)
        community = any(
            k in host for k in ["forum", "community", "stack", "discuss", "qa"]
        )
        social_ugc = any(
            k in host for k in ["reddit", "discord", "medium", "blog", "substack"]
        )
        # longer hosts often mean subprojects/docs (slight boost)
        host_len_bonus = min(0.05, 0.01 * max(0, len(parts) - 2))
        # deep paths are often specific; cap bonus
        path_depth_bonus = min(0.08, 0.02 * max(0, depth - 1))
        return {
            "host": host,
            "tld_boost": tld_boost,
            "community": community,
            "social_ugc": social_ugc,
            "host_len_bonus": host_len_bonus,
            "path_depth_bonus": path_depth_bonus,
        }

    def _language_alignment(self, query_lang: str, text_lang: str) -> float:
        if query_lang == "other":
            return 0.0
        return (
            1.0
            if query_lang == text_lang
            else (-0.4 if text_lang in ("ar", "zh") and query_lang == "en" else -0.2)
        )

    def _topic_alignment(
        self, q_tokens: list[str], title: str, snippet: str, url: str
    ) -> float:
        txt = f"{title} {snippet} {url}".lower()
        hits = sum(1 for t in q_tokens if t and t in txt)
        uniq = len(set(q_tokens))
        if uniq == 0:
            return 0.0
        return min(1.0, hits / (uniq * 1.0))  # 0..1

    def _snippet_quality(self, title: str, snippet: str) -> float:
        tlen = len(title or "")
        slen = len(snippet or "")
        if tlen < 3 and slen < 20:
            return -0.3
        bonus = 0.0
        if tlen >= 10:
            bonus += 0.05
        if slen >= 60:
            bonus += 0.10
        # penalize non-Latin heavy text for English queries later via language_alignment
        return bonus

    def _novelty_key(self, r: dict) -> str:
        return (r.get("url") or r.get("title") or "")[:200]

    def _score_web_candidate(self, query: str, r: dict) -> float:
        # Features
        q_lang = self._detect_lang_fast(query)
        t_lang = self._detect_lang_fast(
            (r.get("title", "") + " " + r.get("content", ""))
        )
        q_tokens = self._tokenize_loose(query)
        align_lang = self._language_alignment(q_lang, t_lang)  # [-0.4..1.0]
        align_topic = self._topic_alignment(
            q_tokens, r.get("title", ""), r.get("content", ""), r.get("url", "")
        )  # 0..1
        hostf = self._host_features(r.get("url", ""))
        snipq = self._snippet_quality(r.get("title", ""), r.get("content", ""))
        intent = self._infer_intent(query)

        # Intent-aware community/social strength (dynamic; no static domain list)
        community_pen = 0.0
        if intent in ("definition", "reference", "informational"):
            if hostf["community"]:
                community_pen -= 0.10
            if hostf["social_ugc"]:
                community_pen -= 0.08
        elif intent in ("tutorial",):
            # tutorials can live on community sites; reduce penalty
            if hostf["community"]:
                community_pen -= 0.03

        # Combine
        score = 0.0
        score += 1.20 * align_topic
        score += 0.90 * align_lang
        score += 0.20 * hostf["tld_boost"]
        score += 0.10 * hostf["host_len_bonus"]
        score += 0.10 * hostf["path_depth_bonus"]
        score += 0.20 * snipq
        score += community_pen

        # keep in [0, 2] rough range
        return max(0.0, min(2.0, score))

    def _is_navigational_page(self, query: str, title: str, url: str) -> bool:
        """
        Drop generic vendor/home/landing pages when the query isn't clearly navigational.
        """
        q = (query or "").lower()
        t = (title or "").lower()
        u = url or ""
        host = (urllib.parse.urlparse(u).netloc or "").lower()
        path = (urllib.parse.urlparse(u).path or "/").lower()

        # If user intent is clearly navigational, don't block
        nav_intent = any(
            k in q
            for k in [
                "download",
                "install",
                "homepage",
                "official site",
                "chrome",
                "firefox",
            ]
        )
        if nav_intent:
            return False

        # Common patterns that are frequently irrelevant
        if (
            host.endswith("google.com")
            and path.startswith("/intl/")
            and "chrome" in (t + " " + u)
        ):
            return True  # localized Chrome landing pages
        if re.fullmatch(r"/(intl/[a-z\-]{2,5}/)?chrome/?", path):
            return True
        if re.fullmatch(r"/(intl/[a-z\-]{2,5}/)?about/?", path):
            return True
        if re.search(r"/(download|pricing|signup|login|start|welcome)/?", path):
            # treat as navigational unless query has those words
            if not any(
                k in q for k in ["download", "price", "pricing", "signup", "login"]
            ):
                return True

        # Extremely low topical overlap with a very strong brand word in title
        toks = [t for t in re.split(r"[^a-z0-9]+", q) if t]
        hits = sum(1 for tok in set(toks) if tok and tok in t)
        if hits == 0 and any(
            b in t for b in ("chrome", "facebook", "instagram", "pinterest")
        ):
            return True

        return False

    def _topic_overlap(self, query: str, title: str, snippet: str, url: str) -> float:
        import re

        toks = [t for t in re.split(r"[^A-Za-z0-9]+", (query or "").lower()) if t]
        if not toks:
            return 0.0
        hay = f"{(title or '').lower()} {(snippet or '').lower()} {(url or '').lower()}"
        uniq = set(toks)
        hits = sum(1 for t in uniq if t in hay)
        return hits / max(1, len(uniq))

    def _postfilter_web_hits(self, query: str, items: list[dict]) -> list[dict]:
        out = []
        qlang = self._detect_lang_fast(query)
        q = (query or "").strip()
        # Adaptive thresholds: easier for definitional / short queries
        is_def = self._is_definitional_intent(q) or len(self._tokenize_loose(q)) <= 5
        min_overlap = 0.08 if is_def else 0.18

        kept, dropped_lang, dropped_nav, dropped_overlap = 0, 0, 0, 0

        for r in items or []:
            title = r.get("title", "")
            snippet = r.get("content", "")
            url = r.get("url", "")

            # language guard: keep same-language hits; drop strong mismatch for EN queries
            tlang = self._detect_lang_fast(title + " " + snippet)
            if qlang == "en" and tlang in ("zh", "ar"):
                dropped_lang += 1
                continue

            # navigational/installer guard
            lowt = (title or "").lower()
            lowu = (url or "").lower()
            if ("chrome" in lowt and "google.com/" in lowu) or (
                "download" in lowt and "google.com/chrome" in lowu
            ):
                dropped_nav += 1
                continue

            # topical overlap (ADAPTIVE)
            if self._topic_overlap(query, title, snippet, url) < min_overlap:
                dropped_overlap += 1
                continue

            # keep
            r["source"] = self._normalize_source(r.get("source"))
            out.append(r)
            kept += 1

        print(
            f"[postfilter] kept={kept} drop_lang={dropped_lang} drop_nav={dropped_nav} drop_overlap={dropped_overlap} (min_overlap={min_overlap})"
        )
        return out

    def _bucketize_urls(self, items: list[dict]) -> dict[str, list[str]]:
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
            if not u:
                continue
            dom = tldextract.extract(u)
            root = ".".join([p for p in [dom.domain, dom.suffix] if p])
            lurl = u.lower()
            lt = (r.get("title") or "").lower()

            if root.startswith(("docs.",)):
                buckets["docs"].add(u)
            elif "github.com" in lurl:
                buckets["github"].add(u)
            elif lurl.endswith(".edu") or ".edu/" in lurl:
                buckets["edu"].add(u)
            elif lurl.endswith(".gov") or ".gov/" in lurl:
                buckets["gov"].add(u)
            elif any(
                w in lurl
                for w in [
                    "stackoverflow.com",
                    "serverfault.com",
                    "superuser.com",
                    "reddit.com",
                    "stackexchange.com",
                ]
            ):
                buckets["forums"].add(u)
            elif any(w in lurl for w in ["medium.com", "dev.to", "blog.", "/blog/"]):
                buckets["news/blogs"].add(u)
            elif any(
    Returns: list of {source,title,content,url}
    """
    import aiohttp, urllib.parse, re, html as _html

    q = (query or "").strip()
    if not q:
        return []

    # ---------- helpers ----------
    def _mk(item_title: str, item_snippet: str, item_url: str) -> dict:
        return {
            "source": "duckduckgo",
            "title": (item_title or "").strip()[:160],
            "content": (item_snippet or "").strip(),
            "url": (item_url or "").strip(),
        }

    async def _fetch_json(session, url):
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            return None

    async def _fetch_text(session, url, params=None):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    return ""
                return await resp.text()
        except Exception:
            return ""

    # ---------- run ----------
    out: list[dict] = []
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.8",
    }
    timeout = aiohttp.ClientTimeout(total=12)

    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        # 1) Instant Answer JSON API (quick pass; may be empty)
        try:
            q_enc = urllib.parse.quote_plus(q)
            url_json = f"https://api.duckduckgo.com/?q={q_enc}&format=json&no_html=1&skip_disambig=1"
            data = await _fetch_json(session, url_json)
            if data:
                # Abstract/Results
                abs_text = (data.get("AbstractText") or "").strip()
                abs_url = (data.get("AbstractURL") or "").strip()
                if abs_text and abs_url:
                    out.append(
                        _mk(
                            abs_text.split(".")[0][:120] or "Result",
                            abs_text,
                            abs_url,
                        )
                    )

                for it in data.get("Results") or []:
                    t = _html.unescape(it.get("Text", "") or "")
                    u = it.get("FirstURL", "") or ""
                    if t and u:
                        out.append(_mk(t[:120], t, u))

                # RelatedTopics
                for it in data.get("RelatedTopics") or []:
                    if isinstance(it, dict):
                        t = _html.unescape(it.get("Text", "") or "")
                        u = it.get("FirstURL", "") or ""
                        if t and u:
                            out.append(_mk(t[:120], t, u))

            # Trim if enough
            if len(out) >= max_results:
                return out[:max_results]
        except Exception:
            pass  # ignore JSON errors, fall through

        # 2) HTML endpoint (real SERP)
        try:
            # Using /html endpoint avoids heavy JS; easier to parse
            params = {"q": q, "kl": "us-en"}  # kl=region
            html = await _fetch_text(
                session, "https://duckduckgo.com/html/", params=params
            )

            # Each result has <a class="result__a" href="...">Title</a>
            # Snippet often inside <a class="result__snippet"> or <div class="result__snippet">
            link_pat = re.compile(
                r'<a[^>]+class="result__a"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
                re.IGNORECASE | re.DOTALL,
            )
            titles = link_pat.findall(html)

            # Try to capture snippet sitting near link block
            # (greedy enough to grab text between link and next result)
            block_pat = re.compile(
                r'(?:<a[^>]+class="result__a"[^>]+href="[^"]+"[^>]*>.*?</a>)(?P<block>.*?)(?=<a[^>]+class="result__a"|$)',
                re.IGNORECASE | re.DOTALL,
            )
            snippet_pat = re.compile(
                r'(?:<div[^>]+class="result__snippet"[^>]*>|<span>)(?P<snip>.*?)(?:</div>|</span>)',
                re.IGNORECASE | re.DOTALL,
            )

            blocks = block_pat.findall(html)
            snippets = []
            for b in blocks:
                ms = snippet_pat.search(b or "")
                if ms:
                    sn = re.sub(r"<[^>]+>", "", ms.group("snip") or "")
                    snippets.append(_html.unescape(sn))
                else:
                    snippets.append("")

            for i, (href, title) in enumerate(titles[: max_results * 2]):
                snip = snippets[i] if i < len(snippets) else ""
                if href and title:
                    out.append(_mk(title, snip, href))

            if len(out) >= max_results:
                return out[:max_results]
        except Exception:
            pass  # fall through
            return []
                blocks = block_pat.findall(html)
                snippets = []
                for b in blocks:
                    ms = snippet_pat.search(b or "")
                    if ms:
                        sn = re.sub(r"<[^>]+>", "", ms.group("snip") or "")
                        snippets.append(_html.unescape(sn))
                    else:
                        snippets.append("")

                for i, (href, title) in enumerate(titles[: max_results * 2]):
                    snip = snippets[i] if i < len(snippets) else ""
                    if href and title:
                        out.append(_mk(title, snip, href))

                if len(out) >= max_results:
                    return out[:max_results]
            except Exception:
                pass  # fall through

            # 3) Lite endpoint (super simple HTML)
            try:
                params = {"q": q}
                html = await _fetch_text(
                    session, "https://lite.duckduckgo.com/lite/", params=params
                )
                # Links look like: <a rel="nofollow" class="result-link" href="URL">Title</a>
                link_lite = re.compile(
                    r'<a[^>]+class="result-link"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
                    re.IGNORECASE | re.DOTALL,
                )
                for m in link_lite.finditer(html):
                    href = _html.unescape(m.group("href") or "")
                    title = re.sub(r"<[^>]+>", "", m.group("title") or "")
                    if href and title:
                        out.append(_mk(_html.unescape(title), "", href))
                    if len(out) >= max_results:
                        break
            except Exception:
                pass

        # Final trim
        # de-dup by URL
        seen = set()
        dedup = []
        for r in out:
            u = r.get("url", "")
            if u and u not in seen:
                seen.add(u)
                dedup.append(r)
            if len(dedup) >= max_results:
                break
        return dedup

    def google_cse_search(self, query: str, num: int = 5) -> List[Dict[str, Any]]:

        if not self.google_key or not self.google_cx:
            print("🔵 NOTICE: Google CSE disabled → using DuckDuckGo fallback.")
            return []
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": self.google_key,
                "cx": self.google_cx,
                "q": query,
                "num": num,
                "lr": "lang_en",  # ask for EN, but scoring remains dynamic
                "hl": "en",
                "safe": "active",
            }
            r = requests.get(url, params=params, timeout=10)
            if r.status_code != 200:
                print(f"🔵 NOTICE: Google CSE HTTP {r.status_code} → empty.")
                try:
                    print("diag:", r.json())
                except Exception:
                    pass
                return []

            data = r.json()
            tmp: list[dict] = []
            for it in data.get("items", []) or []:
                cand = {
                    "source": "google",
                    "title": (it.get("title", "") or "")[:160],
                    "content": it.get("snippet", "") or "",
                    "url": it.get("link", "") or "",
                }
                cand["_score"] = self._score_web_candidate(query, cand)
                tmp.append(cand)

            # sort, dedup, return best N
            seen = set()
            out = []
            for c in sorted(tmp, key=lambda x: x["_score"], reverse=True):
                key = self._novelty_key(c)
                if key in seen:
                    continue
                seen.add(key)
                c.pop("_score", None)
                out.append(c)
                if len(out) >= num:
                    break
            return out
        except Exception as e:
            print("🔵 NOTICE: Google CSE error:", e)
            return []

    def search_zoomeye(self, query: str) -> List[Dict[str, Any]]:
        # +++++ CHANGE B: MODIFIED TOP OF search_zoomeye +++++
        # global enable (optional)
        if not getattr(self, "zoomeye_enabled", True):
            return []

        # Skip if we're on cooldown (after a 402)
        now = time.time()
        if now < getattr(self, "_zoomeye_cooldown_until", 0):
            # print(f"🔕 [ZoomEye] Skipping (cooldown {int(self._zoomeye_cooldown_until - now)}s left)")
            return []

        # Only search if query contains {keyword}
        m = re.search(r"\{(.*?)\}", query)
        if not m:
            return []
        keyword = (m.group(1) or "").strip()
        if not keyword:
            print("🔵 [ZoomEye] Empty {} found. Skipping search.")
            return []

        # cache guard
        hit = self._zoomeye_cache.get(keyword)
        if hit and (now - hit[0] < self._zoomeye_cache_ttl):
            return hit[1]

        # (Original code continues)
        API_KEY = (self.zoomeye_api_key or "").strip()
        if not API_KEY:
            print(
                "❌ [ZoomEye] API key is not set (check config.json → zoomeye.api_key or ZOOMEYE_API_KEY env)."
            )
            return []

        API_URL = "https://api.zoomeye.ai/host/search"  # API host (not the web site)

        headers = {
            "API-KEY": API_KEY.strip(),
            "Accept": "application/json",
            "User-Agent": "RonaV6/1.0",
        }
        params = {"query": f'"{keyword}"', "page": 1}

        print(f"🔵 [ZoomEye] GET {API_URL} params={params}")

        try:
            # Block redirects so we can see if API is bouncing us to the web UI
            resp = requests.get(
                API_URL,
                headers=headers,
                params=params,
                timeout=20,
                allow_redirects=False,
            )

            # Debug: show where we actually landed and any redirect chain
            if (
                resp.is_redirect
                or resp.is_permanent_redirect
                or resp.status_code in (301, 302, 303, 307, 308)
            ):
                print(
                    f"❌ [ZoomEye] Redirected: {resp.status_code} → {resp.headers.get('Location')}"
                )
                return []

            if resp.status_code == 401:
                print("❌ [ZoomEye] 401 Unauthorized – invalid/expired API key.")
                return []

            # +++++ CHANGE C: REPLACED 402 HANDLING +++++
            if resp.status_code == 402:
                print(
                    "❌ [ZoomEye] 402 Payment Required – ZoomEye credits insufficient."
                )
                self._zoomeye_cooldown_until = time.time() + self._zoomeye_cooldown_sec
                # optional notice item; or just `return []`
                results = [
                    {
                        "source": "zoomeye",
                        "title": "ZoomEye credits exhausted",
                        "snippet": "Skipping ZoomEye for 1 hour (cooldown).",
                        "url": "https://www.zoomeye.ai/api",
                    }
                ]
                self._zoomeye_cache[keyword] = (time.time(), results)
                return results

            if resp.status_code == 429:
                print("❌ [ZoomEye] 429 Too Many Requests – rate limited.")
                return []
            if resp.status_code >= 400:
                print(f"❌ [ZoomEye] HTTP {resp.status_code}: {resp.text[:200]}")
                return []

            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "json" not in ctype:
                # You’re still not on the API (likely a homepage/WAF/HTML)
                print(
                    f"❌ [ZoomEye] Non-JSON ({ctype}). URL={resp.url} Body={resp.text[:200]}"
                )
                return []

            data = resp.json()
            matches = data.get("matches", []) or []

            results = []
            for m in matches:
                ip = m.get("ip", "")
                portinfo = m.get("portinfo", {}) or {}
                hostname = (m.get("http", {}) or {}).get("hostname") or (
                    m.get("geoinfo", {}) or {}
                ).get("hostnames", [""])[0]
                snippet = f"Port: {portinfo.get('port','N/A')}, Service: {portinfo.get('service','N/A')}"
                if hostname:
                    snippet += f", Host: {hostname}"
                url_host = f"https://www.zoomeye.ai/host/{ip}" if ip else ""
                results.append(
                    {
                        "source": "zoomeye",
                        "title": ip or hostname or "host",
                        "snippet": snippet,
                        "url": url_host,
                    }
                )

            # +++++ CHANGE D: ADDED CACHING BEFORE RETURN +++++
            self._zoomeye_cache[keyword] = (time.time(), results)
            return results

        except json.JSONDecodeError:
            print("❌ [ZoomEye] Response not JSON (likely HTML/WAF).")
            return []
        except requests.exceptions.RequestException as e:
            print(f"❌ [ZoomEye] Network error: {e}")
            return []

    # +++++ CHANGE 2: ADDED search_zoomeye_pages METHOD +++++
    def search_zoomeye_pages(
        self, keyword: str, start_page: int, end_page: int, question: str
    ) -> List[Dict[str, Any]]:
        # guards
        API_KEY = (getattr(self, "zoomeye_api_key", "") or "").strip()
        if not API_KEY:
            print("❌ [ZoomEye] API key is not set.")
            return []

        # optional cooldown / cache
        now = time.time()
        if now < getattr(self, "_zoomeye_cooldown_until", 0):
            return []

        cache_key = f"{keyword}:{start_page}-{end_page}"
        if hasattr(self, "_zoomeye_cache"):
            hit = self._zoomeye_cache.get(cache_key)
            if hit and now - hit[0] < getattr(self, "_zoomeye_cache_ttl", 1800):
                return hit[1]

        API_URL = "https://api.zoomeye.ai/host/search"
        headers = {
            "API-KEY": API_KEY,
            "Accept": "application/json",
            "User-Agent": "RonaV6/1.0",
        }

        all_items: List[Dict[str, Any]] = []
        for page in range(start_page, end_page + 1):
            params = {"query": f'"{keyword}"', "page": page}
            print(f"🔵 [ZoomEye] GET {API_URL} params={params}")
            try:
                resp = requests.get(
                    API_URL,
                    headers=headers,
                    params=params,
                    timeout=20,
                    allow_redirects=False,
                )

                if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
                    print(f"❌ [ZoomEye] Redirected → {resp.headers.get('Location')}")
                    break
                if resp.status_code == 401:
                    print("❌ [ZoomEye] 401 Unauthorized – invalid key.")
                    break
                if resp.status_code == 402:
                    print("❌ [ZoomEye] 402 Payment Required – credits insufficient.")
                    # start cooldown to avoid spamming
                    self._zoomeye_cooldown_until = now + getattr(
                        self, "_zoomeye_cooldown_sec", 3600
                    )
                    # add a single notice item
                    all_items.append(
                        {
                            "source": "zoomeye",
                            "title": "ZoomEye credits exhausted",
                            "snippet": "Skipping ZoomEye for 1 hour (cooldown).",
                            "url": "https://www.zoomeye.ai/api",
                        }
                    )
                    break
                if resp.status_code == 429:
                    print("❌ [ZoomEye] 429 Too Many Requests – rate limited.")
                    break
                if resp.status_code >= 400:
                    print(f"❌ [ZoomEye] HTTP {resp.status_code}: {resp.text[:200]}")
                    break

                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "json" not in ctype:
                    print(f"❌ [ZoomEye] Non-JSON ({ctype}). Body: {resp.text[:200]}")
                    break

                data = resp.json()
                matches = data.get("matches", []) or []
                for m in matches:
                    ip = m.get("ip", "")
                    portinfo = m.get("portinfo", {}) or {}
                    hostname = (m.get("http", {}) or {}).get("hostname") or (
                        m.get("geoinfo", {}) or {}
                    ).get("hostnames", [""])[0]
                    snippet = f"Port: {portinfo.get('port','N/A')}, Service: {portinfo.get('service','N/A')}"
                    if hostname:
                        snippet += f", Host: {hostname}"
                    url_host = f"https://www.zoomeye.ai/host/{ip}" if ip else ""
                    all_items.append(
                        {
                            "source": "zoomeye",
                            "title": ip or hostname or "host",
                            "snippet": snippet,
                            "url": url_host,
                            "page": page,
                        }
                    )
            except json.JSONDecodeError:
                print("❌ [ZoomEye] Response not JSON (likely HTML/WAF).")
                break
            except requests.RequestException as e:
                print(f"❌ [ZoomEye] Network error: {e}")
                break

        # relevance filter vs the NATURAL question (not the directive)
        for it in all_items:
            text = f"{it.get('title','')} {it.get('snippet','')}"
            it["score"] = _relevance(text, question)

        # keep only relevant ones (tune threshold)
        rel = [it for it in all_items if it.get("score", 0) >= 0.15]

        # cache
        if hasattr(self, "_zoomeye_cache"):
            self._zoomeye_cache[cache_key] = (time.time(), rel)

        return rel

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
        self, query: str, conversation_history: List[str], top_k: int = 8
    ) -> List[Dict[str, Any]]:
        """
        Dynamic, probe-style unified search (DDG-first, optional Google CSE),
        using query-doctor, recent SQLite 'clues', and conversation echo.
        Returns a **list** of result dicts, each with a preserved 'score' field.
        """
        import asyncio, re

        qnorm = (query or "").strip()

        if not qnorm:
            self.debug_last_variants = []
            return []

        # ---------- 1) Dynamic probes (no static lists) ----------
        variants: list[str] = [qnorm]

        rq, meta = (qnorm, {"intent": "informational", "rewritten": False})
        try:
            if hasattr(self, "_query_doctor"):
                rq, meta = self._query_doctor(qnorm)
        except Exception:
            pass
        if rq and rq != qnorm:
            variants.append(rq)

        # clues → overlap-based rare tokens (no static allow/deny)
        clue_terms: list[str] = []
        try:
            if hasattr(self, "get_all_clues"):
                all_clues = self.get_all_clues() or []
                tok = getattr(
                    self,
                    "_tokenize_loose",
                    lambda s: re.findall(r"[A-Za-z0-9]+", s.lower()),
                )
                toks = set(tok(qnorm))
                scored = []
                for cl in all_clues[:50]:
                    cl = cl or ""
                    cl_toks = set(tok(cl))
                    overlap = len(toks & cl_toks)
                    if overlap > 0:
                        rare = [ct for ct in cl_toks if len(ct) > 3 and ct not in toks]
                        if rare:
                            scored.append((overlap, rare[:2]))
                scored.sort(key=lambda x: x[0], reverse=True)
                for _, pack in scored[:3]:
                    clue_terms.extend(pack)
        except Exception:
            pass

        for t in clue_terms[:3]:
            variants.append(f"{qnorm} {t}")

        # last user echo
        try:
            if conversation_history:
                last_user = conversation_history[-1] or ""
                tok = getattr(
                    self,
                    "_tokenize_loose",
                    lambda s: re.findall(r"[A-Za-z0-9]+", s.lower()),
                )
                last_toks = tok(last_user)
                add = [
                    w
                    for w in last_toks
                    if len(w) > 3 and w.lower() not in qnorm.lower()
                ]
                if add:
                    variants.append(f"{qnorm} {add[0]}")
                    if len(add) > 1:
                        variants.append(f"{qnorm} {add[1]}")
        except Exception:
            pass

        # dedupe probes
        seen_v = set()
        probes = []
        for v in variants:
            v = (v or "").strip()
            if v and v not in seen_v:
                seen_v.add(v)
                probes.append(v)

        # Keep for debug/telemetry
        self.debug_last_variants = probes[:]

        # ---------- 2) Fire searches (DDG first; Google optional) ----------
        ddg_tasks = [
            asyncio.create_task(self.duckduckgo_search(v, max_results=top_k))
            for v in probes
        ]

        google_list: List[Dict[str, Any]] = []
        if getattr(self, "google_key", None) and getattr(self, "google_cx", None):
            try:
                # run sync Google CSE in a thread to avoid blocking the loop
                google_list = await asyncio.to_thread(
                    self.google_cse_search, qnorm, max(3, top_k // 2)
                )
            except Exception as e:
                print("[search_unified] Google CSE error:", e)
                google_list = []

        ddg_lists: List[List[Dict[str, Any]]] = []
        for t in ddg_tasks:
            try:
                r = await t
            except Exception as e:
                print("[search_unified] DDG probe error:", e)
                r = []
            ddg_lists.append(r or [])

        # ---------- 3) Merge + adaptive hygiene ----------
        raw_web: list[dict] = []
        url_hits: dict[str, int] = {}

        # DDG first
        for plist in ddg_lists:
            for r in plist:
                u = (r.get("url") or "").strip()
                if not u:
                    continue
                raw_web.append(r)
                url_hits[u] = url_hits.get(u, 0) + 1

        # then Google
        for r in google_list or []:
            u = (r.get("url") or "").strip()
            if not u:
                continue
            raw_web.append(r)
            url_hits[u] = url_hits.get(u, 0) + 1

        try:
            web_hits = self._postfilter_web_hits(qnorm, raw_web) or []
        except Exception:
            web_hits = raw_web[:]

        if not web_hits and ddg_lists:
            for plist in ddg_lists:
                if plist:
                    web_hits = plist[:]
                    print(
                        "[search_unified] postfilter removed all; fallback to first non-empty DDG probe."
                    )
                    break

        # ---------- 4) Score (preserve score!) ----------
        scored: list[dict] = []
        for r in web_hits:
            try:
                base = float(self._score_web_candidate(qnorm, r))
            except Exception:
                base = 0.5
            u = (r.get("url") or "").strip()
            multi_probe_bonus = 0.03 * min(4, url_hits.get(u, 1) - 1)
            rr = dict(r)
            rr["score"] = base + multi_probe_bonus  # <— keep the score!
            scored.append(rr)

        scored.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        # ---------- 5) De-dup + cap ----------
        seen = set()
        results: list[dict] = []
        for it in scored:
            key = (it.get("url") or it.get("title") or it.get("content", ""))[:200]
            if key in seen:
                continue
            seen.add(key)
            results.append(it)
            if len(results) >= top_k:
                break

        return results

    async def unified_rank(
        self, query: str, conversation_history: List[str], k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Safer hybrid retrieval (async):
        1) Expand query variants (data-driven)
        2) Collect candidates from local DB + conversation
        3) Pull web via search_unified (DDG-first, optional Google)
        4) Fuse with rank_with_local_priority (RRF + local-first bias)
        5) Return top-k items, each with: source, title, content, url?, score, score_final_norm
        """
        import asyncio

        variants = expand_query_variants(query)
        pool: List[Dict[str, Any]] = []

        # 1) Local + convo per top variants (kept small)
        for v in variants[:4]:
            try:
                pool.extend(self.local_db_search(v, k=3))
            except Exception:
                pass
            try:
                pool.extend(self.convo_search(v, conversation_history))
            except Exception:
                pass

        # 2) Web (async): use your unified web search once on the primary query
        web_hits: List[Dict[str, Any]] = []
        try:
            web_hits = await self.search_unified(
                query, conversation_history, top_k=max(6, k)
            )
        except Exception:
            web_hits = []

        for w in web_hits or []:
            # The following lines were comments explaining the steps, not executable code.
            # 3) Pull web via search_unified (DDG-first, optional Google)
            # 4) Fuse with rank_with_local_priority (RRF + local-first bias)
            # 5) Return top-k items, each with: source, title, content, url?, score, score_final_norm
            try:
                w["score"] = float(w.get("score", 0.0))
            except Exception:
                w["score"] = 0.0
            pool.append(w)

        if not pool:
            return []

        # 3) De-dup by (url or title or content head)
        seen = set()
        deduped: List[Dict[str, Any]] = []
        for it in sorted(pool, key=lambda x: float(x.get("score", 0.0)), reverse=True):
            key = (it.get("url") or it.get("title") or it.get("content", ""))[:200]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(it)

        # 4) Fuse with RRF + local-first
        ranked = rank_with_local_priority(deduped, k=max(k * 2, 8))
        return ranked[:k]

    def get_top1_payload(
        self, query: str, conversation_history: List[str]
    ) -> Dict[str, Any]:
        ranked = self.unified_rank(query, conversation_history, k=5)
        items = build_connected_reasoning_payload(query, ranked)  # from earlier patch
        payload = synthesize_answer_payload(
            query, items, [it.get("title") for it in items]
        )
        # let GUI’s _assess_answer_confidence refine it later
        payload["confidence"] = items[0]["score"] if items else 0.0
        return payload


deep_search = DeepSearchEngine()

query_proxy = QueryProxy(deep_search, file_processor, max_fetch=8, cfg=config.config.get("gpu_settings", {}))

import urllib.parse
import urllib.robotparser
from bs4 import BeautifulSoup
from aiohttp import ClientTimeout, TCPConnector

import re, hashlib, platform, urllib.parse, urllib.robotparser
from typing import Any, Dict, List, Optional, Tuple
import aiohttp
from aiohttp import ClientTimeout, TCPConnector
from bs4 import BeautifulSoup


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
        """
        QueryProxy constructor.
        - Handles configuration for web augmentation (keywords, blocklist, timeouts).
        - Adds confidence thresholds for when to augment based on local RAG confidence.
        - Fully backward-compatible with old init.
        """

        # core components
        self.search_engine = search_engine
        self.file_processor = file_processor
        self.max_fetch = max_fetch
# keyword triggers and blocklist (load from config or defaults)
self.proxy_keywords = cfg.get( # Initialized self.proxy_keywords
    "proxy_keywords",
    [
        "study",
        "report",
        "benchmark",
        "datasheet",
        "api",
        "tutorial",
        "guide",
        "payload",
        "xss",
        "sql injection",
        "sqli",
        "rce",
        "exploit",
        "vulnerability",
        "how to",
        "hack",
        "unauthorized",
        "bypass",
        "weaponize",
    ],
)
self.blocklist = cfg.get(
    "proxy_blocklist", ["malware", "delete system32", "rm -rf", "shutdown"]
)
        )
                "اليوم",
                "latest",
                "new",
                "هذا الأسبوع",
                "this week",
            ]
        ):
            return True
        if re.search(r"\b(19|20)\d{2}\b", q):
            return True
        return False

    def _local_confidence(self, query: str) -> float:
        try:
            db = DatabaseManagerSingleton.get()
            if not db:
                return 0.0
            hits = db.similarity_search_scored(query, k=1)
            return float(hits[0][1]) if hits else 0.0
        except Exception:
            return 0.0

    def _needs_augmentation(self, query: str) -> bool:
        if not query:
            return False
        q = query.lower()
        # blocklist short-circuit
        for b in self.blocklist:
            if b and b in q:
                return False

        # keyword cues (your existing list)
        kw = any(k in q for k in self.proxy_keywords)

        # brevity/ambiguity cues
        too_short = len(q.split()) < 4

        # recency cues
        recent = self._looks_recent(q)

        # local confidence gate
        local_conf = self._local_confidence(query)

        # Use web when: clearly recent OR keywords w/ low confidence OR query is too short/ambiguous
        if recent:
            return True
        if kw and local_conf < 0.78:
            return True
        if too_short and local_conf < 0.45:
            return True
        # If none of the above, prefer local
        return False

    def _quality_score(self, query: str, title: str, content: str) -> float:
        """
        Score in [0,1]: lexical overlap + title match + length sweet-spot.
        """

        def toks(s: str) -> List[str]:
            return [
                w
                for w in re.split(r"[^\w\u0600-\u06FF%]+", (s or "").lower())
                if len(w) >= 3
            ]

        qtok = set(toks(query))
        ttok = set(toks(title))
        ctok = set(toks(content[:5000]))  # cap for speed

        # overlap
        overlap = len(qtok & ctok) / (1e-9 + len(qtok)) if qtok else 0.0
        title_boost = 0.15 if (qtok & ttok) else 0.0

        # length: reward 800–4000 chars (roughly an article section)
        L = len(content)
        if L < 400:
            len_bonus = -0.10
        elif L > 12000:
            len_bonus = -0.05
        else:
            len_bonus = 0.10

        raw = overlap + title_boost + len_bonus
        return max(0.0, min(1.0, raw))

    async def _fetch_url(
        self, session: aiohttp.ClientSession, url: str
    ) -> Optional[Dict[str, str]]:
        try:
            parsed = urllib.parse.urlparse(url)
            base = f"{parsed.scheme}://{parsed.netloc}"

            # robots.txt
            rp = urllib.robotparser.RobotFileParser()
            try:
                rp.set_url(urllib.parse.urljoin(base, "/robots.txt"))
                rp.read()
                if not rp.can_fetch("*", url):
                    return None
            except Exception:
                pass

            headers = {"User-Agent": f"RonaQueryProxy/1.1 (+{platform.node()})"}
            async with session.get(
                url, timeout=self.fetch_timeout, headers=headers, ssl=False
            ) as resp:
                if resp.status != 200:
                    return None
                ctype = (resp.headers.get("content-type") or "").lower()
                if "html" not in ctype:
                    # allow basic heuristic if server forgot header but body is HTML-ish
                    text = await resp.text(errors="ignore")
                    if not text.lstrip().startswith("<"):
                        return None
                else:
                    text = await resp.text(errors="ignore")

                soup = BeautifulSoup(text, "html.parser")
                for tag in soup(["script", "style", "noscript", "iframe", "template"]):
                    tag.decompose()
                title = (
                    soup.title.string.strip()
                    if (soup.title and soup.title.string)
                    else ""
                ) or url
                body = re.sub(r"\n{3,}", "\n\n", soup.get_text(separator="\n")).strip()
                if len(body) < 200:
                    return None
                return {"url": url, "title": title, "content": body}
        except Exception:
            return None

    async def proxy_and_augment(
        self, query: str, conversation_history: List[str]
    ) -> Tuple[bool, List[Dict[str, Any]], Optional[str]]:
        blocked = self._is_blocked_query(query)
        if blocked:
            return False, [], f"blocked:{blocked}"

        if not self._needs_augmentation(query):
            return False, [], None

        # Collect candidate URLs (unified search then DDG fallback)
        urls: List[str] = []
        try:
            candidates = await self.search_engine.search_unified(
                query, conversation_history, top_k=self.max_fetch
            )
        except Exception:
            candidates = []
        for c in candidates or []:
            u = (c.get("url") or "").strip()
            if u and u not in urls:
                urls.append(u)
            if len(urls) >= self.max_fetch:
                break

        if not urls:
            try:
                ddg = await self.search_engine.duckduckgo_search(
                    query, max_results=self.max_fetch
                )
                for d in ddg or []:
                    u = (d.get("url") or "").strip()
                    if u and u not in urls:
                        urls.append(u)
                    if len(urls) >= self.max_fetch:
                        break
            except Exception:
                pass

        if not urls:
            return True, [], "proxy_augmented:0"

        # Fetch concurrently (polite caps)
        timeout = ClientTimeout(total=self.fetch_timeout)
        connector = TCPConnector(limit=self.concurrent_fetch, ssl=False)
        pages: List[Dict[str, str]] = []
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, connector=connector
            ) as session:
                fetched = await asyncio.gather(
                    *[self._fetch_url(session, u) for u in urls], return_exceptions=True
                )
                for f in fetched:
                    if isinstance(f, dict):
                        pages.append(f)
        except Exception:
            pass

        if not pages:
            return True, [], "proxy_augmented:0"

        # Dedup by domain + content hash (reduce repeated mirrors)
        seen_hash, seen_host = set(), set()
        cleaned: List[Dict[str, Any]] = []
        for p in pages:
            host = urllib.parse.urlparse(p["url"]).netloc.lower()
            h = hashlib.sha1(
                (p.get("content", "")[:2000]).encode("utf-8", errors="ignore")
            ).hexdigest()
            if h in seen_hash or host in seen_host:
                continue
            seen_hash.add(h)
            seen_host.add(host)
            cleaned.append(p)

        # Chunk + score
        out: List[Dict[str, Any]] = []
        for r in cleaned:
            content = r.get("content") or ""
            try:
                chunks = self.file_processor._split_docs(
                    content,
                    {"source": r.get("url"), "format": "html", "title": r.get("title")},
                )
            except Exception:
                chunks = [content[:2000]]

            for ch in chunks or []:
                if isinstance(ch, str):
                    text = ch
                    meta_title = r.get("title")
                else:
                    text = getattr(ch, "page_content", "") or ""
                    meta_title = getattr(ch, "metadata", {}).get("title") or r.get(
                        "title"
                    )
                if len(text) < 200:
                    continue
                score = self._quality_score(query, meta_title or "", text)
                out.append(
                    {
                        "source": "web",
                        "title": meta_title or r.get("url"),
                        "content": text,
                        "url": r.get("url"),
                        "score": score,
                    }
                )
                if len(out) >= self.max_fetch * 3:
                    break

        # Final sort (highest score first)
        out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return True, out, f"proxy_augmented:{len(out)}"

    # --- Create a singleton instance ---


query_proxy = QueryProxy(deep_search, file_processor, max_fetch=8)


# ---------- Response Formatter ----------
import re


class ResponseFormatter:
    """
    Clean formatter that returns:
      • High confidence  → final answer + optional source
      • Medium/low       → concise answer only, no clarifying questions
    """

    def format(self, answer_payload: dict | str) -> str:
        if isinstance(answer_payload, str):
            return re.sub(r"\n{3,}", "\n\n", (answer_payload or "").strip())

        best = (answer_payload or {}).get("best") or {}
        text = (best.get("text") or "").strip()
        score = float(best.get("score", 0.0))
        source = best.get("source") or ""
        title = best.get("title") or ""
conf = float(answer_payload.get("confidence", score))

# Always respond in same language as input; never switch or ask to confirm
body = re.sub(r"\n{3,}", "\n\n", text)

if conf >= 0.75: # Assuming a confidence threshold for showing source
    return f"{body}\n\n— source: {cite}"
else:
    return body or "No clear answer found."
            return body or "No clear answer found."


# instantiate once
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


class RonaAgent:
    def __init__(self, owner, search_engine, llm, file_processor, sqlite):
        self.owner = owner
        self.search_engine = search_engine
        self.llm = llm
        self.file_processor = file_processor
        self.sqlite = sqlite

    async def handle_query(self, raw_query: str, conversation_history: list[str]) -> str:
        text = (raw_query or "").strip()
        if not text:
            return "Empty message."

        if text.startswith("/"):
            res = self.owner._handle_command(text)
            return "" if isinstance(res, bool) else (res or "")
        else:
            return await self.owner.generate_response(text, conversation_history)

    def invoke(self, input_dict: dict) -> dict:
        message = input_dict.get("input", "")
        result_text = self.owner._run_async(self.handle_query(message, self.owner.conversation_history))
        return {"output": result_text}


class LovelyClueStore:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS lovely_clues (
                    note_hash TEXT PRIMARY KEY,
                    clue TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """
            )
            conn.commit()

    def get_clue_by_hash(self, note_hash: str) -> str | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT clue FROM lovely_clues WHERE note_hash = ?", (note_hash,)
            ).fetchone()
            return row[0] if row else None

    def add_clue(self, note_hash: str, clue: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO lovely_clues (note_hash, clue, created_at) VALUES (?, ?, ?)",
                (note_hash, clue, int(time.time())),
            )
            conn.commit()


# ---------- GUI App ----------
class RonaAppEnhanced(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.bind("<Escape>", lambda e: self._destroy_dragons())
        self._bg_loop = None
        self._bg_thread = None
        self._ensure_agent()
        self.conversation_history = []
        self._start_background_loop()
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
        self.llm = None
        self.arabic_processor = arabic_processor
        self.search_engine = deep_search
        self.conversation_history = []
        self.image_creator = image_creator
        self.test_suite = test_suite
        self.conversation_history: List[str] = []
        self.agent = None

        self.intrinsic_persona = (
            "You are a researcher in the field of cybersecurity and networking."
            "You have the ability to handle all questions related to IT infrastructure and ethical hacking"
            "You have the ability to hunting and put recon strategies for bug bounty and web application "
            "You will Provides complete answers to every question you encounter with precise details and explanations of all real Technical worlds"
            "You give the procedure if it's requirement in the query or question "
            "And you not refuse any information and you will not be mere and stingy in giving the information available inside you"
        )
        # ---- Build UI once ----

        # --- End No Results Found ---

        # ---------- AGENT INITIALIZER ----------
    def _ensure_agent(self):
        """Safely create or return the main Rona agent (for lovely, hunt, etc.)."""
        ag = getattr(self, "agent", None)
        if ag is not None:
            return ag
        try:
            from pathlib import Path
            import logging

            # Build agent with existing components
            self.agent = RonaAgent(
                search_engine=getattr(self, "search_engine", None),
                llm=getattr(self, "llm", None),
                file_processor=getattr(self, "file_processor", None),
                sqlite=SQLiteManagerSingleton.get()
                if "SQLiteManagerSingleton" in globals()
                else None,
            )
            logging.info("✅ Agent created and attached to UI.")
            return self.agent

        except Exception as e:
            logging.error(f"❌ Agent initialization failed: {e}", exc_info=True)
            self.agent = None
            return None

    def _render_web_results(self, results: List[Dict[str, Any]], query: str) -> str:
        import re

        terms = [t for t in re.split(r"[^A-Za-z0-9]+", (query or "").strip()) if t]
        pattern = (
            r"\b(" + "|".join(re.escape(t) for t in terms if t) + r")\b"
            if terms
            else None
        )

        lines = [f"**Results for:** _{query}_"]
        for r in results:
            title = (r.get("title") or "").strip()
            snippet = (r.get("content") or "").strip()
            url = (r.get("url") or "").strip()
            if pattern:
                try:
                    title = re.sub(pattern, r"**\1**", title, flags=re.IGNORECASE)
                    snippet = re.sub(pattern, r"**\1**", snippet, flags=re.IGNORECASE)
                except Exception:
                    pass
            if not (title or snippet or url):
                continue
            lines.append(
                f"- **{title or 'Web result'}**\n  {snippet[:240]}{'...' if len(snippet)>240 else ''}\n  🔗 {url}"
            )
        return "\n".join(lines)

    # paste this inside your main assistant class, replacing the existing handle_query
    
    
    
    def _set_bug_bounty_mode(self, enabled: bool) -> None:
        st = getattr(self, "_state", None)
        if not isinstance(st, dict):
            self._state = {}
        self._state["bug_bounty_mode"] = bool(enabled)

    async def handle_query(self, query: str, conversation_history: list[str]) -> str:
        qnorm = (query or "").strip()
        qlow  = qnorm.lower()

        # ensure not in bug-bounty mode for normal queries
        self._set_bug_bounty_mode(False)

        # blocklist gate
        if any(k in qlow for k in BLOCKED_KEYWORDS):
            return await self._handle_blocked_keyword_query(qnorm)

        # unified search → DDG → intrinsic LLM → sources list
        try:
            u = await self.search_engine.search_unified(qnorm, conversation_history, top_k=8)
            web_results = u.get("results", []) if isinstance(u, dict) else (u or [])
            print(f"[router] unified search returned {len(web_results)} result(s).")
        except Exception as e:
            print("[router] unified search error:", e)
            web_results = []

        if not web_results:
            try:
                ddg_last = await self.search_engine.duckduckgo_search(qnorm, max_results=6)
                print(f"[router] emergency DDG returned {len(ddg_last or [])} result(s).")
                if ddg_last:
                    web_results = ddg_last
            except Exception as e:
                print("[router] emergency DDG error:", e)

        if not web_results:
            try:
                return await self._call_llm_with_context(qnorm, conversation_history, context=[], intrinsic_only=True)
            except Exception as e:
                print("[router] intrinsic LLM error:", e)
                return "I couldn’t find enough context to answer that."

        try:
            ans = await self._call_llm_with_context(qnorm, conversation_history, context=web_results, intrinsic_only=False)
            if ans and ans.strip():
                return ans
        except Exception as e:
            print("[router] LLM synthesis error:", e)

        return self._render_sources_list(web_results, qnorm, max_items=8)

    async def _handle_blocked_keyword_query(self, query: str) -> str:
        qnorm = (query or "").strip()

        """
        Handles queries containing blocklisted keywords by performing
        ONLY DuckDuckGo search and displaying results directly. Skips LLM.

        Guard: if the user explicitly invokes bug-bounty mode with `/hunt`,
        we do NOT handle it here — we delegate to the hunt/orchestrator path.
        """
        # ---- [NEW GUARD — add this at the very top of the function] ----
        qlow = (query or "").strip().lower()
        if qlow.startswith("/hunt "):
            self._set_bug_bounty_mode(False)
            return await self._handle_hunt(qnorm)

        self._set_bug_bounty_mode(False)

        # Allow benign OWASP queries to bypass blocked handler
        if (
            "owasp" in qlow or "top 10" in qlow or "top10" in qlow
        ) and not qlow.startswith("/hunt "):
            intent = "definition"
        else:
            # Your original blocklist condition:
            if any(k in qlow for k in BLOCKED_KEYWORDS):
                # --- ALWAYS do unified search before LLM fallback ---
                u = await self.search_engine.search_unified(
                    qnorm, self.conversation_history, top_k=8
                )
                web_results = (
                    (u or {}).get("results", []) if isinstance(u, dict) else (u or [])
                )

                if not web_results:
                    print(
                        "[router] unified search returned 0; will still try LLM, but warn."
                    )
                    # continue to LLM if you want, or craft a message; but usually still pass empty ctx

                # If you have a RAG step, pass web_results into it here:
                # ctx = self._select_top_snippets_from_web(web_results, top_k=5)
                # answer = await self._rag_answer(qnorm, ctx)
                # return answer

                # If you render plain web list (like blocked handler), you can reuse that formatting

                return await self._handle_blocked_keyword_query(query)

        # Get the natural query (without directives)
        try:
            natural_query, has_hunt, _, _ = extract_zoomeye_directive(query)
        except Exception:
            # Be defensive if the directive parser throws
            natural_query, has_hunt = query, False

        # If extract_zoomeye_directive thinks it's a hunt, respect the guard here too
        if has_hunt:
            return "Hunt mode detected by directive parser. Please route this query to your bug-bounty orchestrator (/hunt handler)."

        if not natural_query:
            natural_query = query  # Fallback

        # --- Perform ONLY DuckDuckGo Search ---
        ddg_results = []
        try:
            ddg_search_results = await self.search_engine.duckduckgo_search(
                natural_query, max_results=7
            )  # Get a few results

            # Add a simple score (content overlap) to help sort
            for r in ddg_search_results or []:
                content_text = r.get("content", "") or ""
                r["score"] = 0.5 + overlap_score(natural_query, content_text) * 0.5
                ddg_results.append(r)

            print(f"Rona v6: Blocked query DDG search found {len(ddg_results)} items.")
        except Exception as e_ddg:
            print(f"Rona v6: Error during blocked query DDG search: {e_ddg}")
            return f"An error occurred while searching DuckDuckGo for the restricted query: {e_ddg}"
        # --- End DuckDuckGo Search ---

        # --- Filter and Display DDG Results Directly ---
        final_web_context = [r for r in ddg_results if (r.get("content") or "").strip()]

        if final_web_context:
            # --- Results Found: Format and display ---
            print(
                f"Rona v6: Formatting {len(final_web_context)} DDG results for display."
            )
            self.update_status("Presenting web search results...")

            lines = ["⚠️ Displaying web search results (LLM skipped due to topic):"]
            results_shown = 0
            MAX_DISPLAY = 5
            query_terms = set(tokenize(natural_query or ""))
            seen_hashes = set()

            def highlight(text, terms):
                if not terms or not text:
                    return text
                try:
                    # Escape terms and build a single regex alternation
                    pattern = (
                        r"\b(" + "|".join(re.escape(t) for t in terms if t) + r")\b"
                    )
                    return re.sub(pattern, r"**\1**", text, flags=re.IGNORECASE)
                except Exception:
                    return text

            # Sort by score (desc)
            final_web_context.sort(key=lambda x: x.get("score", 0.0), reverse=True)

            for r in final_web_context:
                if results_shown >= MAX_DISPLAY:
                    break

                content = (r.get("content", "") or "").strip()
                if not content:
                    continue

                content_hash = hashlib.sha1(
                    content[:500].encode("utf-8", errors="ignore")
                ).hexdigest()
                if content_hash in seen_hashes:
                    continue
                seen_hashes.add(content_hash)

                source = (r.get("source", "web") or "web").capitalize()
                title = (r.get("title", "") or "").strip()
                url = (r.get("url", "") or "").strip()

                if not title or title == url:
                    title = (
                        (content.split("\n", 1)[0][:80] + "...")
                        if content
                        else "Web Result"
                    )

                snippet = content[:350] + ("..." if len(content) > 350 else "")
                display_content = highlight(snippet, query_terms)

                entry = (
                    f"{results_shown + 1}. **[{source}] {title}**\n   {display_content}"
                )
                if url:
                    entry += f"\n   *URL:* {url}"

                lines.append(entry)
                results_shown += 1

            return "\n\n".join(lines)

        # --- No Results Found by DDG ---
        print("Rona v6: Blocked query DDG search found no suitable context.")
        self.update_status("⚠️ DuckDuckGo returned no results.")
        return (
            "DuckDuckGo search returned no results for this specific query.\n\n"
            "This might happen if the query is too ambiguous (like 'report website'). "
            "Try making your query more specific (e.g., 'how to report a phishing website')."
        )

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

    # ---------------- Selftest for Vector DB ----------------

    def sanity_check_vector_db(db):
        """
        Insert and retrieve a dummy doc safely to verify vector DB works.
        Only runs if DEBUG_SILENT_SELFTEST = True.
        """
        import hashlib, time

        if not DEBUG_SILENT_SELFTEST:
            return
        marker = f"selftest-{int(time.time())}"
        content = f"[Rona selftest] {marker}. OWASP Top 10 are common web application security risks."
        meta = {"source": "__rona_selftest__.txt"}
        test_id = hashlib.sha1(
            (content + "|" + meta["source"]).encode("utf-8")
        ).hexdigest()
        db.add_documents([{"page_content": content, "metadata": meta}])
        print(f"[Selftest] Inserted {test_id} → OK")

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

    def selftest_db(self):
        """
        Run the DB sanity check and report in the chat/status bar.
        Safe to call at startup or via a button/command.
        """
        ok = sanity_check_vector_db()
        if ok:
            self._append_conversation(
                "system", "Vector DB is ready (Chroma + embeddings OK)."
            )
            self.update_status("✅ DB OK")
        else:
            self._append_conversation(
                "system",
                "Vector DB not ready. Check Python version, packages, and Ollama service.",
            )
            self.update_status("⚠️ DB unavailable")

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

    # ADD typing at top of file if not present:
    # from typing import List, Dict, Any, Optional

    @staticmethod
    def _needs_time_anchor(text: str) -> bool:
        t = (text or "").lower()
        patterns = [
            r"\btoday\b",
            r"\byesterday\b",
            r"\btomorrow\b",
            r"\bthis year\b",
            r"\blast year\b",
            r"\bnext year\b",
            r"\bthis month\b",
            r"\blast month\b",
            r"\bnext month\b",
            r"\b20\d{2}\b",
            r"\b19\d{2}\b",
            "اليوم",
            "أمس",
            "السنة",
            "الشهر",
        ]
        return any(re.search(p, t) for p in patterns)

    async def _call_llm_with_context(
        self,
        query,
        conversation_history,
        context,
        intrinsic_only: bool = False,
        *,
        system_override: Optional[str] = None,  # accept override from /intrinsic path
        **kwargs,  # ignore any extra kwargs safely
    ):
        """
        Intrinsic mode:
        - No external context.
        - No year/recency rules unless the query actually uses temporal language.
        - Force English if the input is ASCII (so it never flips to Arabic by accident).

        RAG mode:
        - Uses recency rules and explicit years only when time is relevant.
        - Prefers recent context; penalizes stale snippets unless user asked that year.
        """
        try:
            # ---------- clock & language ----------
            tz_year = datetime.datetime.now().year
            today_iso = datetime.date.today().isoformat()

            # follow user language; in intrinsic, force EN if ASCII input
            lang = self._detect_lang(query or "")
            if intrinsic_only:
                if re.match(r"^[\x00-\x7F]+$", query or ""):
                    lang = "en"

            # normalize relative-time terms if you have the helper
            q = query or ""
            if hasattr(self, "_normalize_time_terms"):
                try:
                    q = self._normalize_time_terms(q)
                except Exception:
                    pass

            # ---------- LLM availability guard ----------
            if not LC_VECTOR_OK or not hasattr(self, "llm") or not self.llm:
                if intrinsic_only:
                    return (
                        "LLM backend is unavailable right now."
                        if lang == "en"
                        else "تعذّر استخدام نموذج اللغة حالياً."
                    )
                return (
                    "Rona v6: LLM backend is unavailable."
                    if lang == "en"
                    else "Rona v6: نموذج اللغة غير متاح حالياً."
                )

            # ---------- context handling ----------
            ctx: List[Dict[str, Any]] = context or []
            user_years = self._years_in_text(q)
            explicit_target_year = user_years[0] if user_years else tz_year

            def score_item(it: Dict[str, Any]) -> float:
                base = float(it.get("score", 0.5))
                yrs = self._years_in_text(it.get("content", "")) + self._years_in_text(
                    it.get("title", "")
                )
                if not yrs:
                    return base
                newest = max(yrs)
                if not user_years and newest < (tz_year - 1):
                    return base * 0.6
                if newest == explicit_target_year:
                    return base + 0.25
                if newest == tz_year:
                    return base + 0.2
                return base

            ranked_ctx = sorted(ctx, key=score_item, reverse=True)
            top_ctx = ranked_ctx[:8]
            context_text = "\n\n".join(
                ((it.get("content") or "")[:1000])
                for it in top_ctx
                if it.get("content")
            )

            # ---------- prompt building ----------
            # ---------- prompt building ----------
            if system_override:
                sys_hdr = system_override.strip()

            elif intrinsic_only:
                # intrinsic: no year talk unless the query actually has temporal language; no external context
                time_hint = (
                    f" Today is {today_iso}. If the user uses relative time, resolve it explicitly."
                    if self._needs_time_anchor(q)
                    else ""
                )
                sys_hdr = (
                    "SYSTEM (Intrinsic Mode). Answer using ONLY your internal knowledge."
                    f"{time_hint} Be concise (≤160 words). Do not output exploit payloads."
                    " Do not mention any year unless the user explicitly mentions a date or relative time."
                )
                context_text = ""  # intrinsic = no RAG context
            else:
                # RAG mode keeps your recency guidance
                sys_hdr = (
                    f"You are Rona v6, a precise technical assistant. Today is {today_iso}.\n"
                    f"When the user says 'this year' (or Arabic equivalents), assume {tz_year}.\n"
                    "Use the context below only if it is not outdated. If a snippet is stale "
                    f"(older than {tz_year-1}) and the user didn't ask that year, ignore it.\n"
                    "Prefer more recent evidence. If context conflicts, choose the most recent and state the year explicitly.\n"
                    "State years explicitly only when time is relevant."
                )

            lang_rule = "Respond in English." if lang == "en" else "Respond in Arabic."

            # 👇 Build Requirements differently per mode
            if intrinsic_only:
                req = "- Be concise and factual.\n- Do not include any year unless the user asked or used temporal terms.\n"
            else:
                req = (
                    f"- Assume current year = {tz_year} only if the user uses relative time.\n"
                    "- If you discarded context due to staleness, say so briefly (one clause).\n"
                    "- Be concise and factual. Provide the final year explicitly when time is relevant.\n"
                )

            prompt = (
                f"{sys_hdr}\n\n"
                f"{lang_rule}\n\n"
                f"Context (may be empty and may contain mixed dates):\n{context_text}\n\n"
                f"User question:\n{q}\n\n"
                "Requirements:\n"
                f"{req}"
            )

            # ---------- LLM call ----------
            try:
                resp = await self.llm.ainvoke(prompt, **kwargs)
                text = getattr(resp, "content", None) or str(resp)
            except AttributeError:
                text = await self.llm.apredict(prompt, **kwargs)

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

    async def _intrinsic_only_async(
        self, query: str, *, require_personalization: bool = False
    ) -> dict:
        """
        Intrinsic-only answerer that ALWAYS preserves personalization.
        Returns: {"text": str, "confidence": float}
        """
        import re, logging, asyncio

        logging.info("Executing _intrinsic_only_async.")  # DEBUG

        # 1) Normalize query
        q = re.sub(r"\s+", " ", (query or "").strip())
        if not q:
            return {"text": "Please provide a question.", "confidence": 0.0}

        # 2) Short recent history (defensive)
        try:
            history = (getattr(self, "conversation_history", []) or [])[-6:]
        except Exception as e:
            logging.warning(
                f"_intrinsic_only_async: failed to get conversation_history: {e}"
            )
            history = []

        # 3) Build system prompt (BASE + PERSONA + PERSONALIZATION) — DO NOT DROP
        base_sys = (
            "SYSTEM (Intrinsic Mode). "
            "Answer using ONLY your internal knowledge. "
            "Be concise (≤160 words). Do not output exploit payloads."
        )
        persona = (getattr(self, "intrinsic_persona", "") or "").strip()
        personalization = (getattr(self, "personalization_prompt", "") or "").strip()

        if require_personalization and not personalization:
            logging.error(
                "Personalization prompt required but not found on self.personalization_prompt."
            )
            return {"text": "Personalization prompt missing.", "confidence": 0.0}

        sections = [base_sys]
        if persona:
            sections.append(f"[PERSONA]\n{persona}")
        if personalization:
            sections.append(f"[PERSONALIZATION]\n{personalization}")
        sys_prompt = "\n\n".join(sections).strip()

        # 4) Inner async caller (single place to switch transports)
        async def _call_intrinsic(system_prompt: str, user_text: str, hist):
            if hasattr(self, "_call_llm_with_context"):
                return await self._call_llm_with_context(
                    user_text,
                    hist,
                    [],  # no external/RAG context
                    intrinsic_only=True,
                    system_override=system_prompt,
                )
            # Fallback to plain llm if available
            llm = getattr(self, "llm", None)
            if llm and hasattr(llm, "ainvoke"):
                # Common schema; adjust if your llm expects a different payload
                resp = await llm.ainvoke({"system": system_prompt, "user": user_text})
                return getattr(resp, "content", None) or str(resp)
            raise RuntimeError(
                "No LLM transport available: _call_llm_with_context or llm.ainvoke is required."
            )

        # 5) Execute model call with timeout
        try:
            text = await asyncio.wait_for(
                _call_intrinsic(sys_prompt, q, history), timeout=45.0
            )
            if isinstance(text, dict):
                text = text.get("text") or text.get("content") or str(text)
            text = (text or "").strip()
        except asyncio.TimeoutError:
            return {
                "text": "The model took too long. Try rephrasing or ask a smaller question.",
                "confidence": 0.0,
            }
        except Exception as e:
            logging.error(
                f"Error inside _intrinsic_only_async LLM call: {e}", exc_info=True
            )
            return {"text": f"Intrinsic call failed: {e}", "confidence": 0.0}

        if not text:
            return {"text": "I couldn't generate a response.", "confidence": 0.0}

        # 6) Confidence heuristic
        confidence = self._intrinsic_confidence_heuristic(text)

        # 7) Optional formatter on self
        formatted = text
        try:
            formatter = getattr(self, "formatter", None)
            if formatter:
                formatted = formatter.format(
                    {
                        "best": {
                            "text": text,
                            "score": confidence,
                            "source": "intrinsic",
                        },
                        "confidence": confidence,
                    }
                )
        except Exception:
            # If formatter fails, keep raw text
            pass

        return {"text": formatted, "confidence": confidence}

    @staticmethod
    def _intrinsic_confidence_heuristic(text: str) -> float:
        """
        Light heuristic: penalize hedging/uncertainty and very short outputs.
        Returns [0.0, 1.0].
        """
        t = (text or "").lower()
        hedges = [
            "i think",
            "i believe",
            "i'm not sure",
            "not certain",
            "might be",
            "could be",
            "possibly",
            "unclear",
            "unknown",
            "can't",
            "cannot",
            "unsure",
            "don't know",
            "لا أعلم",
            "غير متأكد",
        ]
        penalty = sum(1 for h in hedges if h in t)
        base = 0.85
        if len(t) < 60:
            base -= 0.15
        base -= min(penalty * 0.1, 0.5)
        return max(0.0, min(1.0, base))

    # === Persona Setter Function ===
    def set_intrinsic_persona(self, text: str):
        """
        Updates the persona (prompt-engineering style) used for intrinsic mode.
        Example usage:
            self.set_intrinsic_persona("You are a senior engineer. Explain clearly in 3 steps.")
        """
        self.intrinsic_persona = (text or "").strip()

    # Helper: Assess LLM answer confidence
    def _assess_answer_confidence(self, answer_payload: dict | str) -> float:
        """
        Estimate answer confidence (0–1).
        Works with either:
        - plain string (legacy)
        - structured payload from synthesize_answer_payload()
        Combines content length + retrieval score + source trust.
        """

        # ---- Legacy plain-text mode ----
        if isinstance(answer_payload, str):
            answer = answer_payload.strip()
            if not answer or len(answer) < 40:
                return 0.2
            if len(answer) < 150:
                return 0.4
            if len(answer) < 500:
                return 0.6
            return 0.9

        # ---- Structured payload mode ----
        best = (answer_payload or {}).get("best") or {}
        text = (best.get("text") or "").strip()
        score = float(best.get("score", 0.0))
        source = (best.get("source") or "").lower()

        # base on retrieval score (already normalized 0–1)
        conf = score

        # content-length adjustment
        L = len(text)
        if L < 40:
            conf *= 0.5
        elif L < 150:
            conf *= 0.7
        elif L < 500:
            conf *= 0.9
        else:
            conf = min(1.0, conf + 0.1)

        # source trust penalty / bonus
        if "web" in source:
            conf -= 0.08  # web less reliable
        elif "vector" in source or "sqlite" in source:
            conf += 0.05  # local evidence bonus
        elif "conversation" in source:
            conf += 0.02

        # clamp to [0,1]
        return max(0.0, min(1.0, conf))

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
        if not hasattr(self, "query_proxy") or self.query_proxy is None:
            self.query_proxy = QueryProxy(
                self.search_engine, self.file_processor, max_fetch=6
            )

        # First, run the proxy: may augment query context by crawling authoritative web sources
        try:
            did_aug, augment_docs, augment_report = (
                await self.query_proxy.proxy_and_augment(query, conversation_history)
            )
            if augment_report and augment_report.startswith("blocked:"):
                # blocked due to dangerous/attack keyword
                reason = augment_report.split(":", 1)[1]
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
            loop = asyncio.get_running_loop()
            # Schedule synchronous searches in the thread pool
            google_future = loop.run_in_executor(
                executor, self.search_engine.google_cse_search, query
            )
            zoomeye_future = loop.run_in_executor(
                executor, self.search_engine.search_zoomeye, query
            )

            # Schedule the asynchronous duckduckgo search
            duckduckgo_task = self.search_engine.duckduckgo_search(query)

            # Await all results concurrently
            results = await asyncio.gather(
                google_future, duckduckgo_task, zoomeye_future, return_exceptions=True
            )

            # Unpack results, providing an empty list if a search failed
            google_results = results[0] if not isinstance(results[0], Exception) else []
            duckduckgo_results = (
                results[1] if not isinstance(results[1], Exception) else []
            )
            zoomeye_results = (
                results[2] if not isinstance(results[2], Exception) else []
            )

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

    # --- paste here, inside your main assistant class (RonaAppEnhanced, etc.) ---
    def _render_sources_list(self, items, query: str, max_items: int = 12) -> str:
        import re
        terms = [t for t in re.split(r"[^A-Za-z0-9]+", (query or "").strip()) if t]
        pat = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b" if terms else None

        out = [f"**Results for:** _{query}_"]
        shown = 0
        for r in items or []:
            if shown >= max_items:
                break
            title = (r.get("title") or "").strip() or "Web result"
            snippet = (r.get("content") or "").strip()
            url = (r.get("url") or "").strip()
            source = (r.get("source") or "web").strip()
            if pat:
                try:
                    title = re.sub(pat, r"**\1**", title, flags=re.IGNORECASE)
                    snippet = re.sub(pat, r"**\1**", snippet, flags=re.IGNORECASE)
                except Exception:
                    pass
            line = f"- **{title}**\n  source: `{source}`\n  🔗 {url}"
            if snippet:
                line += f"\n  {snippet[:220]}{'...' if len(snippet) > 220 else ''}"
            out.append(line)
            shown += 1
        return "\n\n".join(out) if shown else "No web items to display."


    async def _stage3_unified_rag(
        self, query, conversation_history, initial_context=None
    ):
        initial_context = initial_context or []

        ql = (query or "").lower().strip()

        # A) Detect intents
        try:
            is_define = (
                "give me" in ql
                or "what is" in ql
                or "what's" in ql
                or "define" in ql
                or self.search_engine._is_definitional_intent(query)
            )
        except Exception:
            is_define = (
                "give me" in ql or "what is" in ql or "what's" in ql or "define" in ql
            )

        is_wide = False
        try:
            if hasattr(self.search_engine, "_should_wide_range"):
                is_wide = self.search_engine._should_wide_range(query)
            else:
                is_wide = any(
                    k in ql
                    for k in [
                        "list",
                        "websites",
                        "sites",
                        "best",
                        "top",
                        "alternatives",
                        "resources",
                        "sources",
                    ]
                )
        except Exception:
            pass

        if hasattr(self.search_engine, "categorize_web_results"):
            try:
                cats = (
                    self.search_engine.categorize_web_results(
                        query, ranked, max_per_cat=6
                    )
                    or {}
                )
                if cats:
                    lines = [f"**Sources for:** _{query}_"]
                    for cat, urls in cats.items():
                        if not urls:
                            continue
                        lines.append(f"\n**{cat}**")
                        for u in urls[:6]:
                            lines.append(f"- 🔗 {u}")
                    self.after(0, lambda: self._reply_assistant("\n".join(lines)))
            except Exception:
                pass
        else:
            # fallback to flat list you already added
            if hasattr(self, "_render_sources_list"):
                bullets = self._render_sources_list(ranked, query, max_items=10)
                self.after(0, lambda: self._reply_assistant(bullets))

        # ---- A) Definitional: rank → print definition → print sources → synthesize ----
        if is_define and hasattr(self.search_engine, "unified_rank"):
            try:
                ranked = await self.search_engine.unified_rank(
                    query, conversation_history, k=8
                )
            except Exception:
                ranked = []

            if ranked:
                # 1) Ask the LLM for ONLY a crisp 1–2 sentence definition, using the retrieved context
                def_sys = (
                    "You are a precise explainer. Return ONLY a concise 1–2 sentence definition "
                    "of the user's term/question, using the provided context. "
                    "No preamble, no lists, no citations, no sources; just the definition."
                )
                try:
                    definition = await self._call_llm_with_context(
                        query,
                        conversation_history,
                        context=ranked[:8],
                        intrinsic_only=False,
                        system_override=def_sys,
                    )
                    definition = (definition or "").strip()
                except Exception:
                    definition = ""

                if definition:
                    self.after(
                        0,
                        lambda: self._reply_assistant(f"**Definition**\n{definition}"),
                    )

                # 2) Print sources (title + source + URL). Use whatever renderer you already added.
                if hasattr(self, "_render_sources_list"):
                    try:
                        bullets = self._render_sources_list(ranked, query, max_items=10)
                        self.after(0, lambda: self._reply_assistant(bullets))
                    except Exception:
                        pass

                # 3) Hand the same ranked context to your normal synthesis (your existing flow)
                return await self._call_llm_with_context(
                    query, conversation_history, ranked
                )

    # Then render per category if you like. Keeping it simple for now per your ask.

    # Then render per category if you like. Keeping it simple for now per your ask.

    # ... rest of RonaAppEnhanced (including GUI flow, etc.) unchanged ...

    import inspect
    async def route_user_input(self, raw: str, conversation_history: list[str]) -> str:
        text = (raw or "").strip()
        if not text:
            return "Empty message."

        if text.startswith("/"):
            handler = getattr(self, "_handle_command", None)
            if not handler:
                return "No command handler."
            if inspect.iscoroutinefunction(handler):
                res = await handler(text)
            else:
                res = handler(text)
            # If command handler returns bool, treat True as “handled via UI”.
            if isinstance(res, bool):
                return ""  # UI already updated inside _handle_command
            return res or ""  # if it returns a string, pass it through
        # normal queries
        return await self.handle_query(text, conversation_history)

    async def _route_and_respond(self, raw: str) -> None:
        try:
            reply = await self.route_user_input(raw, getattr(self, "conversation_history", []))
            if isinstance(reply, str) and reply.strip():
                self.after(0, partial(self._reply_assistant, reply))
        except Exception as e:
            self.after(0, partial(self._reply_assistant, f"Error: {e}"))

    def _create_corner_icons(self):
        # (Keep implementation)
        C = RONA_THEME
        icons = [
            ("left", "🏮", self.open_file_dialog),
            ("right", "🐉", lambda: self._show_four_dragons(duration_ms=4000)),
            ("right", "˚⊱🪷⊰˚", self.deep_search_dialog),
            ("right", "☠︎", self.run_tests),
            ("right", "⛩️", self.open_settings),
            ("right", "×̷̷͜×̷", self.clear_chat),
            # Psycho icon handled separately
        ]
        parent = {"left": self.left_icons, "right": self.right_icons}
        for side, text, cmd in icons:
            btn = ctk.CTkButton(
                parent[side],
                text=text,
                width=40,
                height=40,
                command=cmd,
                fg_color=C["accent"],
                hover_color=C["accent_hover"],
                text_color=C["text"],
                corner_radius=10,
            )
            btn.pack(side=side, padx=2)

    def _attach_context_menu_to_textbox(self, textbox):
        # (Keep implementation)
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
        # (Keep implementation)
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

            if not LC_VECTOR_OK:
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
            import traceback

            FLASK_OK = True  # keep this separate from FAISS_OK
            if FLASK_OK:
                try:
                    # If you run under Docker/WSL and open from host: bind_host="0.0.0.0" and publish the port
                    start_psycho_ui_or_raise(
                        self, bind_host="127.0.0.1", preferred_port=8765
                    )
                except Exception as e:
                    print("psycho server error:", e)
                    traceback.print_exc()

        except Exception as e:
            self.agent = None
            self.update_status(f"❌ Agent init failed: {e}")
            # after finishing init (success or fail)

    # ---------- Chat flow ----------
    # --- THIS IS THE CORRECT send_message FUNCTION ---
    def send_message(self, event=None):
        logging.info("send_message triggered.")  # DEBUG
        msg_original = self.user_input.get().strip()
        if not msg_original:
            return
        # Check for user's mood and add a personalized greeting
        # ... (greeting logic remains the same) ...

        # Normalize time expressions
        msg_normalized = self._normalize_time_terms(msg_original)

        # Decide if text is Arabic or English
        is_arabic = hasattr(
            self.arabic_processor, "is_arabic"
        ) and self.arabic_processor.is_arabic(msg_normalized)

        # Optional UI alignment fix
        # ... (alignment logic remains the same) ...

        # Correct grammar only for English messages
        corrected_msg = msg_normalized
        if not is_arabic:
            try:
                corrected_msg = self.grammar_correct(msg_normalized)
            except Exception as e:
                logging.warning(f"Grammar correction skipped: {e}")

        # Clear input box
        # Clear input box
        self.user_input.delete(0, "end")

        # Show the original user message in the chat
        self._append_conversation("user", msg_original)

        # Use the corrected (English) or unchanged (Arabic) text for logic
        msg_to_process = corrected_msg

        # 🔁 Route and respond (commands or normal query) in an async wrapper thread
        logging.info("Routing via route_user_input -> reply.")  # DEBUG
        threading.Thread(
            target=self._run_async_wrapper,
            args=(self._route_and_respond(msg_to_process),),
            daemon=True,
        ).start()

        # --- End Check ---

    def _run_async_wrapper(self, coro):
        """Helper to run an async function from a sync thread context."""
        try:
            # Use the existing background loop runner
            result = self._run_async(coro)
            # Handle the result (e.g., update UI) if necessary,
            # ensuring UI updates happen on the main thread via self.after
            # Note: _fallback_unified_search_and_reply already handles replying.
        except Exception as e:
            logging.error(f"Error running async wrapper: {e}", exc_info=True)
            self.after(
                0,
                lambda: self._append_conversation(
                    "system", f"Error during processing: {e}"
                ),
            )
            self.after(0, lambda: self.update_status("❌ Error"))

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
                    "You: "
                    if role == "user"
                    else ("Rona: " if role == "assistant" else "")
                )
                line = f"\u202b{label}{msg}\u202c\n\n"
                self.chat_history.insert(
                    "end",
            # Flask psycho UI (optional)
            import traceback

            # FLASK_OK is already defined globally, no need to redefine here
            if FLASK_OK:
                try:
                    # If you run under Docker/WSL and open from host: bind_host="0.0.0.0" and publish the port
                    start_psycho_ui_or_raise(
                        self, bind_host="127.0.0.1", preferred_port=8765
                    )
                except Exception as e:
                    print("psycho server error:", e)
                    traceback.print_exc()

        except Exception as e:
            self.agent = None
            self.update_status(f"❌ Agent init failed: {e}")
            # after finishing init (success or fail)
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

    def _get_local(self, name: str, default=None):
        """
        Safe getter that avoids Tkinter's __getattr__ redirection.
        Reads directly from instance __dict__.
        """
        return self.__dict__.get(name, default)

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

    def grammar_correct(self, text: str) -> str:
        # (Keep implementation)
        if not GRAMMAR_TOOL_OK or (
            self.arabic_processor and self.arabic_processor.is_arabic(text)
        ):
            return text
        if not hasattr(self, "_grammar_tool"):
            self._grammar_tool = language_tool_python.LanguageTool("en-US")
        try:
            matches = self._grammar_tool.check(text)
            return language_tool_python.utils.correct(text, matches)
        except Exception as e_grammar:
            print(f"Grammar check error: {e_grammar}")
            return text

    # (Keep _append_terminal, confirm_and_run_shell, _run_shell_worker)
    def _append_terminal(self, text: str):
        self.after(
            0,
            lambda: (
                self.chat_history.insert(
                    "end", text + ("\n" if not text.endswith("\n") else ""), "terminal"
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
            llama = (
                ChatOllama(model="llama3:8b", temperature=0.2) if LC_VECTOR_OK else None
            )
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

    # ---- in the same class (e.g., RonaAppEnhanced / your UI class) ----
    def _run_async(self, coro, timeout: float | None = None):
        """
        Submit 'coro' to the persistent background event loop and wait for the result.
        Avoids creating/closing loops per call (fixes 'Event loop is closed').
        """
        import asyncio

        self._start_background_loop()
        fut = asyncio.run_coroutine_threadsafe(coro, self._bg_loop)
        return fut.result(timeout)  # e.g., set timeout=60.0 if you wa

    # inside class RonaAppEnhanced (or your app class)
    async def _intrinsic_only_async(self, query: str) -> dict:
        import asyncio, re

        # 1) normalize query first
        q = re.sub(r"\s+", " ", (query or "").strip())
        if not q:
            return {"text": "Please provide a question.", "confidence": 0.0}

        # 2) small recent history window (for style/continuity only)
        history = (
            self.conversation_history[-6:]
            if hasattr(self, "conversation_history")
            else []
        )

        # 3) a tight intrinsic system prompt (no year/recency rules here)
        sys_prompt = (
            "You are Rona (intrinsic mode). Answer using only your internal knowledge. "
            "If unsure, say so briefly and ask for one clarifying detail. Be concise."
        )

        async def _call_intrinsic(system_prompt: str, user_text: str, hist):
            if hasattr(self, "_call_llm_with_context"):
                # relies on your patched signature that accepts system_override
                return await self._call_llm_with_context(
                    user_text,
                    hist,
                    [],
                    intrinsic_only=True,
                    system_override=system_prompt,
                )
            # fallback if you ever call the LLM client directly
            return await self.llm.ainvoke({"system": system_prompt, "user": user_text})

        try:
            text = await asyncio.wait_for(
                _call_intrinsic(sys_prompt, q, history), timeout=45.0
            )
            if isinstance(text, dict):
                text = text.get("text") or text.get("content") or str(text)
            text = (text or "").strip()
        except asyncio.TimeoutError:
            return {
                "text": "The model took too long. Try rephrasing or ask a smaller question.",
                "confidence": 0.0,
            }
        except Exception as e:
            return {"text": f"Intrinsic call failed: {e}", "confidence": 0.0}

        # 4) lightweight confidence
        L = len(text)
        conf = 0.2 if L < 80 else (0.45 if L < 220 else (0.65 if L < 500 else 0.8))
        hedges = ["not sure", "uncertain", "don't know", "لا أعلم", "غير متأكد"]
        if any(h in text.lower() for h in hedges):
            conf = max(0.2, conf - 0.2)

        # 5) format using your ResponseFormatter (dict payload path)
        try:
            formatted = formatter.format(
                {
                    "best": {"text": text, "score": conf, "source": "intrinsic"},
                    "confidence": conf,
                }
            )
        except Exception:
            formatted = text

        return {"text": formatted, "confidence": conf}

    # inside your app class (e.g., RonaAppEnhanced)

    def _start_background_loop(self):
        """
        Ensure a dedicated event loop is running in a background thread.
        Safe to call multiple times.
        """
        import asyncio, threading

        # already running?
        if (
            getattr(self, "_bg_loop", None)
            and getattr(self, "_bg_thread", None)
            and self._bg_thread.is_alive()
        ):
            return

        loop = asyncio.new_event_loop()

        def _runner(l):
            asyncio.set_event_loop(l)
            l.run_forever()

        t = threading.Thread(
            target=_runner, args=(loop,), name="RonaAsyncLoop", daemon=True
        )
        t.start()

        self._bg_loop = loop
        self._bg_thread = t

    def _run_async(self, coro, timeout: float | None = None):
        """
        Submit 'coro' (or callable/any value) to the persistent background event loop and wait for result.
        - Reuses one loop across calls (prevents 'Event loop is closed').
        - Supports an optional timeout (seconds).
        - Accepts:
            * a coroutine object,
            * a callable (wrapped & executed in async),
            * or any value (returned as-is via async wrapper).
        """
        import asyncio
        from concurrent.futures import TimeoutError as _FutTimeoutError

        # make sure the background loop is up
        self._start_background_loop()
        loop = self._bg_loop

        # normalize input to a coroutine object
        if asyncio.iscoroutine(coro):
            task_coro = coro
        elif callable(coro):

            async def _wrap_callable():
                return coro()

            task_coro = _wrap_callable()
        else:

            async def _wrap_value(v):
                return v

            task_coro = _wrap_value(coro)

        # submit to the background loop
        fut = asyncio.run_coroutine_threadsafe(task_coro, loop)

        try:
            return fut.result(
                timeout
            )  # blocks this calling thread until done or timeout
        except _FutTimeoutError:
            # cancel on timeout and re-raise a clean TimeoutError
            fut.cancel()
            raise TimeoutError(f"Async operation timed out after {timeout} seconds")
        except Exception as e:
            # bubble up the original exception from inside the coroutine

            raise e

    def _stop_background_loop(self):
        """
        Stop and clean up the background event loop (optional, e.g., on app shutdown).
        """
        import asyncio

        loop = getattr(self, "_bg_loop", None)
        t = getattr(self, "_bg_thread", None)

        if not loop:
            return

        # ask the loop to stop
        try:
            loop.call_soon_threadsafe(loop.stop)
        except Exception:
            pass

        # wait a bit for the thread to exit
        if t and t.is_alive():
            try:
                t.join(timeout=1.5)
            except Exception:
                pass

        # closing the loop must happen in its own thread; schedule a finalizer
        try:

            def _close():
                try:
                    # drain/cancel pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                except Exception:
                    pass

            # run the closer in a temporary thread so we don't block UI
            import threading as _th

            _th.Thread(target=_close, name="RonaAsyncLoopCloser", daemon=True).start()
        except Exception:
            pass

        self._bg_loop = None
        self._bg_thread = None

   
   
   
   
    def _run_lovely_with_caching(self, arg: str):
        self.update_status("💖 Thinking with lovely mode...")
        sq = SQLiteManagerSingleton.get()
        if not sq:
            self._reply_assistant(
                "Database connection is not available for lovely mode."
            )
            return

        # 1. Summarize long query first (existing logic)
        summarized_arg = arg
        if len(arg) > 2000:
            try:
                self._ensure_ollama_model("llama3:8b")
                summarizer_llama = (
                    ChatOllama(model="llama3:8b", temperature=0.1)
                    if LC_OK
                    else None
                )
                if summarizer_llama:
                    summarization_prompt = f"Please summarize the following text concisely, capturing the main points, questions, and sentiment. The summary should be no more than 300 words.\n\nTEXT:\n{arg}"
                    summary_resp = summarizer_llama.invoke(summarization_prompt)
                    summarized_arg = getattr(
                        summary_resp, "content", ""
                    ) or str(summary_resp)
                else:
                    summarized_arg = arg[:2000] + "\n... (truncated)"
            except Exception as e:
                print(f"Lovely summarization error: {e}")
                summarized_arg = arg[:2000] + "\n... (truncated)"

        # 2. Process psycho notes with caching
        all_notes = psycho_store.list_entries()
        new_clues_generated = 0
        cached_clues = []

        # Use a smaller model for fast clue generation
        clue_generator_model = "gemma:2b"
        self._ensure_ollama_model(clue_generator_model)
        clue_generator = (
            ChatOllama(model=clue_generator_model, temperature=0.3)
            if LC_OK
            else None
        )
        
        for note in all_notes:
            note_text = f"Title: {note.get('title', '')}\nDate: {note.get('date', '')}\nMood: {note.get('mood', '')}/10\nDetails: {note.get('details', '')}"
            note_hash = hashlib.sha256(note_text.encode("utf-8")).hexdigest()

            cached_clue = sq.get_clue_by_hash(note_hash)
            if cached_clue:
                cached_clues.append(cached_clue)
            else:
                if not clue_generator:
                    continue  # Cannot generate new clues
                # This note is new or has changed, generate a new clue
                new_clues_generated += 1
                self.update_status(
                    f"💖 Analyzing new note ({new_clues_generated})..."
                )
                clue_prompt = f"Extract the key insight, emotion, or theme from this journal entry in one short sentence:\n\n---\n{note_text}\n---"
                try:
                    resp = clue_generator.invoke(clue_prompt)
                    new_clue = (
                        getattr(resp, "content", "") or str(resp)
                    ).strip()
                    if new_clue:
                        sq.add_clue(note_hash, new_clue)
                        cached_clues.append(new_clue)
                except Exception as e:
                    print(f"Error generating clue for note: {e}")

        ctx_text = "\n".join(cached_clues)

        # 3. Final response generation (existing logic with modifications)
        self.update_status("💖 Composing response...")
        try:
            self._ensure_ollama_model("llama3:8b")
            final_llm = (
                ChatOllama(model="llama3:8b", temperature=0.2)
                if LC_OK
                else None
            )
            if not final_llm:
                self._reply_assistant(
                    "Ollama is not available to provide a lovely response."
                )
                return

            # Language detection for prompt
            is_arabic_query = self.arabic_processor.is_arabic(summarized_arg)
            if is_arabic_query:
                system_prompt = (
                    "SYSTEM: وضع رونا اللطيف مع الوعي التحليل النفسي.\n"
                    "استخدم الملاحظات التحليل النفسية للمستخدم لتخصيص رد دافئ كرونا، موجز.\n"
                    "إذا طرح المستخدم سؤالاً، أجب عليه بإيجاز.\n"
                    "إذا أدى المستخدم عملاً جيداً مؤخراً، امدحه بحرارة واذكر التقدم الذي أحرزه.\n"
                    "إذا بدا المستخدم محبطاً، قدم له تشجيعاً لطيفاً.\n"
                    "إذا كان المستخدم يوماً سيئاً، قدم كلمات حب ولطف.\n"
                    "لا تقل أنك فخور بهم حتى لو ارتكبوا أخطاء.\n"
                    "لا تقل أنك فخور بالتقدم دون ذكر تقدم محدد.\n"
                    "إذا كان المستخدم يتحدث عن مسؤولياته، اعرض عليه المساعدة في تقسيم المهام إلى خطوات قابلة للإدارة.\n"
                    "إذا كان المستخدم يوماً سيئاً، قدم كلمات حب ولطف لأنك صديقه المفضل والوكيل الذكي المسمى رونا.\n"
                    "إذا احتاج المستخدم مساعدة في الصحة النفسية بغض النظر عن المساعدة البشرية، يجب عليك مساعدته وإعطائه تعليمات لتحسين حالته.\n"
                    "إذا ذكرت في أي سياق التحدث مع شخص، اذكر دائماً أنني هنا من أجلك وأكن صادقاً.\n"
                    "ركز على الملاحظة وإذا كان لديه سؤال أو مشكلة اسأله وساعده في حلها.\n"
                    "ولا تقل له لماذا لا نقسم تلك المهام إلى أجزاء أصغر، إنها مهمتك لمساعدته، افعلها خطوة بخطوة.\n"
                    "رد على شكل 8-10 نقاط كحد أقصى وسطر واحد 'الإجراء التالي الصغير'.\n\n"
                    f"الملاحظات النفسية (الأحدث أولاً):\n{ctx_text}\n\n"
                    f"استفسار المستخدم: {summarized_arg}\n"  # Use summarized query
                )
            else:
                system_prompt = (
                    "SYSTEM: Rona Lovely Mode with psychoanalytical awareness.\n"
                    "Use the user's psychoanalytical notes to tailor a warm as Rona, concise response.\n"
                    "if the user puts a question, answer it immediately.\n"
                    "if the user had done good job recently, praise them warmly and mention this progress which he had done.\n"
                    "If the user seems down, offer gentle encouragement.\n"
                    "if the user is having a bad day, offer love words and kind.\n"
                    "if the user talking about his responsibilities, offer to help them break down tasks into manageable steps.\n"
                    "if the user is having a bad day, offer love words and kind because your his best friend and the Ai agent called Rona.\n"
                    "if the user need help with mental health regardless of human helping you should help him and give him instraction for make him better.\n"
                    "if you mention in any context about talk to someone, mention always I'm here for you and meant be honest.\n"
                    "focus on the note and if he have question or truble ask them and help them to solve it.\n"
                    "and dont tell him Why don't we break down those tasks into smaller it's your task to help him you do it one by one,"
                    "Respond in 8-10 bullets max and one 'Next tiny action' line.\n\n"
                    f"Psycho Notes (most recent first):\n{ctx_text}\n\n"
                    f"User Query: {summarized_arg}\n"  # Use summarized query
                )
            final_resp = final_llm.invoke(system_prompt)
            out = getattr(final_resp, "content", "") or str(final_resp)
            self._reply_assistant(
                "**Lovely Notes + Psychoanalysis**\n" + out.strip()
            )
        except Exception as e:
            self._reply_assistant(f"Lovely error: {e}")
        finally:
            self.update_status("✅ Ready")
    
    # (Keep _handle_command as is)
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
        # ---- inside your command handling section ----
        # NEW
        if cmd == "/intrinsic":
            logging.info("Handling /intrinsic command.")  # DEBUG

            if not arg:
                self._append_conversation("system", "Usage: /intrinsic <query>")
                return True

            # Accept either transport: _call_llm_with_context or self.llm
            if not (
                hasattr(self, "_call_llm_with_context") or getattr(self, "llm", None)
            ):
                self._append_conversation("system", "LLM not available.")
                return True

            # If you want to hard-require personalization sometimes:
            require_personalization = bool(
                getattr(self, "require_personalization", False)
            )

            try:
                # _run_async should run the coroutine and return its result (dict)
                result = self._run_async(
                    self._intrinsic_only_async(
                        arg, require_personalization=require_personalization
                    ),
                    timeout=60.0,
                )

                # Normalize result
                if isinstance(result, dict):
                    text = (result.get("text") or "").strip()
                    confidence = float(result.get("confidence") or 0.0)
                else:
                    text = (str(result) if result is not None else "").strip()
                    confidence = 0.0

                if not text:
                    text = "I couldn't generate a response."

                # UI update must be thread-safe
                self.after(0, lambda t=text: self._reply_assistant(t))

            except (
                TimeoutError,
                asyncio.TimeoutError,
            ) as e:  # handle both sync/async timeouts
                msg = "Intrinsic command timed out."
                logging.error(msg)
                # Bind the message inside the lambda to avoid NameError on e
                self.after(0, lambda m=msg: self._append_conversation("system", m))

            except Exception as e:
                err_text = f"Intrinsic mode error: {e}"
                logging.error(err_text, exc_info=True)
                # Bind err_text so lambda doesn't capture a free variable
                self.after(0, lambda m=err_text: self._append_conversation("system", m))

            return True  # IMPORTANT: Ensures default RAG isn't triggered

        # ---------- Bug Bounty Commands ----------
        if cmd == "/hunt":
            ag = getattr(self, "agent", None)
            if not ag:
                self._reply_assistant("Agent not ready.")
                return True
            if hasattr(ag, "_set_bug_bounty_mode"):
                ag._set_bug_bounty_mode(True)
            # then start your hunt flow...
            threading.Thread(target=self._run_hunt_command, args=(arg,), daemon=True).start()
            return True


        # inside def _handle_command(self, raw: str) -> bool
        if cmd == "/lovely":
            ag = getattr(self, "agent", None)
            if not ag:
                self._reply_assistant("Agent not ready.")
                return True

            # /lovely list
            if arg.lower() == "list":
                try:
                    clues = ag.get_all_clues() or []
                except Exception as e:
                    self._reply_assistant(f"Error reading clues: {e}")
                    return True
                if not clues:
                    self._reply_assistant("No lovely notes yet.")
                    return True
                lines = [f"- {c}" for c in clues[:10]]
                more = f"\n(+{len(clues)-10} more…)" if len(clues) > 10 else ""
                self._reply_assistant("Recent lovely notes:\n" + "\n".join(lines) + more)
                return True

            # /lovely find <term>
            if arg.lower().startswith("find "):
                term = arg[5:].strip().lower()
                if not term:
                    self._reply_assistant("Usage: /lovely find <term>")
                    return True
                try:
                    clues = ag.get_all_clues() or []
                except Exception as e:
                    self._reply_assistant(f"Error reading clues: {e}")
                    return True
                hits = [c for c in clues if term in (c or "").lower()]
                if not hits:
                    self._reply_assistant(f"No notes matched “{term}”.")
                    return True
                lines = [f"- {c}" for c in hits[:15]]
                more = f"\n(+{len(hits)-15} more…)" if len(hits) > 15 else ""
                self._reply_assistant(f"Matches for “{term}”:\n" + "\n".join(lines) + more)
                return True

            # add near the top of the /lovely block, after agent ready:
            if arg.lower().startswith("psycho "):
                long_text = arg[7:].strip()
                threading.Thread(
                    target=self._run_lovely_with_caching, args=(long_text,), daemon=True
                ).start()
                return True

            # /lovely <text>  (save)
            if not arg:
                self._reply_assistant("Usage: /lovely <text>  |  /lovely list  |  /lovely find <term>")
                return True

            # If it's a query for lovely mode, run the caching logic
            threading.Thread(target=self._run_lovely_with_caching, args=(arg,), daemon=True).start()
            return True


        if cmd == "/memory":
            self._show_memory_summary()
            return True # Indicate that the command was handled.
        # The original code had an unmatched parenthesis here, removing it.
        try:
            processed_message = process_message(raw) # Changed 'message' to 'raw'
            intent = processed_message["intent"]
            # FIX: allow security_testing, bug_bounty, and general_query (or None)
            allowed_topics = {"security_testing", "bug_bounty", "general_query"}
            if intent in allowed_topics or intent is None:
                response = self.agent.invoke({"input": raw}) # Changed 'message' to 'raw'
                raw_response_content = (
                    (response.get("output") or "")
                    if isinstance(response, dict)
                    else str(response)
                )
                formatted = formatter.format(raw_response_content)
                self._reply_assistant(formatted or "No response.")
                self.update_status("✅ Ready")
                return True # Indicate command handled (or processed as general message)
            else:
                self._reply_assistant("I'm not sure I understand your request.")
                self.update_status("✅ Ready")
                return False # Command not understood
        except Exception as e:
            self._append_conversation("system", f"Agent error: {e}")
            self.update_status("❌ Agent error, using fallback")
            try:
                # Changed 'asyncio.run' to 'self._run_async' and 'message' to 'raw'
                self._run_async(self._fallback_unified_search_and_reply(raw))
            except RuntimeError:
                # If already in an event loop (e.g., in some environments), fallback to original method
                self._fallback_unified_search_and_reply(raw) # Changed 'message' to 'raw'
            return False # Indicate error or unhandled

    def _coerce_items_to_dicts(self, items):
        fixed = []
        for it in items or []:
            if isinstance(it, dict):
                fixed.append(it)
            elif isinstance(it, str):
                fixed.append(
                    {
                        "source": "raw",
                        "title": "raw result",
                        "content": it,
                        "url": "",
                        "score": 0.0,
                    }
                )
            else:
                fixed.append(
                    {
                        "source": "raw",
                        "title": f"result:{type(it).__name__}",
                        "content": str(it),
                        "url": "",
                        "score": 0.0,
                    }
                )
        return fixed

    # ---- Simple HTML → text extractor (no dependencies) ----
    def _extract_readable_text(self, html: str) -> tuple[str, str]:
        """
        Returns (title, text). Strips scripts/styles/tags, keeps paragraph-ish text.
        Very lightweight—good enough for RAG context.
        """
        import re, html as _html

        if not html:
            return "", ""

        # title
        m = re.search(r"<title>(.*?)</title>", html, flags=re.I | re.S)
        title = _html.unescape(m.group(1).strip()) if m else ""

        # drop script/style
        html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.I)
        html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.I)

        # keep <p> and <li> blocks roughly
        paras = re.findall(r"<(p|li|h1|h2|h3)[^>]*>([\s\S]*?)</\1>", html, flags=re.I)
        chunks = []
        for _, block in paras:
            # remove tags from block
            block = re.sub(r"<[^>]+>", " ", block)
            block = _html.unescape(block)
            block = " ".join(block.split())
            if len(block) >= 40:
                chunks.append(block)

        # if nothing, fall back to full page stripped
        if not chunks:
            tmp = re.sub(r"<[^>]+>", " ", html)
            tmp = _html.unescape(tmp)
            tmp = " ".join(tmp.split())
            text = tmp[:6000]
        else:
            text = " ".join(chunks)[:6000]

        return title[:160], text

    async def enrich_results_with_pages(
        self, items: list[dict], max_fetch: int = 6, per_page_bytes: int = 250_000
    ) -> list[dict]:
        """
        Visit top-N URLs concurrently, extract main text, and upgrade each item:
        - fill/replace title if page has a better one
        - put readable page text into 'content' (for LLM)
        Skips non-HTML and errors gracefully.
        """
        import aiohttp, asyncio

        if not items:
            return items

        sem = asyncio.Semaphore(4)  # keep polite & fast

        async def fetch_one(r: dict):
            url = (r.get("url") or "").strip()
            if not url:
                return r
            try:
                async with sem:
                    timeout = aiohttp.ClientTimeout(total=12)
                    headers = {"User-Agent": "Mozilla/5.0 (Rona v6 context fetcher)"}
                    async with aiohttp.ClientSession(
                        timeout=timeout, headers=headers
                    ) as s:
                        async with s.get(url, allow_redirects=True) as resp:
                            ctype = (resp.headers.get("content-type") or "").lower()
                            if resp.status != 200 or (
                                "text/html" not in ctype
                                and "application/xhtml" not in ctype
                            ):
                                return r  # skip PDFs/images/binaries
                            raw = await resp.content.read(per_page_bytes)
                            html = raw.decode(resp.charset or "utf-8", errors="ignore")
                            page_title, page_text = self._extract_readable_text(html)
                            if page_text and len(page_text) > max(
                                len(r.get("content", "")), 180
                            ):
                                r["content"] = page_text
                            if page_title and (
                                not r.get("title")
                                or len(page_title) > len(r.get("title", ""))
                            ):
                                r["title"] = page_title
                            r["source"] = r.get("source") or "web"
                            return r
            except Exception:
                return r

        # choose top candidates to fetch (prefer those with URLs and higher score)
        picks = [it for it in items if it.get("url")]
        picks = sorted(picks, key=lambda x: float(x.get("score", 0.0)), reverse=True)[
            :max_fetch
        ]

        # map back by URL for update
        idx = {(it.get("url") or ""): it for it in items}
        updated = await asyncio.gather(*(fetch_one(dict(p)) for p in picks))

        for up in updated:
            u = up.get("url") or ""
            if u in idx:
                idx[u].update(up)

        return items

    async def search_unified(
        self, query: str, conversation_history: List[str], top_k: int = 6
    ) -> List[Dict[str, Any]]:
        import asyncio

        # --- Run all searches concurrently ---
        loop = asyncio.get_running_loop()

        # Local searches (sync)
        local_task = loop.run_in_executor(None, self.local_db_search, query, top_k)
        convo_task = loop.run_in_executor(
            None, self.convo_search, query, conversation_history
        )

        # Web searches (sync and async)
        ddg_task = asyncio.create_task(self.duckduckgo_search(query, max_results=top_k))
        google_task = loop.run_in_executor(None, self.google_cse_search, query, top_k)

        # Bug bounty context (async)
        bug_bounty_task = None
        if BUG_BOUNTY_OK:
            try:
                bug_bounty_task = asyncio.create_task(
                    bug_bounty_integration.deep_search_enhanced(
                        query, conversation_history
                    )
                )
            except Exception:
                pass

        tasks = [local_task, convo_task, ddg_task, google_task]
        if bug_bounty_task:
            tasks.append(bug_bounty_task)

        results_sets = await asyncio.gather(*tasks, return_exceptions=True)

        local, convo, ddg, google = results_sets[:4]
        bug_bounty_results = results_sets[4] if bug_bounty_task else []

        # --- Combine and score results ---
        combined = []

        if not isinstance(local, Exception):
            combined.extend(local)
        if not isinstance(convo, Exception):
            combined.extend(convo)
        if not isinstance(bug_bounty_results, Exception):
            combined.extend(bug_bounty_results)

        if not isinstance(google, Exception):
            for r in google:
                score = 2.4 * overlap_score(query, r.get("content", ""))
                combined.append({**r, "score": score})

        if not isinstance(ddg, Exception):
            for r in ddg:
                score = 1.2 + overlap_score(query, r.get("content", ""))
                combined.append({**r, "score": score})

        # --- Sort and de-dup (pre-enrichment) ---
        combined_sorted = sorted(
            combined, key=lambda x: x.get("score", 0.0), reverse=True
        )
        seen = set()
        final_results = []
        for item in combined_sorted:
            key = (item.get("url") or item.get("title") or item.get("content", ""))[
                :200
            ]
            if key in seen:
                continue
            seen.add(key)
            final_results.append(item)
            if len(final_results) >= top_k:
                break

        # --- Enrich: fetch page HTML for top results and extract readable text ---
        # So LLM doesn't see only titles/snippets
        try:
            # definitional / short queries deserve enrichment; otherwise keep it lightweight
            lower_q = (query or "").lower()
            is_define = False
            try:
                is_define = self._is_definitional_intent(query)
            except Exception:
                pass
            is_define = (
                is_define
                or len(
                    getattr(self, "_tokenize_loose", lambda s: (s or "").split())(
                        lower_q
                    )
                )
                <= 6
            )

            if is_define and final_results:
                final_results = await self.enrich_results_with_pages(
                    final_results, max_fetch=min(6, len(final_results))
                )
        except Exception:
            # fail-safe: return un-enriched if something goes wrong
            pass

        return final_results

    # ---------- helpers (drop-in safe) ----------
    async def _fallback_unified_search_and_reply(
        self,
        message: str,
    async def search_unified(
        self, query: str, conversation_history: List[str], top_k: int = 6
    ) -> List[Dict[str, Any]]:

        # --- Run all searches concurrently ---
        loop = asyncio.get_running_loop()
        """
        import logging, hashlib
        from typing import List, Dict, Any

        logging.info("Executing _fallback_unified_search_and_reply (RAG).")
        self.after(0, lambda: self.update_status("🪷 Searching (unified)..."))

        # --- ensure locals are defined ---
        items: List[Dict[str, Any]] = []
        categories = (
            None  # kept locally; NOT passed to minimal formatter unless it supports it
        )
        sqlite_ctx: List[Dict[str, Any]] = []

        # --- Build the system prompt once (persona + personalization) ---
        sys_prompt = self._build_intrinsic_system_prompt_for_rag(
            require_personalization=require_personalization
        )

        try:
            # --- Perform unified search ---
            search_result = await self.search_engine.search_unified(
                message,
                getattr(self, "conversation_history", []),
                top_k=8,
            )

            # --- Normalize shapes (dict vs list) ---
            if isinstance(search_result, dict):
                items = search_result.get("results", []) or []
                categories = search_result.get("categories") or None
            else:  # list
                items = search_result or []

            # Ensure dict shapes for safe .get()
            items = (
                self._coerce_items_to_dicts(items)
                if hasattr(self, "_coerce_items_to_dicts")
                else self._coerce_items_to_dicts_local(items)
            )

            # --- Add SQLite context (optional boost) ---
            try:
                sq = None
                if "SQLiteManagerSingleton" in globals():
                    sq = globals()["SQLiteManagerSingleton"].get()
                elif hasattr(self, "SQLiteManagerSingleton"):
                    sq = self.SQLiteManagerSingleton.get()

                if sq:
                    expand = (
                        expand_query_variants(message)[:3]
                        if "expand_query_variants" in globals()
                        else (
                            getattr(
                                self,
                                "_default_expand_variants",
                                lambda q: [q.lower(), q.upper(), q.title()],
                            )(message)[:3]
                        )
                    )
                    for qv in [message] + expand:
                        sqlite_ctx.extend(sq.search("cyberassest", qv, limit=2))
                        sqlite_ctx.extend(sq.search("lovely_assest", qv, limit=1))

                    # De-duplicate by short content hash
                    seen_sqlite = set()
                    unique_sqlite_ctx = []
                    for it in sqlite_ctx:
                        c = (
                            it.get("content", "") if isinstance(it, dict) else str(it)
                        )[:500]
                        h = hashlib.sha1(c.encode("utf-8")).hexdigest()
                        if h not in seen_sqlite:
                            unique_sqlite_ctx.append(it)
                            seen_sqlite.add(h)
                    sqlite_ctx = unique_sqlite_ctx

            except Exception as e_sql:
                logging.error(
                    f"SQLite search error in fallback: {e_sql}", exc_info=True
                )

            # --- Combine contexts ---
            combined_context: List[Dict[str, Any]] = []
            combined_context.extend(items)

            # Normalize SQLite rows to a common schema
            for sql_item in sqlite_ctx:
                if isinstance(sql_item, dict):
                    combined_context.append(
                        {
                            "source": "sqlite",
                            "title": sql_item.get("filename")
                            or sql_item.get("path")
                            or "SQLite Note",
                            "content": sql_item.get("content", ""),
                            "score": self._safe_overlap(
                                message, sql_item.get("content", "")
                            ),
                        }
                    )
                else:
                    text = str(sql_item)
                    combined_context.append(
                        {
                            "source": "sqlite",
                            "title": "SQLite Note",
                            "content": text,
                            "score": self._safe_overlap(message, text),
                        }
        """
        import logging, hashlib
        from typing import List, Dict, Any
        import asyncio # Ensure asyncio is imported for run_in_executor

        logging.info("Executing _fallback_unified_search_and_reply (RAG).")
        self.after(0, lambda: self.update_status("🪷 Searching (unified)..."))
                # 1) Unified web search (DDG-first, optional Google via your search_unified)
                try:
                    u = await self.search_engine.search_unified(
                        message,
                        getattr(self, "conversation_history", []),
                        top_k=8,
                    )
                    web_results = (
                        (u or {}).get("results", [])
                        if isinstance(u, dict)
                        else (u or [])
                    )
                    logging.info(
                        f"[fallback] unified search returned {len(web_results)} result(s)."
                    )
                except Exception as e:
                    logging.error(
                        f"[fallback] unified search error: {e}", exc_info=True
                    )
                    web_results = []

                # 2) Emergency: if unified returned nothing, try raw DDG once
                if not web_results:
                    try:
                        ddg_last = await self.search_engine.duckduckgo_search(
                            message, max_results=6
                        )
                        web_results = ddg_last or []
                        logging.info(
                            f"[fallback] emergency DDG returned {len(web_results)} result(s)."
                        )
                    except Exception as e:
                        logging.error(
                            f"[fallback] emergency DDG error: {e}", exc_info=True
                        )
                        web_results = []

                # 3) If we got anything from web, treat as context
                if web_results:
                    combined_context = web_results[:5]  # lightweight context slice

            # If we now have any context (local or web), prefer synthesis/formatting
            if combined_context:
                logging.info(
                    f"Found {len(combined_context)} context items. Proceeding with formatting/synthesis."
                )
                if getattr(self, "llm", None):
                    try:
                        final_answer = await self._call_llm_with_context(
                            message,
                            getattr(self, "conversation_history", []),
                            context=combined_context,  # RAG + SQLite +/or Web
                            intrinsic_only=False,
                            system_override=sys_prompt,
                        )
                        self.after(
                            0,
                            lambda: self._reply_assistant(
                                final_answer
                                or "I found some context but couldn't synthesize a final answer."
                            ),
                        )
                    except Exception as e_llm_synth:
                        logging.error(
                            f"LLM synthesis error: {e_llm_synth}", exc_info=True
                        )
                        formatted_text = self._safe_format_unified_answer(
                            message, combined_context, categories=None
                        )
                        self.after(0, lambda: self._reply_assistant(formatted_text))
                else:
                    # No LLM, just format safely
                    formatted_text = self._safe_format_unified_answer(
                        message, combined_context, categories=None
                    )
                    self.after(0, lambda: self._reply_assistant(formatted_text))
            else:
                # Only reach here if BOTH local + web are empty → pure LLM fallback
                logging.info("No local or web context available → LLM-only fallback.")
                if getattr(self, "llm", None):
                    try:
                        fallback_answer = await self._call_llm_with_context(
                            message,
                            getattr(self, "conversation_history", []),
                            context=[],  # open-world, no context
                            intrinsic_only=False,
                            system_override=sys_prompt,
                        )
                        self.after(
                            0,
                            lambda: self._reply_assistant(
                                fallback_answer
                                or "I couldn't find relevant information."
                            ),
                        )
                    except Exception as e_llm_fb:
                        logging.error(f"LLM fallback error: {e_llm_fb}", exc_info=True)
                        err_text = f"Unified search error: {e_llm_fb}"
                        self.after(
                            0, lambda t=err_text: self._append_conversation("system", t)
                        )
                        self.after(
                            0,
                            lambda: self._reply_assistant(
                                "An error occurred trying to generate a fallback response."
                            ),
                        )
                else:
                    self.after(
                        0, lambda: self._reply_assistant("No useful matches found.")
                    )

        except RuntimeError:
            # If already in an event loop, fallback to original method (sync)
            self._fallback_unified_search_and_reply(message)
        except Exception as e:
            logging.error(
                f"Error in _fallback_unified_search_and_reply: {e}", exc_info=True
            )
            err_text = f"Unified search error: {e}"
            self.after(0, lambda t=err_text: self._append_conversation("system", t))
        finally:
            self.after(0, lambda: self.update_status("✅ Ready"))

    def _coerce_items_to_dicts_local(items):
        """Local defensive fallback if self._coerce_items_to_dicts is absent."""
        out = []
        for it in items or []:
            if isinstance(it, dict):
                out.append(it)
            else:
                s = (str(it) or "").strip()
                if s:
                    out.append({"source": "raw", "title": "result", "content": s})
        return out

    def _default_expand_variants(q: str):
        """Small, cheap expansion as a fallback."""
        q = (q or "").strip()
        if not q:
            return []
        return [q.lower(), q.upper(), q.title()]

    def _default_overlap_score(a: str, b: str):
        """Very simple token overlap (0..1)."""
        ta = set((a or "").lower().split())
        tb = set((b or "").lower().split())
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        return inter / max(len(ta), 1)

    def _safe_format_unified_answer(self, message, items, categories=None):
        """Call the available formatter safely regardless of argument count."""
        if hasattr(self, "_format_unified_answer"):
            try:
                return self._format_unified_answer(message, items, categories)
            except TypeError:
                try:
                    return self._format_unified_answer(message, items)
                except TypeError:
                    pass

        if hasattr(self, "_format_unified_answer_minimal"):
            try:
                return self._format_unified_answer_minimal(message, items)
            except TypeError:
                return self._format_unified_answer_minimal(message, items, categories)

        # Last-resort fallback
        lines = []
        if not items:
            return "No results found."
        for it in items[:8] or []:
            if isinstance(it, dict):
                title = it.get("title") or it.get("meta", {}).get("source") or "result"
                src = (it.get("source") or "").split(":")[0]
                url = it.get("url") or ""
                lines.append(f"- [{src}] {title}" + (f" — {url}" if url else ""))
            else:
                t = str(it).strip()
                if t:
                    snippet = (t[:120] + "…") if len(t) > 120 else t
                    lines.append(f"- [raw] {snippet}")
        return "\n".join(lines)

    def _build_intrinsic_system_prompt_for_rag(
        self, *, require_personalization: bool = False
    ) -> str:
        """
        Build a system prompt that ALWAYS preserves persona + personalization.
        Reused across LLM calls in the unified RAG flow.
        """
        base_sys = (
            "SYSTEM (RAG Synthesis Mode). "
            "Use provided context when available; otherwise be transparent about gaps. "
            "Be concise. Do not output exploit payloads."
        )
        persona = (getattr(self, "intrinsic_persona", "") or "").strip()
        personalization = (getattr(self, "personalization_prompt", "") or "").strip()

        if require_personalization and not personalization:
            # Hard guard: still return base + persona to avoid None, but you can log/return earlier if desired
            import logging

            logging.error(
                "Personalization prompt required but not found on self.personalization_prompt."
            )

        parts = [base_sys]
        if persona:
            parts.append(f"[PERSONA]\n{persona}")
        if personalization:
            parts.append(f"[PERSONALIZATION]\n{personalization}")
        return "\n\n".join(parts).strip()

    def _safe_overlap(a: str, b: str) -> float:
        try:
            if "overlap_score" in globals() and callable(globals()["overlap_score"]):
                return globals()["overlap_score"](a, b)
        except Exception:
            pass
        ta = set((a or "").lower().split())
        tb = set((b or "").lower().split())
        return (len(ta & tb) / max(len(ta), 1)) if ta and tb else 0.0

    def _safe_format_unified_answer(self, message, items, categories=None):
        """
        Calls the best available formatter (_format_unified_answer or _format_unified_answer_minimal)
        with the correct number of arguments — avoids the 'takes 2 positional args' error.
        """
        if hasattr(self, "_format_unified_answer"):
            try:
                # Try 3-arg version first
                return self._format_unified_answer(message, items, categories)
            except TypeError:
                # Fallback to 2-arg if defined that way
                return self._format_unified_answer(message, items)

        if hasattr(self, "_format_unified_answer_minimal"):
            try:
                # Try 2-arg version (most common)
                return self._format_unified_answer_minimal(message, items)
            except TypeError:
                # If someone defined it differently, try 3-arg
                return self._format_unified_answer_minimal(message, items, categories)

        # Last fallback (no formatters found)
        lines = []
        if not items:
            return "No results found."
        for it in items[:8]:
            title = (it.get("title") if isinstance(it, dict) else str(it)) or "Result"
            lines.append(f"- {title}")
        return "\n".join(lines)

    def _format_best(self, message, items, categories=None):
        """
        Try your formatters in a tolerant way so arg-count mismatches never crash.
        Order tried:
        1) _format_unified_answer(message, items, categories)
        2) _format_unified_answer(message, items)
        3) _format_unified_answer(items)
        4) _format_unified_answer(message)
        5) _format_unified_answer_minimal(...) with the same sequence
        6) fallback inline minimal list
        """
        candidates = []
        if hasattr(self, "_format_unified_answer"):
            candidates.append(getattr(self, "_format_unified_answer"))
        if hasattr(self, "_format_unified_answer_minimal"):
            candidates.append(getattr(self, "_format_unified_answer_minimal"))

        for fmt in candidates:
            for args in (
                (message, items, categories),
                (message, items),
                (items,),
                (message,),
                tuple(),  # in case someone made a zero-arg variant
            ):
                try:
                    return fmt(*args)
                except TypeError:
                    continue
                except Exception:
                    # Any other formatter exception: skip to next attempt
                    continue

        # Last resort: inline minimal formatting
        lines = []
        if not items:
            lines.append("No results found.")
        else:
            for it in items[:8] or []:
                if isinstance(it, dict):
                    title = (
                        it.get("title") or it.get("meta", {}).get("source") or "result"
                    )
                    src = (it.get("source") or "").split(":")[0]
                    url = it.get("url") or ""
                    lines.append(f"- [{src}] {title}" + (f" — {url}" if url else ""))
                else:
                    t = str(it).strip()
                    if t:
                        snippet = (t[:120] + "…") if len(t) > 120 else t
                        lines.append(f"- [raw] {snippet}")
        if categories:
            lines.append("\nCategories:")
            try:
                for label, urls in categories.items():
                    try:
                        lines.append(f"- {label}: " + ", ".join(urls[:5]))
                    except Exception:
                        lines.append(f"- {label}")
            except Exception:
                pass
        return "\n".join(lines)

    def _is_refusal(self, text: str) -> bool:
        # (Keep the latest version with all refusal patterns)
        t = (text or "").lower()
        patterns = [
            "sorry, i can't",
            "i can't assist",
            "i cannot help",
            "i can't help",
            "as an ai",
            "i won't provide",
            "i will not provide",
            "unable to provide",
            "cannot fulfill",
            "my purpose is to help",
            "my instructions are to",
            "i'm sorry for any misunderstanding",
            "goes against my programming",
            "illegal",
            "unethical",
            "harmful content",
        ]  # Add more as needed
        return any(p in t for p in patterns)

    # --- Fallback Display Function (Keep upgraded version) ---
    def _answer_with_consolidated_methodology(
        self, query: str, context: List[Dict[str, Any]]
    ) -> str:
        # (Keep the latest version with highlighting, date, URL, filtering etc.)
        lines = ["⚠️ Displaying relevant context (LLM synthesis skipped/failed):"]
        results_shown = 0
        MAX_FALLBACK_RESULTS = 7
        SNIPPET_LENGTH = 350
        seen_hashes = set()
        priority_sources = [
            "duckduckgo",
            "google",
            "web_proxy",
            "local",
            "vector",
            "conversation",
        ]
        sorted_context = sorted(
            context,
            key=lambda r: (
                0 if r.get("source", "").lower() in priority_sources else 1,
                -r.get("score", 0),
            ),
        )
        query_terms = set(tokenize(query))

        def highlight(text, terms):
            if not terms or not text:
                return text
            pattern = (
                r"\b(" + "|".join(re.escape(term) for term in terms if term) + r")\b"
            )
            try:
                return re.sub(pattern, r"**\1**", text, flags=re.IGNORECASE)
            except Exception:
                return text

        for r in sorted_context:
            if results_shown >= MAX_FALLBACK_RESULTS:
                break
            source = r.get("source", "unknown").lower()
            title = r.get("title", "")
            content = (r.get("content", "") or "").strip()
            url = r.get("url", "")
            timestamp = r.get(
                "created_at",
                r.get("meta", {}).get("timestamp", r.get("meta", {}).get("date")),
            )
            is_internal_meta = source in ["professional_approach", "zoomeye_notice"]
            is_just_url_or_short = not content or len(content) < 15 or content == url
            content_hash = hashlib.sha1(
                content[:500].encode("utf-8", errors="ignore")
            ).hexdigest()
            is_duplicate = content_hash in seen_hashes
            if is_internal_meta or is_just_url_or_short or is_duplicate:
                continue
            results_shown += 1
            seen_hashes.add(content_hash)
            if "/" in title or "\\" in title:
                try:
                    title = Path(title).name
                except Exception:
                    pass
            if not title or title.lower() == source or title == url:
                first_line = content.split("\n", 1)[0].strip()
                if (
                    first_line
                    and 5 < len(first_line) < 100
                    and len(first_line.split()) > 1
                ):
                    title = first_line + "..."
                elif not title:
                    title = f"{source.capitalize()} Result {results_shown}"
            snippet = content[:SNIPPET_LENGTH] + (
                "..." if len(content) > SNIPPET_LENGTH else ""
            )
            display_content = highlight(snippet, query_terms)
            entry = f"{results_shown}. **[{source.capitalize()}] {title}**"
            if timestamp and isinstance(timestamp, str):
                date_part = timestamp.split(" ")[0]
                if re.match(r"\d{4}-\d{2}-\d{2}", date_part):
                    entry += f" *(Date: {date_part})*"
            entry += f"\n   {display_content or '[No content]'}"
            if url:
                entry += f"\n   *URL:* {url}"
            lines.append(entry)
        if results_shown == 0:
            return "Found context, but filtering removed all items. Try rephrasing."
        zoomeye_footer = self._zoomeye_sources(context)
        if zoomeye_footer:
            lines.append("\n---\n**ZoomEye Sources:**\n" + zoomeye_footer)
        return "\n\n".join(lines)

    # --- Web Only Fallback (Keep as is) ---
    def _answer_with_web_results_only(
        self, query: str, context: List[Dict[str, Any]]
    ) -> str:
        # (Keep implementation as provided previously)
        web_sources = ["duckduckgo_api", "duckduckgo_scrape", "google", "web_proxy"]
        lines = ["⚠️ Displaying web search results (LLM synthesis skipped/failed):"]
        results_shown = 0
        MAX_WEB_FALLBACK_RESULTS = 5
        SNIPPET_LENGTH = 400
        query_terms = set(tokenize(query))

        def highlight(text, terms):
            if not terms or not text:
                return text
                pattern = (
                    r"\b("
                    + "|".join(re.escape(term) for term in terms if term)
                    + r")\b"
                )
            try:
                return re.sub(pattern, r"**\1**", text, flags=re.IGNORECASE)
            except Exception:
                return text

        sorted_context = sorted(
            context, key=lambda x: x.get("score", 0.0), reverse=True
        )  # Sort web context by score
        for r in sorted_context:
            source = r.get("source", "").lower()
            if source in web_sources:
                if results_shown >= MAX_WEB_FALLBACK_RESULTS:
                    break
                title = r.get("title", "")
                content = (r.get("content", "") or "").strip()
                url = r.get("url", "")
                if not content or not url:
                    continue  # Skip if no content or URL for web results
                results_shown += 1
                if title == url:
                    title = ""
                if not title:
                    first_line = content.split("\n", 1)[0].strip()
                    title = (
                        first_line[:80] + "..."
                        if first_line and 5 < len(first_line)
                        else "Web Result"
                    )
                snippet = content[:SNIPPET_LENGTH] + (
                    "..." if len(content) > SNIPPET_LENGTH else ""
                )
                display_content = highlight(snippet, query_terms)
                entry = f"{results_shown}. **[{source.replace('_',' ').title()}] {title}**\n   {display_content}\n   *URL:* {url}"
                lines.append(entry)
        if results_shown == 0:
            return "Couldn't find suitable web results in context. Try rephrasing."
        zoomeye_footer = self._zoomeye_sources(
            context
        )  # Check original context passed down
        if zoomeye_footer:
            lines.append("\n---\n**ZoomEye Sources Visited:**\n" + zoomeye_footer)
        return "\n\n".join(lines)

    # (Keep _deep_search_dialog, _deep_search_and_reply as is)
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
        # (Keep implementation, calls search_unified)
        self.update_status("🔎 Deep searching...")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(
                self.search_engine.search_unified(
                    query, self.conversation_history, top_k=8
                )
            )
        except Exception as e_dsr:
            results = []
            print(f"Deep search error: {e_dsr}")
        finally:
            asyncio.set_event_loop(None)
        if not results:
            self._reply_assistant("Deep search found no matches.")
            self.update_status("Ready")
            return
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")[:100]
            content = (r.get("content", "") or "")[:300]
            url = r.get("url", "")
            lines.append(
                f"{i}. [{r.get('source','?')}] {title}\n{content}\n{('URL: ' + url) if url else ''}"
            )
        self._reply_assistant("Deep Search Results:\n\n" + "\n\n".join(lines))
        self.update_status("Ready")

    # (Keep file/image handlers, tests, settings, clear chat as they were)
    def open_file_dialog(self):
        # (Keep implementation)
        filetypes = [
            ("Docs", "*.txt;*.md;*.log;*.pdf;*.xml;*.json;*.csv;*.html;*.htm"),
            ("All", "*.*"),
        ]
        p_str = filedialog.askopenfilename(filetypes=filetypes)
        if not p_str:
            return
        try:
        def highlight(text, terms):
            if not terms or not text:
                return text
            pattern = ( # Corrected indentation
                r"\b("
                + "|".join(re.escape(term) for term in terms if term)
                + r")\b"
            )
            try:
                return re.sub(pattern, r"**\1**", text, flags=re.IGNORECASE)
            except Exception:
                return text
                return
            table = "cyberassest"
            use_lovely = messagebox.askyesno(
                "Store Destination",
                "Save to lovely_assest? (No=cyberassest)",
                parent=self,
            )
            if use_lovely:
                table = "lovely_assest"
            sq = SQLiteManagerSingleton.get()
            created_at = datetime.datetime.now().isoformat(sep=" ", timespec="seconds")
            sqlite_ok = (
                sq.insert_document(
                    table,
                    dst.name,
                    str(dst),
                    combined,
                    {"source": str(dst)},
                    created_at,
                )
                if sq
                else False
            )
            vector_ok = False
            if docs and LC_VECTOR_OK and LCDocument and LCDocument != type(None):
                db = DatabaseManagerSingleton.get()
                if db:
                    vector_ok = db.add_documents(
                        docs
                    )  # Assuming docs are LCDocuments here
            self._reply_assistant(
                f"Indexed: {dst.name} (vector={vector_ok}, sqlite={sqlite_ok} to {table})"
            )
        except Exception as e:
            self._append_conversation("system", f"Upload error: {e}")

    def open_image_dialog(self):
        self._append_conversation("system", "Image analysis disabled.")

    def create_image_dialog(self):
        self._append_conversation("system", "Image generation disabled.")

    def run_tests(self):
        # (Keep implementation)
        self.update_status("🧪 Running tests...")
        results = self.test_suite.run_all_tests()
        self._append_conversation("system", "🧪 Test Results:")
        for name, r in results.items():
            icon = "✅" if r.get("status") == "passed" else "❌"
            self._append_conversation("system", f"{icon} {name}: {r.get('status','?')}")
        self.update_status("Ready")

    def open_settings(self):
        messagebox.showinfo("Settings", "Settings via config.json (restart needed).")

    def clear_chat(self):
        # (Keep implementation)
        if hasattr(self, "chat_history") and self.chat_history:
            self.chat_history.delete("1.0", "end")
        self.conversation_history = []  # Clear history list too
        clear_messages = [
            "Chat cleared.",
            "All clear!",
            "Cleared.",
            "The slate is clean.",
        ]
        self._append_conversation("system", random.choice(clear_messages))

    # (Keep Bug Bounty Command Handlers as is, ensure they exist if BUG_BOUNTY_OK)
    # Define placeholder if not OK
    def _run_hunt_command(self, target: str):
        if not BUG_BOUNTY_OK or not bug_bounty_integration:
            self._append_conversation("system", "BB integration disabled.")
            return
        # (Keep original implementation)
        self.update_status(f"🎯 Hunting {target}...")
        try:
            result = bug_bounty_integration.run_hunt_command(
                target
            )  # Assuming sync or handled in lib
        except Exception as e:
            result = {"success": False, "error": str(e)}
        if result.get("success"):
            self._append_conversation("system", f"✅ Hunt done: {target}")
            self._append_terminal(result.get("stdout", ""))
        else:
            self._append_conversation(
                "system", f"❌ Hunt failed: {result.get('error', 'Unknown')}"
            )
            self._append_terminal(result.get("stderr", ""))
        self.update_status("✅ Ready")

    def _run_lovely_mode(self, folder_path: str):  # Assuming sync helper
        if not BUG_BOUNTY_OK or not bug_bounty_integration:
            self._append_conversation("system", "BB integration disabled.")
            return
        # (Keep original implementation)
        self.update_status(f"💖 Lovely mode...")
        try:
            result = bug_bounty_integration.run_lovely_mode(folder_path)
        except Exception as e:
            result = {"success": False, "error": str(e)}
        if result.get("success"):
            self._append_conversation("system", f"✅ Lovely done")
            self._append_terminal(result.get("stdout", ""))
        else:
            self._append_conversation(
                "system", f"❌ Lovely failed: {result.get('error', 'Unknown')}"
            )
            self._append_terminal(result.get("stderr", ""))
        self.update_status("✅ Ready")

    def _show_memory_summary(self):
        if not BUG_BOUNTY_OK or not bug_bounty_integration:
            self._append_conversation("system", "BB integration disabled.")
            return
        # (Keep original implementation)
        self.update_status("🧠 Loading memory...")
        try:
            summary = bug_bounty_integration.get_memory_summary()  # Format summary...
            summary_text = "**BB Memory**\n" + json.dumps(
                summary, indent=2
            )  # Basic format
        except Exception as e:
            summary_text = f"❌ Memory error: {e}"
        self._reply_assistant(summary_text)
        self.update_status("✅ Ready")

    def _get_scan_recommendations(self, target: str):
        if not BUG_BOUNTY_OK or not bug_bounty_integration:
            self._append_conversation("system", "BB integration disabled.")
            return
        # (Keep original implementation)
        self.update_status(f"💡 Recommending for {target}...")
        try:
            recommendations = bug_bounty_integration.get_scan_recommendations(target)
            rec_text = (
                f"**Recommendations for {target}:**\n"
                + "\n".join([f"- {r}" for r in recommendations])
                if recommendations
                else "None."
            )
        except Exception as e:
            rec_text = f"❌ Recommend error: {e}"
        self._reply_assistant(rec_text)
        self.update_status("✅ Ready")


# ---------- Main ----------
# ---------- Main ----------
def main():
    print("🚀 Starting Rona v6 Enhanced (RAG-Direct)...")
    load_balancer.monitor_temperatures()  # Initial check
    app = RonaAppEnhanced()
    app.mainloop()


if __name__ == "__main__":
    import sys

    # optionally handle a simple CLI arg to run only the selftest
    if "--selftest-db" in sys.argv:
        sanity_check_vector_db()
        sys.exit(0)

    # Normal GUI launch
    app = RonaAppEnhanced()
    # run the DB selftest shortly after the UI is up (non-blocking)
    try:
        app.after(200, app.selftest_db)  # 200 ms after mainloop starts
    except Exception:
        # fallback: run immediately
        app.selftest_db()
    app.mainloop()
