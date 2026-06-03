"""
Arwanos utility functions — pure helpers with no GUI or LLM dependencies.
"""
from __future__ import annotations

import re as _re
import urllib.parse as _urlp
import json as _json
from pathlib import Path
from typing import List, Dict, Any


# ─── Sound toggle ────────────────────────────────────────────────────────────
# 0 = sound ON  |  1 = sound OFF (default)
ARWANOS_SOUND_ENABLED: int = 1


# ─── Username cache ───────────────────────────────────────────────────────────
_USERNAME_CACHE: str = ""


def _get_username() -> str:
    """Return the configured username from config.json (cached after first read)."""
    global _USERNAME_CACHE
    if _USERNAME_CACHE:
        return _USERNAME_CACHE
    try:
        cfg = Path(__file__).parent / "config.json"
        _USERNAME_CACHE = (_json.loads(cfg.read_text()).get("username") or "GMM").strip() or "GMM"
    except Exception:
        _USERNAME_CACHE = "GMM"
    return _USERNAME_CACHE


# ─── URL helpers ──────────────────────────────────────────────────────────────

def _unwrap_duckduckgo(u: str) -> str:
    """Unwrap a DuckDuckGo redirect URL to reveal the actual destination."""
    if not u:
        return u
    try:
        if "duckduckgo.com/l/?" in u and "uddg=" in u:
            if u.startswith("//"):
                u = "https:" + u
            parsed = _urlp.urlparse(u)
            qs = _urlp.parse_qs(parsed.query)
            if "uddg" in qs and qs["uddg"]:
                return _urlp.unquote(qs["uddg"][0])
    except Exception:
        pass
    return u


def _normalize_url(url: str) -> str:
    """Normalise a URL: unwrap DDG redirects, add scheme, lowercase host, strip query/fragment."""
    if not url:
        return ""
    url = _unwrap_duckduckgo(url)
    try:
        u = _urlp.urlparse(url.strip())
        if not u.scheme:
            u = _urlp.urlparse("https://" + url.strip())
        return _urlp.urlunparse((
            u.scheme.lower(),
            u.netloc.lower(),
            _re.sub(r"/{2,}", "/", u.path or "/").rstrip("/"),
            "", "", "",
        ))
    except Exception:
        return url.strip()


# ─── Display / zoom ───────────────────────────────────────────────────────────

def _detect_initial_zoom_delta() -> int:
    """Return zoom delta based on connected monitors: 6 (external) or 4 (laptop)."""
    try:
        import subprocess
        r = subprocess.run(
            ["xrandr", "--listmonitors"],
            capture_output=True, text=True, timeout=2,
        )
        m = _re.search(r"Monitors:\s*(\d+)", r.stdout)
        count = int(m.group(1)) if m else 1
        return 6 if count > 1 else 4
    except Exception:
        return 4


# ─── Query intent detection ───────────────────────────────────────────────────

def _is_definitional_query(text: str) -> bool:
    """Return True if the query looks like a definition/explanation request."""
    t = (text or "").strip().lower()
    if not t:
        return False
    triggers = (
        "what is ", "what's ", "what's ", "define ",
        "definition of ", "meaning of ", "means ",
        "شرح ", "ما هو", "تعريف", "يعني",
    )
    if any(t.startswith(p) for p in triggers):
        return True
    tokens = [x for x in _re.split(r"\W+", t) if x]
    return len(tokens) <= 3


# ─── Cybersecurity domain data ────────────────────────────────────────────────

_CYBER_ACRONYMS: set[str] = {
    "xss", "csrf", "sqli", "sql injection", "rce", "lfi", "rfi", "ssrf", "xxe",
    "ssti", "idor", "bola", "bua", "iam", "ppe", "open redirect",
    "clickjacking", "path traversal", "directory traversal", "privilege escalation",
    "owasp", "nist", "mitre", "cvss", "cve", "cwe", "capec", "att&k", "attck",
    "pci dss", "hipaa", "sox", "iso 27001", "soc 2", "gdpr",
    "tls", "ssl", "ipsec", "ssh", "vpn", "pki", "ca", "csr", "jwt", "oauth",
    "saml", "ldap", "kerberos", "radius", "ntlm", "krb", "x509", "hmac", "aes",
    "rsa", "ecdsa", "ecdh", "sha", "md5", "bcrypt", "pbkdf2",
    "nmap", "burp", "metasploit", "wireshark", "sqlmap", "dirbuster", "nikto",
    "hydra", "hashcat", "john the ripper", "aircrack", "mimikatz", "bloodhound",
    "cobalt strike", "impacket", "netcat", "nc", "ncat", "socat",
    "ffuf", "gobuster", "wfuzz", "subfinder", "amass", "shodan", "censys",
    "malware", "ransomware", "spyware", "adware", "rootkit", "trojan", "worm",
    "botnet", "c2", "c&c", "phishing", "spear phishing", "whaling", "vishing",
    "smishing", "apt", "ioc", "ttp", "ttp's", "threat actor", "zero day", "0day",
    "pentest", "penetration testing", "red team", "blue team", "purple team",
    "siem", "soc", "ids", "ips", "waf", "edr", "xdr", "dlp", "mfa", "2fa",
    "sandbox", "honeypot", "deception", "threat hunting", "dfir", "forensics",
    "osint", "recon", "enumeration", "lateral movement", "persistence",
    "exfiltration", "command and control", "dmarc", "spf", "dkim",
    "ceh", "oscp", "osep", "osed", "oswe", "gpen", "gwapt", "ewapt", "ejpt",
    "cpts", "cissp", "cism", "comptia security+", "sec+",
}

_CYBER_DOMAINS: set[str] = {
    "owasp.org", "portswigger.net", "exploit-db.com", "cve.mitre.org",
    "nvd.nist.gov", "attack.mitre.org", "cwe.mitre.org", "capec.mitre.org",
    "hackerone.com", "bugcrowd.com", "hacker101.com", "tryhackme.com",
    "hackthebox.com", "pentesterlab.com", "vulnhub.com", "payloadsallthethings",
    "github.com", "sans.org", "krebs", "thehackernews.com", "bleepingcomputer.com",
    "securityweek.com", "darkreading.com", "rapid7.com", "tenable.com",
    "snyk.io", "cloudflare.com", "shodan.io", "censys.io", "exploit.db",
    "cybersecurity", "security", "hacking", "pentest", "infosec",
}

_ANTI_CYBER_DOMAINS: set[str] = {
    "mayoclinic.org", "webmd.com", "healthline.com", "nih.gov", "medline",
    "drugs.com", "rxlist.com", "medicinenet.com", "pediatrics", "nejm.org",
    "espn.com", "nba.com", "nfl.com", "mlb.com", "sports", "football",
    "recipe", "cooking", "food.com", "allrecipes", "yummly",
    "realestate", "zillow.com", "trulia.com", "realtor.com",
    "imdb.com", "rottentomatoes.com", "movies", "tvguide",
}


def _is_cyber_query(text: str) -> bool:
    """Return True if the query is likely asking about a cybersecurity topic."""
    t = (text or "").strip().lower()
    cyber_kw = (
        "cyber", "hack", "exploit", "vulnerab", "pentest", "cve", "owasp",
        "malware", "ransomware", "phishing", "injection", "payload", "bypass",
        "privilege", "escalat", "reverse shell", "xss", "csrf", "sqli", "ssrf",
        "ctf", "capture the flag", "bugbounty", "bug bounty", "recon", "osint",
        "zero day", "0day", "infosec", "security research",
    )
    if any(kw in t for kw in cyber_kw):
        return True
    term = _extract_term_from_query(t).strip().lower().rstrip("?؟")
    return term in _CYBER_ACRONYMS


def _cyber_domain_boost(url: str, title: str, content: str) -> float:
    """Score multiplier: >1 for cyber sources, <1 for clearly off-topic domains."""
    combined = (" ".join([url or "", title or "", content or ""])).lower()
    for d in _CYBER_DOMAINS:
        if d in combined:
            return 2.2
    for d in _ANTI_CYBER_DOMAINS:
        if d in combined:
            return 0.3
    return 1.0


# ─── Term / acronym helpers ───────────────────────────────────────────────────

def _is_acronym(term: str) -> bool:
    T = (term or "").strip()
    if len(T) < 2:
        return False
    letters = [c for c in T if c.isalpha()]
    if not letters:
        return False
    return sum(1 for c in letters if c.isupper()) >= max(2, int(0.6 * len(letters)))


def _extract_term_from_query(q: str) -> str:
    q = (q or "").strip()
    _uname = _get_username()
    if q.lower().startswith(_uname.lower() + ":"):
        q = q[len(_uname) + 1:].strip()
    m = _re.search(
        r"(what\s+is|what's|what's|define|definition of|meaning of)\s+(.+)",
        q, flags=_re.I,
    )
    if m:
        return (m.group(2) or "").strip().strip("?؟。")
    return q


def _guess_acronym_expansion(term: str, items: List[Dict[str, Any]]) -> str:
    """Find the most likely expansion for an acronym from search result context."""
    T = (term or "").strip()
    if not _is_acronym(T):
        return ""

    cands: Dict[str, int] = {}

    def _add(s: str) -> None:
        s = (s or "").strip()
        if not s or len(s) > 120:
            return
        cands[s] = cands.get(s, 0) + 1

    p1 = _re.compile(rf"\b([A-Z][A-Za-z][A-Za-z ,&/\-]{{2,100}})\s*\(\s*{_re.escape(T)}\s*\)")
    p2 = _re.compile(rf"\b{_re.escape(T)}\s*\(\s*([A-Z][A-Za-z][A-Za-z ,&/\-]{{2,100}})\s*\)")

    for it in items or []:
        for chunk in (it.get("title") or "", it.get("content") or ""):
            if not chunk:
                continue
            for match in p1.findall(chunk):
                _add(match.strip())
            for match in p2.findall(chunk):
                _add(match.strip())

    if not cands:
        return ""
    return sorted(cands.items(), key=lambda kv: (-kv[1], len(kv[0])))[0][0]


def _choose_best_sentence(term: str, items: List[Dict[str, Any]]) -> str:
    """Extract the most informative definitional sentence for *term* from search results."""
    esc = _re.escape(term)
    pat  = _re.compile(rf"\b{esc}\b\s+(is|are|stands for|refers to|means)\b[^.?!]{{10,300}}[.?!]", flags=_re.I)
    soft = _re.compile(rf"\b{esc}\b[^.?!]{{0,80}}\bis\b[^.?!]{{10,300}}[.?!]", flags=_re.I)

    cand = []
    for it in items or []:
        txt = (it.get("content") or "").strip()
        if not txt or len(txt) < 60:
            continue
        for sent in _re.findall(r"[^.?!]*[.?!]", txt):
            s = sent.strip()
            if pat.search(s) or soft.search(s):
                wc = len(s.split())
                if 6 <= wc <= 45:
                    cand.append(s)

    seen: set[str] = set()
    uniq = []
    for s in cand:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(s)
    uniq.sort(key=lambda s: (len(s), s))
    return uniq[0].strip() if uniq else ""


def _definition_from_context(query: str, items: List[Dict[str, Any]]) -> str:
    """Build a best-effort definition for the query term from search result context."""
    term = _extract_term_from_query(query)
    s = _choose_best_sentence(term, items)
    if s:
        return s
    if _is_acronym(term):
        exp = _guess_acronym_expansion(term, items)
        if exp:
            return f"{term} stands for {exp}."
    return ""
