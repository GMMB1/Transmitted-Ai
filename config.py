# config.py — auto hardware detection with RAM + GPU tiers
from __future__ import annotations
import os, shutil, subprocess, json
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Hardware probe helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None


def _gpu_vram_mib() -> int | None:
    """Total VRAM across all NVIDIA GPUs (MiB). None if no NVIDIA GPU found."""
    if not _has_nvidia_smi():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=2.0,
        ).strip().splitlines()
        return sum(int(x.strip()) for x in out if x.strip().isdigit()) or None
    except Exception:
        return None


def _has_amd_gpu() -> bool:
    """Check for AMD ROCm GPU."""
    return shutil.which("rocm-smi") is not None or os.path.exists("/dev/kfd")


def _cpu_threads() -> int:
    try:
        import multiprocessing as mp
        cores = mp.cpu_count() or 8
        return max(2, cores - 1)   # leave 1 core free for the OS
    except Exception:
        return 8


def _ram_gb() -> float:
    """System RAM in GB. Falls back to 8 GB if detection fails."""
    # Linux / macOS via psutil (preferred)
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    # Linux fallback: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
    except Exception:
        pass
    # macOS fallback: sysctl
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], timeout=2.0, text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:
        pass
    return 8.0   # safe conservative fallback


# ─────────────────────────────────────────────────────────────────────────────
# Hardware tier logic
# ─────────────────────────────────────────────────────────────────────────────

def _detect_tier() -> dict:
    """
    Probe GPU + RAM and return a dict of recommended Ollama/LLM settings.

    Tiers
    ─────
    POWER   : GPU (NVIDIA/AMD) + ≥ 24 GB RAM   → max context, full GPU offload
    HIGH    : GPU + 16–24 GB RAM                → high context, full GPU offload
    MID     : GPU + 8–16 GB RAM   OR no GPU + ≥ 24 GB RAM
    NORMAL  : No GPU + 16–24 GB RAM
    LIGHT   : No GPU + 8–16 GB RAM
    MINIMAL : No GPU + < 8 GB RAM               → survival mode
    """
    ram = _ram_gb()
    vram = _gpu_vram_mib()                  # MiB or None
    has_gpu = _has_nvidia_smi() or _has_amd_gpu()
    threads = _cpu_threads()

    # ------- pick base settings per tier -----------------------------------
    if has_gpu and ram >= 24:
        tier = "POWER"
        num_ctx      = 8192
        num_predict  = 1024
        num_gpu_layers = 999
        temperature  = 0.2
    elif has_gpu and ram >= 16:
        tier = "HIGH"
        num_ctx      = 6144
        num_predict  = 512
        num_gpu_layers = 999
        temperature  = 0.2
    elif has_gpu and ram >= 8:
        tier = "MID-GPU"
        num_ctx      = 4096
        num_predict  = 256
        num_gpu_layers = 28          # partial offload to save VRAM
        temperature  = 0.25
    elif not has_gpu and ram >= 24:
        tier = "MID-CPU"
        num_ctx      = 4096
        num_predict  = 512
        num_gpu_layers = 0
        temperature  = 0.25
    elif not has_gpu and ram >= 16:
        tier = "NORMAL"
        num_ctx      = 3072
        num_predict  = 256
        num_gpu_layers = 0
        temperature  = 0.3
    elif not has_gpu and ram >= 8:
        tier = "LIGHT"
        num_ctx      = 2048
        num_predict  = 192
        num_gpu_layers = 0
        temperature  = 0.3
    else:
        tier = "MINIMAL"
        num_ctx      = 1024
        num_predict  = 128
        num_gpu_layers = 0
        temperature  = 0.35

    # ------- model choice (only switch if larger VRAM detected) -----------
    model = "llama3.1:8b"          # safe default for all tiers

    gpu_opts: dict = {}
    if has_gpu:
        gpu_opts = {
            "num_gpu":        1,
            "num_gpu_layers": num_gpu_layers,
            "main_gpu":       0,
        }

    return {
        "tier":       tier,
        "model":      model,
        "ram_gb":     round(ram, 1),
        "vram_mib":   vram,
        "use_gpu":    has_gpu,
        "threads":    threads,
        "options": {
            "num_ctx":        num_ctx,
            "num_predict":    num_predict,
            "num_thread":     threads,
            "temperature":    temperature,
            "top_p":          0.9,
            "top_k":          40,
            "repeat_penalty": 1.05,
            **gpu_opts,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    model:   str  = "llama3.1:8b"
    threads: int  = field(default_factory=_cpu_threads)
    use_gpu: bool = field(default_factory=_has_nvidia_smi)
    options: dict = field(default_factory=dict)

    def as_ollama_options(self) -> dict:
        base = {
            "temperature":    0.2,
            "top_p":          0.9,
            "repeat_penalty": 1.05,
            "num_ctx":        4096,
            "num_thread":     self.threads,
        }
        if self.use_gpu:
            base.update({"num_gpu": 1, "num_gpu_layers": 999, "main_gpu": 0})
        base.update(self.options or {})
        return base


@dataclass
class AppConfig:
    llm: LLMConfig

    # ── cached probe so we only run detection once per process ───────────────
    _tier_info: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def auto() -> "AppConfig":
        """
        Auto-detect GPU + RAM and pick the best settings for this machine.

        Tier table (examples):
          POWER   : NVIDIA GPU + ≥ 24 GB RAM  → ctx 8192, full GPU offload
          HIGH    : NVIDIA GPU + 16–24 GB RAM → ctx 6144, full GPU offload
          MID-GPU : NVIDIA GPU + 8–16 GB RAM  → ctx 4096, partial GPU offload
          MID-CPU : No GPU + ≥ 24 GB RAM      → ctx 4096, CPU only
          NORMAL  : No GPU + 16–24 GB RAM     → ctx 3072, CPU only
          LIGHT   : No GPU + 8–16 GB RAM      → ctx 2048, CPU only
          MINIMAL : No GPU + < 8 GB RAM       → ctx 1024, survival mode
        """
        info = _detect_tier()
        # ── colorful terminal banner ─────────────────────────────────────────
        _P  = "\033[38;5;208m"   # orange
        _C  = "\033[96m"         # cyan
        _Y  = "\033[93m"         # yellow
        _G  = "\033[92m"         # green
        _R  = "\033[91m"         # red
        _W  = "\033[97m"         # white
        _M  = "\033[95m"         # magenta
        _D  = "\033[2m"          # dim
        _N  = "\033[0m"          # reset
        _BD = "\033[1m"          # bold

        tier_color = {
            "POWER":   _R + _BD,
            "HIGH":    _P + _BD,
            "MID-GPU": _Y + _BD,
            "MID-CPU": _Y,
            "NORMAL":  _C,
            "LIGHT":   _G,
            "MINIMAL": _D,
        }.get(info["tier"], _W)

        gpu_disp = (
            f"{_G}✔ Yes  VRAM {info['vram_mib']} MiB{_N}"
            if info["use_gpu"] else f"{_D}✘ No (CPU only){_N}"
        )

        print(
            f"\n{_M}{_BD}╔══════════════════════════════════════════╗{_N}\n"
            f"{_M}{_BD}║  ⚙️  Rona Hardware Auto-Config            ║{_N}\n"
            f"{_M}{_BD}╚══════════════════════════════════════════╝{_N}\n"
            f"  {_D}Tier    {_N}: {tier_color}{info['tier']}{_N}\n"
            f"  {_D}RAM     {_N}: {_C}{_BD}{info['ram_gb']} GB{_N}\n"
            f"  {_D}GPU     {_N}: {gpu_disp}\n"
            f"  {_D}Threads {_N}: {_Y}{info['threads']}{_N}\n"
            f"  {_D}Model   {_N}: {_W}{info['model']}{_N}\n"
            f"  {_D}ctx     {_N}: {_P}{_BD}{info['options']['num_ctx']} tokens{_N}\n"
            f"  {_D}predict {_N}: {_G}{info['options']['num_predict']} tokens{_N}\n"
            f"{_D}{'─' * 44}{_N}\n"
        )
        cfg = AppConfig(
            llm=LLMConfig(
                model=info["model"],
                use_gpu=info["use_gpu"],
                options=info["options"],
            )
        )
        cfg._tier_info = info
        return cfg

    def apply_env(self) -> None:
        """Set thread-count env vars for BLAS/OpenMP libs."""
        t = str(self.llm.threads)
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, t)

    @property
    def tier(self) -> str:
        return self._tier_info.get("tier", "UNKNOWN")

    @property
    def ram_gb(self) -> float:
        return self._tier_info.get("ram_gb", 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Old API — kept for backward compatibility
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AutoConfig:
    model_name:  str   = "llama3:8b"
    temperature: float = 0.2
    num_ctx:     int   = 4096
    num_predict: int   = 512
    num_thread:  int   = field(default_factory=_cpu_threads)
    use_gpu:     bool  = False
    gpu_count:   int   = 0

    @staticmethod
    def _has_nvidia() -> tuple[bool, int]:
        try:
            if not _has_nvidia_smi():
                return (False, 0)
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=2.0,
            ).splitlines()
            gpus = [ln.strip() for ln in out if ln.strip()]
            return (len(gpus) > 0, len(gpus))
        except Exception:
            return (False, 0)

    @classmethod
    def from_system(cls, model_name: str | None = None) -> "AutoConfig":
        has, cnt = cls._has_nvidia()
        return cls(
            model_name=model_name or "llama3.1:8b",
            use_gpu=has,
            gpu_count=cnt,
            num_thread=_cpu_threads(),
            num_ctx=4096,
            num_predict=512,
            temperature=0.2,
        )

    def apply_env(self) -> None:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, str(self.num_thread))

    def ollama_options(self) -> dict:
        base = {
            "temperature": self.temperature,
            "num_ctx":     self.num_ctx,
            "num_predict": self.num_predict,
            "num_thread":  self.num_thread,
        }
        if self.use_gpu:
            base.update({"num_gpu": 1, "num_gpu_layers": 999, "main_gpu": 0})
        return base

    def as_json(self) -> str:
        return json.dumps({
            "model_name":  self.model_name,
            "temperature": self.temperature,
            "num_ctx":     self.num_ctx,
            "num_predict": self.num_predict,
            "num_thread":  self.num_thread,
            "use_gpu":     self.use_gpu,
            "gpu_count":   self.gpu_count,
        }, indent=2)
