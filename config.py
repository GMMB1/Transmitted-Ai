# config.py
from __future__ import annotations
import os, shutil, subprocess, json
from dataclasses import dataclass, field

# ---------- helpers ----------
def _has_nvidia_smi() -> bool:
    return shutil.which("nvidia-smi") is not None

def _gpu_vram_mib() -> int | None:
    """Total VRAM across GPUs (MiB). None if unknown."""
    if not _has_nvidia_smi():
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=2.0
        ).strip().splitlines()
        return sum(int(x.strip()) for x in out if x.strip().isdigit())
    except Exception:
        return None

def _cpu_threads() -> int:
    try:
        import multiprocessing as mp
        cores = mp.cpu_count() or 8
        # leave one core free
        return max(2, cores - 1)
    except Exception:
        return 8

# ---------- NEW API (preferred) ----------
@dataclass
class LLMConfig:
    model: str = "llama3.1:8b"
    threads: int = field(default_factory=_cpu_threads)
    use_gpu: bool = field(default_factory=_has_nvidia_smi)
    options: dict = field(default_factory=dict)

    def as_ollama_options(self) -> dict:
        base = {
            "temperature": 0.2,
            "top_p": 0.9,
            "repeat_penalty": 1.05,
            "num_ctx": 4096,
            "num_thread": self.threads,
            # num_predict intentionally omitted; set in code if you want a cap
        }
        if self.use_gpu:
            base.update({
                "num_gpu": 1,
                "num_gpu_layers": 999,  # offload as many layers as supported
                "main_gpu": 0,
            })
        base.update(self.options or {})
        return base

@dataclass
class AppConfig:
    llm: LLMConfig

    @staticmethod
    def auto() -> "AppConfig":
        vram = _gpu_vram_mib()
        use_gpu = _has_nvidia_smi()

        # Select a sensible quant based on VRAM
        model = "llama3.1:8b"     # 8 GB friendly
        if vram and vram >= 12288:
            model = "llama3.1:8b" # 12 GB+
        if vram and vram >= 16384:
            model = "llama3.1:8b"   # 16 GB+

        return AppConfig(
            llm=LLMConfig(
                model=model,
                use_gpu=use_gpu,
                options={
                    # speed-focused defaults; keep or tweak later
                    "num_ctx": 3072,
                    "num_predict": 256,
                    "num_thread": _cpu_threads(),
                    "top_p": 0.9,
                    "top_k": 30,
                    "temperature": 0.3,
                    # full GPU offload if CUDA exists:
                    **({"num_gpu": 1, "num_gpu_layers": 999, "main_gpu": 0} if use_gpu else {})
                },
            )
        )

    # Convenience: old code may expect these
    def apply_env(self) -> None:
        """Thread/BLAS env hints (safe no-ops if libs absent)."""
        os.environ.setdefault("OMP_NUM_THREADS", str(self.llm.threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.llm.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.llm.threads))

# ---------- OLD API (kept for backward compatibility) ----------
@dataclass
class AutoConfig:
    model_name: str = "llama3:8b"   # will be overridden by from_system()
    temperature: float = 0.2
    num_ctx: int = 4096
    num_predict: int = 512
    num_thread: int = field(default_factory=_cpu_threads)
    use_gpu: bool = False
    gpu_count: int = 0

    @staticmethod
    def _has_nvidia() -> tuple[bool, int]:
        try:
            if not _has_nvidia_smi():
                return (False, 0)
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True, timeout=2.0
            ).splitlines()
            gpus = [ln.strip() for ln in out if ln.strip()]
            return (len(gpus) > 0, len(gpus))
        except Exception:
            return (False, 0)

    @classmethod
    def from_system(cls, model_name: str | None = None) -> "AutoConfig":
        has, cnt = cls._has_nvidia()
        cfg = cls(
            model_name=model_name or "llama3.1:8b",  # upgraded default
            use_gpu=has,
            gpu_count=cnt,
            num_thread=_cpu_threads(),
            num_ctx=4096,
            num_predict=512,
            temperature=0.2,
        )
        return cfg

    def apply_env(self) -> None:
        os.environ.setdefault("OMP_NUM_THREADS", str(self.num_thread))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(self.num_thread))
        os.environ.setdefault("MKL_NUM_THREADS", str(self.num_thread))
        # Optional: force CUDA
        # if self.use_gpu: os.environ.setdefault("OLLAMA_USE_CUDA", "1")

    def ollama_options(self) -> dict:
        base = {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "num_thread": self.num_thread,
        }
        if self.use_gpu:
            base.update({
                "num_gpu": 1,
                "num_gpu_layers": 999,
                "main_gpu": 0,
            })
        return base

    def as_json(self) -> str:
        return json.dumps({
            "model_name": self.model_name,
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "num_thread": self.num_thread,
            "use_gpu": self.use_gpu,
            "gpu_count": self.gpu_count,
        }, indent=2)
