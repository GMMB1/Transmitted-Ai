#!/bin/bash

PROJECT="$(cd "$(dirname "$0")" && pwd)"
ICON="$PROJECT/Arwanos_icon.png"
LOGDIR="$PROJECT/logs"
LOGFILE="$LOGDIR/arwanos_startup.log"

# ── Wayland / HiDPI normalization ──────────────────────────────────────────────
export GDK_SCALE=1
export GDK_DPI_SCALE=1

# Ensure DISPLAY is set for XWayland sessions
if [ -z "$DISPLAY" ] && [ -n "$WAYLAND_DISPLAY" ]; then
    export DISPLAY=:0
fi

# ── CUDA / GPU environment ──────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
export CUDA_PATH=/opt/cuda
export LD_LIBRARY_PATH=/usr/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH}

# ── GPU Detection & Startup Log ────────────────────────────────────────────────
mkdir -p "$LOGDIR"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

# Gather GPU info
GPU_NAME=$(nvidia-smi --query-gpu=name          --format=csv,noheader 2>/dev/null || echo "N/A")
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total  --format=csv,noheader 2>/dev/null || echo "N/A")
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "N/A")
GPU_VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || echo "N/A")
OLLAMA_VER=$(/usr/local/bin/ollama --version 2>/dev/null || ollama --version 2>/dev/null || echo "N/A")

# Check CUDA backend presence
CUDA_BACKEND="NOT FOUND ⚠️"
if ls /usr/local/lib/ollama/cuda_v12/libggml-cuda.so \
      /usr/local/lib/ollama/cuda_v13/libggml-cuda.so \
      /usr/lib/ollama/libggml-cuda*.so 2>/dev/null | grep -q .; then
    CUDA_BACKEND="FOUND ✅"
fi

# Check Ollama GPU/CPU mode
OLLAMA_PROC="offline"
if pgrep -x "ollama" > /dev/null; then
    PROC=$(ollama ps 2>/dev/null | grep -v '^NAME' | grep -oP '\d+% (GPU|CPU)')
    if [ -n "$PROC" ]; then
        OLLAMA_PROC="$PROC"
    else
        OLLAMA_PROC="running (no model loaded yet)"
    fi
fi

# ── Write log ──────────────────────────────────────────────────────────────────
{
echo "══════════════════════════════════════════════════════"
echo "  Arwanos Startup — $TIMESTAMP"
echo "══════════════════════════════════════════════════════"
echo "  GPU        : $GPU_NAME"
echo "  VRAM Total : $GPU_VRAM"
echo "  VRAM Used  : $GPU_VRAM_USED"
echo "  Driver     : $GPU_DRIVER"
echo "  CUDA Back  : $CUDA_BACKEND"
echo "  Ollama     : $OLLAMA_VER"
echo "  Processor  : $OLLAMA_PROC"
echo "══════════════════════════════════════════════════════"
echo ""
} | tee -a "$LOGFILE"

# ── Check Ollama status & GPU mode ─────────────────────────────────────────────
if pgrep -x "ollama" > /dev/null; then
    OLLAMA_RUNNING=true
    PROC_COL=$(ollama ps 2>/dev/null | grep -v '^NAME' | awk '{print $4}')
    if echo "$PROC_COL" | grep -qi "100% CPU"; then
        OLLAMA_STATUS="⚠️  Ollama running — but on CPU (GPU backend missing)\nFix: sudo curl -fsSL https://ollama.com/install.sh | sh"
    elif echo "$PROC_COL" | grep -qi "gpu\|cuda"; then
        OLLAMA_STATUS="✅  Ollama running on GPU"
    else
        if [ "$CUDA_BACKEND" = "FOUND ✅" ]; then
            OLLAMA_STATUS="✅  Ollama running (GPU backend ready)"
        else
            OLLAMA_STATUS="⚠️  Ollama running — CPU-only build\nFix: sudo curl -fsSL https://ollama.com/install.sh | sh"
        fi
    fi
else
    OLLAMA_RUNNING=false
    OLLAMA_STATUS="Ollama offline — will start with GPU env"
fi

# ── Get current model from config.json ────────────────────────────────────────
MODEL=$(python3 -c "import json; d=json.load(open('$PROJECT/config.json')); print(d.get('model_name','llama3:8b'))" 2>/dev/null || echo "llama3:8b")

# ── Launch dialog ──────────────────────────────────────────────────────────────
zenity --question \
    --title="Arwanos" \
    --window-icon="$ICON" \
    --text="<span size='large'><b>Arwanos AI Agent  v10</b></span>\n\n$OLLAMA_STATUS\nModel:  <b>$MODEL</b>\n\nReady to launch?" \
    --ok-label="  Launch  " \
    --cancel-label="Cancel" \
    --width=420 \
    --height=200

[ $? -ne 0 ] && exit 0

# ── Start Ollama if needed ──────────────────────────────────────────────────────
if [ "$OLLAMA_RUNNING" = false ]; then
    CUDA_VISIBLE_DEVICES=0 \
    LD_LIBRARY_PATH=/usr/lib:/opt/cuda/lib64:${LD_LIBRARY_PATH} \
    ollama serve > /dev/null 2>&1 &
    sleep 3
fi

# ── Launch Arwanos ─────────────────────────────────────────────────────────────
cd "$PROJECT" || exit
source .venv/bin/activate
python Arwanos_v10.py
deactivate
