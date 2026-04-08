#!/bin/bash

PROJECT="/home/gmm/Project/Transmitted-Ai"
ICON="$PROJECT/Arwanos_icon.png"

# ── Wayland / HiDPI normalization ──────────────────────────────────────────────
# GNOME's compositor-level scaling can confuse Tkinter's DPI detector on XWayland,
# making CTk double widget heights while place(y=N) coords stay unscaled → buttons
# overlap.  These variables tell the GDK/X layer to NOT apply additional scaling on
# top of what the compositor already does, giving CTk a clean 1:1 baseline.
export GDK_SCALE=1
export GDK_DPI_SCALE=1

# Ensure DISPLAY is set for XWayland sessions (some launchers drop it)
if [ -z "$DISPLAY" ] && [ -n "$WAYLAND_DISPLAY" ]; then
    export DISPLAY=:0
fi

# ── Check Ollama status ────────────────────────────────────────────────────────
if pgrep -x "ollama" > /dev/null; then
    OLLAMA_STATUS="Ollama is running"
    OLLAMA_RUNNING=true
else
    OLLAMA_STATUS="Ollama offline - will start in GPU mode"
    OLLAMA_RUNNING=false
fi

# ── Get current model from config.json ────────────────────────────────────────
MODEL=$(python3 -c "import json; d=json.load(open('$PROJECT/config.json')); print(d.get('model_name','llama3:8b'))" 2>/dev/null || echo "llama3:8b")

# ── Launch dialog ──────────────────────────────────────────────────────────────
zenity --question \
    --title="Arwanos" \
    --window-icon="$ICON" \
    --text="<span size='large'><b>Arwanos AI Agent  v9.8</b></span>\n\n$OLLAMA_STATUS\nModel:  <b>$MODEL</b>\n\nReady to launch?" \
    --ok-label="  Launch  " \
    --cancel-label="Cancel" \
    --width=380 \
    --height=180

# User clicked Cancel
[ $? -ne 0 ] && exit 0

# ── Start Ollama if needed ─────────────────────────────────────────────────────
if [ "$OLLAMA_RUNNING" = false ]; then
    CUDA_VISIBLE_DEVICES=0 ollama serve > /dev/null 2>&1 &
    sleep 2
fi

# ── Launch Arwanos ─────────────────────────────────────────────────────────────
cd "$PROJECT" || exit
source venv/bin/activate
python Arwanos_v9_8.py
deactivate
