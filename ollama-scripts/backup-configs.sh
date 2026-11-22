#!/usr/bin/env bash
# Copia la configuración y las herramientas a una carpeta con fecha.
set -u
DEST="/home/onion/Backups.Ollama/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$DEST" || exit 1
cp /etc/systemd/system/ollama.service.d/override.conf "$DEST"/ 2>/dev/null || true
cp /etc/systemd/system/cpu-performance.service "$DEST"/ 2>/dev/null || true
cp /etc/systemd/system/cpu-performance-enforce.timer "$DEST"/ 2>/dev/null || true
cp /etc/sysctl.d/99-ollama.conf "$DEST"/ 2>/dev/null || true
cp /home/onion/ollama-cli.py /home/onion/ollama-commands.json \
   /home/onion/ollama-kit.md /home/onion/ollama-bench.sh "$DEST"/ 2>/dev/null || true
cp /home/onion/.bashrc "$DEST/bashrc" 2>/dev/null || true
cp /home/onion/.ollama-prompts.json "$DEST"/ 2>/dev/null || true
cp /home/onion/.ollama-bench-history.jsonl "$DEST"/ 2>/dev/null || true
cp -r /home/onion/.ollama-sessions "$DEST"/sessions 2>/dev/null || true
cp -r /home/onion/ollama-scripts "$DEST"/ 2>/dev/null || true
mkdir -p "$DEST/bash-completion" 2>/dev/null || true
cp /home/onion/.local/share/bash-completion/completions/ollama-cli \
   "$DEST/bash-completion"/ 2>/dev/null || true
echo "Respaldo creado en: $DEST"
ls -1 "$DEST"
