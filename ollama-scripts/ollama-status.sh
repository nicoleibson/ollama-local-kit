#!/usr/bin/env bash
echo "=== SERVICIO ==="
systemctl is-active ollama
echo
echo "=== MODELOS CARGADOS ==="
ollama ps
echo
echo "=== GPU DETECTADA EN LOGS ==="
journalctl -u ollama -n 5000 --no-pager 2>/dev/null | grep -oE 'RADV [A-Z]+|offloaded [0-9]+/[0-9]+ layers' | tail -5
echo
echo "=== MODELOS INSTALADOS ==="
ollama list
