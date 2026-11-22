#!/usr/bin/env bash
# Descarga todos los modelos actualmente cargados en Ollama.
set -u
MODELOS=$(curl -s http://127.0.0.1:11434/api/ps 2>/dev/null | python3 -c '
import json,sys
try:
    d=json.load(sys.stdin)
    print(" ".join(m["name"] for m in d.get("models",[])))
except Exception:
    pass
')
if [ -z "$MODELOS" ]; then
  echo "No hay modelos cargados."
  exit 0
fi
for m in $MODELOS; do
  echo "Descargando $m ..."
  ollama stop "$m"
done
