#!/usr/bin/env bash
# Comprobaciones rápidas del kit: sintaxis, catálogo, índice RAG, doctor y pruebas.
set -u
fail=0
TESTS="${OLLAMA_KIT_TESTS:-/ia-local/project/ollama-local-kit/tests}"

step() { printf '[%s] %s\n' "$1" "$2"; }

step 1/6 'Sintaxis de ollama-cli.py'
python3 -m py_compile /ia-local/project/ollama-local-kit/ollama-cli.py || fail=1

step 2/6 'Sintaxis de los scripts'
for f in /ia-local/project/ollama-local-kit/ollama-scripts/*.sh; do
    bash -n "$f" || fail=1
done

step 3/6 'Catálogo JSON'
python3 -m json.tool /ia-local/project/ollama-local-kit/ollama-commands.json >/dev/null || fail=1

step 4/6 'Índice RAG'
python3 - <<'PY' || fail=1
import json
p = '/home/onion/.ollama-rag/index.json'
d = json.load(open(p))
print('%d fragmentos, embedding %s' % (len(d.get('chunks', [])), d.get('embedding_model')))
PY

step 5/6 'Doctor'
/ia-local/project/ollama-local-kit/ollama-cli.py doctor | tail -1

step 6/6 'Pruebas versionadas'
if [ -d "$TESTS" ]; then
    for f in "$TESTS"/test-*.py; do
        [ -f "$f" ] || continue
        printf '  %s\n' "$(basename "$f")"
        if ! python3 "$f" >/dev/null; then
            echo "  FALLÓ: $f"
            fail=1
        fi
    done
else
    echo "  Sin carpeta tests/ en $HERE"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo 'SELFTEST FALLÓ: revisa las líneas marcadas arriba.'
    exit 1
fi
echo 'SELFTEST OK'
