#!/usr/bin/env bash
# Comprobaciones rápidas del kit: sintaxis, catálogo, índice RAG y doctor.
set -u
fail=0

step() { printf '[%s] %s\n' "$1" "$2"; }

step 1/5 'Sintaxis de ollama-cli.py'
python3 -m py_compile /home/onion/ollama-cli.py || fail=1

step 2/5 'Sintaxis de los scripts'
for f in /home/onion/ollama-scripts/*.sh; do
    bash -n "$f" || fail=1
done

step 3/5 'Catálogo JSON'
python3 -m json.tool /home/onion/ollama-commands.json >/dev/null || fail=1

step 4/5 'Índice RAG'
python3 - <<'PY' || fail=1
import json
p = '/home/onion/.ollama-rag/index.json'
d = json.load(open(p))
print('%d fragmentos, embedding %s' % (len(d.get('chunks', [])), d.get('embedding_model')))
PY

step 5/5 'Doctor'
/home/onion/ollama-cli.py doctor | tail -1

if [ "$fail" -ne 0 ]; then
    echo 'SELFTEST FALLÓ: revisa las líneas marcadas arriba.'
    exit 1
fi
echo 'SELFTEST OK'
