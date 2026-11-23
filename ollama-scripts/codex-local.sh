#!/usr/bin/env bash
# Codex con modelos Ollama locales de la iGPU: sin :dgpu y sin créditos.
# `ollama launch codex` crea el perfil; este wrapper fija el contexto real
# (64k) en el catálogo y ofrece un modo --exec no interactivo para pruebas.
set -euo pipefail

MODEL="${CODEX_MODEL:-nemotron3:codex}"
CTX="${CODEX_CTX:-65536}"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"
PROFILE="ollama-launch"
CATALOG="$CODEX_DIR/model.json"

case "$MODEL" in
    *:dgpu*)
        echo "ERROR: $MODEL pertenece a la R7 260X; usa nemotron3:codex o qwen35:codex." >&2
        exit 1
        ;;
esac
if ! ollama show "$MODEL" >/dev/null 2>&1; then
    echo "ERROR: falta el tag $MODEL. Créalo desde ~/ollama-local-kit/config/Modelfile.${MODEL/:/-}" >&2
    exit 1
fi

# Genera/actualiza ~/.codex/ollama-launch.config.toml y model.json. En un
# terminal no interactivo puede terminar con "stdin is not a terminal" después
# de escribir los archivos; por eso el error se ignora y se lanza Codex aquí.
ollama launch codex --model "$MODEL" --config -y >/dev/null 2>&1 || true

# Ollama anuncia el contexto nativo del modelo (1M), pero el tag Codex se creó
# con 65536. Alinear el catálogo evita que Codex pida más contexto del real.
python3 - "$CATALOG" "$MODEL" "$CTX" <<'PY'
import json
import sys

path, model, ctx = sys.argv[1], sys.argv[2], int(sys.argv[3])
try:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
except (OSError, ValueError):
    sys.exit(0)
changed = False
for item in data.get("models", []):
    if item.get("slug") == model and item.get("context_window") != ctx:
        item["context_window"] = ctx
        changed = True
if changed:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
PY

if [[ "${1:-}" == "--config" ]]; then
    echo "Perfil: $PROFILE | modelo: $MODEL | ctx real: $CTX | base_url: 127.0.0.1:11434/v1/"
    exit 0
fi
if [[ "${1:-}" == "--exec" ]]; then
    shift
    exec codex exec --profile "$PROFILE" --skip-git-repo-check --ephemeral \
        -s read-only --color never "$@"
fi
exec codex --profile "$PROFILE" "$@"
