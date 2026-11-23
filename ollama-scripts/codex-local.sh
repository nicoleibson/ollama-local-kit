#!/usr/bin/env bash
# Codex con modelos Ollama locales de la iGPU: sin :dgpu y sin créditos.
# Codex 0.169.0 excluye el daemon compartido cuando se pasa --profile, así que
# el perfil que genera `ollama launch codex` se fusiona en ~/.codex/config.toml
# y Codex se lanza sin --profile. El catálogo se alinea a 64k de contexto real.
set -euo pipefail

MODEL="${CODEX_MODEL:-nemotron3:codex}"
CTX="${CODEX_CTX:-65536}"
CODEX_DIR="${CODEX_HOME:-$HOME/.codex}"
CATALOG="$CODEX_DIR/model.json"
CONFIG="$CODEX_DIR/config.toml"
GENERATED="$CODEX_DIR/ollama-launch.config.toml"

case "$MODEL" in
    *:dgpu*)
        echo "ERROR: $MODEL pertenece a la R7 260X; usa nemotron3:codex." >&2
        exit 1
        ;;
esac
if ! ollama show "$MODEL" >/dev/null 2>&1; then
    echo "ERROR: falta el tag $MODEL. Créalo desde ~/ollama-local-kit/config/Modelfile.${MODEL/:/-}" >&2
    exit 1
fi

# Genera/actualiza el catálogo de modelos y el perfil dedicado de Ollama. En un
# terminal no interactivo puede terminar con "stdin is not a terminal" después
# de escribir los archivos; por eso el error se ignora y se continúa aquí.
ollama launch codex --model "$MODEL" --config -y >/dev/null 2>&1 || true

python3 - "$CATALOG" "$CONFIG" "$GENERATED" "$MODEL" "$CTX" <<'PY'
import json
import sys
import tomllib
from pathlib import Path

catalog_path, config_path, generated_path, model, ctx = sys.argv[1:]
ctx = int(ctx)
ROOT_TAG = "ocodex root"
SECTIONS_TAG = "ocodex sections"

# 1) Ollama anuncia el contexto nativo (1M), pero el tag Codex se creó con 64k.
try:
    catalog = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
except (OSError, ValueError):
    catalog = None
if catalog is not None:
    changed = False
    for item in catalog.get("models", []):
        if item.get("slug") == model and item.get("context_window") != ctx:
            item["context_window"] = ctx
            changed = True
    if changed:
        Path(catalog_path).write_text(
            json.dumps(catalog, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )

# 2) Extrae las claves raíz y las secciones del perfil generado.
generated = Path(generated_path).read_text(encoding="utf-8").splitlines()
split = next((i for i, line in enumerate(generated) if line.lstrip().startswith("[")),
             len(generated))
root_block = "\n".join(generated[:split]).strip()
sections_block = "\n".join(generated[split:]).strip()
if not root_block or not sections_block:
    sys.exit("ERROR: el perfil generado por ollama launch codex está incompleto")

# 3) Quita bloques gestionados en ejecuciones anteriores (idempotente).
def strip_managed(text, tag):
    start, end = f"# >>> {tag} >>>", f"# <<< {tag} <<<"
    out, skipping = [], False
    for line in text.splitlines():
        if line.strip() == start:
            skipping = True
            continue
        if skipping:
            if line.strip() == end:
                skipping = False
            continue
        out.append(line)
    return "\n".join(out).rstrip() + "\n"

config_path = Path(config_path)
base = strip_managed(strip_managed(
    config_path.read_text(encoding="utf-8") if config_path.exists() else "",
    ROOT_TAG,
), SECTIONS_TAG)

# Las claves raíz deben ir antes del primer encabezado [tabla] en TOML.
lines = base.splitlines()
insert_at = next((i for i, line in enumerate(lines) if line.lstrip().startswith("[")),
                 len(lines))
merged = (
    "\n".join(lines[:insert_at]).rstrip()
    + f"\n\n# >>> {ROOT_TAG} >>>\n{root_block}\n# <<< {ROOT_TAG} <<<\n"
    + "\n".join(lines[insert_at:]).lstrip("\n")
)
merged = merged.rstrip() + f"\n\n# >>> {SECTIONS_TAG} >>>\n{sections_block}\n# <<< {SECTIONS_TAG} <<<\n"

# No escribir nada si el resultado no es TOML válido.
tomllib.loads(merged)
config_path.write_text(merged, encoding="utf-8")
PY

if [[ "${1:-}" == "--config" ]]; then
    echo "Codex local: modelo=$MODEL | ctx real=$CTX | config=$CONFIG | sin --profile (daemon compartido disponible)"
    exit 0
fi
if [[ "${1:-}" == "--exec" ]]; then
    shift
    exec codex exec --skip-git-repo-check --ephemeral \
        -s read-only --color never "$@"
fi
exec codex "$@"
