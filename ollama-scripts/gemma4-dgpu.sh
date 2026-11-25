#!/usr/bin/env bash
# Gemma 4 E4B en la R7 260X, sin tocar el servicio principal (iGPU).
# Arranca una instancia temporal en 11435, ejecuta la petición y la apaga siempre.
#
# Uso:
#   gemma4-dgpu.sh "tu pregunta"          # una respuesta
#   gemma4-dgpu.sh --chat                 # chat interactivo (Ctrl+C para salir)
#   gemma4-dgpu.sh --bench                # medir 80 tokens
#   gemma4-dgpu.sh --stats "tu pregunta"  # cualquier opción de `ollama-cli.py chat`
#   gemma4-dgpu.sh --safe-fit "pregunta"  # ctx 8192, capas ajustadas por Ollama
#   gemma4-dgpu.sh --print-env            # muestra el plan sin arrancar nada
set -euo pipefail

PORT=11435
HOST="http://127.0.0.1:$PORT"
MODEL="gemma4:dgpu"
CLI="/ia-local/project/ollama-local-kit/ollama-cli.py"
LOG="${TMPDIR:-/tmp}/ollama-gemma4-dgpu.log"
PIDFILE="${TMPDIR:-/tmp}/ollama-gemma4-dgpu.pid"
SAFE_MODEL="gemma4:dgpu-safe"
SAFE_MODELFILE="/ia-local/project/ollama-local-kit/config/Modelfile.gemma4-dgpu-safe"
SAFE_FIT=0
CTX=2048

if [[ "${1:-}" == "--safe-fit" ]]; then
    SAFE_FIT=1
    CTX=8192
    MODEL="$SAFE_MODEL"
    shift
fi

if [[ "${1:-}" == "--print-env" ]]; then
    printf 'modelo=%s | contexto=%s | puerto=%s | GPU=Vulkan 1\n' "$MODEL" "$CTX" "$PORT"
    if (( SAFE_FIT )); then
        printf 'receta=%s (sin num_gpu fijo: Ollama ajusta capas)\n' "$SAFE_MODELFILE"
    fi
    exit 0
fi

stop_pidfile() {
    local announce="${1:-0}"
    [[ -f "$PIDFILE" ]] || return 0
    local pid
    pid=$(cat "$PIDFILE" 2>/dev/null || true)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        if (( announce )); then
            echo "Aviso: PID $pid de una ejecución anterior sigue vivo; se termina." >&2
        fi
        kill -TERM "$pid" 2>/dev/null || true
    fi
    rm -f "$PIDFILE"
}

cleanup() { stop_pidfile 0; }
trap cleanup EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

if curl -sf --max-time 2 "$HOST/api/tags" >/dev/null 2>&1; then
    echo "ERROR: ya hay una instancia escuchando en $PORT. Ciérrala antes de continuar." >&2
    exit 1
fi

if ! MAIN_TAGS=$(curl -sf --max-time 2 http://127.0.0.1:11434/api/tags); then
    echo "ERROR: el servicio principal no responde en 11434. Arranca Ollama y reintenta." >&2
    exit 1
fi
if ! grep -q "$MODEL" <<<"$MAIN_TAGS"; then
    if (( SAFE_FIT )); then
        echo "Creando $MODEL con la receta sin num_gpu fijo..." >&2
        ollama create "$MODEL" -f "$SAFE_MODELFILE" >&2
    else
        echo "ERROR: falta el tag $MODEL. Créalo con:" >&2
        echo "  ollama create $MODEL -f /ia-local/project/ollama-local-kit/config/Modelfile.gemma4-dgpu" >&2
        exit 1
    fi
fi

stop_pidfile 1

nohup env \
    OLLAMA_HOST="$HOST" \
    OLLAMA_MODELS=/usr/share/ollama/.ollama/models \
    GGML_VK_VISIBLE_DEVICES=1 \
    OLLAMA_CONTEXT_LENGTH="$CTX" \
    OLLAMA_NUM_PARALLEL=1 \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_KEEP_ALIVE=5m \
    OLLAMA_KV_CACHE_TYPE=q8_0 \
    OLLAMA_IGPU_ENABLE=1 \
    OLLAMA_NO_CLOUD=true \
    OLLAMA_NOPRUNE=true \
    ollama serve > "$LOG" 2>&1 &
echo $! > "$PIDFILE"

ready=0
for _ in $(seq 1 20); do
    if curl -sf --max-time 2 "$HOST/api/tags" >/dev/null 2>&1; then ready=1; break; fi
    sleep 1
done
if (( ! ready )); then
    echo "ERROR: la instancia temporal no arrancó. Revisa $LOG" >&2
    exit 1
fi

echo "R7 260X aislada en $PORT (GPU 1, ctx $CTX$( (( SAFE_FIT )) && printf ', capas automáticas' )). Se apagará al terminar." >&2

case "${1:-}" in
    --bench)
        OLLAMA_HOST="$HOST" "$CLI" bench "$MODEL" --tokens 80 --ctx "$CTX" --no-save
        ;;
    --chat)
        shift
        OLLAMA_HOST="$HOST" "$CLI" chat -m "$MODEL" --ctx "$CTX" "$@"
        ;;
    *)
        OLLAMA_HOST="$HOST" "$CLI" chat -m "$MODEL" --ctx "$CTX" "$@"
        ;;
esac
