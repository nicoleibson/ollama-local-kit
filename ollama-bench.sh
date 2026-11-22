#!/usr/bin/env bash
# Benchmark rápido de los modelos instalados. Uso: ./ollama-bench.sh [modelo ...]
set -u
URL=${OLLAMA_HOST:-http://127.0.0.1:11434}
MODELOS=("$@")
[ ${#MODELOS[@]} -eq 0 ] && MODELOS=(gemma3:q4 gemma3:gpu nemotron3:gpu)   # qwen:gpu es solo embedding

for m in "${MODELOS[@]}"; do
  printf '%-16s ' "$m"
  curl -s "$URL/api/generate" -d "{\"model\":\"$m\",\"prompt\":\"Explica por qué el cielo es azul\",\"stream\":false,\"options\":{\"num_ctx\":2048,\"num_predict\":20}}" \
  | python3 -c '
import json,sys
d=json.load(sys.stdin)
if d.get("error"): print("ERROR:", d["error"][:80])
else:
    n=d["eval_count"]; t=d["eval_duration"]/1e9
    print(f"{n} tokens / {t:.2f}s = {n/t:.2f} tok/s" if t else "sin datos")
'
done
printf '\n%-16s ' "PROCESSOR"
curl -s "$URL/api/ps" | python3 -c '
import json,sys
for m in json.load(sys.stdin).get("models",[]):
    print("{}: {:.2f} GiB en VRAM".format(m["name"], m.get("size_vram",0)/2**30))
' 2>/dev/null || true
