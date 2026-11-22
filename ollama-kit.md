# Manual Ollama — A8-9600 / A10-9700 (iGPU Carrizo)

**Estado: configuración aplicada el 21/11/2026.** Lo que aparece en la sección 1 ya está
en marcha; se conserva como referencia para reproducirlo tras una reinstalación.

> El equipo se conoce como A8-9600, aunque `lscpu` reporta `AMD A10-9700 RADEON R7`.
> En ambos casos la iGPU es Carrizo y es el dispositivo Vulkan 0 usado por Ollama.

## TL;DR — lo que más usarás

```bash
ollama run gemma3:q4 "tu pregunta"        # chat rápido (9,3 tok/s)
~/ollama-act.py "muestra la fecha"        # ejecutar acción (6-8 tokens)
~/ollama-bench.sh gemma3:q4               # medir velocidad
orag index /home/onion/ollama-kit.md      # indexar archivos para RAG
orag ask "cómo funciona el modo act"      # preguntar sobre lo indexado
orun "muestra la fecha"                   # enrutador: act / rag / chat
owatch                                    # monitor en vivo
ollama ps                                 # comprobar que dice 100% GPU
ollama stop gemma3:q4                     # liberar memoria al terminar
```

## 0. CLI personal (`~/ollama-cli.py`)

Todo lo anterior se puede hacer con una sola herramienta. Está instalada y probada:

```bash
ollama-cli.py --help                        # ver todos los comandos
ollama-cli.py list                          # modelos + uso recomendado
ollama-cli.py chat "hola"                   # chat streaming
ollama-cli.py chat -m gemma3:gpu --stats "explica esto"
ollama-cli.py chat                          # modo interactivo (salir = 'salir')
ollama-cli.py act --list                          # catálogo editable (41 entradas)
ollama-cli.py act --dry-run "cuánto espacio hay"
ollama-cli.py act --free --dry-run "procesos que más CPU usan"
ollama-cli.py bench gemma3:q4 gemma3:gpu
ollama-cli.py status                        # servicio, GPU, frecuencia, sensores
ollama-cli.py doctor                        # comprueba toda la configuración
ollama-cli.py embed -m qwen:gpu "texto"
ollama-cli.py rag index /home/onion/ollama-kit.md
ollama-cli.py rag ask "cómo funciona el modo act" -k 3
ollama-cli.py rag list
ollama-cli.py rag clear --yes
ollama-cli.py run "muestra la fecha"             # enrutador automático
ollama-cli.py run --intent rag "qué dice el manual..."
ollama-cli.py watch --count 5 --interval 1
ollama-cli.py import gemma3:q2 RUTA.gguf    # importar otro GGUF local
ollama-cli.py show gemma3:q4
ollama-cli.py stop gemma3:q4
ollama-cli.py rm NOMBRE --yes
ollama-cli.py config                        # variables efectivas y atajos
```

Aliases ya añadidos a `~/.bashrc` (se activan en una terminal nueva o con `source ~/.bashrc`):

```bash
ocli                       # ollama-cli.py
ochat                      # ollama-cli.py chat
oact                       # ollama-cli.py act
orag                       # ollama-cli.py rag
orun                       # ollama-cli.py run
owatch                     # ollama-cli.py watch
```

**Autocompletado con Tab** instalado en
`~/.local/share/bash-completion/completions/ollama-cli` y cargado desde `~/.bashrc`.
Completa subcomandos, modelos de `ollama list`, rutas de archivos, identificadores
del catálogo y opciones. Prueba: `ocli rag <Tab>`, `ocli rag ask -m <Tab>`,
`orag <Tab>`, `ochat -m <Tab>`.

La CLI respeta `OLLAMA_HOST` si quieres apuntarla a otra instancia. Usa solo
librería estándar de Python.

---

## 1. Inventario: GGUF descargados ↔ modelos utilizables

Ollama no ejecuta un `.gguf` directamente: se importa una vez con `ollama create` y
después se usa un tag. Al importar, los pesos se copian a
`/usr/share/ollama/.ollama/models` (el archivo original queda como respaldo).

| GGUF en `/home/onion/Documentos/ModelsIA/` | Tag | Uso |
|---|---|---|
| `Gemma-...-Q6_k.gguf` (965 M) | `gemma3:gpu`, `gemma3:latest` | Calidad, 7,8 tok/s |
| `Gemma-...-Q4_k_m.gguf` (769 M) | **`gemma3:q4`** | Principal, 9,3 tok/s |
| `Gemma-...-Q2_k.gguf` (658 M) | — | Solo emergencia, baja calidad |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` (610 M) | `qwen-embed:q8` (si se importa) | Embeddings, menos RAM |
| `Qwen3-Embedding-0.6B-f16.gguf` (1,2 G) | — | Innecesario, ya hay F16 |
| `Qwen3-4B-...Q4_K_M.gguf` (2,4 G) | — | ❌ No importar: ~2 tok/s |
| `Qwen3.5-4B-...Q4_K_M.gguf` (2,6 G) | — | ❌ No importar: ~2 tok/s |

`qwen:gpu` no viene de esos archivos: es `Qwen3-0.6B-Base` F16, **solo embedding**
(`/api/chat` y `/api/generate` responden `does not support`). `nemotron3:gpu` es
Nemotron-H 4B IQ2_M, 2,6 tok/s, solo para tareas puntuales.

Prefiere siempre los tags `:gpu` / `:q4`: los `:latest` no tienen `num_gpu` fijo y rinden menos.

---

## 2. Configuración aplicada

`/etc/systemd/system/ollama.service.d/override.conf`:

| Variable | Para qué sirve |
|---|---|
| `OLLAMA_IGPU_ENABLE=1` | Imprescindible: sin esto Ollama dice `no usable GPU found` |
| `GGML_VK_VISIBLE_DEVICES=0` | GPU0 = CARRIZO; la R7 260X (GPU1) no se usa |
| `OLLAMA_CONTEXT_LENGTH=2048` | Límite de historial; si te pasas se corta |
| `OLLAMA_MAX_LOADED_MODELS=1` | Solo un modelo residente; al cambiar se recarga |
| `OLLAMA_NUM_PARALLEL=1` | Una petición a la vez; el resto hace cola |
| `OLLAMA_KV_CACHE_TYPE=q8_0` | KV cuantizada: cabe mejor en la iGPU |
| `OLLAMA_KEEP_ALIVE=5m` | El modelo se descarga solo tras 5 min inactivo |
| `OLLAMA_NO_CLOUD=true` | Nunca usa modelos remotos |

Otros cambios:

- `vm.vfs_cache_pressure=50` en `/etc/sysctl.d/99-ollama.conf`.
- Gobernador CPU `performance` a 3,7 GHz.
- **`powerprofilesctl set performance` NO existe en este sistema** (solo ofrece
  `power-saver` y `balanced`). Se usa `cpupower`.
- Se probó `After=power-profiles-daemon.service` en `cpu-performance.service`,
  pero **se quitó**: creaba el ciclo
  `PPD -> multi-user.target -> cpu-performance.service -> PPD` y el temporizador
  quedaba `failed` tras reiniciar.
- Algo (PPD o `cpupower-gui-helper`) volvía a poner `powersave` horas después.
  Se usa `/etc/systemd/system/cpu-performance-enforce.timer`, que reaplica
  `performance` cada minuto (`OnBootSec=30s`). Verificado dos veces: al forzar
  `powersave`, volvió solo a `performance` en menos de 70 s.
- Copias de seguridad: `override.conf.bak`, `cpu-performance.service.bak` y
  `cpu-performance.service.bak2`.

Para reproducirlo desde cero:

```bash
printf '%s\n' 'TU_PASSWORD' | sudo -S -v
sudo cpupower frequency-set -g performance
printf '%s\n' 'vm.vfs_cache_pressure = 50' | sudo tee /etc/sysctl.d/99-ollama.conf
sudo sysctl --system
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

---

## 3. Verificación

```bash
systemctl is-active ollama                                   # active
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor    # performance
powerprofilesctl get                                         # balanced (normal)
sysctl -n vm.vfs_cache_pressure                               # 50
vulkaninfo --summary | grep -E 'GPU[0-9]|deviceName'          # GPU0 = CARRIZO
ollama ps                                                    # 100% GPU
journalctl -u ollama -n 200 --no-pager | grep -Ei 'using device|no usable GPU|offloaded'
```

Esperado en el log:

```text
using device Vulkan0 (AMD Radeon R7 Graphics (RADV CARRIZO))
offloaded 27/27 layers to GPU
```

---

## 4. La mejor forma de usar cada modelo

| Necesitas | Modelo | Por qué |
|---|---|---|
| Chat/consultas rápidas | `gemma3:q4` | 9,3 tok/s, responde bien y ocupa poca VRAM |
| Redacción con más precisión | `gemma3:gpu` | Q6_K, 7,8 tok/s |
| Razonamiento más complejo | `nemotron3:gpu` | 4B, pero 2,6 tok/s |
| Ejecutar acciones en terminal | `ocli act` / `ocli act --free` | JSON forzado, catálogo editable |
| Vectores para RAG | `qwen:gpu` | Solo `/api/embed`; `qwen-embed:q8` quedó opcional, no importado |
| Responder sobre archivos (RAG) | `nemotron3:gpu` | Sintetiza y cita; `gemma3:q4` copia fragmentos |

```bash
ollama run gemma3:q4 "¿por qué el cielo es azul?"   # una pregunta
ollama run gemma3:q4                                # interactivo: /clear, /bye
ollama run gemma3:gpu "reescribe esto con más precisión: ..."
```

Reglas prácticas:

- **Una petición a la vez**: `NUM_PARALLEL=1` y un modelo cargado; lanzar varias en paralelo solo las pone en cola.
- **Contexto 2048**: trocea documentos largos; el historial se recorta al pasar de 2048.
- **`temperature 0` para ejecutar**, 0,3 para chatear (ya viene en el modelo).
- **`ollama stop`** cuando termines: con 6,7 GB de RAM conviene no retener modelos.

---

## 5. Acciones desde terminal (JSON forzado)

**Ya no estás limitado a una lista fija.** El catálogo es un JSON editable:

```bash
ocli act --list                                   # ver las 25 entradas actuales
ocli act --dry-run "cuánto espacio libre queda"   # elige df -> df -h
ocli act "dime la temperatura de la gpu"          # elige temp -> sensors | grep ...
```

Edita `/home/onion/ollama-commands.json` para añadir o quitar comandos:

```json
"mi_comando": {"cmd": "comando real --opcion", "desc": "cuándo elegirlo"}
```

El modelo elige solo el **identificador**; el comando real sale del JSON, nunca del
modelo. Con el catálogo actual se usa automáticamente `nemotron3:gpu` (el Gemma 1B se
confunde con más de 8 opciones): 6 tokens y ~2 s.

### Modo libre (cualquier comando)

```bash
ocli act --free --dry-run "muestra los procesos que más CPU consumen"   # top
ocli act --free --dry-run "muestra las direcciones ip"                  # ip -4
ocli act --free "listar archivos"                                       # pide confirmación
ocli act --free --yes "date"                                            # sin confirmar
```

Aquí el modelo propone un comando arbitrario. La CLI comprueba que el primer token
exista (`which`), muestra lo propuesto y **pide confirmación** antes de ejecutarlo.

Además hay una **barrera de seguridad**: comandos como `rm -rf /...`, `mkfs`, `dd if=`,
`reboot`, `curl ... | sh` o escribir en `/dev/sdX` se marcan `[PELIGROSO]` y se bloquean.
Para permitirlos de forma explícita:

```bash
ocli act --free --allow-dangerous "borra /tmp/prueba con rm -rf"   # sigue pidiendo confirmación
```

Usa `shell=True`: ten mucho cuidado con `--yes`, y muchísimo más si lo combinas con
`--allow-dangerous`. Si propone algo inventado, prueba `-m nemotron3:gpu` (es el que
mejor obedece) o vuelve al catálogo.

También puedes pasar otro catálogo con `--catalog RUTA.json`.

El script antiguo `~/ollama-act.py` sigue funcionando, pero usa su lista de 6 comandos
codificada; para cualquier uso nuevo, prefiere `ocli act`.

Usamos `format` + JSON Schema + `think:false` + `temperature:0` porque el tool calling
nativo con Gemma 1B no sirve: 96 tokens, ~24 s y `tool_calls: null`.

Equivalente por API (catálogo corto):

```bash
curl -s http://127.0.0.1:11434/api/chat -d '{
  "model":"gemma3:q4",
  "stream":false,
  "think":false,
  "format":{"type":"object","properties":{"command":{"type":"string","enum":["date","pwd","free"]}},"required":["command"]},
  "messages":[{"role":"system","content":"Eres un ejecutor. Devuelve solo JSON."},{"role":"user","content":"muestra la fecha"}],
  "options":{"temperature":0,"num_ctx":2048,"num_predict":16}
}'
```

### Catálogo ampliado y scripts (~/ollama-scripts)

El catálogo tiene **41 entradas**: 25 comandos inline y 16 scripts en
`/home/onion/ollama-scripts/`. El modelo elige solo el identificador y la CLI
ejecuta el script asociado con `bash`.

| Identificador | Script | Para qué |
|---|---|---|
| `sys_info` | `sys-info.sh` | Informe completo del sistema |
| `cpu_info` | `cpu-info.sh` | CPU, gobernador y frecuencias |
| `mem_info` | `mem-info.sh` | RAM, swap y procesos pesados |
| `disk_info` | `disk-info.sh` | Discos, inodos y carpetas grandes |
| `gpu_info` | `gpu-info.sh` | GPUs, Vulkan y sensores |
| `net_info` | `net-info.sh` | Interfaces, rutas y puertos |
| `proc_info` | `proc-info.sh` | Procesos que más CPU consumen |
| `logs_tail` | `logs-tail.sh` | Últimos 50 mensajes del sistema |
| `ollama_report` | `ollama-status.sh` | Estado completo de Ollama |
| `ollama_bench` | `ollama-bench.sh` | Benchmark de modelos |
| `ollama_doctor` | `ollama-doctor.sh` | Comprobar configuración |
| `ollama_models` | `ollama-models.sh` | Listar modelos |
| `ollama_stop_all` | `ollama-stop-all.sh` | Descargar modelos cargados |
| `backup_configs` | `backup-configs.sh` | Copia con fecha de la configuración |
| `big_files` | `big-files.sh` | Archivos >100 MB |
| `systemd_failed` | `systemd-failed.sh` | Unidades fallidas |

Para añadir uno nuevo: crea el script y agrega al JSON:

```json
"mi_reporte": {"script": "/home/onion/ollama-scripts/mi-reporte.sh", "desc": "cuándo elegirlo"}
```

Ejemplos ya probados:

```bash
ocli act "haz un informe completo del sistema"   # -> sys_info
ocli act "comprueba la configuración"             # -> ollama_doctor
ocli act "haz una copia de seguridad"             # -> backup_configs
```

Con 41 opciones se usa siempre `nemotron3:gpu`: el Gemma 1B se confunde con
catálogos grandes. Puedes forzar otro modelo con `-m`.

---

## 6. RAG local: preguntar sobre tus archivos

Indexa archivos de texto con `qwen:gpu` (embeddings) y responde con un modelo chat.
El índice vive en `/home/onion/.ollama-rag/index.json`.

```bash
orag index /home/onion/ollama-kit.md              # un archivo
orag index /home/onion/ollama-scripts             # directorio (recursivo)
orag index manual.md scripts/ --max-files 50      # limitar cuántos archivos
orag list                                         # ver archivos y fragmentos
orag ask "cómo funciona el modo act" -k 3
orag ask "qué script hace la copia" -m gemma3:q4  # forzar modelo
orag clear                                        # pide confirmación
orag clear --yes
```

Opciones principales de `rag ask`:

| Opción | Por defecto | Para qué |
|---|---|---|
| `-k, --top-k` | `3` | cuántos fragmentos recuperar |
| `-m, --model` | `nemotron3:gpu` | modelo que redacta la respuesta |
| `--embed-model` | `qwen:gpu` | modelo de embeddings |
| `--min-score` | `0.0` | descartar fragmentos poco similares |
| `--ctx` | `2048` | contexto del modelo chat |
| `--max-tokens` | `384` | longitud máxima de la respuesta |
| `--context-chars` | `3600` | tope de texto recuperado inyectado |
| `--no-stream` | no | imprimir la respuesta de una vez |

**Cómo funciona:**

1. Divide cada archivo en fragmentos de 900 caracteres con 150 de solape
   (`--chunk-chars` y `--overlap` en `rag index`).
2. `qwen:gpu` genera un vector de 1024 dimensiones por fragmento mediante
   `/api/embed` (lotes de 32).
3. La pregunta se convierte en vector con el mismo modelo. Los fragmentos se
   ordenan por **similitud coseno + 0,5 × solapamiento léxico ponderado por IDF**
   (el número mostrado en `Fuentes` es esa puntuación combinada). Así evitamos que
   un script con la palabra `GOVERNOR` gane a la sección del manual que explica el
   temporizador del gobernador.
4. Se eliminan fragmentos casi duplicados del mismo archivo, se inyectan los
   mejores como contexto y el modelo chat sintetiza la respuesta citando `[1]`, `[2]`...

**Modelos:** el embedding siempre es `qwen:gpu` (solo `/api/embed`). Para redactar,
el predeterminado es `nemotron3:gpu`: con `gemma3:q4` el Gemma 1B tiende a copiar
los fragmentos en vez de resumirlos. Como `OLLAMA_MAX_LOADED_MODELS=1`, al pasar de
embeddings a chat Ollama descarga `qwen:gpu` y carga el modelo chat; eso añade un
par de segundos por consulta y es normal ver `ollama ps` con un solo modelo.

**Verificado:** el manual y los 16 scripts generan unos 40 fragmentos (el número
crece al ampliar este documento). Preguntas reales:

- *"cómo funciona el modo act y qué modelos usa"* → respuesta resumida con citas
  `[2] [3]`.
- *"qué script sirve para hacer una copia de seguridad"* → `backup-configs.sh [1]`.
- *"cuál es la receta de pan casero"* → `No lo encuentro en el índice`.

Volver a indexar la misma ruta reemplaza solo los fragmentos de esa ruta; el resto
del índice se conserva. `rag clear` borra todo.

---

## 7. Enrutador `run` y monitor `watch`

### Enrutador automático

Una sola entrada para todo. Clasifica la intención con `nemotron3:gpu` y delega:

```bash
orun "muestra la fecha"                              # -> act
orun "cuánta memoria libre hay"                      # -> act
orun "qué dice el manual del temporizador"           # -> rag
orun "en los scripts, cuál hace la copia"            # -> rag
orun "por qué el cielo es azul"                      # -> chat
orun --dry-run "haz una copia de seguridad"          # solo muestra la decisión
orun --intent chat --no-stream "responde OK"         # saltar clasificación
```

| Opción | Para qué |
|---|---|
| `--intent act\|rag\|chat` | forzar intención y ahorrar la llamada de clasificación |
| `-m, --model` | modelo para `chat` o para responder en `rag` |
| `--act-model` | modelo usado por `act` |
| `-k, --top-k` | fragmentos a recuperar si decide `rag` |
| `--dry-run` | clasifica y muestra qué haría, sin ejecutar |
| `--no-stream`, `--stats`, `--system`, `--catalog` | se pasan al subcomando correspondiente |

Clasificaciones verificadas: fecha, memoria y copia de seguridad → `act`; preguntas
sobre el manual o los scripts → `rag`; explicaciones generales → `chat`. Si no hay
índice RAG, la opción `rag` ni se ofrece al clasificador.

### Monitor en vivo

```bash
owatch                                  # hasta Ctrl+C, cada 2 s
owatch --interval 1 --count 10          # 10 muestras, cada 1 s
owatch --no-clear                       # no limpiar la pantalla
```

Muestra gobernador y frecuencia de CPU, carga, estado de Ollama, modelos cargados y
VRAM, memoria, disco, sensores de temperatura/ventilador y los 3 procesos que más
CPU consumen. Es solo lectura y usa la API local de Ollama.

---

## 8. Otras recetas de API

```bash
# Generar texto
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"gemma3:q4","prompt":"hola","stream":false,"options":{"num_predict":40}}'

# Chat con historial
curl -s http://127.0.0.1:11434/api/chat \
  -d '{"model":"gemma3:q4","stream":false,"messages":[{"role":"user","content":"hola"}]}'

# Embeddings (NO usar /api/chat ni /api/generate con qwen)
curl -s http://127.0.0.1:11434/api/embed \
  -d '{"model":"qwen:gpu","input":"texto a vectorizar"}' | head -c 200
```

---

## 9. Benchmarks y decisiones ya probadas

| Configuración | tok/s |
|---|---:|
| `gemma3:q4` en CARRIZO | **9,3** |
| `gemma3:gpu` (Q6) en CARRIZO | 7,8 |
| CPU con gobernador powersave | 4,2 |
| BONAIRE (dGPU, descartada) | ~35 |
| Dos instancias (iGPU 8,4 + dGPU 22,7) | ~31 agregado |
| `nemotron3:gpu` | 2,6 |

- `OLLAMA_SCHED_SPREAD=true` **no reparte capas** en Ollama 0.33.2 + Vulkan: el modelo
  va entero a un dispositivo. Por eso se descartó usar ambas GPUs.
- Tool calling nativo con Gemma 1B: descartado (ver sección 5).
- Los GGUF 4B locales no se importan: con 2 GB de VRAM y 6,7 GB de RAM rinden peor que `nemotron3`.

---

## 10. Importar otro GGUF (plantilla)

```bash
cat > /home/onion/Documentos/ModelsIA/Modelfile.NOMBRE <<'EOF'
FROM /home/onion/Documentos/ModelsIA/ARCHIVO.gguf
PARAMETER temperature 0.3
PARAMETER num_ctx 2048
PARAMETER num_gpu 27
PARAMETER stop <end_of_turn>
EOF
ollama create NOMBRE -f /home/onion/Documentos/ModelsIA/Modelfile.NOMBRE
ollama show NOMBRE
```

`/home/onion/Documentos/ModelsIA/Modelfile.gemma-tools-fixed` activa la capability
`tools` con un bloque `{{ .Tools }}`, pero **no produce tool_calls útiles**: se dejó
como experimento documentado, no como modelo de uso diario.

---

## 11. Solución de problemas

| Síntoma | Causa / solución |
|---|---|
| `no usable GPU found`, `100% CPU` | Falta `OLLAMA_IGPU_ENABLE=1`; revisar drop-in y reiniciar Ollama |
| `model 'gemma3:q4' not found` en instancia temporal | Falta `OLLAMA_MODELS=/usr/share/ollama/.ollama/models` |
| Gobernador vuelve a `powersave` o el timer está `failed` | Quitar `After=power-profiles-daemon.service` de `cpu-performance.service`, `systemctl daemon-reload`, `systemctl enable --now cpu-performance-enforce.timer`; revisar `journalctl -u cpu-performance.service` |
| `powerprofilesctl set performance: invalid choice` | Normal: este sistema solo tiene `power-saver`/`balanced`; usar `cpupower` |
| `does not support chat` / `generate` con qwen | Es embedding: usar `/api/embed` |
| El enrutador `run` elige mal | Usar `orun --intent act\|rag\|chat ...` o revisar con `--dry-run` |
| Peticiones lentas o coladas | `NUM_PARALLEL=1` y un modelo cargado: enviar de una en una |
| Poco espacio en `/var` o `/home` | `sudo journalctl --vacuum-size=200M`; revisar `df -h` |

---

## 12. Mantenimiento y reversión

```bash
watch -n 2 sensors                 # temperatura, consumo, ventilador
watch -n 2 ollama ps               # modelo cargado y VRAM
journalctl -u ollama -f            # logs en vivo
ollama list                        # modelos instalados
ollama show gemma3:q4              # detalle del modelo
ollama stop gemma3:q4              # descargarlo de memoria
```

Volver a la R7 260X: cambiar `GGML_VK_VISIBLE_DEVICES=0` por `1` en el override,
`sudo systemctl daemon-reload && sudo systemctl restart ollama`.

Deshacer los cambios:

```bash
sudo cp /etc/systemd/system/ollama.service.d/override.conf.bak \
        /etc/systemd/system/ollama.service.d/override.conf
sudo rm /etc/sysctl.d/99-ollama.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama rm gemma3:q4
```
