# Manual Ollama — A8-9600 / A10-9700 (iGPU Carrizo)

**Estado: configuración aplicada el 21/11/2026.** Lo que aparece en la sección 1 ya está
en marcha; se conserva como referencia para reproducirlo tras una reinstalación.

> El equipo se conoce como A8-9600, aunque `lscpu` reporta `AMD A10-9700 RADEON R7`.
> En ambos casos la iGPU es Carrizo y es el dispositivo Vulkan 0 usado por Ollama.

## TL;DR — lo que más usarás

```bash
ocli config                               # perfil tareas y modelos por rol
ollama run gemma3:gpu "tu pregunta"       # principal: más exactitud (7,8 tok/s)
oagent --prompt "resume este repo"        # agente de una sola tarea, sin REPL
~/ollama-act.py "muestra la fecha"        # ejecutar acción (6-8 tokens)
~/ollama-bench.sh gemma3:gpu              # medir velocidad
orag index /home/onion/ollama-kit.md      # indexar archivos para RAG
orag ask "cómo funciona el modo act"      # preguntar sobre lo indexado
ochat --precise "explica esto con detalle" # máxima precisión: lento, sin prisa
orun "muestra la fecha"                   # enrutador: act / rag / chat
owatch                                    # monitor en vivo
odgpu "tu pregunta"                       # Gemma4 en la R7 260X (instancia aislada)
ocode review programa.py                  # revisar código con JSON Schema
odebug                                    # ver errores y trazas de la CLI
ochat --session proyecto "hola"           # conversación persistente
oprompt use resumen "texto a resumir"     # plantilla reutilizable
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
ollama-cli.py run --reasoning "analiza este diff" # respaldo nemotron3:gpu
ollama-cli.py run --intent rag "qué dice el manual..."
ollama-cli.py watch --count 5 --interval 1
gemma4-dgpu.sh --bench                            # Gemma4 en dGPU, instancia temporal
ollama-cli.py code review archivo.py --dry-run    # prompt y schema sin llamar al modelo
ollama-cli.py debug --full                        # tracebacks de errores
ollama-cli.py chat --session proyecto "hola"     # conversación persistente
ollama-cli.py session list
ollama-cli.py prompt add resumen --text 'Resume: {{args}}'
ollama-cli.py prompt use resumen "texto largo"
ollama-cli.py bench --history                     # historial de rendimiento
ollama-cli.py import gemma3:q2 RUTA.gguf    # importar otro GGUF local
ollama-cli.py show gemma3:q4
ollama-cli.py stop gemma3:q4
ollama-cli.py stop --all                      # descargar todos los modelos cargados
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
oprompt                    # ollama-cli.py prompt
osess                      # ollama-cli.py session
odgpu                      # gemma4-dgpu.sh (R7 260X aislada)
ocode                      # ollama-cli.py code
odebug                     # ollama-cli.py debug
ogui                       # ollama-kit-gui.py (ventana local)
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
| `Gemma-...-Q6_k.gguf` (965 M) | **`gemma3:gpu`**, `gemma3:latest` | **Principal**, 7,8 tok/s |
| `Gemma-...-Q4_k_m.gguf` (769 M) | `gemma3:q4` | Alternativa rápida, 9,3 tok/s |
| `Gemma-...-Q2_k.gguf` (658 M) | — | Solo emergencia, baja calidad |
| `Qwen3-Embedding-0.6B-Q8_0.gguf` (610 M) | `qwen-embed:q8` (si se importa) | Embeddings, menos RAM |
| `Qwen3-Embedding-0.6B-f16.gguf` (1,2 G) | — | Innecesario, ya hay F16 |
| `Qwen3-4B-...Q4_K_M.gguf` (2,4 G) | — | ❌ No importar: ~2 tok/s |
| `Qwen3.5-4B-...Q4_K_M.gguf` (2,6 G) | `qwen35:gpu`, `qwen35:embed` | 2,61 tok/s; ver nota abajo |
| `Gemma-4-E4B-...Q2_K_P.gguf` (4,2 G) | `gemma4:cpu`, `gemma4:dgpu` | 2,5 tok/s CPU; 8,5 tok/s en dGPU aislada; no cabe en la iGPU |

`qwen:gpu` no viene de esos archivos: es `Qwen3-0.6B-Base` F16, **solo embedding**
(`/api/chat` y `/api/generate` responden `does not support`). `nemotron3:gpu` es
Nemotron-H 4B IQ2_M, 2,6–3,6 tok/s, modelo de respaldo para razonamiento. Junto
con `qwen3:06b` son los que declaran `thinking`; la CLI pide razonamiento a
Nemotron3 pero envía `think:false` a los tags `qwen3:*`, que piensan por su
cuenta y agotaban el presupuesto de salida en respuestas cortas. Gemma 3 solo
declara `completion` y rechaza `think=true` con HTTP 400.

### Modelos nuevos evaluados el 22/11/2026

Importados desde `~/Descargas/models` con tags nuevos (sin sobrescribir nada):

| GGUF | Tag | Arquitectura | Contexto | tok/s (80 tok, ctx 2048, media de 3) |
|---|---|---|---:|---:|
| `Bonsai-1.7B-Q1_0.gguf` (237 M) | `bonsai:q1` | Qwen3 1.7B, Q1_0 | 32k YaRN | 25,06 |
| `gemma-3-1b-it-Q4_K_M.gguf` (769 M) | `gemma3:1b` | Gemma 3 1B-it | 32k | 9,11 |
| `LFM2.5-350M-Q4_K_M.gguf` (219 M) | `lfm25:350m` | LFM2.5 350M híbrido | 128k | **43,34** |
| `Qwen_Qwen3-0.6B-Q4_K_M.gguf` (462 M) | `qwen3:06b` | Qwen3 0.6B GQA | 32k | 25,21 |

Referencia: `gemma3:q4` 8,86 tok/s en las mismas condiciones. El bench usó
`--no-save`, así que el historial no se modificó. Los cuatro responden en
español y respetan un JSON Schema simple; `act --dry-run` con `qwen3:06b` y
`bonsai:q1` propone `date` correctamente.

Recomendación práctica: `lfm25:350m` para enrutado/clasificación (es el más
rápido, pero confundió dispersión con refracción al explicar el cielo),
`qwen3:06b` para chat, JSON ligero y RAG rápida con contexto, `bonsai:q1` como
experimento de 1 bit y `gemma3:1b` solo si se quiere la versión stock en vez de
los pesos «Heretic» de `gemma3:q4`.

### RAG y precisión factual (23/11/2026)

El índice estaba desactualizado (19:47 del 22/11 frente a documentos editados a
las 23:49), así que se reconstruyó con `qwen:gpu`, chunk 900 y solape 150:
**697 fragmentos en 52 archivos**. La copia anterior quedó en
`/tmp/opencode/index-before-reindex.json`. Tras incorporar los resultados de esta
sección se reindexó de nuevo a las 01:45: **más de 700 fragmentos en 52 archivos**, con
copia previa en `/tmp/opencode/index-before-final-reindex.json`. Con el índice
fresco se preguntó lo
mismo a `gemma3:gpu` y `qwen3:06b` (`-k 4 --ctx 2048 --max-tokens 300`):

| Pregunta | `gemma3:gpu` | `qwen3:06b` |
|---|---|---|
| `¿Qué hace odgpu --safe-fit?` | Definición genérica de `odgpu`, repite texto | Correcto: ctx 8192, capas ajustadas, 2,64 vs 8,54 tok/s |
| Embeddings en RAG | Dice que se usa `qwen35:embed` (❌) | Dice que se usa Qwen3.5-4B y recomienda `qwen:gpu` (❌ parcial) |
| Perfil de máxima precisión | Mezcla parámetros del perfil tareas (❌) | Cita `--precise`/`OLLAMA_KIT_PRECISE=1`, pero mezcla modelos (❌ parcial) |

Ambos recibieron exactamente las mismas fuentes. Qwen3 respondió en 80–116 tokens
(20,0 tok/s) frente a los 136–300 de Gemma 3 (7,3–7,5 tok/s). Conclusión: **se
mantiene `gemma3:gpu` como default**; `orag ask -m qwen3:06b` es una opción válida
para RAG rápida donde la velocidad importe más que la redacción, y `--precise`
sigue siendo la vía para respuestas críticas (sin importar el tiempo).

Comparación contra el perfil de máxima precisión (`--precise`, mismas tres
preguntas): `nemotron3:precise` acertó la de embeddings y la de `safe-fit`, con
respuestas citadas de 268 y 804 tokens a 3,4 tok/s; `qwen3:06b` solo acertó
`safe-fit`, con 80–116 tokens a ~20 tok/s. La pregunta sobre el perfil preciso
**agotó los 1024 tokens** en 304 s y dejó la respuesta final cortada; con
`orag ask -m nemotron3:precise --max-tokens 600` (misma temp 0 y ctx 8192, sin el
añadido «exhaustivo» de `--precise`) respondió completa en 21 s. Conclusión:
`--precise` es más fiable, pero conviene vigilar su presupuesto; para preguntas
cortas, el modelo directo con 600 tokens es más práctico. El timeout de la API
en RAG `--precise` se amplió de 300 a 900 s (`PRECISE_API_TIMEOUT`) para que no
falle antes de agotar el presupuesto.

Sonda factual local ampliada (20 preguntas, 5 modelos, salida JSON, acentos y
subíndices normalizados; los 100 JSON fueron válidos):

| Modelo | Aciertos | Fallos destacados |
|---|---:|---|
| `gemma3:gpu` | **17/20** | CO₂ (`C2O`), Júpiter (dijo Mercurio), días de febrero |
| `gemma3:q4` | 16/20 | CO₂, Júpiter (dijo Tierra), planeta más cercano, anillos |
| `bonsai:q1` | 14/20 | planeta rojo, anillos, océano, continente, Au |
| `lfm25:350m` | 12/20 | meses, Júpiter, Mercurio, anillos, océano, continente |
| `qwen3:06b` | 10/20 | Armstrong, Venus ×3, Cervantes, Los miserables, etc. |

Esto confirma que Qwen3 0.6B debe usarse con grounding (RAG) y no como fuente de
conocimiento libre; Bonsai aguanta mejor los datos generales pese a ser Q1_0,
pero `gemma3:gpu` sigue siendo el modelo principal. Todas las pruebas locales,
sin APIs externas ni créditos.

Sobre `qwen35:gpu` / `qwen35:embed` (probados el 22/11/2026):

- `qwen35:gpu` usa el GGUF original con `num_gpu 33` → **33/33 capas en GPU**, 100% GPU,
  2,66 GiB de VRAM y 2,61 tok/s. Con `num_gpu 32` solo cargaban 32/33 capas (18%/82%
  CPU/GPU). Es un modelo generativo con capacidades `tools`, `thinking`, `completion`.
- `qwen35:embed` parte de una copia del GGUF con el metadato añadido
  `qwen35.pooling_type = 3`, porque Ollama 0.33.2 solo habilita `/api/embed` si el GGUF
  declara `pooling_type`. Requiere `num_ctx 512`: con 2048 se agota la VRAM compartida.
  Genera vectores de **2560 dimensiones** y tiene capability `embedding` (ya no `completion`).
- **Decisión:** RAG sigue usando `qwen:gpu`. Qwen3.5 es ~5x más lento como embedder y
  algo menos preciso (4/6 vs 5/6 en una prueba de recuperación con 6 temas únicos).
  `qwen35:embed` queda instalado como alternativa probada. La receta está en
  `tools/make_qwen35_embed.py` y `config/Modelfile.qwen35-embed` del repositorio.

Sobre `gemma4:cpu` (probado el 22/11/2026):

- Es `gemma4`, 42 bloques, 7,5 B parámetros, Q2_K, embedding 2560. Ollama 0.33.2 lo
  reconoce y ofrece `completion`, `tools` y `thinking`.
- **No cabe en la iGPU.** Con `num_gpu 42` carga 42/43 capas pero muere con
  `radv/amdgpu: Not enough memory for command submission`. Bajando a 36, 32, 28, 24,
  20, 16, 12 y 8 capas sigue fallando: el backend reserva ~0,8–1,5 GiB en Vulkan0 y
  **~4 GiB más en `Vulkan_Host` (RAM)**, por encima de los ~3,4 GiB libres.
- Con `num_gpu 0` y `num_ctx 512` funciona: **2,5–2,7 tok/s**, `ollama ps` muestra
  `100% CPU` y **0,00 GiB de VRAM**, así que la R7 260X sigue intacta. El bench de la
  CLI usa 2048 de contexto y también carga.
- Responde de forma coherente, pero es más lento que `gemma3:q4` (9,3 tok/s) y que
  `nemotron3:gpu` (3,6 tok/s). Se conserva como modelo de prueba, no como default.

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
| Chat, redacción, `oflow`, acciones | `gemma3:gpu` | Q6_K (uncensored): principal; más exactitud, 7,8 tok/s |
| Máxima velocidad | `gemma3:q4` | Q4_K_M, 9,3 tok/s; misma base, algo menos de calidad |
| Razonamiento más complejo | `nemotron3:gpu` | 4B, pero 2,6 tok/s |
| Máxima precisión (sin importar velocidad) | `nemotron3:precise` | `--precise`: temp 0, ctx 8192, 1024 tokens, ~3 tok/s |
| Ejecutar acciones en terminal | `ocli act` / `ocli act --free` | JSON forzado, catálogo editable |
| Vectores para RAG | `qwen:gpu` | Solo `/api/embed`; `qwen-embed:q8` quedó opcional, no importado |
| Responder sobre archivos (RAG) | `gemma3:gpu` | Default del perfil tareas; cita fuentes. Con `--reasoning` pasa a `nemotron3:gpu` |
| RAG rápida (opcional) | `qwen3:06b` | `orag ask --fast`, ~3× más rápido; requiere contexto recuperado |
| Probar Gemma 4 en la R7 260X | `odgpu` → `gemma4:dgpu` | 8,5 tok/s, instancia aislada; no cambia el default |
| Probar Gemma 4 en CPU | `gemma4:cpu` | 2,5 tok/s, 0 GiB VRAM; no es práctico como default |
| Código/razonamiento muy largo | `ocloud` → `glm-5.3`, `kimi-k2.7-code` | nube bajo demanda, sin VRAM; consume créditos |

**Decisión de exactitud (22/11/2026):** el principal pasa a ser `gemma3:gpu`
(Q6_K, uncensored) en chat, `oflow`, acciones y explicaciones: es el mismo modelo
y fine-tune que `gemma3:q4`, pero la cuantización de 6 bits conserva mejor los
pesos que la de 4 bits. El coste es pasar de 9,3 a 7,8 tok/s. Desde la tanda 18
también es el modelo de agente, RAG y enrutador del **perfil tareas**;
`nemotron3:gpu` queda como respaldo de razonamiento (`--reasoning`) y para
`act --free`, además de `code review` y `commit`. No hay en la carpeta ningún
candidato medido que lo supere. Los embeddings siguen con `qwen:gpu` y Gemma 4
queda para uso puntual.

```bash
ollama run gemma3:gpu "¿por qué el cielo es azul?"  # una pregunta
ollama run gemma3:gpu                               # interactivo: /clear, /bye
ollama run gemma3:q4 "respuesta rápida: ..."        # alternativa más veloz
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
  "model":"gemma3:gpu",
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
orag ask "qué hace safe-fit" --fast              # RAG rápida: qwen3:06b, 300 tok
orag clear                                        # pide confirmación
orag clear --yes
```

Opciones principales de `rag ask`:

| Opción | Por defecto | Para qué |
|---|---|---|
| `-k, --top-k` | `3` | cuántos fragmentos recuperar |
| `-m, --model` | `gemma3:gpu` (`nemotron3:precise` con `--precise`) | modelo que redacta la respuesta |
| `--embed-model` | `qwen:gpu` | modelo de embeddings |
| `--min-score` | `0.10` | descartar fragmentos poco similares (`0` desactiva) |
| `--ctx` | `2048` (`8192` con `--precise`) | contexto del modelo chat |
| `--max-tokens` | `192` (`384` con `--reasoning`, `1024` con `--precise`) | longitud máxima de la respuesta |
| `--context-chars` | `3600` | tope de texto recuperado inyectado, recortado además por el presupuesto |
| `--precise` | no | perfil de máxima precisión: modelo dedicado, temp 0 y razonamiento |
| `--fast` | no | RAG rápida: `qwen3:06b` y 300 tokens; excluye `--reasoning` y `--precise` |

Para usar la alternativa Qwen3.5 hay que reindexar **todo** el índice con el mismo modelo
(no se pueden mezclar vectores de 1024 y 2560 dimensiones):

```bash
orag index manual.md scripts/ --embed-model qwen35:embed
orag ask "cómo funciona el modo act" --embed-model qwen35:embed
```

No se recomienda cambiar el default: `qwen:gpu` es más rápido y preciso. Si se cambia,
hay que actualizar también `-m/--model` de `embed`, el default de `rag index` y la opción
`--embed-model` del autocompletado.
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
el predeterminado del perfil tareas es `gemma3:gpu`; con `--reasoning` pasa a
`nemotron3:gpu`, que sintetiza mejor fragmentos largos. `qwen3:06b` es una opción
de RAG rápida con contexto (`--fast` o `-m qwen3:06b`). Como `OLLAMA_MAX_LOADED_MODELS=1`, al pasar de
embeddings a chat Ollama descarga `qwen:gpu` y carga el modelo chat; eso añade un
par de segundos por consulta y es normal ver `ollama ps` con un solo modelo.

**Precedencia del modelo:** `-m` explícito > `--precise` > `--reasoning` > perfil
tareas. En perfil tareas el default es `gemma3:gpu`; `--precise` usa
`nemotron3:precise` aunque no se pase `-m`; `--reasoning` usa `nemotron3:gpu` en
agente y RAG. `--precise` y `--reasoning` fijan temperatura 0 y contexto amplio,
pero un `-m` explícito conserva ese modelo. `--fast` usa `qwen3:06b` con 300
tokens y no se combina con `--precise` ni `--reasoning`; un `-m` explícito
también manda sobre `--fast`. Para saber qué modelo está activo en cada rol,
`ocli config` es más fiable que preguntárselo a RAG.

**Verificado:** el manual y los 16 scripts generan unos 40 fragmentos (el número
crece al ampliar este documento). Preguntas reales:

- *"cómo funciona el modo act y qué modelos usa"* → respuesta resumida con citas
  `[2] [3]`.
- *"qué script sirve para hacer una copia de seguridad"* → `backup-configs.sh [1]`.
- *"cuál es la receta de pan casero"* → `No lo encuentro en el índice`.

Volver a indexar la misma ruta reemplaza solo los fragmentos de esa ruta; el resto
del índice se conserva. `rag clear` borra todo.

Este índice es global: si trabajas con varios proyectos, conviene usar `oflow`
(sección 12), que mantiene una búsqueda efímera por documento y no lo sobrescribe.

---

## 7. Enrutador `run` y monitor `watch`

### Enrutador automático

Una sola entrada para todo. Clasifica la intención con el modelo del perfil
(`gemma3:gpu`; `nemotron3:gpu` con `--reasoning`) y delega:

```bash
orun "muestra la fecha"                              # -> act
orun "cuánta memoria libre hay"                      # -> act
orun "qué dice el manual del temporizador"           # -> rag
orun "en los scripts, cuál hace la copia"            # -> rag
orun "por qué el cielo es azul"                      # -> chat
orun --dry-run "haz una copia de seguridad"          # solo muestra la decisión
orun --intent chat --no-stream "responde OK"         # saltar clasificación
orun --intent rag --fast "qué dice el manual"        # RAG rápida con qwen3:06b
```

| Opción | Para qué |
|---|---|
| `--intent act\|rag\|chat` | forzar intención y ahorrar la llamada de clasificación |
| `-m, --model` | modelo para `chat` o para responder en `rag` |
| `--act-model` | modelo usado por `act` |
| `-k, --top-k` | fragmentos a recuperar si decide `rag` |
| `--dry-run` | clasifica y muestra qué haría, sin ejecutar |
| `--fast` | si la intención es `rag`, usa `qwen3:06b` y 300 tokens; excluye `--reasoning` y `--precise` |
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

## 8. Gemma 4 en la R7 260X (instancia aislada)

Gemma 4 E4B (7,5 B, Q2_K) **no cabe en la iGPU**, pero sí en la R7 260X.
`gemma4:dgpu` fija `num_gpu 42` y `num_ctx 2048`, pero la GPU no la elige el
Modelfile: la decide la variable de proceso `GGML_VK_VISIBLE_DEVICES=1`.
Por eso `odgpu` levanta una instancia temporal en el puerto 11435, ejecuta la
petición y la apaga al terminar (también con Ctrl+C):

```bash
odgpu "tu pregunta"                     # una respuesta
odgpu --stats "tu pregunta"             # con tokens/s
odgpu --chat                            # conversación interactiva
odgpu --bench                           # 80 tokens, sin guardar en el historial
odgpu --safe-fit "texto largo"          # ctx 8192, capas ajustadas por Ollama
odgpu --print-env                       # muestra el plan sin arrancar la instancia
```

Para prompts largos, `--safe-fit` cambia a `gemma4:dgpu-safe` (`num_ctx 8192`,
receta `config/Modelfile.gemma4-dgpu-safe` **sin `num_gpu` fijo**) y fija
`OLLAMA_CONTEXT_LENGTH=8192` en la instancia temporal. Así Ollama elige cuántas
capas caben en los ~2,0 GiB libres. **Probado con una inferencia real**
(`odgpu --safe-fit "di HOLA"`): el log emite `cannot meet free memory target of
1024 MiB, need to reduce device memory by 1268 MiB`, pero es informativo —no
aborta—; Ollama carga 9/43 capas en GPU, mantiene `n_ctx = 8192` y devuelve la
respuesta. La primera llamada crea el tag si no existe. Tras la prueba, el
puerto 11435 quedó cerrado, el PIDFILE limpio y el servicio de 11434 sigue
respondiendo. Es mucho más lento que el modo de 42 capas; véase la tabla
siguiente.

El script **no toca `override.conf` ni el servicio principal**: este sigue con
`GGML_VK_VISIBLE_DEVICES=0` y la iGPU continúa siendo el default de todos los
modelos. Si el puerto 11435 ya está ocupado, se niega a arrancar. El log de la
instancia temporal se escribe en `${TMPDIR:-/tmp}/ollama-gemma4-dgpu.log`: por
defecto `/tmp/ollama-gemma4-dgpu.log`, o dentro de `$TMPDIR` si lo defines.

### Resultados medidos (22/11/2026)

| Configuración | tok/s |
|---|---|
| Gemma4 en dGPU, 42 capas, ctx 2048 | **8,54** |
| Gemma4 en dGPU, safe-fit, ctx 8192 (9/43 capas) | 2,64 |
| Gemma4 en dGPU, 36 capas | 4,39 |
| Gemma4 en CPU | 2,33–2,56 |
| Nemotron3 en iGPU | 3,47 |
| Nemotron3 en dGPU | 11,22 |

Con 42 capas se cargan 42/43 en GPU y se usan ~1,92 GiB de VRAM de ~2,0 GiB; el
margen es justo, por eso bajar a 36 capas empeora mucho el rendimiento. Con
`num_gpu > 0` en la iGPU sigue fallando (`Not enough memory for command
submission`). La R7 260X se usa **solo** bajo este procedimiento.

Nota: Gemma4 expone el razonamiento en `message.thinking` y la CLI lo muestra en
`RAZONAMIENTO:`; el texto final llega en `message.content`. En la prueba de
safe-fit, 296 tokens evaluados incluyeron ese razonamiento, por lo que 2,64
tok/s no son comparables directamente con una respuesta sin pensar. Las cifras
anteriores corresponden a sus configuraciones respectivas.

---

## 9. Programar con el kit

### Modelos por tarea

| Tarea | Modelo | Por qué |
|---|---|---|
| Iterar, explicar errores y generar tests | `gemma3:gpu` | Q6_K, más exacto, 7,8 tok/s |
| Revisar código, arquitectura, diffs | `nemotron3:gpu` | razona mejor |
| Código extenso o respuesta larga | `odgpu` → `gemma4:dgpu` | 8,5 tok/s en dGPU aislada |
| Buscar en tu propio código | `qwen:gpu` + `gemma3:gpu` | embeddings + RAG (`nemotron3:gpu` con `--reasoning`) |

### `code`: revisar, testear, explicar y commitear

`ocli code` (alias `ocode`) lee archivos o texto, trocea a 120 líneas por defecto
para no desbordar los 2048 tokens y fuerza un JSON Schema por acción:

```bash
ocode review ~/ollama-cli.py --lines 80 --stats
ocode review ~/ollama-cli.py --start 81 --lines 80   # siguiente trozo
ocode tests modulo.py
ocode explain "def suma(a, b): return a + b"
echo 'def f(x): return x*2' | ocli code review --stdin
ocode commit /home/onion/ollama-local-kit
ocode review archivo.py --dry-run      # ver system, prompt y schema sin llamar al modelo
```

| Acción | Modelo | JSON devuelto |
|---|---|---|
| `review` | `nemotron3:gpu` | `problemas[]` (severidad, sugerencia) + `resumen` |
| `tests` | `gemma3:gpu` | `tests`, `casos[]`, `notas` |
| `explain` | `gemma3:gpu` | `explicacion`, `mejoras[]` |
| `commit` | `nemotron3:gpu` | `tipo`, `alcance`, `mensaje`, `cuerpo` |

Opciones útiles: `--from/--start` y `--lines` para trocear, `--stdin`, `-m` para
cambiar el modelo, `--no-schema` para salida libre, `--dry-run` para inspeccionar,
`--session`, `--stats` y `--max-tokens`. Si el prompt supera ~7000 caracteres,
la CLI avisa de que puede desbordar el contexto.

### RAG sobre tu código

`rag index` acepta directorios, entiende `.py`, `.sh`, `.json`, `.md` y omite
`.git`, `__pycache__` y `node_modules`:

```bash
orag index /home/onion/ollama-local-kit
orag ask "¿dónde se guardan las sesiones de chat?" -k 4
orag ask "cómo funciona el modo act" -m gemma3:gpu -k 3
```

### `act` ampliado y selftest

El catálogo tiene 53 entradas. Las nuevas para programar son `py_compile`,
`shell_check`, `json_check`, `run_tests`, `git_status`, `git_diff`, `git_log`,
`kit_doctor`, `kit_bench`, `rag_reindex`, `debug_log` y `review_cli`.

```bash
oact "comprueba la sintaxis de los scripts" --dry-run
oact "cambios pendientes en el repo"
bash ~/ollama-scripts/selftest.sh     # 5 pasos, sin dependencias nuevas
```

`selftest.sh` valida sintaxis Python y bash, el JSON del catálogo, el índice RAG y
`doctor`. No necesita pytest (no está instalado).

### Indicadores y debug

Cada proceso lanzado por `act` muestra `[->] comando`, su salida, la duración y
`[OK]` o `[X] exit N`. Todo se registra en `~/.ollama-debug.log` (JSONL):
peticiones API, errores HTTP, comandos ejecutados, timeouts y tracebacks.

```bash
odebug                      # últimos 15 eventos
odebug -n 3 --full          # tracebacks completos
odebug -n 50
odebug --clear
ocli --debug code review archivo.py          # trazas en stderr
OLLAMA_KIT_DEBUG=1 ocli act "muestra la fecha"
```

Si una orden falla, verás `ERROR: ... | Detalle en ~/.ollama-debug.log`. Con
`--debug` se muestra además el traceback en pantalla. El log se incluye en el
backup y se excluye del repositorio.

### Modo agente interactivo (`oagent`)

`oagent` convierte la CLI en un REPL con herramientas, parecido a un agente de
código, pero con permisos explícitos y sin servidor ni TUI. En perfil tareas
arranca con `gemma3:gpu` local y usa `format` + JSON Schema (no tool calling
nativo), que es lo que mejor funciona con los modelos pequeños de este kit:

```bash
cd ~/ollama-local-kit
oagent                                         # REPL, raíz = directorio actual
oagent --prompt "resume este repo"             # una sola tarea y salir
oagent --reasoning --prompt "analiza este diff"  # respaldo nemotron3:gpu
oagent --root ~/ollama-local-kit --session trabajo
oagent -m gemma3:q4 --steps 4                  # más rápido, menos pasos
ogemini agent --root ~/ollama-local-kit        # Gemini también admite schema
```

El bucle funciona así: escribes una petición, el modelo devuelve un JSON con
`thought`, `tool`, `args` y, si ya puede responder, `answer`. La CLI ejecuta la
herramienta, añade el resultado al historial y vuelve a llamar al modelo hasta
que el modelo devuelve `answer` o se alcanza `--steps` (4 por defecto en perfil
tareas, 8 con `--reasoning`). Cada decisión se imprime como
`RAZONAMIENTO AGENTE:`; con `--prompt` se ejecuta una sola petición y se sale sin
REPL. La pestaña Agente de `ogui` usa esta modalidad y nunca pasa `--yes` ni
`--allow-dangerous`.

| Herramienta | Qué hace | Permiso |
|---|---|---|
| `read` | Lee un archivo con números de línea (200 líneas por llamada) | automático |
| `list` | Lista una carpeta | automático |
| `grep` | Busca un regex, máximo 30 coincidencias | automático |
| `rag` | Recupera fragmentos del índice con `qwen:gpu` | automático |
| `write` | Crea o sobrescribe un archivo | **confirmación** |
| `edit` | Reemplaza un texto que debe aparecer una sola vez | **confirmación** |
| `bash` | Ejecuta un comando (valida que exista y filtra patrones peligrosos) | **confirmación** |

Comandos internos: `/help`, `/model NAME`, `/provider`, `/session NAME`,
`/clear`, `/root PATH`, `/tools`, `/rag PREGUNTA`, `/bash COMANDO`, `/undo`,
`/yes` (alterna confirmaciones) y `/exit`.

Seguridad y límites:

- Las rutas se resuelven dentro de `--root`; se rechazan absolutas, `..`,
  secretos (`~/.nvidia-api-key`, `~/.gemini-api-key`, `~/.ollama-cloud-key`) y
  `.git`. Nada se escribe fuera de esa carpeta.
- Toda escritura/edición guarda una copia en `~/.ollama-agent-undo`; `/undo`
  restaura el último cambio o borra el archivo recién creado.
- Los comandos peligrosos (`rm -rf /`, `mkfs`, `dd if=`, `curl … | sh`, …)
  se bloquean salvo `--allow-dangerous`, y aun así piden confirmación. `--yes`
  desactiva las confirmaciones: úsalo solo en directorios desechables.
- Cada petición consume contexto: el historial se recorta a `--history` mensajes
  (20) y el modelo va con `--ctx 2048`. Para tareas largas usa `ogemini agent`
  o un modelo cloud con más contexto.
- Ollama Cloud y NVIDIA no soportan JSON Schema aquí: el modo agente se
  bloquea; puedes forzar `--no-schema` para que intente parsear el JSON del
  texto, con menor fiabilidad. El proveedor no cambia en caliente: `/provider`
  indica qué prefijo usar al relanzar.

---

## 10. Sesiones, prompts e historial de benchmarks

### Conversaciones persistentes

`chat --session NOMBRE` guarda el historial tras cada respuesta, tanto en modo
interactivo como con una sola pregunta. La siguiente llamada con el mismo nombre
recupera el contexto; `--new-session` lo descarta y empieza de cero.

**Al cerrar una sesión, la CLI imprime el comando exacto para volver a ella.**
También lo deja en `~/.ollama-sessions/last.txt`, así que no hace falta
acordarse del nombre:

```bash
ochat --session proyecto "responde solo con HOLA"
# Sesión 'proyecto' guardada: 2 mensajes.
# Para continuar: ocli chat -m gemma3:gpu --session proyecto

osess last                                           # el último comando guardado
osess resume proyecto                                # comando de esa sesión
ocli chat --auto-session                             # nombre automático si no pones --session
ocli chat --session proyecto                         # modo interactivo
ocli chat --session proyecto --new-session "otro tema"
osess list                                           # sesiones y nº de mensajes
osess show proyecto                                   # volcar la conversación
osess clear proyecto                                  # pide confirmación
osess clear --all --yes                               # borrar todas (y el last.txt)
```

El comando devuelto incluye el prefijo correcto según el proveedor (`ocli`,
`ocloud`, `onvidia`, `ogemini` u `ofree`), el `-m` usado y, en el agente, `--root`
y `--no-schema` si corresponden. Si cierras un chat sin `--session`, se avisa de
que esa conversación no se guardó; `--auto-session` crea un nombre
`auto-AAAAMMDD-HHMMSS`.

Los archivos viven en `~/.ollama-sessions/<nombre>.json` (el nombre se normaliza a
caracteres alfanuméricos, `-` y `_`). Son pequeños y se incluyen en el backup.
`last.txt` es solo texto y no cuenta como sesión en `osess list`.

Ojo con el contexto: `num_ctx` sigue siendo 2048. Una sesión muy larga puede
desbordarlo; la CLI no trunca el historial automáticamente.

### Biblioteca de prompts

`~/.ollama-prompts.json` guarda plantillas con variables `{{1}}`, `{{2}}`, `{{args}}`
y `{{clave}}`. Se usa con `oprompt` / `ocli prompt`.

```bash
ocli prompt add resumen --text 'Resume en una frase: {{args}}' --desc 'resumen rápido'
ocli prompt add saludo --text 'Di hola a {{persona}} y menciona {{1}}'
ocli prompt list
ocli prompt show resumen
oprompt use resumen "texto largo a resumir" --no-stream
oprompt use saludo "Mundo" --var persona=Nico --no-stream
ocli prompt rm saludo
```

| Opción de `prompt use` | Para qué |
|---|---|
| `-m, --model` | sustituye el modelo guardado en la plantilla |
| `--system` | sustituye el mensaje de sistema |
| `--var CLAVE=VALOR` | rellena `{{clave}}` (se puede repetir) |
| `--session` | guarda la respuesta en esa sesión |
| `--no-stream`, `--stats` | igual que en `chat` |

Si queda alguna variable `{{...}}` sin rellenar, la CLI avisa por stderr.

### Historial de benchmarks

Cada `ocli bench` guarda una línea JSON con fecha, modelo, tokens, segundos, tok/s y
VRAM en `~/.ollama-bench-history.jsonl`.

```bash
ocli bench                       # guarda el resultado de 3 modelos
ocli bench --history             # últimos 15 registros
ocli bench --history 5           # últimos 5
ocli bench gemma3:q4 --no-save   # medir sin registrar
ocli bench gemma3:ctx4k --ctx 4096 --no-save  # forzar contexto
```

Por defecto `bench` usa el `num_ctx` declarado en el Modelfile de cada modelo;
`--ctx` solo lo fuerza cuando se indica explícitamente. Es útil para detectar
regresiones tras cambiar el gobernador, el drop-in de Ollama o los parámetros de
un Modelfile.

---

## 11. Otras recetas de API

```bash
# Generar texto
curl -s http://127.0.0.1:11434/api/generate \
  -d '{"model":"gemma3:gpu","prompt":"hola","stream":false,"options":{"num_predict":40}}'

# Chat con historial
curl -s http://127.0.0.1:11434/api/chat \
  -d '{"model":"gemma3:gpu","stream":false,"messages":[{"role":"user","content":"hola"}]}'

# Embeddings (NO usar /api/chat ni /api/generate con qwen)
curl -s http://127.0.0.1:11434/api/embed \
  -d '{"model":"qwen:gpu","input":"texto a vectorizar"}' | head -c 200
```

### Modo nube con API key (`ocloud`)

El servicio local sigue con `OLLAMA_NO_CLOUD=true`. Para usar modelos de servidor
sin tocar el override, la CLI habla directamente con `https://ollama.com` cuando
encuentra `OLLAMA_API_KEY`:

```bash
# 1) Crea una key en https://ollama.com/settings/keys
# 2) Pégala en un archivo de un solo usuario:
printf 'TU_API_KEY' > ~/.ollama-cloud-key
chmod 600 ~/.ollama-cloud-key

# 3) Alterna con un prefijo, sin cambiar nada del servicio local:
ocloud config                                  # MODO: CLOUD
ocloud list                                    # catálogo de modelos cloud
ocloud chat -m glm-5.3 "revisa este diff"
ocloud chat -m gemma4:31b --no-stream "explica..."
ocli chat -m nemotron3:gpu "hola"              # vuelve al local
```

`ocli` y `ocode` siguen yendo a tu iGPU. La key no se guarda en el repo, en los
logs ni en los backups; `~/.ollama-cloud-key` está en `.gitignore`.

**Nombres:** en la API directa se usa el nombre real (`gemma4:31b`), no el
sufijo `:cloud` que se usa desde la app/CLI de Ollama.

El archivo de key **no activa el modo cloud por sí solo**: los prefijos
`ocloud`/`onvidia`/`ogemini` lo leen y exportan la variable correspondiente.
Puedes tener la key guardada y `ocli` seguirá yendo al servidor local.

**Límites:** los modelos cloud no soportan structured outputs. `act` está
bloqueado en modo cloud y `code` exige `--no-schema`. Los prompts se procesan en
la nube (según Ollama, no se usan para entrenar) y consumen créditos del plan.

**Importante: ningún modelo cloud de Ollama es gratis.** El plan "Pay as you go"
no tiene suscripción, pero cobra por token y exige créditos; por eso
`glm-5.3-flash` responde
`Add usage credits or subscribe and save up to 66%`. Ejecutar modelos en tu
hardware sí es ilimitado y gratuito. Para no depender de créditos, el kit tiene
`ofree`:

```bash
ofree config                                  # GRATIS: ignora keys y HOST remoto
ofree chat -m glm-5.3-flash "hola"            # usa nemotron3:gpu en local
ofree agent --root ~/ollama-local-kit         # agente local, sin créditos
```

`ofree` fuerza `OLLAMA_KIT_LOCAL=1`, borra las variables de key y traduce los
nombres cloud habituales a modelos locales (`glm-5.3-flash`, `glm-5.3`,
`kimi-k2.7-code`, `deepseek-v4.1-flash`, `nemotron-3-super` → `nemotron3:gpu`;
`beam`, `clef:27b` → `gemma3:gpu`; `gemma4:31b`/`gemma4:cloud` → `gemma4:cpu`).
Un modelo que no esté en esa tabla se pasa tal cual, así que conviene usar los
nombres que muestra `ocli list`. Si una llamada cloud devuelve el error de
créditos, la CLI añade el aviso de usar `ofree`.

**Modelos cloud para desarrollo** (catálogo consultado el 2026-11-22):

| Modelo | Para qué | Contexto |
|---|---|---|
| `glm-5.3` | Referencia para código y tareas agénticas largas; el más capaz para desarrollar | 1M |
| `kimi-k2.7-code` | Enfocado en código, ~30 % menos tokens de razonamiento; mejor coste | 256K |
| `beam` | Eficiente en tokens, bueno para iterar sin disparar el coste | 1M |
| `deepseek-v4.1-flash` | Respuestas rápidas y mucho throughput; tareas cortas y baratas | 1M |
| `gemma4:31b` | Misma familia que tus `gemma4:cpu`/`:dgpu`; útil para comparar local vs nube | 256K |
| `nemotron-3-super` | 120B MoE (12B activos), alto rendimiento por coste | 256K |
| `kimi-k3.1-pro`, `hy4`, `mimo-v2.6-pro` | Reserva para arquitectura crítica; son de los más caros | 1M |
| `clef:27b` | Clasificación y decisiones sí/no; **no** es para programar | 125K |

**Flujo recomendado para optimizar coste y calidad:**

1. **Pensar en cloud:** `glm-5.3` o `kimi-k2.7-code` para arquitectura, revisión
   de diffs grandes, generación de tests y explicación de repos completos.
2. **Fijar el conocimiento en local:** guarda las decisiones con
   `ocli prompt add ...` e indexa el repo con `orag index DIRECTORIO`.
3. **Iterar en local:** `nemotron3:gpu` para revisar, `gemma3:gpu` para
   tests/explicaciones y `odgpu` para generación larga puntual.
4. **Volver a cloud solo cuando el local se quede corto** por contexto o
   razonamiento. Así el gasto se limita a la fase de diseño.

### Otros proveedores: NVIDIA NIM y Google Gemini

Los modelos de Ollama Cloud no son los únicos que puede usar la CLI. Con una key
propia se puede hablar con NVIDIA NIM y con Gemini mediante un adaptador
OpenAI-compatible: la CLI sigue enviando comandos Ollama (`/api/chat`,
`/api/embed`, `/api/tags`) y la capa `api_prepare()` traduce las rutas y los
cuerpos, y `api_stream()` convierte el SSE (`data: ...`, `[DONE]`) al NDJSON que
ya consume `cmd_chat`. El resultado se normaliza a forma Ollama para no
reescribir los comandos.

```bash
printf 'TU_KEY_DE_NVIDIA' > ~/.nvidia-api-key; chmod 600 ~/.nvidia-api-key
printf 'TU_KEY_DE_GEMINI' > ~/.gemini-api-key; chmod 600 ~/.gemini-api-key

onvidia config                                       # PROVEEDOR: nvidia
onvidia chat -m nvidia/nemotron-3-ultra-550b-a55b "revisa este diff"
onvidia chat "resume esto"                           # usa el modelo por defecto
ogemini config                                       # PROVEEDOR: gemini
ogemini chat -m gemini-4-hydrogen "explica esto"
ocli chat -m nemotron3:gpu "hola"                    # vuelves al local
```

| | NVIDIA NIM | Google Gemini |
|---|---|---|
| Base | `https://integrate.api.nvidia.com/v1` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| Variable / archivo | `NVIDIA_API_KEY` / `~/.nvidia-api-key` | `GEMINI_API_KEY` / `~/.gemini-api-key` |
| Modelo de ejemplo | `nvidia/nemotron-3-ultra-550b-a55b` (1M ctx) | `gemini-4-hydrogen` |
| Razonamiento | `reasoning_effort: none\|medium\|high`; `--think` → `medium`, `think:false` → `none` | El thinking de 2.5 Pro/3/4 **no se puede desactivar** |
| JSON Schema | No aparece en el schema del endpoint: `act` bloqueado y `code` exige `--no-schema` | Sí (`response_format`): `act` y `code` funcionan |
| Embeddings | No expone `/embeddings` en esa API: la CLI lo bloquea y sugiere `qwen:gpu` u `ogemini` | Sí, `/embeddings` con `gemini-embedding-001` |
| Listado | `/models` | `/models` |

`onvidia` tiene como modelo por defecto `nvidia/nemotron-3.5-lightning-30b-a3b`
(30B totales / 3B activos, hasta 1M de contexto y soporte en español). Se aplica
a `chat`, `ask`, `agent`, `rag ask`, `code`, `run` y `prompt use` cuando no se
pasa `-m`; `ocli`/`ofree` conservan sus defaults locales. `onvidia config` lo
muestra como `Modelo por defecto`.

### Acceso gratuito del NVIDIA Developer Program

Unirse al [NVIDIA Developer Program](https://developer.nvidia.com/developer-program)
es gratis y da acceso a los endpoints NIM alojados en
[build.nvidia.com](https://build.nvidia.com/), además de contenedores NIM
descargables para self-host en hasta 16 GPUs mientras dure la membresía. Es un
acceso de **evaluación y prototipado**, no producción, y está sujeto a límites:
el [FAQ oficial de NIM](https://forums.developer.nvidia.com/t/nvidia-nim-faq/300317)
indica que el número de llamadas depende del modelo y de la carga concurrente, y
que se consulta en la esquina superior derecha de build.nvidia.com. En foros se
reportan ~1.000 créditos de prueba y ~40 RPM como referencia habitual, pero
NVIDIA no garantiza esa cifra ni ofrece una recarga gratuita documentada; para
más capacidad hay que desplegar o pagar.

Por eso conviene reservar `onvidia` para consultas puntuales que necesitan mucho
contexto o razonamiento, y dejar `ofree` para programar, RAG y agente. Como
NVIDIA no admite JSON Schema aquí, `act` está bloqueado y `code`/`agent`
requieren `--no-schema`. El parámetro `--ctx` no se envía al endpoint (este lo
gestiona), pero `--max-tokens` sí; `--think` activa `reasoning_effort: medium`.

Detalles operativos:

- `OLLAMA_KIT_PROVIDER=nvidia|gemini|ollama` selecciona el proveedor sin los
  alias; `OLLAMA_HOST` explícito sigue mandando sobre la base por defecto.
- `config` muestra PROVEEDOR, MODO, HOST, key y ruta del archivo de key.
- `list` consulta el catálogo remoto; `show` y `stop` avisan de que solo operan
  sobre el Ollama local, para no fingir que se descarga un modelo remoto.
- Las keys van en archivos `chmod 600`, nunca en el repo, los logs ni los
  backups, y están en `.gitignore`. Los errores 401/402/403/429 añaden pista
  sobre la key y el plan.
- No se lanzó ninguna inferencia real con estas keys durante la integración: la
  verificación usó `urlopen` simulado. Cada llamada real consume créditos.

---

## 12. `oflow`: flujos con modelos de contexto corto

Tu hardware no puede pasarle un libro entero a `gemma3:gpu` (2048 tokens de
contexto). `oflow` invierte la responsabilidad: **el modelo no recuerda nada; la
herramienta guarda el estado en disco**. Cada llamada recibe un fragmento y, como
mucho, un resumen parcial; nunca el documento completo. Es local por diseño: los
proveedores externos se rechazan porque estos modos dependen de JSON Schema y de
archivos en este equipo.

### Los cuatro modos

| Modo | Arquitectura | Cuándo usarlo |
|---|---|---|
| `map` | Dividir y conquistar | resumir o extraer lo mismo de cada fragmento |
| `slide` | Ventana deslizante | traducir o resumir con coherencia entre párrafos |
| `ask` | RAG efímero por documento | preguntar un dato concreto sin índice global |
| `classify` | Micro-clasificación + informe | etiquetar cláusulas, detectar patrones |

```bash
oflow map informe.md --prompt "Resume en 3 viñetas" --out informe
oflow map libro.txt --prompt "Extrae nombres propios" --chunk-chars 700 --limit 20
oflow slide novela.txt --prompt "Continúa el resumen" --carry 300 --out novela
oflow classify contrato.md --labels rescision,confidencialidad --out contrato
oflow ask --path contrato.md "¿hay cláusula de rescisión?" -k 3
```

`--out` relativo cuelga de `~/.ollama-flow/` (configurable con `OLLAMA_FLOW_DIR`).
Si no se da, se deriva del nombre del archivo. Estado por trabajo:

```
~/.ollama-flow/<trabajo>/
├── state.json       # firma de la configuración; detecta cambios
├── results.jsonl    # una fila por fragmento ya procesado
├── carry.txt        # memoria externa de slide (recortada a --carry)
└── report.md        # informe final agrupado
```

Si se interrumpe, vuelve a ejecutar **el mismo comando**: salta los fragmentos
cuyo ID ya está en `results.jsonl`. Si cambias el prompt, el modelo o el tamaño de
fragmento, la firma no coincide y pide `--restart` (o usa otro `--out`).
`--limit N` procesa N fragmentos por ejecución y también es reanudable.

### Detalles de cada modo

- **`map`**: fragmenta por párrafos completos (los párrafos largos se dividen por
  frase) y aplica `--prompt` a cada uno. No se pasa ningún resultado anterior:
  cada respuesta es independiente.
- **`slide`**: antepone el resumen parcial de `carry.txt` al fragmento actual y
  vuelve a recortar el resultado a `--carry` caracteres. Es la memoria del modelo,
  pero fuera del modelo.
- **`classify`**: fuerza un JSON Schema con `matches[{label,evidence}]` y `enum`
  con tus etiquetas, temperatura 0 y `think=False`. El informe agrupa
  coincidencias por etiqueta y separa las respuestas que no fueron JSON válido.
- **`ask`**: calcula embeddings efímeros con `qwen:gpu`, combina similitud coseno
  con léxico IDF, descarta fragmentos vecinos repetidos e inyecta solo `k` bloques
  hasta `--context-chars`. **No escribe** `~/.ollama-rag/index.json`: cada
  documento tiene su propia búsqueda en memoria.

Opciones comunes: `-m` (default `gemma3:gpu`), `--chunk-chars` (700), `--overlap`
(100), `--ctx` (2048), `--max-tokens` (256), `-t` (0), `--out`, `--restart`.
En `ask`: `--path` (obligatorio), `--embed-model`, `-k` (3) y `--context-chars`
(3600).

> `oflow` no llama a la API de ningún proveedor de pago. Si el entorno apunta a un
> HOST externo, el comando se detiene con un mensaje que sugiere `ofree flow`.

---

## 13. Benchmarks y decisiones ya probadas

| Configuración | tok/s |
|---|---:|
| `gemma3:q4` en CARRIZO | **9,3** |
| `gemma3:gpu` (Q6) en CARRIZO | 7,8 |
| CPU con gobernador powersave | 4,2 |
| BONAIRE (dGPU, descartada) | ~35 |
| Dos instancias (iGPU 8,4 + dGPU 22,7) | ~31 agregado |
| `nemotron3:gpu` | 2,6–3,6 |
| `nemotron3:precise` (ctx 8192, 1024 tokens de salida) | 3,08 |
| `qwen35:gpu` (33/33 capas en GPU) | 2,61 |
| `gemma4:cpu` (100% CPU, 0 GiB VRAM) | 2,56 |
| `gemma4:dgpu` (42/43 capas, instancia aislada) | 8,54 |

- `OLLAMA_SCHED_SPREAD=true` **no reparte capas** en Ollama 0.33.2 + Vulkan: el modelo
  va entero a un dispositivo. Por eso se descartó usar ambas GPUs.
- Tool calling nativo con Gemma 1B: descartado (ver sección 5).
- Los GGUF 4B locales rinden peor que `nemotron3` en generación (Qwen3.5: 2,61 tok/s),
  pero `qwen35:gpu` sí se importó y sirve como modelo alternativo con 100% GPU.
- Comparativa de embeddings (misma prueba de 6 preguntas, 6 documentos de un solo tema):

| Modelo | Dimensiones | top-1 | MRR | Tiempo |
|---|---:|---:|---:|---:|
| `qwen:gpu` | 1024 | **5/6** | **0,917** | ~1 s |
| `qwen35:embed` (`pooling_type=3`) | 2560 | 4/6 | 0,806 | ~5 s |
| `qwen35:mean` (`pooling_type=1`) | 2560 | 3/6 | 0,722 | ~5 s |
| `qwen35:cls` (`pooling_type=2`) | 2560 | 1/6 | 0,350 | ~5 s |

  Las variantes `mean` y `cls` se desinstalaron; solo queda `qwen35:embed` (la mejor de
  las tres variantes de Qwen3.5). El parche del GGUF inserta 96 bytes de metadato para
  conservar la alineación a 32 bytes; con 35 bytes llama.cpp rechaza el archivo.

---

## 14. Importar otro GGUF (plantilla)

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

## 15. Solución de problemas

| Síntoma | Causa / solución |
|---|---|
| `no usable GPU found`, `100% CPU` | Falta `OLLAMA_IGPU_ENABLE=1`; revisar drop-in y reiniciar Ollama |
| `model 'gemma3:q4' not found` en instancia temporal | Falta `OLLAMA_MODELS=/usr/share/ollama/.ollama/models` |
| Gobernador vuelve a `powersave` o el timer está `failed` | Quitar `After=power-profiles-daemon.service` de `cpu-performance.service`, `systemctl daemon-reload`, `systemctl enable --now cpu-performance-enforce.timer`; revisar `journalctl -u cpu-performance.service` |
| `powerprofilesctl set performance: invalid choice` | Normal: este sistema solo tiene `power-saver`/`balanced`; usar `cpupower` |
| `does not support chat` / `generate` con qwen | Es embedding: usar `/api/embed` |
| `Not enough memory for command submission` con Gemma4 en la iGPU | El modelo no cabe ahí; usa `gemma4:cpu` (`num_gpu 0`) u `odgpu` para la R7 260X |
| Gemma4 devuelve `response` vacío | Está gastando los tokens en `<think>`; la CLI y `odgpu` envían `think:false` |
| El enrutador `run` elige mal | Usar `orun --intent act\|rag\|chat ...` o revisar con `--dry-run` |
| El chat con `--session` olvida cosas | El historial supera `num_ctx=2048`; usa `--new-session` o conversaciones más cortas |
| Un prompt no sustituye `{{variable}}` | Pasar `--var clave=valor` o argumentos posicionales; la CLI avisa si quedan variables |
| `code` se queja de contexto | Reduce `--lines`, usa `--start` para el siguiente trozo o indexa con `orag` |
| Un comando de `act` termina con `[X] exit N` | Mira la salida `[stderr]` y ejecuta `odebug -n 3 --full` |
| No sabes qué pasó antes del error | `odebug` muestra todas las peticiones y comandos con su marca de tiempo |
| Peticiones lentas o coladas | `NUM_PARALLEL=1` y un modelo cargado: enviar de una en una |
| Poco espacio en `/var` o `/home` | `sudo journalctl --vacuum-size=200M`; revisar `df -h` |

---

## 16. Mantenimiento y reversión

```bash
watch -n 2 sensors                 # temperatura, consumo, ventilador
watch -n 2 ollama ps               # modelo cargado y VRAM
journalctl -u ollama -f            # logs en vivo
ollama list                        # modelos instalados
ollama show gemma3:q4              # detalle del modelo
ollama stop gemma3:q4              # descargarlo de memoria
```

Volver a la R7 260X de forma global: cambiar `GGML_VK_VISIBLE_DEVICES=0` por `1` en el
override, `sudo systemctl daemon-reload && sudo systemctl restart ollama`. Para Gemma4
no hace falta: `odgpu` usa la dedicada solo durante su propia llamada.

Deshacer los cambios:

```bash
sudo cp /etc/systemd/system/ollama.service.d/override.conf.bak \
        /etc/systemd/system/ollama.service.d/override.conf
sudo rm /etc/sysctl.d/99-ollama.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
ollama rm gemma3:q4
```

---

## 17. Ventana interactiva (`ogui`)

`ogui` abre `ollama-kit-gui.py` con Tkinter nativo: no levanta servidor, no abre
navegador y no necesita dependencias nuevas. Es un panel para **ver, ejecutar y
chatear** en modo local.

```bash
ogui
```

También hay lanzador de escritorio: `~/.local/share/applications/ollama-kit.desktop`
(con icono `~/.local/share/icons/ollama-kit.svg`). Tras instalarlo aparece como
**Kit Ollama local** en el menú de aplicaciones.

Pestañas:

| Pestaña | Qué permite |
|---|---|
| Resumen | `status`, `doctor`, `config`, `rag list` e historial de `bench` |
| Modelos | inventario, detalle (`show`), modelos cargados y descarga con confirmación |
| Chat | selector de modelo local, sesión opcional, `system`, `stats` y `chat --no-stream` en hilo |
| RAG | añadir archivos/carpetas, indexar con `qwen:gpu`, preguntar con `orag ask` (gemma3:gpu o respaldo) y limpiar el índice |
| Acciones | subpestañas Act, Run y Code: simular, ejecutar solo entradas permitidas y preparar prompts |
| Agente | `agent --prompt` para una sola tarea: modelo, pasos, raíz y sesión opcional; razonamiento de la decisión visible |
| Prompts | listar, ver, guardar, borrar y usar plantillas de `~/.ollama-prompts.json` con argumentos y variables |
| Bench | medir modelos locales, historial de `~/.ollama-bench-history.jsonl` y gráfico de barras de tok/s |
| Sesiones | listado, mensajes, comando de reanudación y borrado con confirmación |
| Flujos | formulario para `oflow map/slide/classify/ask` y lectura de informes |
| Comandos | catálogo filtrable; solo entradas de solo lectura y ejecución con confirmación |
| Registro | `debug -n 20`, debug completo, historial de benchmarks y limpieza con confirmación |

Seguridad de la ventana:

- Todos los subprocess se lanzan con `OLLAMA_KIT_LOCAL=1` y
  `OLLAMA_KIT_TASK_MODE=1`, y se eliminan `OLLAMA_HOST`, `OLLAMA_API_KEY`,
  `NVIDIA_API_KEY`, `GEMINI_API_KEY`, `OLLAMA_KIT_PROVIDER` y
  `OLLAMA_KIT_PRECISE`. Aunque haya keys guardadas, la GUI no las exporta.
- La casilla global **Máxima precisión** fija `OLLAMA_KIT_PRECISE=1` para todos
  los subprocess; alcanza a las doce pestañas sin tocar cada comando. La
  cabecera y el estado muestran «perfil preciso» o «perfil tareas». No relaja la
  lista blanca ni las confirmaciones. Con perfil preciso el timeout de los
  subprocess de la GUI sube de 300 a 900 s (`gui_timeout()`), para que no corte
  las inferencias largas. El inventario de `ocli list`, que alimenta los
  desplegables, ya describe los tags nuevos (`bonsai:q1`, `gemma3:1b`,
  `lfm25:350m`, `qwen3:06b`), `nemotron3:precise`, `gemma4:dgpu-safe` y los seis
  tags `ctx4k`/`ctx8k` conservados.
- La pestaña Comandos solo muestra una lista blanca de entradas de solo lectura.
  Quedan fuera benchmarks con inferencia, reindexado RAG, `review`, borrados,
  `stop` y copias de seguridad.
- La pestaña Acciones nunca expone `act --free`, `--yes` ni `--allow-dangerous`.
  Primero simula con `act --dry-run`; solo si el identificador elegido está en la
  lista blanca se habilita "Ejecutar segura". `run` solo ejecuta directamente
  las intenciones `chat` y `rag`: `act` y `automático` quedan en simulación.
- La pestaña Agente usa siempre `agent --prompt`, sin REPL. No ofrece `--yes` ni
  `--allow-dangerous`, y muestra el comando exacto antes de ejecutar la tarea.
- `code` no hace commits: `commit` solo redacta el mensaje a partir del diff.
- Descargar modelos de memoria, borrar sesiones, reindexar/borrar RAG, ejecutar
  acciones o limpiar el log pide confirmación y muestra el comando exacto.
- Al abrir la ventana no se lanza ningún modelo: solo se consultan inventario,
  sesiones, flujos y catálogo. El chat y los flujos se ejecutan solo al pulsar
  su botón, siempre con el modelo local elegido.
- El archivo es importable para pruebas; `python3 -m py_compile
  ollama-kit-gui.py` y `tests/test-gui.py` pasan sin inferencias. La prueba
  simula envío de chat, pregunta RAG, cancelación/aceptación de comandos, borrado
  de sesión, simulación y ejecución segura de `act`, bloqueo de `run` automático,
  preparación de prompts de `code`, tarea única del agente, guardado/uso de
  plantillas y medición de `bench`. Ver sección 18.

---

## 18. Perfil de tareas, razonamiento visible y pruebas

### Perfil de tareas

Desde la tanda 18 el modo habitual es **perfil tareas**, no conversación larga.
La idea es pedir acciones cortas y verificables, con el contexto fijo de 2048
tokens de esta máquina. `OLLAMA_KIT_TASK_MODE=1` (valor por defecto) activa estos
defaults:

| Rol | Modelo | Cambio con `--reasoning` o `OLLAMA_KIT_TASK_MODE=0` |
|---|---|---|
| `chat`, `ask`, `run`, `act`, agente, RAG, `flow`/`oflow` | `gemma3:gpu` | `nemotron3:gpu` en agente y RAG |
| Respaldo de razonamiento | `nemotron3:gpu` | es el modelo elegido con `--reasoning` |
| Embeddings | `qwen:gpu` | sin cambios |

Los prompts de RAG y `run` añaden una instrucción de longitud acotada (~80
palabras) y las respuestas usan temperatura 0 y `--max-tokens` más bajo en perfil
tareas. El enrutador y el agente reducen también el número de pasos. `ocli
config` muestra el perfil activo:

```bash
ocli config
# PERFIL: TAREAS (acciones cortas)
# RAZONAMIENTO: siempre visible en todas las respuestas
# Modelos por defecto: tareas=gemma3:gpu, respaldo=nemotron3:gpu (--reasoning), embeddings=qwen:gpu
```

Para volver al comportamiento anterior sin tocar cada comando:

```bash
OLLAMA_KIT_TASK_MODE=0 ocli chat "explica esto"
oagent --reasoning --prompt "analiza este diff"
```

### Perfil de precisión (opt-in)

`--precise` o `OLLAMA_KIT_PRECISE=1` invierte la prioridad: máxima precisión sin
importar el tiempo. Está en `chat`, `act`, `run`, `agent` y `rag ask`.

| Parámetro | Perfil tareas | Perfil preciso |
|---|---|---|
| Modelo | `gemma3:gpu` | `nemotron3:precise` |
| Temperatura | 0,3 (0 en agente/RAG) | 0 |
| Contexto | 2048 | 8192 |
| Salida máxima | 192–512 tokens | 1024 tokens |
| Razonamiento | visible si el modelo lo da | `think` activo y visible |

```bash
ocli chat --precise "analiza este diff"
orag ask "¿qué dice el contrato?" --precise -k 4
OLLAMA_KIT_PRECISE=1 ocli config    # muestra PERFIL DE PRECISIÓN
```

Un `-m` explícito se respeta; se ajustan los demás parámetros. La receta es
`config/Modelfile.nemotron3-precise`. Medición local: 620 tokens en 201 s
(3,08 tok/s), con respuesta final y cita `[3]` correcta. No usa APIs de pago.

En RAG, `--min-score` vale 0.10 por defecto y `rag_context_budget()` recorta el
contexto: `ctx - max_tokens - system - pregunta - reserva (64) - razonamiento
(128)` a 3 caracteres/token. Así el prompt no se trunca aunque el tag tenga 8192.
La cabecera indica fragmentos usados/totales, umbral y presupuesto; sin
candidatos se imprime `No lo encuentro en el índice`. La comparativa de modelos
con el índice de 23/11 y la sonda factual están en la sección 1; resumen: no
cambiar el default, usar `-m qwen3:06b` solo para RAG rápida con contexto y
`--precise` para respuestas críticas.

### Validación de JSON Schema con reintento

`act`, `agent_call` (decisiones del agente) y `flow_chat` con esquema usan
`api_chat_json()`: validan la respuesta con `schema_valid()` (tipos, `required`,
`enum` e `items`) antes de aceptarla. Si no valida, se prepara una segunda
petición con `json_retry_payload()`: el contenido de usuario largo se recorta
conservando principio y final, se añade una instrucción que exige solo JSON y se
amplía `num_predict` hasta el margen disponible del contexto. El bloque
`RAZONAMIENTO:` de ambos intentos se imprime. Si el segundo tampoco valida, la
CLI termina con un error explícito en vez de seguir con datos inventados.

Esto refuerza la preferencia del kit por `format` + JSON Schema frente al tool
calling nativo con modelos pequeños. Sin inferencias: `tests/test-schema.py`
simula una primera respuesta inválida y una segunda correcta, y comprueba el
reintento en `act` y en flujos.

### Razonamiento siempre visible

Toda respuesta de `chat`, `act`, `run`, RAG, agente y flujos empieza por un bloque
`RAZONAMIENTO:`. En streaming, el texto `thinking` se imprime en cuanto llega y
después aparece `RESPUESTA:`; si el modelo no devuelve texto de razonamiento, se
escribe `(el modelo no devolvió texto de razonamiento)`. En el agente, la decisión
se muestra como `RAZONAMIENTO AGENTE:`, aunque el campo `thought` esté vacío.

`think` **no se envía a ciegas**. `gemma3:gpu` declara solo la capacidad
`completion` y responde `HTTP 400: "gemma3:gpu" does not support thinking` si se
le envía `think=true`; `nemotron3:gpu` declara `completion`, `tools` y `thinking`.
La función `thinking_for(modelo)` decide el valor del parámetro: `true` para
modelos que declaran `thinking` y ausencia de clave para el resto. `--think`
sigue aceptándose por compatibilidad, pero ya no desactiva nada.

Comprobación local, sin créditos ni API keys:

```bash
python3 ~/ollama-cli.py show gemma3:gpu     # Capabilities: completion
python3 ~/ollama-cli.py show nemotron3:gpu  # Capabilities: completion, tools, thinking
```

### Pruebas versionadas

Las pruebas están en `tests/` y sustituyen a los antiguos scripts temporales de
`/tmp/opencode`. Ninguna hace inferencias: parchean `api()`, `api_stream()`,
`run_process_text()` y los cuadros de diálogo de Tkinter, o cargan la CLI con
variables de entorno aisladas.

| Prueba | Cubre |
|---|---|
| `test-agent.py` | herramientas del agente, confirmaciones y `/undo` |
| `test-flow.py` | `map`/`slide`/`classify`/`ask`, JSON Schema y reanudación |
| `test-free.py` | `ofree` ignora keys y traduce modelos cloud |
| `test-gui.py` | entorno local, argv, parseo y 12 pestañas de Tkinter |
| `test-ninth.py` | proveedores local/cloud/NVIDIA/Gemini y normalización de API |
| `test-nvidia-default.py` | defaults por proveedor y `--reasoning` |
| `test-session.py` | sesiones persistentes y comandos de reanudación |
| `test-task.py` | perfil tareas, defaults, razonamiento visible y agente de una sola petición |
| `test-precision.py` | perfil preciso, defaults y presupuesto anti-truncado de RAG |
| `test-schema.py` | validación de esquemas, recorte del prompt y reintento en `act` y flujos |

```bash
for f in ~/ollama-local-kit/tests/test-*.py; do python3 "$f"; done
bash ~/ollama-scripts/selftest.sh
# [1/6] sintaxis CLI · [2/6] scripts · [3/6] catálogo · [4/6] índice RAG
# [5/6] doctor · [6/6] pruebas versionadas → SELFTEST OK
```

`selftest.sh` lee las pruebas desde `~/ollama-local-kit/tests` por defecto;
`OLLAMA_KIT_TESTS=/otra/ruta/tests` permite cambiar la ubicación sin copiar los
archivos al directorio de scripts. La revisión visual de `ogui` no pudo hacerse
en este entorno porque las tres vías de captura disponibles (`import`,
`ffmpeg x11grab` y `PIL.ImageGrab`) devolvieron una imagen negra; la comprobación
se hizo por introspección de Tkinter: 12 pestañas, geometría y widgets presentes,
sin excepciones.
