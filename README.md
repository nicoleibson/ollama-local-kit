# ollama-local-kit

Kit de terminal para exprimir modelos locales de Ollama en una APU Carrizo usando
**solo la iGPU**, sin arriesgar la R7 260X dedicada. Incluye CLI, catálogo editable
de acciones, scripts, benchmarks, monitorización, autocompletado, RAG local y una
ventana Tkinter (`ogui`) para ver, ejecutar y chatear sin servidor ni navegador.

> Probado el 22/11/2026 con Ollama 0.33.2 + Vulkan (RADV). Documentación completa en
> [`ollama-kit.md`](ollama-kit.md).

## Equipo y decisiones clave

| Elemento | Valor |
|---|---|
| CPU / iGPU | AMD A8-9600 / A10-9700, Radeon R7 (Carrizo) |
| GPU usada | `Vulkan0` / `GPU0` = RADV CARRIZO |
| dGPU | R7 200 Series (`Vulkan1`) **intacta** |
| VRAM iGPU | ~4,4 GiB compartidos (~3,8 GiB libres) |
| RAM | ~6,7 GiB |
| Contexto | 2048 tokens, 1 modelo cargado, KV `q8_0` |
| Nube | desactivada (`OLLAMA_NO_CLOUD=true`) |

Se descartó repartir capas entre GPUs y usar dos instancias: con esta RAM y este
driver, Ollama coloca el modelo entero en una sola GPU. Dos instancias daban
~31 tok/s agregados, pero activaban la dedicada.

## Modelos recomendados

| Modelo | Uso | Rendimiento medido |
|---|---|---|
| `gemma3:gpu` | **principal**: chat, `oflow`, acciones, RAG, enrutador y tests (Q6_K uncensored); el agente usa `nemotron3:gpu` (4B) | 7,8 tok/s |
| `gemma3:q4` | alternativa rápida (Q4_K_M) | 9,3–9,8 tok/s |
| `nemotron3:gpu` | **respaldo** de razonamiento (`--reasoning` o `OLLAMA_KIT_TASK_MODE=0`) y `act --free` | ~2,6–3,5 tok/s |
| `nemotron3:precise` | **máxima precisión** (`--precise`): temp 0, ctx 8192, 1024 tokens | ~3,1 tok/s |
| `qwen:gpu` | embeddings (solo `/api/embed`) | 1024 dimensiones, ~1 s / 5 textos |
| `qwen35:gpu` | generación alternativa, 100% GPU | 2,61 tok/s, 2,66 GiB VRAM |
| `qwen35:embed` | embeddings alternativos (no recomendado) | 2560 dimensiones, ~5 s / 5 textos |
| `gemma4:cpu` | prueba de Gemma 4, solo CPU | 2,56 tok/s, 0 GiB VRAM |
| `gemma4:dgpu` | Gemma 4 en la R7 260X, vía `odgpu` | 8,54 tok/s, instancia aislada |
| `lfm25:350m` | enrutado/clasificación rápida (LFM2.5 350M) | 43,34 tok/s |
| `qwen3:06b` | chat/JSON ligero y RAG rápida opcional (Qwen3 0.6B, `think:false`) | 25,21 tok/s |
| `bonsai:q1` | experimento Qwen3 1.7B en Q1_0 | 25,06 tok/s |
| `gemma3:1b` | Gemma 3 1B-it stock, alternativa a `gemma3:q4` | 9,11 tok/s |

`qwen:latest`, `gemma3:latest` y `nemotron3:latest` rinden menos: no tienen
`num_gpu` fijo en el Modelfile.

Los cuatro tags nuevos se importaron desde `~/Descargas/models` y se midieron con
`ocli bench --tokens 80 --ctx 2048` (tres pasadas, sin guardar en el historial).
LFM2.5 es el más rápido pero falló una pregunta de física, así que se reserva para
clasificar; Qwen3 y Bonsai cumplen JSON Schema y `act --dry-run`. La CLI envía
`think:false` a `qwen3:*` para que no gaste el presupuesto de salida en razonar.

El 23/11 se reindexó RAG con `qwen:gpu` (697 fragmentos, 52 archivos) y se comparó
`gemma3:gpu`, `qwen3:06b` y `--precise` sobre las mismas fuentes. Qwen3 acertó
`odgpu --safe-fit` y respondió unas 3× más rápido, pero **ambos** fallaron otras
preguntas; `--precise` fue el más fiable. El default sigue siendo `gemma3:gpu`.
Para RAG rápida: `orag ask --fast`; para respuestas críticas: `--precise`
(timeout ampliado a 900 s). Ojo: `--precise` puede agotar 1024 tokens de
razonamiento y cortar la respuesta; para preguntas cortas funciona mejor
`orag ask -m nemotron3:precise --max-tokens 600`.

Sonda factual ampliada (20 preguntas, JSON válido en las 100 respuestas):
`gemma3:gpu` 17/20, `gemma3:q4` 16/20, `bonsai:q1` 14/20, `lfm25:350m` 12/20 y
`qwen3:06b` 10/20. Qwen3 conviene usarlo siempre con contexto recuperado, nunca
como fuente de conocimiento libre. Índice RAG final: más de 700 fragmentos en 52
archivos, reconstruido con la documentación definitiva.

Qwen3.5-4B se importó y se probó como reemplazo de `qwen:gpu`, pero **no se adoptó**
para RAG: es ~5x más lento y menos preciso (4/6 vs 5/6 en recuperación). Se conserva
instalado como alternativa. El GGUF original no se modificó; `qwen35:embed` usa una copia
con `qwen35.pooling_type=3` generada por [`tools/make_qwen35_embed.py`](tools/make_qwen35_embed.py).
Detalles y comparativa completa en [`ollama-kit.md`](ollama-kit.md).

Gemma 4 E4B (7,5 B, Q2_K, 4,2 GB) **no cabe en la iGPU**: cualquier `num_gpu > 0` agota
el backend Vulkan. Existen dos variantes: `gemma4:cpu` (`num_gpu 0`, 2,5 tok/s) y
`gemma4:dgpu` (`num_gpu 42`, **8,5 tok/s** en la R7 260X). La GPU no se elige desde el
Modelfile: [`ollama-scripts/gemma4-dgpu.sh`](ollama-scripts/gemma4-dgpu.sh) levanta una
instancia temporal en el puerto 11435 con `GGML_VK_VISIBLE_DEVICES=1`, ejecuta la
petición y la apaga, sin tocar el servicio principal ni la iGPU:

```bash
odgpu "tu pregunta"
odgpu --chat
odgpu --bench
odgpu --safe-fit "pregunta con texto largo"  # ctx 8192, capas ajustadas
```

`--safe-fit` usa `gemma4:dgpu-safe` (ctx 8192 y **sin `num_gpu` fijo**) para que
Ollama ajuste las capas a la VRAM libre. En la prueba real apareció el aviso
`cannot meet free memory target`, pero no hubo aborto: Ollama situó 9/43 capas en
GPU, mantuvo ctx 8192 y respondió. El coste es alto (2,64 tok/s frente a 8,54 con
42 capas y ctx 2048) y buena parte de los tokens son de razonamiento. La primera
llamada crea el tag desde `config/Modelfile.gemma4-dgpu-safe`.
`odgpu --print-env` muestra el plan sin arrancar la dGPU.

## Inicio rápido

Los alias se instalan desde [`config/bashrc-ollama-block.txt`](config/bashrc-ollama-block.txt).

```bash
ocli doctor                                   # comprobar toda la configuración
ochat "por qué el cielo es azul"              # chat rápido
oact --list                                   # 53 entradas editables
oact "cuánto espacio libre queda"             # elige df -> df -h
oact --free --dry-run "procesos que más CPU usan"
orag index /home/onion/ollama-kit.md
orag ask "cómo funciona el modo act" -k 3
orun "muestra la fecha"                       # enrutador automático
owatch --interval 1 --count 10                # monitor en vivo
odgpu "explica la dispersión de Rayleigh"      # Gemma4 en dGPU aislada
ocode review archivo.py --dry-run              # prompt y schema sin llamar al modelo
ocode tests archivo.py                        # tests unitarios con JSON Schema
odebug -n 5                                   # últimos errores y eventos
ocloud list                                   # modelos cloud (necesita API key)
ocloud chat -m glm-5.3 "revisa este diff"     # nube bajo demanda
onvidia chat "resume esto"                    # Nemotron 3.5 Lightning por defecto
onvidia chat -m nvidia/nemotron-3-ultra-550b-a55b "revisa este diff"
ogemini chat -m gemini-4-hydrogen "explica esto"
oagent --prompt "resume este repo"             # tarea única, sin REPL
oagent --root ~/ollama-local-kit --session trabajo
oagent --reasoning --prompt "analiza este diff"  # respaldo Nemotron
ochat --precise "analiza este diff"         # máxima precisión: nemotron3:precise
orag ask "¿qué dice el contrato?" --precise  # RAG exhaustivo y citado
OLLAMA_KIT_PRECISE=1 ocli config             # confirmar perfil activo
ofree chat -m glm-5.3-flash "hola"            # gratuito: traduce a nemotron3:gpu
oflow map informe.md --prompt "Resume en 3 viñetas" --out informe
oflow slide novela.txt --prompt "Continúa el resumen" --carry 300
oflow classify contrato.md --labels rescision,confidencialidad
oflow ask --path contrato.md "¿hay cláusula de rescisión?" -k 3
ochat --session proyecto "hola"               # conversación persistente
osess last                                    # comando para volver a la última
osess resume proyecto                         # comando de otra sesión
ochat --auto-session                          # guarda con nombre automático
oprompt use resumen "texto a resumir"         # plantilla reutilizable
ocli bench --history                          # historial de rendimiento
ogui                                          # ventana local: ver, ejecutar y chatear
```

`ogui` abre **doce pestañas**: Resumen, Modelos, Chat, RAG, Acciones, **Agente**,
**Prompts**, **Bench**, Sesiones, Flujos, Comandos y Registro. Fuerza
`OLLAMA_KIT_LOCAL=1` y `OLLAMA_KIT_TASK_MODE=1`, elimina variables de nube/API
key, pide confirmación para acciones que escriben o descargan modelos y no lanza
ninguna inferencia al abrirse. El chat usa `chat --no-stream` en un hilo para no
congelar la interfaz.

Las pestañas nuevas cubren el perfil de tareas: **Agente** ejecuta `agent
--prompt` (una sola petición, sin REPL, sin `--yes` ni `--allow-dangerous`) y
muestra `RAZONAMIENTO AGENTE:`; **Prompts** gestiona la biblioteca de plantillas;
**Bench** mide modelos locales y dibuja barras de tok/s con el historial.
Toda respuesta de la CLI incluye un bloque `RAZONAMIENTO:`; si el modelo no puede
pensar, se indica `(el modelo no devolvió texto de razonamiento)`. La casilla
global **Máxima precisión** exporta `OLLAMA_KIT_PRECISE=1` a todos los subprocess
y cambia el estado a «perfil preciso»; afecta a las doce pestañas.

RAG indexa archivos con `qwen:gpu` y pregunta con `gemma3:gpu` (usa
`--reasoning` para sintetizar con `nemotron3:gpu`). Acciones simula con `act
--dry-run` y solo ejecuta entradas de solo lectura tras confirmación; nunca
expone `--free`, `--yes` ni `--allow-dangerous`. `run` solo ejecuta directamente
`chat`/`rag` y `code commit` únicamente redacta el mensaje. También hay lanzador
de escritorio (`Kit Ollama local`). Detalles en
[`ollama-kit.md`](ollama-kit.md#17-ventana-interactiva-ogui).

Funciones principales de `ollama-cli.py`:

| Comando | Qué hace |
|---|---|
| `chat` / `ask` | chat por streaming o interactivo, con `--session` persistente |
| `act` | ejecuta acciones del catálogo (53 entradas) o propone comandos con `--free` |
| `code` | `review`/`tests`/`explain`/`commit` con troceo y JSON Schema |
| `rag` | indexa archivos y responde con fragmentos recuperados |
| `run` / `do` | clasifica la intención y delega en `act`, `rag` o `chat` |
| `prompt` | biblioteca de plantillas con variables (`{{args}}`, `{{1}}`, `{{clave}}`) |
| `session` | listar, mostrar, borrar y devolver el comando de reanudación (`last`, `resume`) |
| `bench` | mide tokens/s y guarda el historial (`--history`) |
| `watch` | monitor de CPU, GPU, memoria, discos y sensores |
| `status` / `doctor` | diagnóstico y verificación |
| `debug` | errores, peticiones y tracebacks de la CLI (`~/.ollama-debug.log`) |
| `embed` | embeddings directos con `qwen:gpu` |
| `flow` / `oflow` | flujos de contexto corto: `map`, `slide`, `classify`, `ask` |
| `agent` | REPL con herramientas o `agent --prompt` para una sola tarea; decisión y razonamiento visibles |
| `list` / `show` / `import` / `rm` / `stop` | gestión de modelos |

## Programar y depurar

`ocode review|tests|explain|commit` trocea archivos (120 líneas por defecto,
`--start` para continuar) y fuerza JSON Schema según la acción. `--dry-run` muestra
el prompt exacto sin consumir tokens. Para buscar en un repo, `orag index DIRECTORIO`
y `orag ask "..."`; para ejecutar comprobaciones, `oact "run_tests"` o
`bash ~/ollama-scripts/selftest.sh`.

Los comandos de `act` muestran `[->]`, duración y `[OK]`/`[X] exit N`. Todo queda en
`~/.ollama-debug.log`: `odebug` resume, `odebug --full` muestra tracebacks y
`ocli --debug ...` añade trazas en directo. Ver manual, sección 9.

## Perfil de tareas y razonamiento visible

Por defecto el kit trabaja en **perfil tareas**: el modelo recibe instrucciones
de respuesta corta y verificable en vez de mantener conversaciones largas.
`gemma3:gpu` atiende `chat`, `act`, `run`, RAG y los flujos; el bucle del agente
requiere `nemotron3:gpu` (4B), porque los modelos de 1B devuelven JSON vacío o
piden instrucciones; la pestaña Agente de `ogui` solo ofrece modelos Nemotron;
`qwen:gpu` sigue en embeddings. `ocli config` lo resume así:

```text
PERFIL: TAREAS (acciones cortas)
RAZONAMIENTO: siempre visible en todas las respuestas
Modelos por defecto: tareas=gemma3:gpu, respaldo=nemotron3:gpu (--reasoning), embeddings=qwen:gpu
```

El razonamiento **siempre se muestra**, con el prefijo `RAZONAMIENTO:`. Se envía
`think` solo a los modelos que declaran la capacidad `thinking`
(`nemotron3:gpu`); Gemma 3 solo declara `completion` y rechaza `think=true` con
HTTP 400, así que en ese caso se imprime `(el modelo no devolvió texto de
razonamiento)`. `--think` queda como bandera histórica compatible. Para volver
al respaldo usa `--reasoning` o `OLLAMA_KIT_TASK_MODE=0`.

### Perfil de precisión (opt-in)

Cuando la calidad importa más que la velocidad:

```bash
ochat --precise "explica este diff"
orag ask "¿qué dice el índice?" --precise -k 4
OLLAMA_KIT_PRECISE=1 ocli chat "explica este diff"
```

`--precise` (o la variable) está disponible en `chat`, `act`, `run`, `agent` y
`rag ask`. Fuerza `nemotron3:precise`, `temperature 0`, `num_ctx 8192` y
`num_predict 1024`, activa `think` y mantiene el razonamiento visible. Un modelo
indicado explícitamente con `-m` se respeta; se ajustan los demás parámetros. La
receta es `config/Modelfile.nemotron3-precise`. Es ~3 tok/s: una respuesta RAG
con cita correcta tardó unos 3 minutos. No usa APIs ni créditos.

En RAG, el perfil preciso filtra con `--min-score 0.10` por defecto, calcula el
presupuesto de contexto desde `ctx - max_tokens - reserva` para que el prompt no
se trunque, pide una sola pasada de razonamiento y exige citas `[n]` con aviso de
contradicciones. La cabecera muestra `fragmentos usados/totales`, umbral y
presupuesto en caracteres. Sin fragmentos, la respuesta es
`No lo encuentro en el índice`.

`act`, el agente y el clasificador de flujos validan el JSON devuelto contra su
esquema. Si la salida no cumple (falta una clave, valor fuera de `enum`, JSON
rodeado de texto), se reenvía una vez con el prompt acortado, una instrucción
más estricta y más margen de salida; el razonamiento de los dos intentos se
muestra. Sigue usándose `format` + JSON Schema, no tool calling.

## Pruebas versionadas

Las pruebas viven en [`tests/`](tests/) y no hacen inferencias: simulan la API,
los subprocess y los widgets de Tkinter. Cubren agente, flujos, modo gratuito,
GUI (12 pestañas), proveedores, defaults, perfil de tareas y perfil de precisión
(incluido el presupuesto anti-truncado de RAG), GUI y reintento de JSON Schema.

```bash
for f in ~/ollama-local-kit/tests/test-*.py; do python3 "$f"; done
bash ~/ollama-scripts/selftest.sh        # sintaxis, catálogo, RAG, doctor y tests
```

`selftest.sh` busca las pruebas en `~/ollama-local-kit/tests`; con
`OLLAMA_KIT_TESTS` se puede apuntar a otra copia. Termina con `SELFTEST OK`.

## Flujos con contexto corto

`oflow` resuelve el cuello de botella de los 2048 tokens: **el modelo no recuerda
nada, la herramienta guarda el estado en disco**. Cuatro modos:

- `map`: dividir y conquistar; aplica un `--prompt` a cada fragmento.
- `slide`: ventana deslizante; realimenta un `carry.txt` acotado con `--carry`.
- `classify`: micro-preguntas con JSON Schema e informe agrupado por etiqueta.
- `ask`: RAG efímero por documento con `qwen:gpu`, sin tocar el índice global.

Cada trabajo vive en `~/.ollama-flow/<nombre>/` (`state.json`,
`results.jsonl`, `carry.txt`, `report.md`). Si se corta, repite el mismo comando:
salta lo ya hecho; si cambia la configuración, pide `--restart`. Solo local:
rechaza proveedores externos. Ver manual, sección 12.

## Nube bajo demanda

`ocloud` es la misma CLI apuntando a `https://ollama.com` con una API key de
`~/.ollama-cloud-key` (chmod 600). `ocli` sigue siendo local por defecto y el
override con `OLLAMA_NO_CLOUD=true` no se toca. Los modelos cloud no soportan
JSON Schema: ahí `act` está bloqueado y `code` requiere `--no-schema`. Para
desarrollar, `glm-5.3` y `kimi-k2.7-code` son la referencia; baja el resultado a
prompts y RAG locales para iterar sin coste.

También hay adaptadores para **NVIDIA NIM** (`onvidia`, key en
`~/.nvidia-api-key`) y **Google Gemini** (`ogemini`, key en
`~/.gemini-api-key`), ambos OpenAI-compatible. NVIDIA controla el razonamiento
con `reasoning_effort` pero no ofrece JSON Schema ni embeddings en esa API;
Gemini sí admite ambos, aunque su thinking no se puede desactivar. `onvidia`
usa `nvidia/nemotron-3.5-lightning-30b-a3b` por defecto (1M de contexto), del
catálogo gratuito del NVIDIA Developer Program para prototipar, con límites de
tasa. `ocli` sigue siendo local por defecto. Ver manual, sección 11.

Ningún modelo cloud de Ollama es gratis: "Pay as you go" no tiene suscripción,
pero cobra por token y necesita créditos. Para trabajar sin créditos usa `ofree`:
fuerza el servidor local, ignora las keys y traduce nombres cloud habituales
(`glm-5.3-flash` → `nemotron3:gpu`, `gemma4:31b` → `gemma4:cpu`, ...).

## Modo agente interactivo

`oagent` es un REPL con herramientas (`read`, `list`, `grep`, `rag`, `write`,
`edit`, `bash`) que encadena llamadas al modelo mediante JSON Schema. En perfil
tareas arranca con `gemma3:gpu` y 4 pasos; `--reasoning` usa `nemotron3:gpu`.
Con `oagent --prompt "..."` ejecuta **una sola tarea y sale**, sin entrar en
REPL; es lo que hace la pestaña Agente de `ogui`. Mantiene sesión y permite
`/undo` de los cambios. Las operaciones de lectura son automáticas; escribir,
editar o ejecutar comandos piden confirmación, y las rutas quedan confinadas a
`--root`. Al salir imprime el comando para reabrir esa sesión con su `--root`.
Toda decisión se imprime como `RAZONAMIENTO AGENTE:` con el protocolo **JSON
máquina**: el prompt del sistema es una especificación JSON (sin prosa) y el
modelo solo emite `{"op":"...","trace":["CODE:valor"],"args":{},"answer":{
"status":"ok","summary":"...","facts":[],"files":[]}}`. `trace` lleva códigos
(`SCAN`, `PICK`, `READ`, `ANSWER`, `MOVE`, `AUTO`...), nunca la respuesta en
prosa; si el modelo la escribe ahí, la CLI la promueve a `answer.summary`. Ver
manual, secciones 9, 10 y 18.

## Codex con modelos locales

`ollama launch codex` crea el perfil `ollama-launch`, pero Codex necesita 64k de
contexto y el servicio principal del kit usa 2048. El kit añade el tag
`nemotron3:codex` (`num_ctx=65536`, `tools`) y el alias `ocodex`:

```bash
ocodex --config                            # regenera perfil y catálogo
ocodex                                     # Codex interactivo con modelos locales
ocodex --exec "Responde exactamente HOLA"   # prueba efímera y read-only
```

El wrapper usa `ollama launch codex --model nemotron3:codex`, bloquea tags
`:dgpu` y ajusta el catálogo a 65536. Verificado: respuesta `HOLA` y
`ollama ps` con `CONTEXT 65536`. Sin créditos ni APIs de pago.

## Seguridad

- El modelo **nunca ejecuta texto generado** en modo catálogo: elige un identificador
  y el comando real sale de `ollama-commands.json`.
- `--free` usa `shell=True`, pero valida que el primer token exista, pide confirmación
  y bloquea patrones peligrosos (`rm -rf /...`, `mkfs`, `dd if=`, `reboot`,
  `curl ... | sh`). `--allow-dangerous` los permite, pero sigue pidiendo confirmación.
  Evita combinarlo con `--yes`.
- El temporizador del gobernador **no debe llevar**
  `After=power-profiles-daemon.service`: crea un ciclo en systemd. Ver
  [`INSTALL.md`](INSTALL.md).

## Estructura del repositorio

```text
README.md                      Esta guía
INSTALL.md                     Instalación paso a paso
CHANGELOG.md                   Historial de cambios
ollama-kit.md                  Manual completo (18 secciones)
ollama-cli.py                  CLI principal
ollama-kit-gui.py              Ventana Tkinter (`ogui`)
ollama-commands.json           Catálogo editable de acciones
ollama-bench.sh                Benchmark independiente
ollama-act.py                  Executor antiguo (6 comandos, legado)
ollama-scripts/                Scripts ejecutables por el modelo y `selftest.sh`
tests/                         Pruebas versionadas (sin inferencias)
completions/ollama-cli         Autocompletado para bash
config/                        Drop-ins, sysctl, Modelfiles y lanzador de escritorio
assets/ollama-kit.svg          Icono del lanzador
tools/make_qwen35_embed.py     Receta para habilitar embeddings en Qwen3.5
```

## Instalación

Consulta [`INSTALL.md`](INSTALL.md). Los archivos de `config/` son copias de referencia:
**no se aplican solos**. El instalador usa `tee`/archivos directos, nunca
`systemctl edit ollama`.

## Estado

Repositorio privado de uso personal. Sin licencia pública; puedes reutilizarlo libremente
en tus equipos, pero los detalles de hardware y rutas son específicos de esta máquina.
