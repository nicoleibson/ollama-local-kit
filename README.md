# ollama-local-kit

Kit de terminal para exprimir modelos locales de Ollama en una APU Carrizo usando
**solo la iGPU**, sin arriesgar la R7 260X dedicada. Incluye CLI, catálogo editable
de acciones, scripts, benchmarks, monitorización, autocompletado y RAG local.

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
| `gemma3:q4` | chat rápido y catálogos cortos | 9,3–9,8 tok/s |
| `gemma3:gpu` | redacción con más calidad (Q6_K) | 7,8 tok/s |
| `nemotron3:gpu` | razonamiento, `act`, RAG, enrutador | ~2,6–3,5 tok/s |
| `qwen:gpu` | embeddings (solo `/api/embed`) | 1024 dimensiones, ~1 s / 5 textos |
| `qwen35:gpu` | generación alternativa, 100% GPU | 2,61 tok/s, 2,66 GiB VRAM |
| `qwen35:embed` | embeddings alternativos (no recomendado) | 2560 dimensiones, ~5 s / 5 textos |
| `gemma4:cpu` | prueba de Gemma 4, solo CPU | 2,56 tok/s, 0 GiB VRAM |
| `gemma4:dgpu` | Gemma 4 en la R7 260X, vía `odgpu` | 8,54 tok/s, instancia aislada |

`qwen:latest`, `gemma3:latest` y `nemotron3:latest` rinden menos: no tienen
`num_gpu` fijo en el Modelfile.

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
```

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
onvidia chat -m nvidia/nemotron-3-ultra-550b-a55b "revisa este diff"
ogemini chat -m gemini-4-hydrogen "explica esto"
oagent                                        # REPL agente con herramientas
oagent --root ~/ollama-local-kit --session trabajo
ofree chat -m glm-5.3-flash "hola"            # gratuito: traduce a nemotron3:gpu
ochat --session proyecto "hola"               # conversación persistente
oprompt use resumen "texto a resumir"         # plantilla reutilizable
ocli bench --history                          # historial de rendimiento
```

Funciones principales de `ollama-cli.py`:

| Comando | Qué hace |
|---|---|
| `chat` / `ask` | chat por streaming o interactivo, con `--session` persistente |
| `act` | ejecuta acciones del catálogo (53 entradas) o propone comandos con `--free` |
| `code` | `review`/`tests`/`explain`/`commit` con troceo y JSON Schema |
| `rag` | indexa archivos y responde con fragmentos recuperados |
| `run` / `do` | clasifica la intención y delega en `act`, `rag` o `chat` |
| `prompt` | biblioteca de plantillas con variables (`{{args}}`, `{{1}}`, `{{clave}}`) |
| `session` | listar, mostrar y borrar conversaciones guardadas |
| `bench` | mide tokens/s y guarda el historial (`--history`) |
| `watch` | monitor de CPU, GPU, memoria, discos y sensores |
| `status` / `doctor` | diagnóstico y verificación |
| `debug` | errores, peticiones y tracebacks de la CLI (`~/.ollama-debug.log`) |
| `embed` | embeddings directos con `qwen:gpu` |
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
Gemini sí admite ambos, aunque su thinking no se puede desactivar. `ocli` sigue
siendo local por defecto. Ver manual, sección 11.

Ningún modelo cloud de Ollama es gratis: "Pay as you go" no tiene suscripción,
pero cobra por token y necesita créditos. Para trabajar sin créditos usa `ofree`:
fuerza el servidor local, ignora las keys y traduce nombres cloud habituales
(`glm-5.3-flash` → `nemotron3:gpu`, `gemma4:31b` → `gemma4:cpu`, ...).

## Modo agente interactivo

`oagent` es un REPL con herramientas (`read`, `list`, `grep`, `rag`, `write`,
`edit`, `bash`) que encadena llamadas al modelo mediante JSON Schema. Arranca con
`nemotron3:gpu`, mantiene sesión y permite `/undo` de los cambios. Las
operaciones de lectura son automáticas; escribir, editar o ejecutar comandos
piden confirmación, y las rutas quedan confinadas a `--root`. Ver manual,
sección 9.

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
ollama-kit.md                  Manual completo (15 secciones)
ollama-cli.py                  CLI principal
ollama-commands.json           Catálogo editable de acciones
ollama-bench.sh                Benchmark independiente
ollama-act.py                  Executor antiguo (6 comandos, legado)
ollama-scripts/                16 scripts ejecutables por el modelo
completions/ollama-cli         Autocompletado para bash
config/                        Drop-ins de systemd, sysctl y Modelfiles
tools/make_qwen35_embed.py     Receta para habilitar embeddings en Qwen3.5
```

## Instalación

Consulta [`INSTALL.md`](INSTALL.md). Los archivos de `config/` son copias de referencia:
**no se aplican solos**. El instalador usa `tee`/archivos directos, nunca
`systemctl edit ollama`.

## Estado

Repositorio privado de uso personal. Sin licencia pública; puedes reutilizarlo libremente
en tus equipos, pero los detalles de hardware y rutas son específicos de esta máquina.
