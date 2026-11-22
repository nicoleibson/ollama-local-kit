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
oact --list                                   # 41 entradas editables
oact "cuánto espacio libre queda"             # elige df -> df -h
oact --free --dry-run "procesos que más CPU usan"
orag index /home/onion/ollama-kit.md
orag ask "cómo funciona el modo act" -k 3
orun "muestra la fecha"                       # enrutador automático
owatch --interval 1 --count 10                # monitor en vivo
odgpu "explica la dispersión de Rayleigh"      # Gemma4 en dGPU aislada
ochat --session proyecto "hola"               # conversación persistente
oprompt use resumen "texto a resumir"         # plantilla reutilizable
ocli bench --history                          # historial de rendimiento
```

Funciones principales de `ollama-cli.py`:

| Comando | Qué hace |
|---|---|
| `chat` / `ask` | chat por streaming o interactivo, con `--session` persistente |
| `act` | ejecuta acciones del catálogo o propone comandos con `--free` |
| `rag` | indexa archivos y responde con fragmentos recuperados |
| `run` / `do` | clasifica la intención y delega en `act`, `rag` o `chat` |
| `prompt` | biblioteca de plantillas con variables (`{{args}}`, `{{1}}`, `{{clave}}`) |
| `session` | listar, mostrar y borrar conversaciones guardadas |
| `bench` | mide tokens/s y guarda el historial (`--history`) |
| `watch` | monitor de CPU, GPU, memoria, discos y sensores |
| `status` / `doctor` | diagnóstico y verificación |
| `embed` | embeddings directos con `qwen:gpu` |
| `list` / `show` / `import` / `rm` / `stop` | gestión de modelos |

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
ollama-kit.md                  Manual completo (13 secciones)
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
