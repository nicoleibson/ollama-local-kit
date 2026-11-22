# Changelog

## 2026-11-22 (decimotercera tanda)

- Nuevo `oflow` (`ollama-cli.py flow`, alias `oflow`) para modelos de contexto
  corto, con cuatro modos: `map` (dividir y conquistar), `slide` (ventana
  deslizante con `carry.txt`), `classify` (JSON Schema con `enum` de etiquetas e
  informe agrupado) y `ask` (RAG efímero por documento con `qwen:gpu`, sin
  escribir el índice global).
- El estado vive en `~/.ollama-flow/<trabajo>/` (`state.json`, `results.jsonl`,
  `carry.txt`, `report.md`); es reanudable repitiendo el comando y se protege
  con una firma de configuración que exige `--restart` si cambias prompt,
  modelo o fragmentación. `--limit N` también reanuda. `OLLAMA_FLOW_DIR`
  permite mover la carpeta.
- El modelo recibe solo `chunk + carry` o `chunk + pregunta`, con `stream=False`,
  `think=False`, temperatura 0, `--ctx 2048` y `--max-tokens 256`. Se bloquean
  proveedores externos (oflow siempre local).
- Añadidos alias en `~/.bashrc` y en `config/bashrc-ollama-block.txt`, línea en
  `config`, autocompletado y documentación en manual (sección 12), README e
  INSTALL. Nueva prueba sin inferencia real: `/tmp/opencode/test-flow.py`.
- Verificado sin llamadas a proveedores de pago: fragmentación por párrafos,
  reanudación, `--restart`, carry acotado, JSON Schema, degradación sin JSON,
  RAG efímero sin índice global, bloqueo externo y defaults `flow`/`oflow`.

## 2026-11-22 (duodécima tanda)

- `onvidia` usa `nvidia/nemotron-3.5-lightning-30b-a3b` como modelo por defecto
  cuando no se pasa `-m` (chat, ask, agent, rag ask, code, run y prompt use).
  Los defaults locales (`gemma3:q4`, `nemotron3:gpu`) no cambian y `ofree`
  sigue traduciendo nombres cloud. `onvidia config` muestra el modelo.
- Documentado el acceso gratuito del NVIDIA Developer Program: endpoints NIM
  para prototipado, sujetos a límites por modelo y carga, ~1.000 créditos y
  ~40 RPM como referencia reportada (no garantizada), y la necesidad de
  desplegar o pagar para producción o límites mayores.
- Verificado sin inferencias: defaults por proveedor, override explícito con
  `-m`, `ofree`, regresión de agente/proveedores y `config`.

## 2026-11-22 (undécima tanda)

- Modo gratuito `ofree`: `OLLAMA_KIT_LOCAL=1` fuerza el Ollama local, ignora
  `OLLAMA_API_KEY`/`OLLAMA_HOST`/keys de otros proveedores y traduce nombres
  cloud a modelos locales (`glm-5.3-flash` → `nemotron3:gpu`, `gemma4:31b` →
  `gemma4:cpu`, etc.). Añadir alias, autocompletado y `config` → GRATIS.
- El error de créditos de Ollama Cloud (`Add usage credits or subscribe`) ahora
  sugiere usar `ofree`. Documentado que ningún modelo cloud es gratis y que
  "Pay as you go" solo exime de suscripción.
- Corrección: tener una key guardada en `~/.ollama-cloud-key` ya no cambia el
  modo por defecto. Solo los prefijos exportan la variable; `ocli` sigue local.
- Verificado sin red: modo local forzado, sin header Bearer, mapeo de modelos y
  aviso. Sin créditos gastados.

## 2026-11-22 (décima tanda)

- Nuevo subcomando `agent` (alias `oagent`): REPL con herramientas `read`,
  `list`, `grep`, `rag`, `write`, `edit` y `bash`, bucle de decisiones con JSON
  Schema y confirmaciones. Arranca con `nemotron3:gpu` y conserva sesión.
- Las rutas se confinan a `--root`; se bloquean secretos y `.git`. Comandos
  peligrosos requieren `--allow-dangerous`, y `/undo` restaura cambios desde
  `~/.ollama-agent-undo`. `--yes` desactiva confirmaciones bajo responsabilidad
  del usuario.
- Comandos internos (`/help`, `/model`, `/provider`, `/session`, `/clear`,
  `/root`, `/tools`, `/rag`, `/bash`, `/undo`, `/yes`, `/exit`), autocompletado y
  documentación en manual, README e INSTALL. Proveedores sin JSON Schema
  bloquean el agente salvo `--no-schema`.
- Verificado con decisiones y `urlopen` simulados: sin inferencias reales ni
  créditos gastados.

## 2026-11-22 (novena tanda)

- Adaptador OpenAI-compatible para NVIDIA NIM y Google Gemini: `api_prepare()`
  traduce rutas/cuerpos, `api_stream()` lee SSE y las respuestas se normalizan a
  forma Ollama. `OLLAMA_KIT_PROVIDER` selecciona proveedor; `OLLAMA_HOST`
  explícito sigue teniendo prioridad.
- Nuevos prefijos `onvidia` y `ogemini` (keys en `~/.nvidia-api-key` y
  `~/.gemini-api-key`, chmod 600, ignoradas por git y backups), con
  autocompletado. `ocli` continúa local por defecto.
- NVIDIA envía `reasoning_effort` y bloquea `/embeddings` y JSON Schema; Gemini
  admite `response_format` y embeddings. `act`/`code` usan `NO_SCHEMA` según el
  proveedor, y `list`/`show`/`stop` distinguen local y remoto.
- `config` muestra PROVEEDOR, MODO, HOST y estado de la key. Manual, README,
  INSTALL y `.gitignore` actualizados. Verificación con `urlopen` simulado: no
  se gastaron créditos.

## 2026-11-22 (octava tanda)

- Soporte de API key directa para modelos cloud: `OLLAMA_API_KEY` añade el
  header Bearer y selecciona `https://ollama.com` cuando no hay `OLLAMA_HOST`
  explícito. `api()` y `api_stream()` registran el host en el log de debug.
- Nuevo prefijo `ocloud` (lee `~/.ollama-cloud-key`, chmod 600) con autocompletado.
  `ocli` sigue local y el override con `OLLAMA_NO_CLOUD=true` permanece intacto.
- `act` se bloquea en cloud y `code` exige `--no-schema`: los modelos cloud no
  soportan structured outputs. Errores 401/402/403/429 dan pista sobre la key.
- `config` muestra MODO, HOST y estado de la key. Manual, README, INSTALL y
  `.gitignore` actualizados con el flujo cloud→local y los modelos recomendados.

## 2026-11-22 (séptima tanda)

- Nuevo subcomando `code` (alias `ocode`) con `review`, `tests`, `explain` y `commit`:
  trocea archivos, elige modelo por acción y fuerza JSON Schema. Incluye `--dry-run`,
  `--start/--lines`, `--stdin` y `--no-schema`.
- Instrumentación: `~/.ollama-debug.log` registra peticiones API, errores HTTP,
  comandos, timeouts y tracebacks. `debug` (alias `odebug`), `--debug` global y
  `OLLAMA_KIT_DEBUG=1`. Toda acción de `act` muestra `[->]`, duración y `[OK]`/`[X]`.
- Catálogo ampliado a 53 entradas (compile, shell/json check, run_tests, git,
  doctor, bench, reindex, debug_log, review_cli) y nuevo `selftest.sh` de 5 pasos.
- Alias `ocode`/`odebug`, autocompletado, backup del log, `.gitignore` y manual
  (nueva sección 9) actualizados.

## 2026-11-22 (sexta tanda)

- Prueba aislada de Gemma4 en la R7 260X: 42/43 capas en GPU, ~1,92 GiB de VRAM y
  **8,54 tok/s** frente a 2,56 en CPU. El servicio principal y `override.conf` no
  se tocaron; la instancia temporal se apagó y el puerto 11435 quedó cerrado.
- Nuevo tag `gemma4:dgpu` (`config/Modelfile.gemma4-dgpu`) y script
  `ollama-scripts/gemma4-dgpu.sh`: arranca Ollama en 11435 con
  `GGML_VK_VISIBLE_DEVICES=1`, ejecuta la petición y la apaga siempre. Alias `odgpu`.
- `bench` envía `think:false` para que tok/s sea comparable con el uso real de la CLI.
- Manual (nueva sección 8), README, INSTALL, autocompletado y `INFO` actualizados.

## 2026-11-22 (quinta tanda)

- Corregido `stop --all`: antes ignoraba la opción y fallaba; ahora descarga todos
  los modelos que figuran en `/api/ps` y avisa cuando no hay ninguno cargado.
- Autocompletado y manual actualizados con `stop --all`.

## 2026-11-22 (cuarta tanda)

- Importado `Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf` como `gemma4:cpu`.
  Ollama 0.33.2 reconoce la arquitectura `gemma4` (42 bloques, 7,5 B, Q2_K).
- Descartado el offload a GPU: con `num_gpu` de 42 a 8 el backend Vulkan agota la
  memoria (`Not enough memory for command submission`, ~4 GiB de buffer en RAM host).
  Con `num_gpu 0` funciona a 2,56 tok/s, `100% CPU` y 0 GiB de VRAM.
- Añadido `config/Modelfile.gemma4-cpu` con `num_ctx 512`; documentado el inventario,
  los benchmarks, la tabla de uso y la solución de problemas.

## 2026-11-22 (tercera tanda)

- `chat --session NOMBRE`: conversaciones persistentes en `~/.ollama-sessions/`,
  con `--new-session` y el subcomando `session list|show|clear`.
- `prompt`: biblioteca de plantillas en `~/.ollama-prompts.json` con variables
  `{{1}}`, `{{args}}` y `{{clave}}` (`--var`), acciones `add|show|rm|list|use`.
- `bench` guarda un registro JSONL en `~/.ollama-bench-history.jsonl` con tok/s y
  VRAM; `bench --history N` muestra los últimos resultados y `--no-save` lo evita.
- Alias `oprompt` y `osess`; autocompletado ampliado para los tres comandos.
- `backup-configs.sh` incluye sesiones, prompts e historial de benchmarks.
- Manual ampliado con la sección 8 y renumeración de las siguientes.

## 2026-11-22 (segunda tanda)

- Importado `Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf` como `qwen35:gpu`.
  Con `num_gpu 33` carga 33/33 capas en la iGPU (100% GPU, 2,66 GiB VRAM, 2,61 tok/s);
  con 32 dejaba una capa en CPU. `nemotron3:gpu` se conserva como modelo principal.
- Añadido `qwen35:embed`: copia del GGUF con `qwen35.pooling_type=3` (96 bytes de
  metadato para mantener la alineación a 32 bytes) y `num_ctx 512`. Sin ese metadato
  Ollama 0.33.2 responde `HTTP 501` en `/api/embed`; con 2048 de contexto hay OOM.
- Comparados `pooling_type` 3/1/2 y `qwen:gpu`: el embedder específico sigue ganando
  (5/6 vs 4/6 en top-1, MRR 0,917 vs 0,806, ~5x más rápido). Se mantiene `qwen:gpu`
  como default de RAG y se desinstalan las variantes `mean` y `cls`.
- Añadidos `tools/make_qwen35_embed.py` y `config/Modelfile.qwen35-embed` para
  reproducir el parche sin tocar el GGUF original.

## 2026-11-22

- Añadido `run` / `orun`: enrutador que clasifica la intención (`act`, `rag`, `chat`)
  con `nemotron3:gpu` y delega en el subcomando correspondiente.
- Añadido `watch` / `owatch`: monitor en vivo de CPU, gobernador, Ollama, VRAM,
  memoria, disco, sensores y procesos.
- Añadida barrera de seguridad en `act --free`: bloquea `rm -rf /...`, `mkfs`,
  `dd if=`, `reboot`, `curl ... | sh` y escritura en `/dev/sdX`. Se puede levantar
  con `--allow-dangerous`, manteniendo la confirmación.
- Corregido el ciclo de systemd del gobernador: eliminado
  `After=power-profiles-daemon.service` de `cpu-performance.service`;
  `cpu-performance-enforce.timer` vuelve a quedar activo tras reinicios.
- Autocompletado ampliado para `run`, `watch`, `--intent`, `--act-model` y
  `--allow-dangerous`, con alias `orun` y `owatch`.
- RAG con ranking híbrido (coseno + solapamiento léxico ponderado por IDF),
  citas `[n]` y modelo de respuesta `nemotron3:gpu` por defecto.
- Catálogo ampliado a 41 entradas: 25 comandos inline y 16 scripts.
- Backup de configuración ampliado con `bashrc`, completion y timer del gobernador.
- Creación de este repositorio con `README.md`, `INSTALL.md` y copias sanitizadas
  de la configuración.

## 2026-11-21

- CLI principal `ollama-cli.py` con `chat`, `act`, `bench`, `status`, `doctor`,
  `list`, `show`, `stop`, `import`, `rm`, `embed` y `config`.
- Configuración persistente para usar exclusivamente la iGPU Carrizo.
- Benchmarks y decisiones de modelo documentados en `ollama-kit.md`.
