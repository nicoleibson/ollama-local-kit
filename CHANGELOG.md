# Changelog

## 2026-11-23 (trigésima primera tanda)

- Nueva opción **`orag ask --fast`**: usa `qwen3:06b` con 300 tokens, mantiene
  `qwen:gpu` como embedding y `--min-score 0.10`. Encapsula la RAG rápida medida
  el 23/11 (~20 tok/s frente a 7,4 de Gemma 3) sin cambiar el default.
  `--fast` no se combina con `--precise` ni `--reasoning`; un `-m` explícito
  manda. Verificado con `¿Qué hace odgpu --safe-fit?`: respuesta correcta con 97
  tokens en 5,2 s. Aserciones nuevas en `tests/test-precision.py`; 10/10 pruebas.

## 2026-11-23 (trigésima tanda)

- Corregidas las referencias que aún atribuían RAG a `nemotron3:gpu`: la tabla de
  «mejor forma de usar cada modelo», la sección de funcionamiento de RAG y la
  tabla de modelos por tarea. El default del perfil tareas es `gemma3:gpu`;
  `nemotron3:gpu` se usa con `--reasoning` (o `OLLAMA_KIT_TASK_MODE=0`) y
  `qwen3:06b` queda como opción rápida con `-m`. README, manual y copia
  instalada sincronizados.
- Añadida la precedencia explícita del modelo en RAG (`-m` > `--precise` >
  `--reasoning` > perfil tareas). Una prueba de humo mostró que preguntar a RAG
  por su propio default mezcla secciones; se recomienda `ocli config` para eso.

## 2026-11-23 (vigésima novena tanda)

- `ocli doctor` sin incidencias (servicio activo, iGPU Carrizo, sin nube,
  gobernador `performance`). Aviso esperado: ningún modelo cargado al medir.
- **Índice RAG final reindexado** con la documentación definitiva: 716 fragmentos
  en 52 archivos. Copia del índice anterior en
  `/tmp/opencode/index-before-final-reindex.json`.
- Prueba de humo sobre los resultados nuevos: `gemma3:gpu` respondió 17/20 de forma
  concisa y con tres citas correctas; `qwen3:06b` mezcló el resultado antiguo de
  6 preguntas con el nuevo de 20. Refuerza mantener `gemma3:gpu` como default.
  Sin APIs externas ni créditos.

## 2026-11-23 (vigésima octava tanda)

- Integración GUI/CLI tras el cambio de timeout: `gui_timeout()` eleva de 300 a
  900 s el límite de los subprocess cuando está activa la casilla global
  **Máxima precisión**, en las seis llamadas que podían cortar inferencias
  largas. Aserciones en `tests/test-gui.py`.
- `ocli list` describe ya los tags nuevos, `nemotron3:precise`,
  `gemma4:dgpu-safe` y los seis tags `ctx4k`/`ctx8k` conservados, en lugar de
  mostrarlos como `(otro)`. Como esta tabla alimenta los desplegables de la GUI,
  todos los modelos aparecen ahora con su uso. Aserciones en
  `tests/test-task.py`.

## 2026-11-23 (vigésima séptima tanda)

- **Sonda factual ampliada**: 20 preguntas × 5 modelos con salida JSON. Resultados:
  `gemma3:gpu` 17/20, `gemma3:q4` 16/20, `bonsai:q1` 14/20, `lfm25:350m` 12/20 y
  `qwen3:06b` 10/20. Las 100 respuestas fueron JSON válido. Gemma 3 sigue siendo
  el más fiable; Qwen3 0.6B queda confirmado como modelo de grounding, no de
  conocimiento libre.
- **Comparativa RAG `--precise` vs `qwen3:06b`** sobre las mismas tres preguntas:
  Nemotron3 acertó embeddings y `safe-fit` (268 y 804 tokens, 3,4 tok/s); Qwen3
  solo `safe-fit` (80–116 tokens, ~20 tok/s). La pregunta sobre el perfil preciso
  agotó 1024 tokens en 304 s y cortó la respuesta final; con
  `-m nemotron3:precise --max-tokens 600` respondió completa en 21 s.
- **Fix**: `PRECISE_API_TIMEOUT = 900` para RAG en perfil preciso (antes 300 s).
  Verificado en inferencia real: la pregunta que daba `ERROR: timed out` terminó
  en 304 s sin error. Nueva aserción en `tests/test-precision.py`. Defaults y
  modelos principales sin cambios.

## 2026-11-23 (vigésima sexta tanda)

- **Índice RAG reindexado** con `qwen:gpu` (mismo embedder y parámetros):
  697 fragmentos, 52 archivos. El índice anterior estaba 15 horas desactualizado;
  copia de seguridad en `/tmp/opencode/index-before-reindex.json`.
- **Comparativa RAG con índice fresco** (`-k 4 --ctx 2048 --max-tokens 300`) entre
  `gemma3:gpu` y `qwen3:06b`: ambos responden con las mismas fuentes, pero Qwen3
  es más conciso y ~3× más rápido (20,0 tok/s vs 7,3–7,5). Acierto claro de
  Qwen3 en `¿Qué hace odgpu --safe-fit?`: ctx 8192, capas ajustadas y 2,64 vs
  8,54 tok/s; Gemma 3 dio una definición genérica de `odgpu`. En las otras dos
  preguntas **ambos cometieron errores** (atribuir el embedding por defecto a
  `qwen35:embed` o mezclar los modelos del perfil preciso). Conclusión: no cambiar
  el default; `qwen3:06b` queda como sintetizador opcional de RAG rápida
  (`orag ask -m qwen3:06b`) y `--precise` sigue siendo la vía para respuestas
  críticas.
- **Sonda factual local** de 6 preguntas con JSON: `bonsai:q1`, `gemma3:q4`,
  `gemma3:gpu` y `lfm25:350m` obtuvieron 5/6; `qwen3:06b` 2/6 (Madrid, 31,
  Venus y Don Quijote mal). Normalizados acentos y subíndices. Refuerza usar
  Qwen3 0.6B solo con contexto recuperado, nunca como fuente de conocimiento
  libre. Sin APIs externas ni créditos.

## 2026-11-22 (vigésima quinta tanda)

- **Cuatro GGUF nuevos importados y medidos** desde `~/Descargas/models`:
  `bonsai:q1` (Qwen3 1.7B Q1_0), `gemma3:1b` (Gemma 3 1B-it Q4_K_M),
  `lfm25:350m` (LFM2.5 350M Q4_K_M) y `qwen3:06b` (Qwen3 0.6B Q4_K_M). Tags
  nuevos, sin sobrescribir los existentes; `--no-save` en el bench.
- Benchmark local con `ocli bench --tokens 80 --ctx 2048`, tres pasadas y
  promedio: `lfm25:350m` 43,34 tok/s, `bonsai:q1` 25,06, `qwen3:06b` 25,21,
  `gemma3:1b` 9,11 y referencia `gemma3:q4` 8,86.
- Comprobación funcional: los cuatro responden en español y cumplen un JSON
  Schema simple; `act --dry-run` con `qwen3:06b` y `bonsai:q1` elige `date`.
- `thinking_for()` envía ahora `think:false` a los tags `qwen3:*`: Qwen3 pensaba
  por su cuenta y agotaba el presupuesto de salida en respuestas cortas. No
  afecta a Gemma 3 ni a `qwen35`. Nueva aserción en `tests/test-task.py`; 10/10
  pruebas, sin APIs externas ni créditos.

## 2026-11-22 (vigésima cuarta tanda)

- **Prueba real de `odgpu --safe-fit`** (autorizada por el usuario): creó
  `gemma4:dgpu-safe`, arrancó la instancia temporal en 11435 con ctx 8192 y
  respondió a `di HOLA`. El log emitió `cannot meet free memory target`, pero no
  abortó: Ollama situó 9/43 capas en GPU y completó la respuesta. Midiendo la
  evaluación total (incluido el razonamiento), 2,64 tok/s frente a 8,54 del modo
  de 42 capas con ctx 2048.
- Comprobada la limpieza posterior: puerto 11435 cerrado, PIDFILE limpio y
  servicio principal en 11434 respondiendo. Sin APIs externas ni créditos.
- Corregida la documentación: el aviso de memoria sí aparece, es informativo;
  la nota sobre `think:false` en Gemma4 estaba desactualizada. README, INSTALL,
  manual y copia instalada sincronizados.

## 2026-11-22 (vigésima tercera tanda)

- `odgpu` añade `--safe-fit`: usa el nuevo tag `gemma4:dgpu-safe` (ctx 8192,
  receta `config/Modelfile.gemma4-dgpu-safe` sin `num_gpu` fijo) para que Ollama
  ajuste las capas a la VRAM libre de la R7 260X y no aborte con 42 capas fijas.
  La primera llamada crea el tag; la velocidad baja, la precisión de contexto
  largo se conserva.
- Nuevo `--print-env`: muestra modelo, contexto, puerto y receta sin arrancar la
  instancia temporal. `--ctx` se pasa ahora también a `chat` y `bench`.
- Verificado inicialmente con `bash -n` y `--print-env`; la prueba real se
  registró en la tanda siguiente. Documentado en README, INSTALL y manual
  (sección 8).

## 2026-11-22 (vigésima segunda tanda)

- **JSON Schema con reintento**: `act`, las decisiones del agente y el
  clasificador de flujos validan la salida contra el esquema (`schema_valid`) y,
  si no cumple, reenvían una vez con el prompt recortado, una instrucción más
  estricta y más margen de salida dentro del contexto. El razonamiento de ambos
  intentos se muestra. Se sigue usando `format`/JSON Schema, no tool calling.
- `ogui` añade una casilla global **Máxima precisión**: exporta
  `OLLAMA_KIT_PRECISE=1` a todos los subprocess, así que alcanza a las doce
  pestañas sin tocar cada comando. La cabecera y el estado indican el perfil
  activo. No relaja confirmaciones ni la lista blanca.
- Nueva `tests/test-schema.py`; diez pruebas versionadas, `SELFTEST OK`, copias
  instaladas sincronizadas. Sin APIs de pago.

## 2026-11-22 (vigésima primera tanda)

- **Perfil de precisión opt-in**: `--precise` o `OLLAMA_KIT_PRECISE=1` en `chat`,
  `act`, `run`, `agent` y `rag ask` usan `nemotron3:precise` con temperatura 0,
  `num_ctx 8192`, `num_predict 1024` y `think`; el razonamiento sigue visible.
  Un `-m` explícito se respeta. `ocli config` informa del perfil activo.
- Corregida la delegación de `run`: `--precise` llega ahora a `act`, RAG y
  `chat` (modelo, temp 0, ctx 8192, 1024 tokens y `min-score 0.10`), y el
  enrutador usa también `nemotron3:precise`. `act` respeta el contexto y la
  salida del perfil en lugar de sus valores fijos.
- Nueva receta `config/Modelfile.nemotron3-precise` (mismo GGUF que el respaldo,
  con contexto amplio y salida determinista); modelo creado y verificado con
  `ollama show`.
- **RAG anti-desbordamiento**: `--min-score` pasa a 0.10 por defecto; el contexto
  se recorta con un presupuesto calculado desde `ctx`, `max_tokens`, system,
  pregunta, reserva y razonamiento (3 caracteres/token), de modo que el prompt no
  se trunque. La cabecera muestra fragmentos usados/totales, umbral y
  presupuesto; sin fragmentos se responde `No lo encuentro en el índice`.
- El prompt preciso pide una sola pasada de razonamiento, sin repetirse, y exige
  citas `[n]` y aviso de contradicciones en RAG.
- Nueva `tests/test-precision.py` (nueve pruebas versionadas en total) y
  documentación en README, INSTALL y manual. Verificado con inferencia local:
  620 tokens en 201 s (3,08 tok/s) y respuesta final con cita correcta. Sin APIs
  de pago.

## 2026-11-22 (vigésima tanda)

- Endurecido `ollama-scripts/gemma4-dgpu.sh`: `trap` separado para `EXIT`, `INT`
  (`130`) y `TERM` (`143`), aviso y limpieza de un PIDFILE huérfano al arrancar,
  y mensaje específico cuando el servicio principal de 11434 no responde. El
  uso normal de `odgpu` no cambia.
- Alineado el manual con el script: el log de la instancia temporal se escribe en
  `${TMPDIR:-/tmp}/ollama-gemma4-dgpu.log`.
- Corregido `ollama-cli.py bench`: ya no fuerza `num_ctx 2048` en cada petición,
  sino que respeta el del modelo. Nueva opción `--ctx N` para forzar otro valor
  solo cuando se pida explícitamente.

## 2026-11-22 (decimonovena tanda)

- Añadido `config/Modelfile.nemotron3-gpu`: ruta canónica en
  `~/Documentos/ModelsIA/`, `num_gpu 43`, `num_ctx 2048` y `temperature 0.3`,
  igual que el modelo instalado. Documentado en `INSTALL.md`.
- Ordenados los GGUF del equipo: `InstallApps/IALocal/` contenía copias de
  `gemma3:gpu` y `qwen:gpu` y el único GGUF suelto de Nemotron. Tras copiar y
  verificar este último por hash en `ModelsIA`, se borró `IALocal/` (~4,1 GiB)
  sin afectar a los modelos de Ollama.

## 2026-11-22 (decimoctava tanda)

- **Perfil de tareas por defecto**: `gemma3:gpu` atiende `chat`, `act`, `run`,
  agente, RAG y flujos con instrucciones cortas y verificables. `nemotron3:gpu`
  queda como respaldo de razonamiento vía `--reasoning` o
  `OLLAMA_KIT_TASK_MODE=0`; `qwen:gpu` sigue en embeddings. `ocli config` muestra
  el perfil y los modelos por rol.
- **Razonamiento siempre visible**: toda respuesta imprime `RAZONAMIENTO:` y el
  texto `thinking` cuando existe, o `(el modelo no devolvió texto de
  razonamiento)` si el modelo no lo devuelve. El parámetro `think` solo se envía
  a modelos con capacidad `thinking`: Gemma 3 rechazaba `think=true` con HTTP
  400. `--think` queda como bandera histórica compatible.
- `agent --prompt` ejecuta **una sola tarea y sale** (sin REPL) e imprime
  `RAZONAMIENTO AGENTE:`. En perfil tareas usa 4 pasos y `gemma3:gpu` por defecto.
- `ogui` pasa a **doce pestañas**: nuevas **Agente** (tarea única, sin `--yes` ni
  `--allow-dangerous`), **Prompts** (biblioteca de plantillas) y **Bench**
  (mediciones locales con gráfico de barras). Fuerza también
  `OLLAMA_KIT_TASK_MODE=1` en los subprocess.
- Pruebas versionadas en `tests/`: ocho archivos sin inferencias que simulan API,
  subprocess y Tkinter, incluida `test-task.py` (perfil, defaults, razonamiento
  visible y agente de una sola petición). `ollama-scripts/selftest.sh` pasa a 6
  pasos y ejecuta todo el conjunto desde el repo.
- Corregido el error real `HTTP 400: "gemma3:gpu" does not support thinking`
  detectado al probar el perfil con inferencia local. Actualizados manual
  (secciones 17 y 18 nueva), README, INSTALL e índice RAG. No se usaron APIs de
  pago.

## 2026-11-22 (decimoséptima tanda)

- La ventana `ogui` pasa a nueve pestañas: nuevas **RAG** (añadir
  archivos/carpetas, indexar con `qwen:gpu`, preguntar con `orag ask`, ver y
  limpiar el índice) y **Acciones** (subpestañas Act/Run/Code).
- La pestaña Acciones simula con `act --dry-run`, extrae el identificador y solo
  habilita "Ejecutar segura" para entradas de la lista blanca de solo lectura;
  nunca expone `--free`, `--yes` ni `--allow-dangerous`. `run` solo ejecuta
  directamente `chat`/`rag`; `act` y `automático` quedan en simulación, y `code
  commit` solo redacta el mensaje.
- Lanzador de escritorio `config/ollama-kit.desktop` e icono
  `assets/ollama-kit.svg`, validados con `desktop-file-validate` e instalados en
  `~/.local/share/applications` y `~/.local/share/icons`.
- `/tmp/opencode/test-gui.py` ampliado: simula envío de chat, pregunta RAG,
  cancelación/aceptación de comandos, borrado de sesión, simulación y ejecución
  segura de `act`, bloqueo de `run` automático y preparación de `code`. Pasa sin
  inferencias ni APIs de pago.
- Actualizados manual (sección 17), README, INSTALL e índice RAG.

## 2026-11-22 (decimosexta tanda)

- Nuevo `ollama-kit-gui.py` (alias `ogui`): ventana Tkinter nativa con siete
  pestañas —Resumen, Modelos, Chat, Sesiones, Flujos, Comandos y Registro—. No
  levanta servidor ni navegador y no necesita dependencias nuevas.
- La GUI fuerza `OLLAMA_KIT_LOCAL=1` en todos los subprocess y elimina
  `OLLAMA_HOST`, `OLLAMA_API_KEY`, `NVIDIA_API_KEY`, `GEMINI_API_KEY` y
  `OLLAMA_KIT_PROVIDER`, para que no se usen keys guardadas ni créditos.
- Permite ver `status`/`doctor`/`config`/`rag list`/`bench --history`/`debug`,
  ejecutar una lista blanca de comandos de solo lectura del catálogo y chatear
  con `chat --no-stream` en un hilo, con selector de modelo, sesión opcional,
  `system` y `stats`.
- Acciones que escriben o liberan recursos (descargar modelos, borrar sesiones,
  limpiar el log, ejecutar flujos) muestran el comando exacto y piden
  confirmación. Al abrir la ventana no se lanza ninguna inferencia.
- Añadidos alias en `~/.bashrc`, bloque instalable y autocompletado; `ocli config`
  muestra `ogui`. Documentado en el manual (sección 17), README e INSTALL.
- Prueba nueva sin inferencias: `/tmp/opencode/test-gui.py` valida entorno, argv,
  parseo, catálogo seguro e instancia Tkinter oculta. No se lanzaron modelos.

## 2026-11-22 (decimoquinta tanda)

- `gemma3:gpu` (Q6_K, uncensored) pasa a ser el modelo principal en `chat`/`ask`,
  `flow`/`oflow`, catálogo corto de `act`, enrutador `run`, `code tests` y
  `code explain`. Misma base y fine-tune que `gemma3:q4`, pero Q6_K conserva
  mejor los pesos; el coste es bajar de 9,3 a 7,8 tok/s.
- Se mantienen `nemotron3:gpu` en razonamiento, agente, RAG, enrutador, `code
  review` y `commit`; `qwen:gpu` en embeddings; Gemma 4 en uso puntual. Los
  mapas de `ofree` (`beam`, `clef:27b`) apuntan al nuevo principal.
- `ocli config` muestra ahora los defaults por rol. Actualizados manual, README,
  INSTALL, `ollama-act.py` y los tests de defaults/sesiones/flujos.
- No se importaron GGUF nuevos ni se lanzaron inferencias. La comparativa
  descartó cambiar Nemotron: ningún modelo de la carpeta lo supera con datos
  medidos, y `qwen35:embed` fue peor en recuperación (4/6 frente a 5/6).

## 2026-11-22 (decimocuarta tanda)

- Al cerrar `chat --session` (interactivo o de una sola pregunta) o el REPL
  `agent`, la CLI imprime el comando exacto para continuar: `Para continuar: ...`.
  Incluye el prefijo del proveedor (`ocli`, `ocloud`, `onvidia`, `ogemini`,
  `ofree`), el `-m` usado y, en el agente, `--root` y `--no-schema`.
- Ese comando se guarda también en `~/.ollama-sessions/last.txt`. Nuevas
  acciones `osess last` (última sesión) y `osess resume NOMBRE` (comando de una
  sesión concreta); `osess list` muestra cómo acceder a ellas.
- `chat --auto-session` crea un nombre `auto-AAAAMMDD-HHMMSS` en modo
  interactivo cuando no se da `--session`. Sin sesión, se avisa de que la
  conversación no se guardó.
- Corregido un fallo preexistente: `osess clear --all` sin nombre caía en la
  rama de listado y no borraba nada. Ahora borra también `last.txt` cuando
  procede.
- Metadatos nuevos en los archivos de sesión (`kind`, `root`, `no_schema`) para
  reconstruir el comando del agente. Verificado sin inferencias:
  `/tmp/opencode/test-session.py`.

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
