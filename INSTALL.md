# Instalación

Los archivos de `config/` son **copias de referencia**: no se aplican automáticamente.
Ejecuta estos pasos desde la raíz del repositorio. No se usa `systemctl edit ollama`;
todo se escribe con `cp`/`tee` directos.

## 1. Requisitos

- Linux con systemd, bash y Python 3.
- Ollama 0.33.2 o superior con soporte Vulkan.
- `sudo` para instalar los drop-ins y el temporizador.
- Opcional: `bash-completion`, `lm-sensors`, `vulkaninfo`, `cpupower`.

## 2. Herramientas en casa del usuario

```bash
mkdir -p ~/ollama-scripts ~/.local/share/bash-completion/completions
cp ollama-cli.py ollama-bench.sh ollama-act.py ollama-commands.json ollama-kit.md ~/
cp ollama-kit-gui.py ~/
cp ollama-scripts/*.sh ~/ollama-scripts/
chmod +x ~/ollama-cli.py ~/ollama-kit-gui.py ~/ollama-bench.sh ~/ollama-act.py ~/ollama-scripts/*.sh
cp completions/ollama-cli ~/.local/share/bash-completion/completions/ollama-cli
```

Añade los alias y el autocompletado a `~/.bashrc` (revisa antes si ya están):

```bash
cat config/bashrc-ollama-block.txt >> ~/.bashrc
source ~/.bashrc
```

El bloque también está en `config/bashrc-ollama-block.txt` por si quieres insertarlo
con un editor en vez de `cat >>`.

## 3. Configuración persistente de Ollama

```bash
sudo install -d -m 755 /etc/systemd/system/ollama.service.d
sudo install -m 644 config/ollama-override.conf \
  /etc/systemd/system/ollama.service.d/override.conf
```

Variables resultantes:

```text
OLLAMA_CONTEXT_LENGTH=2048
OLLAMA_NUM_PARALLEL=1
OLLAMA_MAX_LOADED_MODELS=1
OLLAMA_KEEP_ALIVE=5m
GGML_VK_VISIBLE_DEVICES=0
OLLAMA_KV_CACHE_TYPE=q8_0
OLLAMA_IGPU_ENABLE=1
OLLAMA_NO_CLOUD=true
```

`GGML_VK_VISIBLE_DEVICES=0` fija la iGPU Carrizo. Cambiarlo a `1` usaría la dGPU.

## 4. Caché del sistema

```bash
sudo install -m 644 config/99-ollama.conf /etc/sysctl.d/99-ollama.conf
sudo sysctl --system
```

## 5. Gobernador de CPU

`powerprofilesctl set performance` **no existe** en este equipo; se usa `cpupower`.

```bash
sudo install -m 644 config/cpu-performance.service /etc/systemd/system/cpu-performance.service
sudo install -m 644 config/cpu-performance-enforce.timer /etc/systemd/system/cpu-performance-enforce.timer

# Importante: quitar After= si aparece, porque crea un ciclo con multi-user.target.
sudo sed -i '/^After=power-profiles-daemon.service$/d' \
  /etc/systemd/system/cpu-performance.service

sudo systemctl daemon-reload
sudo systemctl enable --now cpu-performance.service
sudo systemctl enable --now cpu-performance-enforce.timer
sudo cpupower frequency-set -g performance
```

El temporizador reaplica `performance` cada minuto. Verificación:

```bash
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # performance
systemctl is-active cpu-performance-enforce.timer          # active
```

## 6. Reiniciar y comprobar la GPU

```bash
sudo systemctl restart ollama
journalctl -u ollama -n 200 --no-pager | grep -Ei 'using device|offloaded|no usable GPU'
```

Esperado:

```text
using device Vulkan0 (AMD Radeon R7 Graphics (RADV CARRIZO))
offloaded 27/27 layers to GPU
```

`ollama ps` debe mostrar el modelo con `100% GPU`.

## 7. Modelos

`config/Modelfile.gemma-q4` es la plantilla usada para `gemma3:q4`. Ajusta la ruta
del GGUF a tu equipo antes de importarlo:

```bash
# Edita la línea FROM del Modelfile si hace falta
ollama create gemma3:q4 -f config/Modelfile.gemma-q4
```

`gemma3:gpu` (Q6_K) es el modelo principal por exactitud; se crea importando el
GGUF `...Q6_k.gguf` con `ocli import gemma3:gpu /ruta/al/Q6_k.gguf --gpu 27`.
`gemma3:q4` se conserva como alternativa rápida.

Para otros modelos locales usa la propia CLI:

```bash
ocli import gemma3:gpu /ruta/Gemma-...-Q6_k.gguf --gpu 27
ocli import nemotron3:gpu /ruta/nemotron...gguf
```

`config/Modelfile.nemotron3-gpu` es la receta canónica para el respaldo de
razonamiento: apunta al GGUF de `~/Documentos/ModelsIA/` y fija `num_gpu 43`,
`num_ctx 2048` y `temperature 0.3`, igual que el modelo instalado. Permite
recrearlo sin depender del directorio de importación original.

`config/Modelfile.nemotron3-precise` es la receta del perfil de máxima precisión:
el mismo GGUF con `temperature 0`, `num_ctx 8192` y `num_gpu 43`. Se crea con
`ollama create nemotron3:precise -f config/Modelfile.nemotron3-precise` y se
activa con `--precise` o `OLLAMA_KIT_PRECISE=1` en `chat`, `act`, `run`, `agent`
y `rag ask`. `ocli config` indica si el perfil está activo.

`qwen:gpu` se usa solo para embeddings. No lo llames con `/api/chat` ni
`/api/generate`; usa `/api/embed` o `ocli embed` / `ocli rag`.

### Qwen3.5-4B (opcional)

El GGUF original se importa tal cual como modelo generativo. Necesita `num_gpu 33`
para cargar todas las capas en la iGPU:

```bash
# Ajusta la ruta FROM en config/Modelfile.qwen35-gpu si hace falta
ollama create qwen35:gpu -f config/Modelfile.qwen35-gpu
ollama ps    # debe mostrar 100% GPU
```

Para usarlo como embedder hay que parchear una **copia** del GGUF (nunca el original):

```bash
python3 tools/make_qwen35_embed.py 3 /ruta/Qwen3.5-4B-Embed-Q4_K_M.gguf
# Ajusta la ruta FROM en config/Modelfile.qwen35-embed
ollama create qwen35:embed -f config/Modelfile.qwen35-embed
ollama show qwen35:embed | grep embedding
```

El script añade `qwen35.pooling_type` y 61 bytes de relleno para conservar la
alineación a 32 bytes que exige llama.cpp. El Modelfile fija `num_ctx 512` porque
con 2048 se agota la VRAM compartida. No se recomienda como default de RAG: es ~5x
más lento y menos preciso que `qwen:gpu`.

### Gemma 4 E4B (opcional)

El GGUF de 4,2 GB no cabe en la iGPU Carrizo: con `num_gpu > 0` el backend Vulkan
falla con `Not enough memory for command submission`. Importarlo con el Modelfile
preparado, que fuerza CPU:

```bash
# Ajusta la ruta FROM en config/Modelfile.gemma4-cpu si hace falta
ollama create gemma4:cpu -f config/Modelfile.gemma4-cpu
ollama ps    # 100% CPU, 0.00 GiB VRAM
ocli bench gemma4:cpu --tokens 20
```

Rinde ~2,5 tok/s. No se recomienda como modelo principal: `gemma3:gpu` es ~3x más
rápido y usa la iGPU. Se conserva solo para probar la arquitectura `gemma4`.

### Gemma 4 en la R7 260X (opcional, aislada)

Sí cabe en la dedicada. El tag `gemma4:dgpu` fija `num_gpu 42` y `num_ctx 2048`,
pero la GPU la selecciona la variable de proceso `GGML_VK_VISIBLE_DEVICES=1`.
El script `ollama-scripts/gemma4-dgpu.sh` levanta una instancia temporal en el
puerto 11435 con esa variable, ejecuta la petición y apaga el proceso al terminar:

```bash
# Ajusta la ruta FROM en config/Modelfile.gemma4-dgpu si hace falta
ollama create gemma4:dgpu -f config/Modelfile.gemma4-dgpu
chmod +x ollama-scripts/gemma4-dgpu.sh     # o copiarlo a ~/ollama-scripts/
odgpu "tu pregunta"
odgpu --chat
odgpu --bench
odgpu --safe-fit "pregunta con texto largo"  # ctx 8192, sin num_gpu fijo
odgpu --print-env                             # plan sin arrancar nada
```

Para ctx 8192 usa `config/Modelfile.gemma4-dgpu-safe`: sin `PARAMETER num_gpu`,
así Ollama ajusta las capas a la VRAM libre en vez de abortar con 42 capas
fijas. `--safe-fit` crea el tag `gemma4:dgpu-safe` la primera vez y es más lento.
Mide ~8,5 tok/s con 42/43 capas en GPU y ~1,92 GiB de VRAM (de ~2,0 GiB). No
modifica `override.conf` ni el servicio principal: la iGPU sigue siendo el
default del resto de modelos. Si el puerto 11435 está ocupado, el script aborta.
Ver detalles y comparativa en [`ollama-kit.md`](ollama-kit.md), sección 8.

## 8. Ventana local (`ogui`)

`ogui` abre un panel Tkinter nativo con doce pestañas: Resumen, Modelos, Chat,
RAG, Acciones, Agente, Prompts, Bench, Sesiones, Flujos, Comandos y Registro. No
levanta servidor ni navegador; fuerza `OLLAMA_KIT_LOCAL=1` y
`OLLAMA_KIT_TASK_MODE=1`, y elimina variables de nube/API key en todos los
subprocess. Al abrirla no se lanza ningún modelo.

Para tenerla también en el menú de aplicaciones:

```bash
mkdir -p ~/.local/share/applications ~/.local/share/icons
cp config/ollama-kit.desktop ~/.local/share/applications/
cp assets/ollama-kit.svg ~/.local/share/icons/ollama-kit.svg
chmod +x ~/.local/share/applications/ollama-kit.desktop
desktop-file-validate ~/.local/share/applications/ollama-kit.desktop  # opcional
update-desktop-database ~/.local/share/applications 2>/dev/null || true
```

La pestaña RAG indexa archivos con `qwen:gpu` y pregunta con `gemma3:gpu` (marca
"Usar respaldo" para `nemotron3:gpu`). La pestaña Agente usa `agent --prompt`,
sin REPL y sin `--yes`/`--allow-dangerous`, y muestra el razonamiento de la
decisión; Prompts gestiona plantillas y Bench mide modelos locales. La pestaña
Acciones simula con `act --dry-run` y solo habilita "Ejecutar segura" si el
identificador está en la lista blanca de entradas de solo lectura; `run` ejecuta
directamente solo `chat` y `rag`, y `code commit` únicamente redacta el mensaje.
Detalles en [`ollama-kit.md`](ollama-kit.md#17-ventana-interactiva-ogui) y en la
sección 18 (perfil de tareas y pruebas).

## 9. Verificación final

```bash
ocli doctor
ocli list
ocli bench gemma3:q4 gemma3:gpu nemotron3:gpu
ocli act --list
ocli rag index ~/ollama-kit.md
ocli rag ask "cómo funciona el modo act" -k 3
ocli watch --count 3 --interval 1
ocli chat --session prueba --no-stream "di HOLA"
ocli session list
ocli prompt add resumen --text 'Resume: {{args}}'
ocli prompt use resumen "texto de prueba" --no-stream
ocli bench --history
ocli code review ~/ollama-cli.py --lines 20 --dry-run
ocode tests "def suma(a,b): return a+b" --no-stream
bash ~/ollama-scripts/selftest.sh             # debe terminar con SELFTEST OK
odebug -n 5
ocli config                                   # MODO: LOCAL, PERFIL: TAREAS, razonamiento visible
oagent --help                                 # debe mostrar --prompt y --reasoning
oagent --prompt "di HOLA"                     # tarea única, sin REPL
ocli chat -m nemotron3:gpu --no-stream "di HOLA"  # debe imprimir RAZONAMIENTO:
OLLAMA_KIT_PRECISE=1 ocli config               # debe mostrar PERFIL DE PRECISIÓN
ocli chat --precise --no-stream "di HOLA"      # nemotron3:precise (lento), razonamiento visible
ofree config                                  # debe decir GRATIS: solo modelos locales
oflow --help                                  # debe mostrar map/slide/classify/ask
python3 -m py_compile ~/ollama-kit-gui.py      # sintaxis de la ventana Tkinter
ogui                                          # 12 pestañas; marca "Máxima precisión": el estado dice "perfil preciso"
                                              # revisa RAG, Agente y Acciones sin ejecutar nada
for f in ~/ollama-local-kit/tests/test-*.py; do python3 "$f"; done
ofree flow map /home/onion/ollama-kit.md --prompt 'Resume' --out prueba --limit 1
cat ~/.ollama-flow/prueba/report.md           # estado en disco, reanudable
OLLAMA_KIT_PROVIDER=nvidia ollama-cli.py config | grep 'Modelo por defecto'  # opcional
# oagent --root /tmp                          # prueba en carpeta desechable
```

Opcional, solo si tienes cuenta en ollama.com con créditos:

```bash
printf 'TU_API_KEY' > ~/.ollama-cloud-key
chmod 600 ~/.ollama-cloud-key
ocloud config                                 # debe decir MODO: CLOUD
ocloud list                                   # catálogo cloud
# ocloud chat -m glm-5.3 "hola"               # consume créditos
```

También puedes añadir proveedores OpenAI-compatible con sus propias keys:

```bash
printf 'TU_KEY_DE_NVIDIA' > ~/.nvidia-api-key; chmod 600 ~/.nvidia-api-key
printf 'TU_KEY_DE_GEMINI' > ~/.gemini-api-key; chmod 600 ~/.gemini-api-key
onvidia config                                # PROVEEDOR: nvidia
ogemini config                                # PROVEEDOR: gemini
# onvidia chat -m nvidia/nemotron-3-ultra-550b-a55b "hola"   # consume créditos
# ogemini chat -m gemini-4-hydrogen "hola"                   # consume créditos
```

`ocli doctor` debe terminar con **TODO CORRECTO**. El único aviso habitual es
"ningún modelo cargado" cuando no hay ninguna inferencia en curso.

`selftest.sh` comprueba sintaxis, catálogo, índice RAG, `doctor` y las diez
pruebas de `~/ollama-local-kit/tests/` (sin inferencias). Si el repo está en otra
ruta, indícala con `OLLAMA_KIT_TESTS=/ruta/tests bash ~/ollama-scripts/selftest.sh`.

Los datos generados por el usuario (sesiones, plantillas e historial) se guardan en
`~/.ollama-sessions/`, `~/.ollama-prompts.json` y `~/.ollama-bench-history.jsonl`.
`backup-configs.sh` los incluye; no se suben al repositorio.

## 10. Reversión

```bash
sudo rm -f /etc/systemd/system/ollama.service.d/override.conf
sudo rm -f /etc/systemd/system/cpu-performance.service
sudo rm -f /etc/systemd/system/cpu-performance-enforce.timer
sudo rm -f /etc/sysctl.d/99-ollama.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama

rm -f ~/ollama-cli.py ~/ollama-bench.sh ~/ollama-act.py \
      ~/ollama-commands.json ~/ollama-kit.md
rm -rf ~/ollama-scripts ~/.ollama-rag ~/.ollama-sessions ~/.ollama-agent-undo
rm -f ~/.ollama-prompts.json ~/.ollama-bench-history.jsonl ~/.ollama-debug.log
rm -f ~/.ollama-cloud-key ~/.nvidia-api-key ~/.gemini-api-key
```

Después elimina el bloque `>>> ollama-cli personal >>>` de `~/.bashrc` y recarga la
terminal. Si el gobernador queda en `powersave`, vuelve a ponerlo con
`sudo cpupower frequency-set -g performance`.
