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
cp ollama-scripts/*.sh ~/ollama-scripts/
chmod +x ~/ollama-cli.py ~/ollama-bench.sh ~/ollama-act.py ~/ollama-scripts/*.sh
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

Para otros modelos locales usa la propia CLI:

```bash
ocli import gemma3:gpu /ruta/Gemma-...-Q6_k.gguf --gpu 27
ocli import nemotron3:gpu /ruta/nemotron...gguf
```

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

Rinde ~2,5 tok/s. No se recomienda como modelo principal: `gemma3:q4` es ~4x más
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
```

Mide ~8,5 tok/s con 42/43 capas en GPU y ~1,92 GiB de VRAM (de ~2,0 GiB). No
modifica `override.conf` ni el servicio principal: la iGPU sigue siendo el
default del resto de modelos. Si el puerto 11435 está ocupado, el script aborta.
Ver detalles y comparativa en [`ollama-kit.md`](ollama-kit.md), sección 8.

## 8. Verificación final

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
bash ~/ollama-scripts/selftest.sh
odebug -n 5
```

`ocli doctor` debe terminar con **TODO CORRECTO**. El único aviso habitual es
"ningún modelo cargado" cuando no hay ninguna inferencia en curso.

Los datos generados por el usuario (sesiones, plantillas e historial) se guardan en
`~/.ollama-sessions/`, `~/.ollama-prompts.json` y `~/.ollama-bench-history.jsonl`.
`backup-configs.sh` los incluye; no se suben al repositorio.

## 9. Reversión

```bash
sudo rm -f /etc/systemd/system/ollama.service.d/override.conf
sudo rm -f /etc/systemd/system/cpu-performance.service
sudo rm -f /etc/systemd/system/cpu-performance-enforce.timer
sudo rm -f /etc/sysctl.d/99-ollama.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama

rm -f ~/ollama-cli.py ~/ollama-bench.sh ~/ollama-act.py \
      ~/ollama-commands.json ~/ollama-kit.md
rm -rf ~/ollama-scripts ~/.ollama-rag ~/.ollama-sessions
rm -f ~/.ollama-prompts.json ~/.ollama-bench-history.jsonl ~/.ollama-debug.log
```

Después elimina el bloque `>>> ollama-cli personal >>>` de `~/.bashrc` y recarga la
terminal. Si el gobernador queda en `powersave`, vuelve a ponerlo con
`sudo cpupower frequency-set -g performance`.
