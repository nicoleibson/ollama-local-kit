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

## 8. Verificación final

```bash
ocli doctor
ocli list
ocli bench gemma3:q4 gemma3:gpu nemotron3:gpu
ocli act --list
ocli rag index ~/ollama-kit.md
ocli rag ask "cómo funciona el modo act" -k 3
ocli watch --count 3 --interval 1
```

`ocli doctor` debe terminar con **TODO CORRECTO**. El único aviso habitual es
"ningún modelo cargado" cuando no hay ninguna inferencia en curso.

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
rm -rf ~/ollama-scripts ~/.ollama-rag
```

Después elimina el bloque `>>> ollama-cli personal >>>` de `~/.bashrc` y recarga la
terminal. Si el gobernador queda en `powersave`, vuelve a ponerlo con
`sudo cpupower frequency-set -g performance`.
