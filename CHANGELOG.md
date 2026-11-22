# Changelog

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
