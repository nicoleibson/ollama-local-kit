# Changelog

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
