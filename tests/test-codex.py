import json
import os
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "ollama-scripts" / "codex-local.sh"
INSTALLED = Path("/home/onion/ollama-scripts/codex-local.sh")
MODELFILE = REPO / "config" / "Modelfile.nemotron3-codex"
CODEX_DIR = Path.home() / ".codex"
PROFILE = CODEX_DIR / "ollama-launch.config.toml"
CONFIG = CODEX_DIR / "config.toml"
CATALOG = CODEX_DIR / "model.json"

assert SCRIPT.exists() and os.access(SCRIPT, os.X_OK), SCRIPT
assert INSTALLED.exists() and SCRIPT.read_bytes() == INSTALLED.read_bytes()

modelfile = MODELFILE.read_text(encoding="utf-8")
assert "FROM nemotron3:gpu" in modelfile
assert "PARAMETER num_ctx 65536" in modelfile
assert ":dgpu" not in modelfile

script_text = SCRIPT.read_text(encoding="utf-8")
# Codex 0.169.0 fuerza modo embebido con --profile; el wrapper no debe pasarlo.
assert "codex --profile" not in script_text, script_text
assert "exec --profile" not in script_text, script_text
assert "exec codex \"$@\"" in script_text, script_text

profile = PROFILE.read_text(encoding="utf-8")
assert 'model = "nemotron3:codex"' in profile, profile
assert 'base_url = "http://127.0.0.1:11434/v1/"' in profile, profile
assert ":dgpu" not in profile

# El perfil generado se fusiona en config.toml para permitir el daemon compartido.
merged = CONFIG.read_text(encoding="utf-8")
parsed = tomllib.loads(merged)
assert parsed["model"] == "nemotron3:codex", parsed
assert parsed["model_provider"] == "ollama-launch", parsed
assert parsed["model_catalog_json"] == str(CATALOG), parsed
assert parsed["model_providers"]["ollama-launch"]["base_url"] == \
    "http://127.0.0.1:11434/v1/", parsed
assert "# >>> ocodex root >>>" in merged and "# <<< ocodex root <<<" in merged
assert "# >>> ocodex sections >>>" in merged and "# <<< ocodex sections <<<" in merged
assert ":dgpu" not in merged

catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
slugs = [item["slug"] for item in catalog["models"]]
assert "nemotron3:codex" in slugs, slugs
assert all(":dgpu" not in slug for slug in slugs), slugs
for item in catalog["models"]:
    if item["slug"] == "nemotron3:codex":
        assert item["context_window"] == 65536, item

bashrc_block = (REPO / "config" / "bashrc-ollama-block.txt").read_text(encoding="utf-8")
assert "alias ocodex=" in bashrc_block

# La guardia se ejecuta antes de tocar la configuración.
env = dict(os.environ, CODEX_MODEL="gemma4:dgpu")
result = subprocess.run([str(SCRIPT), "--config"], env=env,
                        capture_output=True, text=True, timeout=30)
assert result.returncode != 0, result
assert "R7 260X" in result.stderr, result.stderr

# La fusión es idempotente y deja un TOML válido tras repetirla. Se comparan
# dos ejecuciones seguidas: la primera puede incorporar cambios del perfil
# generado que estuvieran pendientes desde una sesión interactiva anterior.
previous = None
for _ in range(3):
    subprocess.run([str(SCRIPT), "--config"], capture_output=True, text=True,
                   timeout=60)
    current = CONFIG.read_bytes()
    if previous == current:
        break
    previous = current
else:
    raise AssertionError("ocodex --config no es idempotente")
tomllib.loads(current.decode("utf-8"))

print("CODEX LOCAL: TODO OK")
