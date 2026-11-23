import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "ollama-scripts" / "codex-local.sh"
INSTALLED = Path("/home/onion/ollama-scripts/codex-local.sh")
MODELFILE = REPO / "config" / "Modelfile.nemotron3-codex"
PROFILE = Path.home() / ".codex" / "ollama-launch.config.toml"
CATALOG = Path.home() / ".codex" / "model.json"

assert SCRIPT.exists() and os.access(SCRIPT, os.X_OK), SCRIPT
assert INSTALLED.exists() and SCRIPT.read_bytes() == INSTALLED.read_bytes()

modelfile = MODELFILE.read_text(encoding="utf-8")
assert "FROM nemotron3:gpu" in modelfile
assert "PARAMETER num_ctx 65536" in modelfile
assert ":dgpu" not in modelfile

profile = PROFILE.read_text(encoding="utf-8")
assert 'model = "nemotron3:codex"' in profile, profile
assert 'base_url = "http://127.0.0.1:11434/v1/"' in profile, profile
assert ":dgpu" not in profile

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

print("CODEX LOCAL: TODO OK")
