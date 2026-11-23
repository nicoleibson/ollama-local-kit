#!/usr/bin/env python3
"""Índice RAG: huellas por archivo y aviso de desfase, sin inferencias."""
import atexit
import contextlib
import importlib.util
import io
import os
import shutil
import tempfile
from pathlib import Path

CLI = Path(__file__).resolve().parents[1] / "ollama-cli.py"
BASE_ENV = ("OLLAMA_KIT_PROVIDER", "OLLAMA_KIT_LOCAL", "OLLAMA_KIT_TASK_MODE",
            "OLLAMA_KIT_PRECISE", "OLLAMA_API_KEY", "NVIDIA_API_KEY",
            "GEMINI_API_KEY", "OLLAMA_HOST", "OLLAMA_RAG_DIR")


def tempdir(prefix):
    path = Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp/opencode"))
    atexit.register(shutil.rmtree, path, True)
    return path


def load():
    for key in BASE_ENV:
        os.environ.pop(key, None)
    os.environ["OLLAMA_KIT_LOCAL"] = "1"
    spec = importlib.util.spec_from_file_location("cli_index_probe", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load()
root = tempdir("ragtest-")
mod.RAG_DIR = root
mod.RAG_INDEX = root / "index.json"
# Embeddings falsos: el índice no necesita el servicio para probar el desfase.
mod.rag_embed = lambda texts, model: [[1.0, 0.0, 0.0] for _ in texts]

doc = root / "doc.md"
doc.write_text("frase uno\n\nfrase dos\n", encoding="utf-8")

args = mod.build_parser().parse_args(
    ["rag", "index", str(doc), "--embed-model", "fake:emb"])
with contextlib.redirect_stdout(io.StringIO()):
    mod.cmd_rag_index(args)

data = mod.rag_load()
assert len(data["chunks"]) >= 1, data["chunks"]
meta = data["files"][str(doc)]
assert meta["size"] == len(doc.read_bytes()), meta
assert mod.rag_stale(data) == ([], []), mod.rag_stale(data)

# Índice al día: el resumen no avisa.
out = io.StringIO()
with contextlib.redirect_stdout(out):
    mod.rag_summary(data)
assert "aviso" not in out.getvalue(), out.getvalue()

# Contenido distinto y mismo tamaño: lo detecta el hash, no el tamaño.
doc.write_text("otra frase\n", encoding="utf-8")
changed, missing = mod.rag_stale(data)
assert changed == [str(doc)] and missing == [], (changed, missing)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    mod.rag_summary(data)
assert "aviso" in out.getvalue() and "cambió" in out.getvalue(), out.getvalue()

# Archivo borrado: se informa como falta, no como cambio.
doc.unlink()
changed, missing = mod.rag_stale(data)
assert changed == [] and missing == [str(doc)], (changed, missing)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    mod.rag_summary(data)
assert "falta" in out.getvalue(), out.getvalue()

# Índice antiguo sin huellas: no inventa avisos.
legacy = {"chunks": [{"path": "/x", "start": 0, "text": "t", "vector": [0.0]}],
          "embedding_model": "fake:emb", "dim": 1}
assert mod.rag_stale(legacy) == ([], []), mod.rag_stale(legacy)
out = io.StringIO()
with contextlib.redirect_stdout(out):
    mod.rag_summary(legacy)
assert "aviso" not in out.getvalue(), out.getvalue()

# Reindexar limpia el aviso y respeta archivos de otras rutas del índice.
doc.write_text("nueva versión\n", encoding="utf-8")
args = mod.build_parser().parse_args(
    ["rag", "index", str(doc), "--embed-model", "fake:emb"])
with contextlib.redirect_stdout(io.StringIO()):
    mod.cmd_rag_index(args)
data = mod.rag_load()
assert mod.rag_stale(data) == ([], []), mod.rag_stale(data)

print("ÍNDICE RAG Y DESFASE: TODO OK")
