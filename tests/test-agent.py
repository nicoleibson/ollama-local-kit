import atexit
import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

CLI = "/home/onion/ollama-cli.py"

for key in ("OLLAMA_KIT_PROVIDER", "OLLAMA_API_KEY", "NVIDIA_API_KEY",
            "GEMINI_API_KEY", "OLLAMA_HOST"):
    os.environ.pop(key, None)


def tempdir(prefix):
    """Crea el temporal de la prueba y lo borra al salir, aunque falle."""
    path = Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp/opencode"))
    atexit.register(shutil.rmtree, path, True)
    return path


spec = importlib.util.spec_from_file_location("cli_agent_probe", CLI)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

root = tempdir("agente-")
(root / "nota.txt").write_text("hola mundo\n", encoding="utf-8")

decisions = iter([
    {"thought": "Voy a leer", "tool": "read", "args": {"path": "nota.txt"}},
    {"thought": "Ya está", "tool": "answer", "answer": "El archivo dice hola"},
    {"thought": "Voy a escribir", "tool": "write",
     "args": {"path": "salida.txt", "content": "nuevo\n"}},
    {"thought": "Hecho", "tool": "answer", "answer": "Escrito"},
    {"thought": "Voy a editar", "tool": "edit",
     "args": {"path": "salida.txt", "old": "nuevo", "new": "editado"}},
    {"thought": "Hecho", "tool": "answer", "answer": "Editado"},
])


def fake_api(path, payload=None, **kwargs):
    d = next(decisions)
    return {"message": {"content": json.dumps(d)}, "eval_count": 1}


m.api = fake_api

users = iter(["lee nota.txt", "crea salida.txt", "edita salida.txt",
              "/undo", "/undo", "/exit"])


def fake_input(prompt=""):
    if prompt.startswith("¿Permitir"):
        return "s"
    return next(users)


m.input = fake_input
m.rag_load = lambda: {"chunks": [], "embedding_model": "qwen:gpu"}

args = m.build_parser().parse_args(["agent", "--root", str(root)])
out = io.StringIO()
with contextlib.redirect_stdout(out):
    m.cmd_agent(args)
text = out.getvalue()

assert "El archivo dice hola" in text, text
assert "Escrito" in text and "Editado" in text, text
assert "Restaurado" in text and "Eliminado el archivo creado" in text, text
assert (root / "nota.txt").read_text() == "hola mundo\n"
assert not (root / "salida.txt").exists(), "undo no eliminó el archivo creado"
print("OK bucle REPL: read -> answer, write, edit, undo x2")

# Herramientas de solo lectura y bloqueos.
assert "hola mundo" in m.agent_read(root, {"path": "nota.txt"})
assert "nota.txt" in m.agent_list(root, {})
assert "hola" in m.agent_grep(root, {"pattern": "mundo", "max": 5})


def expect_exit(fn, needle):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError("Se esperaba SystemExit: " + needle)


expect_exit(lambda: m.agent_read(root, {"path": "../outside"}), "Ruta no permitida")
expect_exit(lambda: m.agent_read(root, {"path": "/etc/passwd"}), "Ruta no permitida")
expect_exit(lambda: m.agent_bash(root, {"command": "rm -rf /"}, args), "peligroso")
expect_exit(lambda: m.agent_bash(root, {"command": "noexiste123"}, args), "no es un comando")

# Salvaguarda de JSON Schema en proveedores que no lo admiten.
os.environ["OLLAMA_KIT_PROVIDER"] = "nvidia"
os.environ["NVIDIA_API_KEY"] = "k"
spec2 = importlib.util.spec_from_file_location("cli_agent_nvidia", CLI)
m2 = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(m2)
assert m2.NO_SCHEMA is True
expect_exit(lambda: m2.cmd_agent(m.build_parser().parse_args(["agent"])), "JSON Schema")
m2.api = lambda path, payload=None, **kw: {
    "message": {"content": '```json\n{"tool":"answer","answer":"ok"}\n```'}
}
loose = m.build_parser().parse_args(["agent", "--no-schema"])
decision, _ = m2.agent_call([], loose)
assert decision["answer"] == "ok", decision
for key in ("OLLAMA_KIT_PROVIDER", "NVIDIA_API_KEY"):
    os.environ.pop(key, None)

print("Herramientas y salvaguardas: TODO OK")
