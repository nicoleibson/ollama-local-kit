import atexit
import contextlib
import importlib.util
import io
import os
import shutil
import tempfile
from pathlib import Path

CLI = "/home/onion/ollama-cli.py"
ENV_KEYS = ("OLLAMA_KIT_PROVIDER", "OLLAMA_KIT_LOCAL", "OLLAMA_API_KEY",
            "NVIDIA_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST")


def load(**env):
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(
        "cli_session_probe_" + str(abs(hash(tuple(env.items()))))[:6], CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SESSIONS_DIR = tempdir("osesion-")
    return mod


def tempdir(prefix):
    """Crea el temporal de la prueba y lo borra al salir, aunque falle."""
    path = Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp/opencode"))
    atexit.register(shutil.rmtree, path, True)
    return path


def capture(fn, *a):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(*a)
    return out.getvalue()


m = load()
root = tempdir("osesion-root-")


# --- chat interactivo: imprime el comando al cerrar -------------------------
def fake_stream(path, payload):
    return iter([{"message": {"content": "RESPUESTA"}}])


m.api_stream = fake_stream
lines = iter(["hola", "salir"])
m.input = lambda prompt="": next(lines)

args = m.build_parser().parse_args(["chat", "--session", "proyecto",
                                    "-m", "gemma3:gpu"])
out = capture(m.cmd_chat, args)
assert "Sesión 'proyecto' guardada: 2 mensajes." in out, out
assert "Para continuar: ocli chat -m gemma3:gpu --session proyecto" in out, out
assert m.session_last_path().read_text().strip() == \
    "ocli chat -m gemma3:gpu --session proyecto"
print("OK chat guarda y devuelve el comando exacto al salir")

# --- chat de una sola respuesta también lo imprime -------------------------
m.api = lambda path, payload=None, **kw: {"message": {"content": "R"}}
args = m.build_parser().parse_args(["chat", "--session", "puntual",
                                    "--no-stream", "di hola"])
out = capture(m.cmd_chat, args)
assert "Para continuar: ocli chat --session puntual" in out, out
print("OK chat --no-stream también anuncia el comando")

# --- auto-session ----------------------------------------------------------
lines = iter(["hola", "salir"])
args = m.build_parser().parse_args(["chat", "--auto-session"])
out = capture(m.cmd_chat, args)
assert "Sesión automática: auto-" in out, out
name = m.session_last_path().read_text().strip().split("--session ")[1]
assert name.startswith("auto-"), name
assert (m.SESSIONS_DIR / f"{name}.json").exists()
print("OK --auto-session crea nombre, guarda y lo recuerda")

# --- sin sesión: avisa de que no se podrá volver ---------------------------
lines = iter(["hola", "salir"])
args = m.build_parser().parse_args(["chat"])
out = capture(m.cmd_chat, args)
assert "no se guardó" in out, out
print("OK aviso cuando no hay sesión")

# --- agente ----------------------------------------------------------------
m.agent_call = lambda messages, args, readonly=False: ({"tool": "answer", "answer": "listo"}, {})
lines = iter(["hola", "/exit"])
args = m.build_parser().parse_args(["agent", "--root", str(root),
                                    "--session", "trabajo", "--steps", "1"])
out = capture(m.cmd_agent, args)
command = m.session_last_path().read_text().strip()
assert "Para continuar: ocli agent" in out, out
assert f"--root {root} --session trabajo" in command, command
data = m.session_load("trabajo")
assert data["kind"] == "agent" and data["root"] == str(root), data
print("OK agente anuncia su comando con --root")

# --- consultar la última sesión y la de un nombre --------------------------
out = capture(m.cmd_session, m.build_parser().parse_args(["session", "last"]))
assert command in out, out

out = capture(m.cmd_session, m.build_parser().parse_args(
    ["session", "resume", "proyecto"]))
assert out.strip() == "ocli chat -m gemma3:gpu --session proyecto", out

out = capture(m.cmd_session, m.build_parser().parse_args(["session", "list"]))
assert "Para volver a la última: osess last" in out, out
assert "osess resume NOMBRE" in out, out
print("OK osess last / resume / list")

# --- borrar limpia el puntero ----------------------------------------------
capture(m.cmd_session, m.build_parser().parse_args(
    ["session", "clear", "--all", "--yes"]))
assert not m.session_last_path().exists()
print("OK osess clear --all borra también la referencia")

# --- con ofree el prefijo cambia -------------------------------------------
free = load(OLLAMA_KIT_LOCAL="1", OLLAMA_API_KEY="k",
            OLLAMA_HOST="https://ollama.com")
assert free.session_resume_command("x", "chat", "glm-5.3-flash") == \
    "ofree chat -m glm-5.3-flash --session x"
assert free.cli_prefix() == "ofree"
print("OK el prefijo respeta ofree")

for key in ENV_KEYS:
    os.environ.pop(key, None)
print("REANUDAR SESIÓN: TODO OK")
