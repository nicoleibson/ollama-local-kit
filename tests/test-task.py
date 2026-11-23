#!/usr/bin/env python3
"""Perfil de tareas y razonamiento visible, sin inferencias reales."""
import contextlib
import importlib.util
import io
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CLI = Path(__file__).resolve().parents[1] / "ollama-cli.py"
BASE_ENV = ("OLLAMA_KIT_PROVIDER", "OLLAMA_KIT_LOCAL", "OLLAMA_KIT_TASK_MODE",
            "OLLAMA_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST")


def load(**env):
    for key in BASE_ENV:
        os.environ.pop(key, None)
    os.environ.update(env)
    name = "cli_task_" + str(abs(hash(tuple(sorted(env.items())))))[:8]
    spec = importlib.util.spec_from_file_location(name, CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load(OLLAMA_KIT_LOCAL="1")
assert mod.TASK_MODE is True
assert mod.default_model_for("chat") == "gemma3:gpu"
assert mod.default_model_for("agent") == "gemma3:gpu"
assert mod.default_model_for("rag") == "gemma3:gpu"
assert mod.default_model_for("agent", SimpleNamespace(reasoning=True)) == "nemotron3:gpu"
print("PERFIL TAREAS: OK")

# Los defaults resueltos por argparse siguen el perfil.
def resolved(mod, argv):
    args = mod.build_parser().parse_args(argv)
    args.func = lambda a: None

    class FakeParser:
        def parse_args(self):
            return args

    saved = mod.build_parser
    mod.build_parser = lambda: FakeParser()
    mod.main()
    mod.build_parser = saved
    return args.model

assert resolved(mod, ["chat", "hola"]) == "gemma3:gpu"
assert resolved(mod, ["agent"]) == "gemma3:gpu"
assert resolved(mod, ["agent", "--reasoning"]) == "nemotron3:gpu"
assert resolved(mod, ["rag", "ask", "hola"]) == "gemma3:gpu"
assert resolved(mod, ["rag", "ask", "--reasoning", "hola"]) == "nemotron3:gpu"
print("DEFAULTS --reasoning: OK")

# Los payloads piden razonamiento y la salida lo muestra siempre.
seen = {}


def fake_api(path, payload=None, **kwargs):
    seen["path"] = path
    seen["payload"] = payload
    thinking = "PIENSO EN PASOS CORTOS" if (payload or {}).get("think") else ""
    return {"message": {"thinking": thinking, "content": "RESPUESTA"},
            "eval_count": 4}


ns = SimpleNamespace(model="gemma3:gpu", system=None, prompt=["hola"], no_stream=True,
                     temperature=0.0, ctx=2048, max_tokens=128, think=True, stats=False,
                     format=None, session=None, new_session=False)
out = io.StringIO()
with patch.object(mod, "api", fake_api), contextlib.redirect_stdout(out):
    mod.cmd_chat(ns)
assert "think" not in seen["payload"], seen["payload"]  # Gemma 3 no soporta thinking
assert "RAZONAMIENTO:" in out.getvalue(), out.getvalue()
assert "no devolvió texto" in out.getvalue(), out.getvalue()
assert "RESPUESTA" in out.getvalue(), out.getvalue()

seen.clear()
ns_reason = SimpleNamespace(**{**vars(ns), "model": "nemotron3:gpu"})
out_reason = io.StringIO()
with patch.object(mod, "api", fake_api), contextlib.redirect_stdout(out_reason):
    mod.cmd_chat(ns_reason)
assert seen["payload"]["think"] is True, seen["payload"]
assert "PIENSO EN PASOS CORTOS" in out_reason.getvalue(), out_reason.getvalue()

out2 = io.StringIO()
with contextlib.redirect_stdout(out2):
    mod.print_reasoning({"message": {"content": "sin thinking"}})
assert "RAZONAMIENTO:" in out2.getvalue(), out2.getvalue()
assert "no devolvió texto" in out2.getvalue(), out2.getvalue()
print("RAZONAMIENTO VISIBLE: OK")

# Qwen3 piensa por defecto: se desactiva explícitamente sin tocar otros modelos.
assert mod.thinking_for("qwen3:06b") is False
assert mod.thinking_for("Qwen3 0.6B") is None
assert mod.thinking_for("qwen35:gpu") is None
assert mod.thinking_for("gemma3:gpu") is None
seen.clear()
ns_qwen = SimpleNamespace(**{**vars(ns), "model": "qwen3:06b"})
with patch.object(mod, "api", fake_api), contextlib.redirect_stdout(io.StringIO()):
    mod.cmd_chat(ns_qwen)
assert seen["payload"].get("think") is False, seen["payload"]
print("QWEN3 SIN PENSAMIENTO: OK")

# El agente de una sola petición no entra en REPL y muestra su decisión.
def fake_agent_api(path, payload=None, **kwargs):
    seen["payload"] = payload
    return {"message": {"thinking": "ELIJO ANSWER",
                        "content": json.dumps({"tool": "answer",
                                               "thought": "tarea breve",
                                               "args": {},
                                               "answer": "LISTO"})},
            "eval_count": 6}


with tempfile.TemporaryDirectory() as tmp:
    agent_ns = SimpleNamespace(
        prompt="di LISTO", root=tmp, model="gemma3:gpu", steps=1, history=4,
        ctx=2048, max_tokens=128, temperature=0.0, no_schema=False, yes=False,
        allow_dangerous=False, bash_timeout=60, session=None, new_session=False,
        reasoning=False,
    )
    out3 = io.StringIO()
    with patch.object(mod, "api", fake_agent_api), contextlib.redirect_stdout(out3):
        mod.cmd_agent(agent_ns)
    assert seen["payload"].get("think") is False, seen["payload"]
    assert "TAREA ÚNICA" in out3.getvalue(), out3.getvalue()
    assert "RAZONAMIENTO:" in out3.getvalue(), out3.getvalue()
    assert "RAZONAMIENTO AGENTE:" in out3.getvalue(), out3.getvalue()
    assert "LISTO" in out3.getvalue(), out3.getvalue()
print("AGENTE TAREA ÚNICA: OK")

# oflow también pide razonamiento.
seen.clear()
flow_ns = SimpleNamespace(model="gemma3:gpu", temperature=0.0, ctx=2048, max_tokens=64)
out4 = io.StringIO()
with patch.object(mod, "api", fake_api), contextlib.redirect_stdout(out4):
    text = mod.flow_chat("sistema", "usuario", flow_ns)
assert "think" not in seen["payload"], seen["payload"]
assert text == "RESPUESTA", text
print("FLUJOS RAZONAMIENTO: OK")

# Modo respaldo explícito por variable de entorno.
backup = load(OLLAMA_KIT_LOCAL="1", OLLAMA_KIT_TASK_MODE="0")
assert backup.TASK_MODE is False
assert backup.default_model_for("agent") == "nemotron3:gpu"
assert backup.default_model_for("rag") == "nemotron3:gpu"
print("RESPALDO TASK_MODE=0: OK")

out5 = io.StringIO()
with contextlib.redirect_stdout(out5):
    mod.cmd_config(mod.build_parser().parse_args(["config"]))
assert "PERFIL: TAREAS" in out5.getvalue(), out5.getvalue()
assert "RAZONAMIENTO: siempre visible" in out5.getvalue(), out5.getvalue()
print("CONFIG: OK")

# El inventario de `ocli list` (que alimenta los desplegables de la GUI) conoce
# los tags nuevos y el perfil preciso.
for tag in ("bonsai:q1", "gemma3:1b", "lfm25:350m", "qwen3:06b",
            "nemotron3:precise", "gemma4:dgpu-safe",
            "gemma3:ctx4k", "gemma3:ctx8k", "nemotron3:ctx4k",
            "nemotron3:ctx8k", "gemma4:ctx4k", "gemma4:ctx8k"):
    assert tag in mod.INFO and mod.INFO[tag][0] != "(otro)", tag
print("INVENTARIO MODELOS: OK")
print("PERFIL TAREAS: TODO OK")
