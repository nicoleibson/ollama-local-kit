#!/usr/bin/env python3
"""Perfil de precisión y presupuesto anti-truncado, sin inferencias reales."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

CLI = Path(__file__).resolve().parents[1] / "ollama-cli.py"
BASE_ENV = ("OLLAMA_KIT_PROVIDER", "OLLAMA_KIT_LOCAL", "OLLAMA_KIT_TASK_MODE",
            "OLLAMA_KIT_PRECISE", "OLLAMA_API_KEY", "NVIDIA_API_KEY",
            "GEMINI_API_KEY", "OLLAMA_HOST")


def load(**env):
    for key in BASE_ENV:
        os.environ.pop(key, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(
        "cli_precision_" + str(abs(hash(tuple(sorted(env.items())))))[:8], CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load(OLLAMA_KIT_LOCAL="1")
assert mod.PRECISE_MODEL == "nemotron3:precise"
assert mod.PRECISE_CTX == 8192

plain = mod.build_parser().parse_args(["chat", "hola"])
assert mod.apply_precision_profile(plain) is False
assert plain.model is None

precise = mod.build_parser().parse_args(["chat", "--precise", "hola"])
assert mod.apply_precision_profile(precise) is True
assert precise.model == "nemotron3:precise"
assert precise.reasoning is True
assert precise.temperature == 0.0
assert precise.ctx == 8192
assert precise.max_tokens >= 512

rag = mod.build_parser().parse_args(["rag", "ask", "--precise", "qué dice"])
assert rag.min_score == 0.10
mod.apply_precision_profile(rag)
assert rag.model == "nemotron3:precise"
assert rag.ctx == 8192

env_mod = load(OLLAMA_KIT_LOCAL="1", OLLAMA_KIT_PRECISE="1")
env_args = env_mod.build_parser().parse_args(["rag", "ask", "qué dice"])
assert env_mod.apply_precision_profile(env_args) is True
assert env_args.model == "nemotron3:precise"

budget_args = SimpleNamespace(ctx=2048, max_tokens=192, context_chars=3600,
                              model="gemma3:gpu", reasoning=False)
budget = mod.rag_context_budget(budget_args, "s" * 100, "q" * 30)
assert 0 < budget <= 3600, budget

tight = SimpleNamespace(ctx=256, max_tokens=512, context_chars=3600,
                        model="gemma3:gpu", reasoning=False)
try:
    mod.rag_context_budget(tight, "s", "q")
except SystemExit as exc:
    assert "insuficiente" in str(exc), exc
else:
    raise AssertionError("debió rechazar un ctx insuficiente")

config_out = io.StringIO()
with contextlib.redirect_stdout(config_out):
    env_mod.cmd_config(SimpleNamespace(precise=False))
assert "PERFIL DE PRECISIÓN" in config_out.getvalue()

# `run` delega en namespaces nuevos: el perfil debe sobrevivir a la delegación.
run_args = mod.build_parser().parse_args(
    ["run", "--precise", "--intent", "rag", "qué dice el índice"])
mod.apply_precision_profile(run_args)
captured = {}
with contextlib.redirect_stdout(io.StringIO()):
    with patch.object(mod, "cmd_rag_ask", lambda ns: captured.update(vars(ns))):
        mod.cmd_run(run_args)
assert captured["precise"] is True
assert captured["model"] == "nemotron3:precise"
assert captured["ctx"] == 8192
assert captured["max_tokens"] == 1024
assert captured["min_score"] == 0.10

chat_args = mod.build_parser().parse_args(
    ["run", "--precise", "--intent", "chat", "hola"])
mod.apply_precision_profile(chat_args)
captured_chat = {}
with contextlib.redirect_stdout(io.StringIO()):
    with patch.object(mod, "cmd_chat", lambda ns: captured_chat.update(vars(ns))):
        mod.cmd_run(chat_args)
assert captured_chat["precise"] is True
assert captured_chat["temperature"] == 0.0
assert captured_chat["ctx"] == 8192

# Un RAG --precise necesita más de 300 s: la API debe recibir 900 s.
assert mod.PRECISE_API_TIMEOUT == 900
fake_index = {
    "chunks": [{"path": "x.md", "text": "texto", "vector": [1.0], "start": 0}],
    "embedding_model": "qwen:gpu", "chunk_chars": 900,
}
seen_timeout = {}


def fake_api(path, payload=None, **kwargs):
    seen_timeout.update(kwargs)
    return {"message": {"content": "respuesta"}}


def run_fake_rag(precise):
    argv = ["rag", "ask", "--no-stream", "qué dice"]
    if precise:
        argv.insert(3, "--precise")
    ns = mod.build_parser().parse_args(argv)
    if precise:
        mod.apply_precision_profile(ns)
    with patch.object(mod, "rag_load", lambda: fake_index), \
         patch.object(mod, "rag_embed",
                      lambda texts, model: [[1.0] for _ in texts]), \
         patch.object(mod, "rag_cosine", lambda a, b: 1.0), \
         patch.object(mod, "api", fake_api), \
         contextlib.redirect_stdout(io.StringIO()):
        mod.cmd_rag_ask(ns)


run_fake_rag(True)
assert seen_timeout.get("timeout") == 900, seen_timeout
seen_timeout.clear()
run_fake_rag(False)
assert seen_timeout.get("timeout") == 300, seen_timeout

print("PRECISIÓN Y PRESUPUESTO: TODO OK")
