#!/usr/bin/env python3
"""Validación de JSON Schema y reintento, sin inferencias reales."""
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
        "cli_schema_" + str(abs(hash(tuple(sorted(env.items())))))[:8], CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = load(OLLAMA_KIT_LOCAL="1")

schema = {
    "type": "object",
    "required": ["command"],
    "properties": {
        "command": {"type": "string", "enum": ["date", "pwd"]},
        "extra": {"type": "integer"},
    },
}
assert mod.schema_valid({"command": "date"}, schema)
assert not mod.schema_valid({"command": "rm"}, schema)
assert not mod.schema_valid({"extra": 1}, schema)
assert not mod.schema_valid({"command": "date", "extra": "uno"}, schema)

nested = {
    "type": "object",
    "required": ["matches"],
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["label"],
                "properties": {"label": {"type": "string", "enum": ["a", "b"]}},
            },
        }
    },
}
assert mod.schema_valid({"matches": [{"label": "a"}]}, nested)
assert not mod.schema_valid({"matches": [{"label": "c"}]}, nested)
assert not mod.schema_valid({"matches": [{"label": "a"}, {"otro": "a"}]}, nested)
print("VALIDACIÓN DE ESQUEMA: OK")

assert mod.json_from_text("```json\n{\"command\": \"date\"}\n```") == {"command": "date"}
assert mod.json_from_text("claro: {\"command\": \"pwd\"} fin") == {"command": "pwd"}
assert mod.json_from_text("sin json") is None
print("PARSEO DE JSON: OK")

payload = {
    "model": "gemma3:gpu",
    "messages": [
        {"role": "system", "content": "Devuelve JSON"},
        {"role": "user", "content": "muestra la fecha"},
    ],
    "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 24},
}
short = mod.shorten_for_retry("x" * 3000)
assert len(short) < 1300 and "recortado" in short
retry = mod.json_retry_payload(payload)
assert retry["options"]["num_predict"] == 256
assert "no cumplía el esquema" in retry["messages"][-1]["content"]
assert retry["messages"][1]["content"] == "muestra la fecha"
print("RECONSTRUCCIÓN DEL REINTENTO: OK")

calls = []


def fake_api(path, body=None, method=None, timeout=300):
    calls.append((path, body))
    text = "no es json" if len(calls) == 1 else '```json\n{"command": "date"}\n```'
    return {"message": {"content": text}, "eval_count": 10}


with patch.object(mod, "api", fake_api), contextlib.redirect_stdout(io.StringIO()):
    data, result, retried = mod.api_chat_json(payload, schema)
assert retried is True and data == {"command": "date"}
assert len(calls) == 2, calls
assert "no cumplía" in calls[1][1]["messages"][-1]["content"]

calls.clear()
with patch.object(mod, "api", fake_api), patch.object(mod, "thinking_for", lambda m: None), \
        contextlib.redirect_stdout(io.StringIO()) as out:
    args = mod.build_parser().parse_args(["act", "--dry-run", "muestra la fecha"])
    mod.cmd_act(args)
assert len(calls) == 2, calls
assert "JSON revalidado tras un reintento" in out.getvalue()

calls.clear()
with patch.object(mod, "api", fake_api), patch.object(mod, "thinking_for", lambda m: None), \
        contextlib.redirect_stdout(io.StringIO()):
    flow_args = SimpleNamespace(model="gemma3:gpu", temperature=0, ctx=2048,
                                max_tokens=64)
    text = mod.flow_chat("sistema", "fragmento", flow_args, schema=schema)
assert len(calls) == 2, calls
assert mod.json_from_text(text) == {"command": "date"}
print("REINTENTO ACT / FLUJO: OK")

print("JSON SCHEMA Y REINTENTO: TODO OK")
