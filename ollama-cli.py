#!/usr/bin/env python3
"""CLI personal para exprimir los modelos locales de Ollama en este equipo.

Todo hereda la configuración ya aplicada: iGPU Carrizo, contexto 2048,
un modelo cargado, KV q8_0, sin nube. Uso: ollama-cli.py --help
"""
import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

CLOUD_HOST = "https://ollama.com"
LOCAL_HOST = "http://127.0.0.1:11434"
PROVIDER = os.environ.get("OLLAMA_KIT_PROVIDER", "ollama").strip().lower()
if PROVIDER not in ("ollama", "nvidia", "gemini"):
    PROVIDER = "ollama"

PROVIDER_KEYVARS = {
    "ollama": "OLLAMA_API_KEY",
    "nvidia": "NVIDIA_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
PROVIDER_KEYFILES = {
    "ollama": Path.home() / ".ollama-cloud-key",
    "nvidia": Path.home() / ".nvidia-api-key",
    "gemini": Path.home() / ".gemini-api-key",
}
PROVIDER_BASES = {
    "ollama": CLOUD_HOST,
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
}


def provider_key():
    # Solo la variable de entorno activa el modo remoto. El archivo de key lo
    # leen los prefijos (ocloud/onvidia/ogemini) y lo exportan aquí.
    return os.environ.get(PROVIDER_KEYVARS[PROVIDER], "").strip()


API_KEY = provider_key()
# OLLAMA_HOST explícito manda; si no, cada proveedor tiene su base por defecto.
if PROVIDER == "ollama":
    HOST = os.environ.get("OLLAMA_HOST") or (CLOUD_HOST if API_KEY else LOCAL_HOST)
else:
    HOST = os.environ.get("OLLAMA_HOST") or PROVIDER_BASES[PROVIDER]
HOST = HOST.rstrip("/")
CLOUD_MODE = "ollama.com" in HOST
EXTERNAL = PROVIDER != "ollama" or CLOUD_MODE
# Gemini admite JSON Schema; Ollama Cloud y NIM (según su schema) no.
NO_SCHEMA = EXTERNAL and not (PROVIDER == "gemini" and not CLOUD_MODE)

# Modo gratuito explícito: ningún modelo cloud de Ollama es gratis; los modelos
# locales sí. OLLAMA_KIT_LOCAL=1 fuerza el servidor de esta máquina e ignora keys.
LOCAL_ONLY = os.environ.get("OLLAMA_KIT_LOCAL", "") == "1"
FREE_MODEL_MAP = {
    "glm-5.3-flash": "nemotron3:gpu",
    "glm-5.3": "nemotron3:gpu",
    "glm-5.3-flash:latest": "nemotron3:gpu",
    "kimi-k2.7-code": "nemotron3:gpu",
    "deepseek-v4.1-flash": "nemotron3:gpu",
    "deepseek-v4.1-flash:1017": "nemotron3:gpu",
    "beam": "gemma3:q4",
    "gemma4:31b": "gemma4:cpu",
    "gemma4:cloud": "gemma4:cpu",
    "nemotron-3-super": "nemotron3:gpu",
    "clef:27b": "gemma3:q4",
}


def free_model(name):
    return FREE_MODEL_MAP.get(name, name)


if LOCAL_ONLY:
    PROVIDER = "ollama"
    API_KEY = ""
    HOST = LOCAL_HOST
    CLOUD_MODE = False
    EXTERNAL = False
    NO_SCHEMA = False


def provider_alias():
    if PROVIDER == "ollama":
        return "ocloud" if CLOUD_MODE else "ocli"
    return f"o{PROVIDER}"


OPENAI_PATHS = {
    "/api/chat": "/chat/completions",
    "/api/generate": "/chat/completions",
    "/api/embed": "/embeddings",
    "/api/tags": "/models",
    "/api/ps": "/models",
}


def api_headers():
    headers = {"Content-Type": "application/json"}
    if API_KEY and EXTERNAL:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


def http_detail(code, body):
    detail = body[:300]
    if "add usage credits" in body.lower() or "subscribe" in body.lower():
        detail += (" | Los modelos cloud no son gratis: usa 'ofree' o un modelo "
                   "local con 'ocli'.")
    if EXTERNAL and code in (401, 402, 403, 429):
        detail += (f" | Revisa {PROVIDER_KEYVARS[PROVIDER]} y el plan/créditos "
                   "del proveedor")
    return detail


def openai_payload(path, payload):
    """Convierte un payload Ollama al formato chat/completions o embeddings."""
    payload = payload or {}
    if path in ("/api/chat", "/api/generate"):
        opts = payload.get("options", {})
        messages = payload.get("messages")
        if path == "/api/generate":
            messages = []
            if payload.get("system"):
                messages.append({"role": "system", "content": payload["system"]})
            messages.append({"role": "user", "content": payload.get("prompt", "")})
        out = {
            "model": payload.get("model"),
            "messages": messages,
            "stream": payload.get("stream", False),
            "temperature": opts.get("temperature", 0.7),
            "max_tokens": opts.get("num_predict") or opts.get("max_tokens") or 512,
        }
        if PROVIDER == "nvidia":
            out["reasoning_effort"] = "medium" if payload.get("think", True) else "none"
        elif PROVIDER == "gemini" and payload.get("format"):
            out["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "salida", "schema": payload["format"]},
            }
        return out
    if path == "/api/embed":
        return {"model": payload.get("model"), "input": payload.get("input")}
    return {}


def openai_response(path, data, started):
    """Devuelve una respuesta con forma Ollama para no tocar los comandos."""
    elapsed = max(time.time() - started, 1e-6)
    if path in ("/api/chat", "/api/generate"):
        choice = (data.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content", "") or ""
        usage = data.get("usage") or {}
        stats = {
            "eval_count": usage.get("completion_tokens", 0),
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "eval_duration": int(elapsed * 1e9),
            "done": True,
        }
        if path == "/api/generate":
            return {"response": content, **stats}
        return {"message": {"content": content}, **stats}
    if path == "/api/embed":
        return {"embeddings": [d.get("embedding", []) for d in data.get("data", [])]}
    if path == "/api/tags":
        return {"models": [{"name": m.get("id") or m.get("name")}
                           for m in data.get("data", [])]}
    if path == "/api/ps":
        return {"models": []}
    return data
OVERRIDE = Path("/etc/systemd/system/ollama.service.d/override.conf")
MODELS_DIR = Path("/home/onion/Documentos/ModelsIA")
SESSIONS_DIR = Path.home() / ".ollama-sessions"
PROMPTS_PATH = Path.home() / ".ollama-prompts.json"
BENCH_HISTORY = Path.home() / ".ollama-bench-history.jsonl"
DEBUG_LOG = Path.home() / ".ollama-debug.log"
UNDO_DIR = Path.home() / ".ollama-agent-undo"
DEBUG = False

INFO = {
    "gemma3:q4": ("Gemma 3 1B Q4_K_M", "9,3 tok/s", "chat rápido y acciones"),
    "gemma3:gpu": ("Gemma 3 1B Q6_K", "7,8 tok/s", "más calidad en chat"),
    "gemma3:latest": ("Gemma 3 1B (sin num_gpu)", "~7 tok/s", "peor que :gpu"),
    "nemotron3:gpu": ("Nemotron-H 4B IQ2_M", "2,6 tok/s", "razonamiento complejo"),
    "nemotron3:latest": ("Nemotron-H 4B (sin num_gpu)", "~2 tok/s", "peor que :gpu"),
    "qwen:gpu": ("Qwen3 0.6B Base F16", "embedding", "solo /api/embed; default RAG"),
    "qwen:latest": ("Qwen3 0.6B Base F16", "embedding", "solo /api/embed"),
    "qwen35:gpu": ("Qwen3.5 4B Q4_K_M", "2,61 tok/s", "generación, 33/33 capas GPU"),
    "qwen35:embed": ("Qwen3.5 4B + pooling_type", "embedding", "alternativa RAG, 2560 dim"),
    "gemma4:cpu": ("Gemma 4 E4B Q2_K", "2,5 tok/s", "CPU: no cabe en la iGPU"),
    "gemma4:dgpu": ("Gemma 4 E4B en R7 260X", "8,5 tok/s", "solo con odgpu (puerto 11435)"),
}

DEFAULT_CATALOG = {
    "date": {"cmd": "date", "desc": "fecha y hora actual"},
    "whoami": {"cmd": "whoami", "desc": "usuario actual"},
    "pwd": {"cmd": "pwd", "desc": "directorio actual"},
    "free": {"cmd": "free -h", "desc": "memoria libre y usada"},
    "uptime": {"cmd": "uptime", "desc": "tiempo encendido"},
    "ls": {"cmd": "ls -la", "desc": "listar archivos"},
}
CATALOG_PATH = Path(
    os.environ.get("OLLAMA_COMMANDS", "/home/onion/ollama-commands.json")
)

RAG_DIR = Path(os.environ.get("OLLAMA_RAG_DIR", "/home/onion/.ollama-rag"))
RAG_INDEX = RAG_DIR / "index.json"
RAG_EXTS = {
    ".md", ".txt", ".rst", ".py", ".sh", ".json", ".jsonl", ".conf",
    ".ini", ".cfg", ".yaml", ".yml", ".csv", ".log", ".toml", ".service",
    ".timer",
}
RAG_SKIP_DIRS = {
    ".git", "__pycache__", ".cache", "node_modules", ".ollama-rag",
    "Backups.Ollama",
}
RAG_STOP = {
    "de", "del", "la", "el", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "en", "por", "para", "con", "sin", "qué", "que", "cómo",
    "como", "cuál", "cuales", "quién", "quienes", "donde", "dónde",
    "cuándo", "cuando", "porque", "es", "son", "al", "lo", "su", "sus",
    "mi", "mis", "me", "se", "dice",
}


def load_catalog(path=None):
    """Catálogo editable: name -> {cmd, desc}. Nunca se ejecuta texto del modelo."""
    p = Path(path) if path else CATALOG_PATH
    try:
        data = json.loads(p.read_text())
        data = {k: v for k, v in data.items() if not k.startswith("_")}
        if data and all(("cmd" in v or "script" in v) for v in data.values()):
            return data
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return DEFAULT_CATALOG


def debug_log(event, **fields):
    """Registro JSONL de eventos y errores. Los errores se guardan siempre."""
    if not DEBUG and event not in ("error", "api_error", "process_error", "process_timeout"):
        return
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "event": event}
    row.update({k: str(v)[:6000] for k, v in fields.items()})
    try:
        DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DEBUG_LOG.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass
    if DEBUG:
        detail = " ".join(f"{k}={str(v)[:160]}" for k, v in fields.items())
        print(f"[debug] {event} {detail}", file=sys.stderr)


def api_prepare(path, payload):
    """Prepara (ruta, cuerpo) según el proveedor. Devuelve None si es nativo."""
    if not (EXTERNAL and path in OPENAI_PATHS):
        return None
    if path == "/api/embed" and PROVIDER == "nvidia":
        raise SystemExit(
            "NVIDIA NIM no expone /embeddings en esta API. Usa 'ocli' (qwen:gpu) "
            "u 'ogemini' para indexar con RAG."
        )
    body = openai_payload(path, payload)
    if not API_KEY:
        raise SystemExit(
            f"Falta la key de {PROVIDER}: exporta {PROVIDER_KEYVARS[PROVIDER]} o "
            f"escríbela en {PROVIDER_KEYFILES[PROVIDER]} (chmod 600)."
        )
    return OPENAI_PATHS[path], json.dumps(body).encode()


def api(path, payload=None, method=None, timeout=300):
    started = time.time()
    prepared = api_prepare(path, payload)
    if prepared:
        req_path, data = prepared
    else:
        req_path = path
        data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        HOST + req_path, data=data, method=method, headers=api_headers(),
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        debug_log("api_error", host=HOST, path=path, code=e.code, body=body[:800],
                  seconds=round(time.time() - started, 3))
        raise SystemExit(f"HTTP {e.code} en {path}: {http_detail(e.code, body)}")
    except urllib.error.URLError as e:
        debug_log("api_error", host=HOST, path=path, reason=str(e.reason))
        raise SystemExit(f"No se puede conectar con {HOST}{path}: {e.reason}")
    debug_log("api", host=HOST, provider=PROVIDER, path=path, status=status,
              bytes=len(raw), seconds=round(time.time() - started, 3))
    parsed = json.loads(raw) if raw else {}
    return openai_response(path, parsed, started) if prepared else parsed


def api_stream(path, payload, timeout=300):
    started = time.time()
    prepared = api_prepare(path, payload)
    if prepared:
        req_path, data = prepared
        body = json.loads(data)
        body["stream"] = True
        data = json.dumps(body).encode()
    else:
        req_path = path
        data = json.dumps(payload).encode()
    req = urllib.request.Request(HOST + req_path, data=data, headers=api_headers())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if prepared:
                total = 0
                prompt_tokens = 0
                for line in resp:
                    text = line.decode(errors="replace").strip()
                    if not text.startswith("data:"):
                        continue
                    raw = text[5:].strip()
                    if raw == "[DONE]":
                        break
                    chunk = json.loads(raw)
                    choice = (chunk.get("choices") or [{}])[0]
                    piece = (choice.get("delta") or {}).get("content", "") or ""
                    usage = chunk.get("usage") or {}
                    if usage:
                        total = usage.get("completion_tokens", total)
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    yield {"message": {"content": piece}}
                yield {
                    "done": True,
                    "eval_count": total,
                    "prompt_eval_count": prompt_tokens,
                    "eval_duration": int(max(time.time() - started, 1e-6) * 1e9),
                }
            else:
                for line in resp:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    yield chunk
                    if chunk.get("done"):
                        break
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        debug_log("api_error", host=HOST, path=path, code=e.code, body=body[:800])
        raise SystemExit(f"HTTP {e.code} en {path}: {http_detail(e.code, body)}")
    except urllib.error.URLError as e:
        debug_log("api_error", host=HOST, path=path, reason=str(e.reason))
        raise SystemExit(f"No se puede conectar con {HOST}{path}: {e.reason}")
    debug_log("api", host=HOST, provider=PROVIDER, path=path, status=200,
              stream=True, seconds=round(time.time() - started, 3))


def sh(cmd, timeout=20):
    try:
        return subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        ).stdout.strip()
    except subprocess.TimeoutExpired:
        return ""


def run_visible(cmd, shell=True, timeout=300):
    """Ejecuta un proceso mostrando inicio, salida, duración y código de salida."""
    started = time.time()
    shown = cmd if isinstance(cmd, str) else " ".join(shlex.quote(c) for c in cmd)
    print(f"[->] {shown}", flush=True)
    try:
        p = subprocess.run(cmd, shell=shell, capture_output=True, text=True,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        debug_log("process_timeout", cmd=cmd, seconds=round(elapsed, 3), timeout=timeout)
        print(f"[X] TIMEOUT tras {timeout}s", file=sys.stderr)
        return 124
    if p.stdout:
        print(p.stdout, end="" if p.stdout.endswith("\n") else "\n")
    if p.stderr:
        print("[stderr] " + p.stderr.replace("\n", "\n[stderr] ").rstrip())
    elapsed = time.time() - started
    if p.returncode:
        debug_log("process_error", cmd=cmd, code=p.returncode,
                  seconds=round(elapsed, 3), stdout=p.stdout[-2000:],
                  stderr=p.stderr[-2000:])
        print(f"[X] exit {p.returncode} en {elapsed:.2f}s | detalle: {DEBUG_LOG}",
              file=sys.stderr)
    else:
        debug_log("process_ok", cmd=cmd, seconds=round(elapsed, 3))
        print(f"[OK] {elapsed:.2f}s")
    return p.returncode


def build_payload(args, prompt, history=None):
    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})
    payload = {
        "model": args.model,
        "stream": not args.no_stream,
        "messages": messages,
        "options": {
            "temperature": args.temperature,
            "num_ctx": args.ctx,
            "num_predict": args.max_tokens,
        },
    }
    if not args.think:
        payload["think"] = False
    if args.format:
        try:
            payload["format"] = json.loads(args.format)
        except json.JSONDecodeError:
            raise SystemExit("--format debe ser un JSON válido")
    return payload


def session_path(name):
    safe = re.sub(r"[^a-zA-Z0-9_-]+", "-", name or "default").strip("-")[:60]
    return SESSIONS_DIR / f"{safe or 'default'}.json"


def session_load(name):
    try:
        data = json.loads(session_path(name).read_text())
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"name": name, "model": None, "messages": []}


def session_save(name, model, messages):
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path(name).write_text(json.dumps({
        "name": name,
        "model": model,
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }, ensure_ascii=False, indent=2))


def cmd_session(args):
    if not SESSIONS_DIR.is_dir():
        print("No hay sesiones guardadas.")
        return
    files = sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if args.action == "list" or not args.name:
        if not files:
            print("No hay sesiones guardadas.")
            return
        print(f"{'SESIÓN':<20} {'MSGS':>5}  MODELO / ACTUALIZADA")
        print("-" * 70)
        for p in files:
            try:
                d = json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            msgs = len(d.get("messages", []))
            print(f"{d.get('name', p.stem):<20} {msgs:>5}  "
                  f"{d.get('model') or '?'} / {d.get('updated', '?')}")
        return

    if args.action == "show":
        data = session_load(args.name)
        if not data["messages"]:
            raise SystemExit(f"Sesión no encontrada o vacía: {args.name}")
        print(f"Sesión: {data['name']}  modelo: {data.get('model') or '?'}  "
              f"actualizada: {data.get('updated', '?')}")
        for msg in data["messages"]:
            print(f"\n[{msg.get('role', '?')}] {msg.get('content', '')}")
        return

    if args.action == "clear":
        targets = list(SESSIONS_DIR.glob("*.json")) if args.all else [session_path(args.name)]
        if not args.yes:
            n = len(targets)
            answer = input(f"¿Borrar {n} sesión(es)? [s/N] ").strip().lower()
            if answer not in ("s", "si", "y", "yes"):
                print("Cancelado.")
                return
        for p in targets:
            p.unlink(missing_ok=True)
        print(f"Sesiones borradas: {len(targets)}")


def cmd_chat(args):
    session = getattr(args, "session", None)
    history = []
    if session:
        data = session_load(session)
        if not getattr(args, "new_session", False):
            history = data["messages"]
            if history:
                print(f"Sesión '{session}': {len(history)} mensajes recuperados.")
    if not args.prompt:
        print("Chat interactivo. 'salir' o Ctrl+C para terminar.\n")
        while True:
            try:
                user = input("tú> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user.strip().lower() in ("salir", "exit", "quit"):
                break
            if not user.strip():
                continue
            args.no_stream = False
            payload = build_payload(args, user, history)
            answer = ""
            for chunk in api_stream("/api/chat", payload):
                if chunk.get("error"):
                    print("ERROR:", chunk["error"])
                    break
                piece = chunk.get("message", {}).get("content", "")
                answer += piece
                print(piece, end="", flush=True)
            print()
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": answer})
            if session:
                session_save(session, args.model, history)
        return

    prompt = " ".join(args.prompt)
    payload = build_payload(args, prompt, history)
    if args.no_stream:
        r = api("/api/chat", payload)
        if r.get("error"):
            raise SystemExit("ERROR: " + str(r["error"]))
        answer = r["message"]["content"]
        print(answer)
        if args.stats:
            print_stats(r)
    else:
        answer = ""
        last = {}
        for chunk in api_stream("/api/chat", payload):
            answer += chunk.get("message", {}).get("content", "")
            print(chunk.get("message", {}).get("content", ""), end="", flush=True)
            last = chunk
        print()
        if args.stats:
            print_stats(last)
    if session:
        history.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ])
        session_save(session, args.model, history)


def prompts_load():
    try:
        data = json.loads(PROMPTS_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def prompts_save(data):
    PROMPTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def render_prompt(text, positional, variables):
    out = text
    for i, value in enumerate(positional, 1):
        out = out.replace("{{%d}}" % i, value)
    out = out.replace("{{args}}", " ".join(positional))
    for key, value in variables.items():
        out = out.replace("{{%s}}" % key, value)
    return out


def cmd_prompt(args):
    data = prompts_load()

    if args.action == "list":
        if not data:
            print(f"No hay prompts. Crea uno con: ocli prompt add nombre --text '...'")
            return
        print(f"{'NOMBRE':<18} MODELO / DESCRIPCIÓN")
        print("-" * 66)
        for name, item in sorted(data.items()):
            print(f"{name:<18} {item.get('model') or 'default'} / {item.get('desc', '')}")
        return

    if args.action == "add":
        if not args.name or not args.text:
            raise SystemExit("Indica nombre y --text. Ej: ocli prompt add resumen --text 'Resume {{args}}'")
        data[args.name] = {
            "text": args.text,
            "desc": args.desc or "",
            "model": args.model,
            "system": args.system,
        }
        prompts_save(data)
        print(f"Prompt guardado: {args.name}")
        return

    if args.action == "show":
        item = data.get(args.name)
        if not item:
            raise SystemExit(f"Prompt no encontrado: {args.name}")
        print(f"Nombre : {args.name}")
        print(f"Modelo : {item.get('model') or 'default'}")
        print(f"Desc   : {item.get('desc', '')}")
        print("Texto  :\n" + item["text"])
        return

    if args.action == "rm":
        if args.name not in data:
            raise SystemExit(f"Prompt no encontrado: {args.name}")
        del data[args.name]
        prompts_save(data)
        print(f"Prompt borrado: {args.name}")
        return

    item = data.get(args.name)
    if not item:
        raise SystemExit(f"Prompt no encontrado: {args.name}. Usa 'ocli prompt list'.")
    variables = {}
    for pair in args.var:
        if "=" not in pair:
            raise SystemExit("--var debe ser CLAVE=VALOR")
        key, value = pair.split("=", 1)
        variables[key] = value
    prompt = render_prompt(item["text"], args.prompt_args, variables)
    if "{{" in prompt:
        print("Aviso: quedan variables sin sustituir en el prompt.", file=sys.stderr)

    ns = argparse.Namespace(
        prompt=[prompt], model=args.model or item.get("model") or "gemma3:q4",
        system=args.system or item.get("system"), temperature=0.3, ctx=2048,
        max_tokens=512, think=False, no_stream=args.no_stream, stats=args.stats,
        format=None, session=args.session, new_session=False,
    )
    cmd_chat(ns)


def print_stats(r):
    n = r.get("eval_count", 0)
    t = r.get("eval_duration", 0) / 1e9
    print(f"\n[{n} tokens / {t:.2f}s = {n/t:.2f} tok/s]" if t else "")


DANGEROUS_PATTERNS = [
    r"\brm\s+-[rRfF]+\s+/",
    r"\brm\s+-[rR]+\s+-[fF]+\s+/",
    r"\brm\s+-[fF]+\s+-[rR]+\s+/",
    r"\brm\s+-rf\s+\*",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r"\b(shutdown|reboot|poweroff|halt)\b",
    r"\bchmod\s+-[rR]+\s+777\s+/",
    r":\(\)\s*\{",
    r"\b(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(sh|bash)\b",
    r">\s*/dev/(sd|nvme|hd)",
    r"\btruncate\s+-s\b",
]


def act_is_dangerous(command):
    text = command.lower()
    return any(re.search(pattern, text) for pattern in DANGEROUS_PATTERNS)


def cmd_act(args):
    if NO_SCHEMA:
        raise SystemExit(
            f"act usa JSON Schema y {PROVIDER} no lo soporta aquí. "
            "Usa 'ocli act ...' (local) o el modo chat del proveedor."
        )
    catalog = load_catalog(args.catalog)
    if args.list:
        print(f"Catálogo: {args.catalog or CATALOG_PATH}  ({len(catalog)} entradas)")
        print(f"{'NOMBRE':<16} COMANDO")
        print("-" * 72)
        for name, item in catalog.items():
            target = item.get("script") or item.get("cmd", "")
            print(f"  {name:<16} {target}")
        print("\nEdita ese JSON para añadir o quitar comandos; no hay que tocar el código.")
        return

    text = " ".join(args.text)
    if not text:
        raise SystemExit("Escribe la instrucción. Ej: ollama-cli.py act 'muestra la fecha'")

    # Con catálogos grandes o modo libre, el Gemma 1B se confunde: usar el 4B.
    model = args.model or ("nemotron3:gpu" if (args.free or len(catalog) > 8) else "gemma3:q4")

    if args.free:
        fmt = {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        }
        system = (
            "Eres un asistente de terminal. Propón UN comando de shell que resuelva "
            "la petición del usuario. Devuelve SOLO JSON: {\"command\":\"...\"}. "
            "No ejecutes nada tú mismo. Prefiere comandos de solo lectura salvo que "
            "el usuario pida expresamente modificar algo."
        )
        max_tokens = 96
    else:
        fmt = {
            "type": "object",
            "properties": {"command": {"type": "string", "enum": list(catalog)}},
            "required": ["command"],
        }
        descs = "; ".join(f"{k}={v.get('desc', '')}" for k, v in catalog.items())
        system = (
            "Eres un ejecutor. Responde SOLO con JSON: {\"command\":\"...\"}. "
            "Elige el identificador de esta lista según lo que pida el usuario: "
            + descs + "."
        )
        max_tokens = 24

    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "format": fmt,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": max_tokens},
    }
    r = api("/api/chat", payload)
    if r.get("error"):
        raise SystemExit("ERROR: " + str(r["error"]))
    content = r["message"]["content"]
    try:
        proposed = json.loads(content)["command"]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise SystemExit("El modelo no devolvió un JSON válido: " + repr(content))

    if args.free:
        command = str(proposed).strip().strip("`")
        if not command:
            raise SystemExit("El modelo propuso un comando vacío.")
        first = command.split()[0]
        if not shutil.which(first):
            raise SystemExit(
                f"El modelo propuso '{first}', que no es un comando instalado. "
                "Prueba con -m nemotron3:gpu o usa el catálogo (sin --free)."
            )
        danger = act_is_dangerous(command)
        flag = " [PELIGROSO]" if danger else ""
        print(f"modelo: {model} | propuesto: {command}{flag}  ({r['eval_count']} tokens)")
        if danger and not getattr(args, "allow_dangerous", False):
            raise SystemExit(
                "Comando bloqueado por peligroso. Si estás seguro, añade "
                "--allow-dangerous (se seguirá pidiendo confirmación)."
            )
        if args.dry_run:
            print("(dry-run: no se ejecutó)")
            return
        if not args.yes:
            answer = input("¿Ejecutar este comando? [s/N] ").strip().lower()
            if answer not in ("s", "si", "y", "yes"):
                print("Cancelado.")
                return
        rc = run_visible(command, timeout=300)
        if rc:
            raise SystemExit(f"El comando terminó con código {rc}")
        return

    if proposed not in catalog:
        raise SystemExit(f"Identificador no admitido: {proposed}")
    item = catalog[proposed]
    if "script" in item:
        script = Path(item["script"]).expanduser()
        if not script.is_file():
            raise SystemExit(f"El script no existe: {script}")
        target = str(script)
        run_fn = lambda: run_visible(["bash", target], shell=False)
    else:
        target = item["cmd"]
        run_fn = lambda: run_visible(target)
    print(f"modelo: {model} | comando: {proposed} -> {target}  ({r['eval_count']} tokens)")
    if args.dry_run:
        return
    rc = run_fn()
    if rc:
        raise SystemExit(f"El comando terminó con código {rc}")


BENCH_PROMPT = "Explica por qué el cielo es azul"


def bench_history_show(limit=15):
    if not BENCH_HISTORY.exists():
        print("Sin historial de benchmarks. Ejecuta: ocli bench")
        return
    rows = []
    for line in BENCH_HISTORY.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = rows[-limit:]
    print(f"{'FECHA':<17} {'MODELO':<18} {'TOK/S':>7} {'TOKENS':>7} {'VRAM':>7}")
    print("-" * 62)
    for r in rows:
        print(f"{r.get('time', '?'):<17} {r.get('model', '?'):<18} "
              f"{r.get('tps', 0):>7.2f} {r.get('tokens', 0):>7} "
              f"{r.get('vram', 0):>6.2f}G")
    print(f"\n{len(rows)} registro(s) de {BENCH_HISTORY}")


def cmd_bench(args):
    if args.history:
        bench_history_show(args.history if isinstance(args.history, int) else 15)
        return
    models = args.models or ["gemma3:q4", "gemma3:gpu", "nemotron3:gpu"]
    results = []
    print(f"{'MODELO':<18} {'RESULTADO':<36}")
    print("-" * 56)
    for model in models:
        r = api("/api/generate", {
            "model": model, "prompt": BENCH_PROMPT, "stream": False,
            "think": False,
            "options": {"num_ctx": 2048, "num_predict": args.tokens},
        })
        if r.get("error"):
            print(f"{model:<18} ERROR: {str(r['error'])[:34]}")
            continue
        n = r.get("eval_count", 0)
        t = r.get("eval_duration", 0) / 1e9
        print(f"{model:<18} {n} tok / {t:.2f}s = {n/t:.2f} tok/s" if t else f"{model:<18} sin datos")
        if t:
            results.append({
                "time": time.strftime("%Y-%m-%d %H:%M"),
                "model": model,
                "tokens": n,
                "seconds": round(t, 3),
                "tps": round(n / t, 3),
                "vram": 0.0,
            })
    print()
    ps = api("/api/ps").get("models", [])
    if ps:
        for m in ps:
            print(f"Cargado: {m['name']}  {m.get('size_vram',0)/2**30:.2f} GiB en VRAM")
        vram = {m["name"]: m.get("size_vram", 0) / 2**30 for m in ps}
        for row in results:
            row["vram"] = round(vram.get(row["model"], 0.0), 2)
    else:
        print("Ningún modelo cargado actualmente.")

    if results and not args.no_save:
        BENCH_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with BENCH_HISTORY.open("a") as fh:
            for row in results:
                fh.write(json.dumps(row) + "\n")
        print("Historial actualizado:", BENCH_HISTORY)


def cmd_status(args):
    governor = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    freq = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    maxfreq = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
    print("=== SERVICIO ===")
    print("  Ollama:", sh("systemctl is-active ollama") or "?")
    freq_text = f"({int(freq)/1e6:.2f} GHz ahora / {int(maxfreq)/1e6:.2f} GHz máx)" if freq.isdigit() and maxfreq.isdigit() else ""
    print("  Gobernador:", governor, freq_text)
    print("  vfs_cache_pressure:", sh("sysctl -n vm.vfs_cache_pressure"))
    print("  powerprofile:", sh("powerprofilesctl get") or "n/d")
    print("\n=== GPU ===")
    gpu = sh("journalctl -u ollama -n 5000 --no-pager 2>/dev/null | grep -o 'RADV [A-Z]*' | tail -1")
    print("  Detectada:", gpu or "sin registro reciente (usa 'ollama ps' para comprobar)")
    ps = api("/api/ps").get("models", [])
    if ps:
        for m in ps:
            proc = "GPU" if m.get("size_vram", 0) > 0 else "CPU"
            print(f"  {m['name']}: {proc}  {m.get('size_vram',0)/2**30:.2f} GiB VRAM")
    else:
        print("  Ningún modelo cargado")
    print("\n=== SENSORES ===")
    sens = sh("sensors 2>/dev/null | grep -E 'edge:|PPT:|fan1:' | head -6")
    print("  " + sens.replace("\n", "\n  ") if sens else "  n/d")
    print("\n=== CONFIGURACIÓN ===")
    try:
        for line in OVERRIDE.read_text().splitlines():
            if line.startswith("Environment="):
                print("  " + line)
    except OSError:
        print("  No se puede leer", OVERRIDE)


def cmd_watch(args):
    """Monitor ligero en vivo: CPU, sensores, memoria y modelos cargados."""
    iteration = 0
    try:
        while args.count == 0 or iteration < args.count:
            iteration += 1
            now = time.strftime("%H:%M:%S")
            governor = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            freq = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
            maxfreq = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")
            if freq.isdigit() and maxfreq.isdigit():
                cpu = f"{int(freq)/1e6:.2f} / {int(maxfreq)/1e6:.2f} GHz"
            else:
                cpu = "n/d"
            load = sh("cut -d' ' -f1-3 /proc/loadavg")
            temp = sh("sensors 2>/dev/null | grep -E 'edge:|Composite:|fan1:' | head -3")
            mem = sh("free -h | head -2")
            disk = sh("df -h / /home 2>/dev/null | tail -n +2")
            top = sh("ps -eo comm,%cpu --sort=-%cpu | head -4 | tail -3")
            try:
                models = api("/api/ps").get("models", [])
            except SystemExit:
                models = []

            if args.clear and iteration > 1 and sys.stdout.isatty():
                print("\033[H\033[2J", end="")
            elif iteration > 1:
                print()

            print(f"=== OLLAMA WATCH  {now} ===")
            print(f"  CPU      : {governor or '?'}  {cpu}  carga {load or '?'}")
            print(f"  Ollama   : {sh('systemctl is-active ollama') or '?'}")
            if models:
                for m in models:
                    vram = m.get("size_vram", 0) / 2**30
                    proc = "GPU" if vram > 0 else "CPU"
                    print(f"  Modelo   : {m['name']}  {proc}  {vram:.2f} GiB VRAM")
            else:
                print("  Modelo   : ninguno cargado")
            print("  Memoria  : " + (mem.replace("\n", "\n             ") if mem else "n/d"))
            print("  Disco    : " + (disk.replace("\n", "\n             ") if disk else "n/d"))
            if temp:
                print("  Sensores : " + temp.replace("\n", "\n             "))
            if top:
                print("  Top CPU  : " + top.replace("\n", "\n             "))

            if args.count == 0 or iteration < args.count:
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nMonitor detenido.")


def check(name, ok, detail, warn_only=False):
    tag = "OK  " if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{tag}] {name}: {detail}")
    return ok or warn_only


def cmd_doctor(args):
    print("=== DOCTOR ===")
    results = []
    results.append(check("servicio Ollama", sh("systemctl is-active ollama") == "active",
                         sh("systemctl is-active ollama") or "inactivo"))
    try:
        env = OVERRIDE.read_text()
    except OSError:
        env = ""
    results.append(check("iGPU habilitada", "OLLAMA_IGPU_ENABLE=1" in env,
                         "OLLAMA_IGPU_ENABLE=1 en override"))
    results.append(check("GPU elegida", "GGML_VK_VISIBLE_DEVICES=0" in env,
                         "CARRIZO (0); para dGPU sería 1"))
    results.append(check("sin nube", "OLLAMA_NO_CLOUD=true" in env,
                         "OLLAMA_NO_CLOUD=true"))
    gov = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    results.append(check("gobernador CPU", gov == "performance", gov or "n/d"))
    pressure = sh("sysctl -n vm.vfs_cache_pressure")
    results.append(check("vfs_cache_pressure", pressure.isdigit() and int(pressure) <= 50,
                         pressure or "n/d", warn_only=True))
    ps = api("/api/ps").get("models", [])
    if ps:
        ok = any(m.get("size_vram", 0) > 0 for m in ps)
        results.append(check("modelo en GPU", ok, ps[0]["name"]))
    else:
        results.append(check("modelo en GPU", False, "ningún modelo cargado", warn_only=True))
    ok = all(results)
    print("\nResultado:", "TODO CORRECTO" if ok else "HAY FALLOS (revisa lo marcado FAIL)")
    sys.exit(0 if ok else 1)


def cmd_list(args):
    if EXTERNAL:
        try:
            models = api("/api/tags").get("models", [])
        except SystemExit:
            print(f"No se pudo consultar el catálogo de {PROVIDER}.")
            return
        if not models:
            print(f"No hay modelos visibles en {PROVIDER}.")
            return
        print(f"{'MODELO':<48} PROVEEDOR")
        print("-" * 62)
        for m in models:
            name = m.get("name", "")
            if name:
                print(f"{name:<48} {PROVIDER}")
        return
    out = sh("ollama list 2>/dev/null")
    lines = out.splitlines()
    if not lines:
        print("No hay modelos o no se pudo consultar Ollama.")
        return
    print(f"{'TAG':<20} {'DESCRIPCIÓN':<34} {'RENDIMIENTO':<12} USO")
    print("-" * 88)
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        tag = parts[0]
        desc, speed, use = INFO.get(tag, ("(otro)", "?", "—"))
        print(f"{tag:<20} {desc:<34} {speed:<12} {use}")


def cmd_show(args):
    if EXTERNAL:
        raise SystemExit(
            f"'show' solo consulta el Ollama local. Usa '{provider_alias()} list' "
            f"para ver los modelos de {PROVIDER}."
        )
    print(sh(f"ollama show {args.model} 2>&1"))


def cmd_stop(args):
    if EXTERNAL:
        print("No hay modelos locales que descargar: los remotos se gestionan en el proveedor.")
        return
    if args.all:
        models = [m.get("name") for m in api("/api/ps").get("models", []) if m.get("name")]
        if not models:
            print("No hay modelos cargados.")
            return
        for name in models:
            api("/api/generate", {"model": name, "keep_alive": 0})
            print(f"Descargado: {name}")
        return
    if not args.model:
        raise SystemExit("Indica un modelo o usa --all.")
    print(sh(f"ollama stop {args.model} 2>&1") or f"Se intentó descargar {args.model}")


def cmd_import(args):
    gguf = Path(args.gguf).expanduser()
    if not gguf.is_file():
        raise SystemExit(f"No existe el GGUF: {gguf}")
    modelfile = MODELS_DIR / f"Modelfile.{args.tag.replace(':', '.')}"
    modelfile.write_text(
        f"FROM {gguf}\nPARAMETER temperature 0.3\nPARAMETER num_ctx 2048\n"
        f"PARAMETER num_gpu {args.gpu}\nPARAMETER stop <end_of_turn>\n"
    )
    existing = sh("ollama list 2>/dev/null")
    if args.tag in existing and not args.force:
        raise SystemExit(f"El tag {args.tag} ya existe. Usa --force para reemplazarlo.")
    if args.force and args.tag in existing:
        sh(f"ollama rm {args.tag}")
    print(f"Importando {args.tag} desde {gguf.name} ...")
    print(sh(f"ollama create {args.tag} -f {modelfile} 2>&1 | tail -4"))
    print("Listo:", args.tag)


def cmd_rm(args):
    if not args.yes:
        resp = input(f"¿Borrar el modelo {args.model}? [s/N] ")
        if resp.strip().lower() not in ("s", "si", "y", "yes"):
            print("Cancelado.")
            return
    print(sh(f"ollama rm {args.model} 2>&1"))


def cmd_embed(args):
    text = " ".join(args.text) if args.text else (
        Path(args.file).read_text() if args.file else ""
    )
    if not text:
        raise SystemExit("Da texto o --file ARCHIVO")
    r = api("/api/embed", {"model": args.model, "input": text})
    if r.get("error"):
        raise SystemExit("ERROR: " + str(r["error"]))
    vecs = r.get("embeddings", [])
    if not vecs:
        raise SystemExit("El modelo no devolvió embeddings.")
    print(json.dumps({
        "modelo": args.model,
        "cantidad": len(vecs),
        "dimensiones": len(vecs[0]),
        "primeros_valores": vecs[0][:args.preview],
    }, ensure_ascii=False))


def cmd_config(args):
    print(f"PROVEEDOR: {PROVIDER}")
    print(f"MODO: {'EXTERNO (API key directa)' if EXTERNAL else 'LOCAL (Ollama en este equipo)'}")
    if LOCAL_ONLY:
        print("GRATIS: solo modelos locales; se ignoran API keys y HOST remoto.")
    print(f"HOST: {HOST}")
    print(f"API key: {'configurada' if API_KEY else f'no configurada ({PROVIDER_KEYVARS[PROVIDER]})'}")
    if EXTERNAL:
        print(f"Archivo de key: {PROVIDER_KEYFILES[PROVIDER]}")
    print(f"Overrides: {OVERRIDE}")
    print(f"Carpeta GGUF: {MODELS_DIR}")
    print("\nVariables activas:")
    try:
        for line in OVERRIDE.read_text().splitlines():
            if line.startswith("Environment="):
                print("  " + line)
    except OSError:
        print("  (no legible)")
    print("\nAtajos activos en ~/.bashrc:")
    print("  ocli   -> ollama-cli.py")
    print("  ochat  -> ollama-cli.py chat")
    print("  oact   -> ollama-cli.py act")
    print("  orag   -> ollama-cli.py rag")
    print("  orun   -> ollama-cli.py run")
    print("  owatch -> ollama-cli.py watch")
    print("  oprompt-> ollama-cli.py prompt")
    print("  osess  -> ollama-cli.py session")
    print("  odgpu  -> ollama-scripts/gemma4-dgpu.sh (R7 260X aislada)")
    print("  ocode  -> ollama-cli.py code")
    print("  odebug -> ollama-cli.py debug")
    print("  ocloud  -> ollama-cli.py contra ollama.com (lee ~/.ollama-cloud-key)")
    print("  onvidia -> ollama-cli.py contra NVIDIA NIM (lee ~/.nvidia-api-key)")
    print("  ogemini -> ollama-cli.py contra Gemini (lee ~/.gemini-api-key)")
    print("  oagent  -> ollama-cli.py agent (REPL con herramientas)")
    print("  ofree   -> solo modelos locales, sin keys ni créditos")
    if EXTERNAL:
        print(f"\nNota: {PROVIDER} habla con la API remota; no toca el override local.")
        if NO_SCHEMA:
            print(" act no está disponible y code exige --no-schema (sin JSON Schema).")
        else:
            print(" JSON Schema disponible: act y code funcionan con este proveedor.")


# ---------------------------------------------------------------------------
# RAG: índice local de fragmentos + embeddings con qwen:gpu
# ---------------------------------------------------------------------------

def rag_load():
    try:
        data = json.loads(RAG_INDEX.read_text())
        if isinstance(data, dict) and "chunks" in data:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {
        "version": 1,
        "embedding_model": None,
        "dim": None,
        "chunk_chars": 900,
        "overlap": 150,
        "chunks": [],
    }


def rag_save(data):
    RAG_DIR.mkdir(parents=True, exist_ok=True)
    RAG_INDEX.write_text(json.dumps(data, ensure_ascii=False))


def rag_split(text, size=900, overlap=150):
    """Divide en fragmentos solapados, procurando cortar por línea/palabra."""
    text = text.replace("\r\n", "\n")
    out = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + size, total)
        if end < total:
            window = text[start + size // 2:end]
            cut = window.rfind("\n")
            if cut < 0:
                cut = window.rfind(". ")
            if cut < 0:
                cut = window.rfind(" ")
            if cut >= 0:
                end = start + size // 2 + cut + 1
        part = text[start:end].strip()
        if part:
            out.append((start, part))
        if end >= total:
            break
        start = max(start + 1, end - overlap)
    return out


def rag_collect(paths, max_files=120, max_mb=2):
    files = []
    seen = set()
    limit = max_mb * 1024 * 1024
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"No existe la ruta: {p}")
        if p.is_file():
            candidates = [p]
        else:
            candidates = sorted(
                f for f in p.rglob("*")
                if f.is_file()
                and not any(part in RAG_SKIP_DIRS for part in f.parts)
                and (f.suffix.lower() in RAG_EXTS or f.name.startswith("Modelfile"))
            )
        for f in candidates:
            if f in seen:
                continue
            try:
                if f.stat().st_size > limit:
                    continue
            except OSError:
                continue
            seen.add(f)
            files.append(f)
            if len(files) >= max_files:
                break
        if len(files) >= max_files:
            break
    return files


def rag_embed(texts, model):
    if not texts:
        return []
    r = api("/api/embed", {
        "model": model,
        "input": texts,
        "truncate": True,
    }, timeout=900)
    if r.get("error"):
        raise SystemExit("ERROR: " + str(r["error"]))
    vecs = r.get("embeddings", [])
    if len(vecs) != len(texts):
        raise SystemExit(
            f"El modelo devolvió {len(vecs)} vectores para {len(texts)} textos."
        )
    return vecs


def rag_in_roots(path, roots):
    p = Path(path)
    return any(p == root or root in p.parents for root in roots)


def rag_summary(data):
    by_file = {}
    for c in data["chunks"]:
        by_file[c["path"]] = by_file.get(c["path"], 0) + 1
    print(f"Índice: {RAG_INDEX}")
    print(f"  embedding: {data.get('embedding_model') or '?'} "
          f"| dimensiones: {data.get('dim') or '?'} "
          f"| fragmentos: {len(data['chunks'])} | archivos: {len(by_file)}")
    for path, count in sorted(by_file.items(), key=lambda x: -x[1])[:20]:
        print(f"    {count:4}  {path}")
    if len(by_file) > 20:
        print(f"    ... y {len(by_file) - 20} archivos más")


def cmd_rag_index(args):
    files = rag_collect(args.paths, args.max_files, args.max_mb)
    if not files:
        raise SystemExit("No hay archivos indexables en las rutas dadas.")

    data = rag_load()
    if data["chunks"] and data.get("embedding_model") not in (None, args.embed_model):
        raise SystemExit(
            f"El índice usa {data['embedding_model']}. Usa el mismo --embed-model "
            "o borra el índice con 'rag clear'."
        )

    roots = [Path(p).expanduser().resolve() for p in args.paths]
    kept = [c for c in data["chunks"] if not rag_in_roots(c["path"], roots)]

    jobs = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "\x00" in text:
            continue
        for start, part in rag_split(text, args.chunk_chars, args.overlap):
            jobs.append((str(f), start, part))

    if not jobs:
        raise SystemExit("No se generaron fragmentos con esas rutas.")

    print(f"Indexando {len(jobs)} fragmentos de {len(files)} archivos "
          f"con {args.embed_model} ...")
    vectors = []
    for i in range(0, len(jobs), args.batch):
        batch = jobs[i:i + args.batch]
        vectors.extend(rag_embed([j[2] for j in batch], args.embed_model))
        print(f"  {min(i + args.batch, len(jobs))}/{len(jobs)}", flush=True)

    data["embedding_model"] = args.embed_model
    data["dim"] = len(vectors[0])
    data["chunk_chars"] = args.chunk_chars
    data["overlap"] = args.overlap
    data["chunks"] = kept + [
        {"path": path, "start": start, "text": text, "vector": vec}
        for (path, start, text), vec in zip(jobs, vectors)
    ]
    rag_save(data)
    print("\nÍndice actualizado:")
    rag_summary(data)


def rag_cosine(a, b):
    if len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else -1.0


def rag_tokens(text):
    return {
        t for t in re.findall(r"[a-z0-9_.-]+", text.lower())
        if len(t) > 2 and t not in RAG_STOP
    }


def rag_idf(chunks):
    """Peso de cada término: los raros importan más (IDF simple)."""
    n = len(chunks)
    df = {}
    for chunk in chunks:
        for token in rag_tokens(chunk["text"] + " " + chunk["path"]):
            df[token] = df.get(token, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}


def rag_lexical(question, chunk, idf):
    terms = rag_tokens(question)
    if not terms:
        return 0.0
    haystack = rag_tokens(chunk["text"] + " " + chunk["path"])
    total = sum(idf.get(t, 1.0) for t in terms)
    hit = sum(idf.get(t, 1.0) for t in terms if t in haystack)
    return hit / total if total else 0.0


def cmd_rag_ask(args):
    data = rag_load()
    if not data["chunks"]:
        raise SystemExit("Índice vacío. Usa primero: ocli rag index ARCHIVO|DIR")
    if data.get("embedding_model") != args.embed_model:
        raise SystemExit(
            f"El índice usa {data.get('embedding_model') or 'otro modelo'}; "
            f"pediste {args.embed_model}. Usa --embed-model "
            f"{data.get('embedding_model') or 'qwen:gpu'}."
        )

    question = " ".join(args.question)
    qvec = rag_embed([question], args.embed_model)[0]

    idf = rag_idf(data["chunks"])
    scored = []
    for chunk in data["chunks"]:
        score = rag_cosine(qvec, chunk["vector"]) + 0.5 * rag_lexical(question, chunk, idf)
        if score >= args.min_score:
            scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)

    picked = []
    for score, chunk in scored:
        if any(chunk["path"] == c["path"]
               and abs(chunk["start"] - c["start"]) < data.get("chunk_chars", 900) // 2
               for _, c in picked):
            continue
        picked.append((score, chunk))
        if len(picked) >= args.k:
            break
    if not picked:
        raise SystemExit("Ningún fragmento supera --min-score.")

    sections = []
    used = 0
    for i, (_, chunk) in enumerate(picked, 1):
        header = f"[{i}] {chunk['path']}\n"
        block = header + chunk["text"]
        if used + len(block) > args.context_chars and sections:
            break
        sections.append(block)
        used += len(block) + 4
    picked = picked[:len(sections)]
    context = "\n\n".join(sections)

    system = (
        "Eres un asistente de documentación. Responde a la pregunta usando SOLO "
        "los fragmentos dados, pero parafrasea y resume en lugar de copiar bloques. "
        "Si la información está presente de forma implícita, responde igualmente. "
        "Di 'No lo encuentro en el índice' SOLO si los fragmentos no tienen nada "
        "que ver con la pregunta. Cita los fragmentos con números entre corchetes, "
        "por ejemplo [1] o [2] (no escribas la letra n)."
    )
    user = f"Pregunta: {question}\n\nFragmentos recuperados:\n{context}"
    payload = {
        "model": args.model,
        "stream": not args.no_stream,
        "think": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "temperature": 0,
            "num_ctx": args.ctx,
            "num_predict": args.max_tokens,
        },
    }

    print(f"modelo: {args.model} | fragmentos: {len(picked)} "
          f"({data['embedding_model']} -> chat)\n")
    if args.no_stream:
        r = api("/api/chat", payload)
        if r.get("error"):
            raise SystemExit("ERROR: " + str(r["error"]))
        print(r["message"]["content"])
        stats = r
    else:
        stats = {}
        for chunk in api_stream("/api/chat", payload):
            piece = chunk.get("message", {}).get("content", "")
            print(piece, end="", flush=True)
            stats = chunk
        print()

    print("\nFuentes:")
    for i, (score, chunk) in enumerate(picked, 1):
        print(f"  [{i}] {score:.3f}  {chunk['path']}  (offset {chunk['start']})")
    if stats.get("eval_count"):
        print_stats(stats)


def cmd_rag_list(args):
    data = rag_load()
    if not data["chunks"]:
        print(f"Índice vacío ({RAG_INDEX}). Usa 'ocli rag index ARCHIVO|DIR'.")
        return
    rag_summary(data)


def cmd_rag_clear(args):
    if not RAG_INDEX.exists():
        print("No hay índice que borrar.")
        return
    if not args.yes:
        resp = input(f"¿Borrar el índice {RAG_INDEX}? [s/N] ")
        if resp.strip().lower() not in ("s", "si", "y", "yes"):
            print("Cancelado.")
            return
    RAG_INDEX.unlink()
    print("Índice borrado.")


def cmd_run(args):
    """Entrada única: clasifica la intención y delega en act, rag o chat."""
    text = " ".join(args.text).strip()
    if not text:
        raise SystemExit("Escribe una instrucción. Ej: ocli run 'muestra la fecha'")

    rag_data = rag_load()
    intents = ["act", "chat"] + (["rag"] if rag_data["chunks"] else [])

    if args.intent:
        intent = args.intent
        confidence = 1.0
    else:
        fmt = {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": intents},
                "confidence": {"type": "number"},
            },
            "required": ["intent", "confidence"],
        }
        system = (
            "Eres un enrutador de terminal. Elige UNA intención para la petición "
            "del usuario. act = ejecutar o consultar algo del sistema (fecha, "
            "memoria, discos, procesos, red, temperatura, scripts, copias, "
            "benchmarks). rag = preguntar sobre el contenido de archivos indexados "
            "localmente. chat = explicaciones, redacción, traducción o preguntas "
            "generales. Devuelve SOLO JSON: {\"intent\":\"...\",\"confidence\":0-1}. "
            "Ejemplos: 'muestra la fecha' y 'cuánta memoria hay' son act; "
            "'qué dice el manual de...', 'según lo indexado...' o 'en los scripts...' "
            "son rag; 'por qué el cielo es azul' es chat."
        )
        if "rag" not in intents:
            system += " No hay índice RAG disponible; no elijas rag."
        r = api("/api/chat", {
            "model": "nemotron3:gpu",
            "stream": False,
            "think": False,
            "format": fmt,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
            "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 24},
        })
        if r.get("error"):
            raise SystemExit("ERROR: " + str(r["error"]))
        try:
            obj = json.loads(r["message"]["content"])
        except (json.JSONDecodeError, KeyError, TypeError):
            raise SystemExit("El enrutador no devolvió JSON válido: " + repr(r["message"]["content"]))
        intent = obj.get("intent") if obj.get("intent") in intents else "chat"
        try:
            confidence = float(obj.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

    print(f"intención: {intent} ({confidence:.2f})")
    if args.dry_run and intent != "act":
        print("(dry-run: no se ejecutó)")
        return

    if intent == "act":
        ns = argparse.Namespace(
            text=text.split(), catalog=args.catalog, list=False,
            model=args.act_model, dry_run=args.dry_run, free=False,
            yes=False, allow_dangerous=False,
        )
        cmd_act(ns)
        return

    if intent == "rag":
        ns = argparse.Namespace(
            question=text.split(),
            model=args.model or "nemotron3:gpu",
            embed_model=rag_data.get("embedding_model") or "qwen:gpu",
            k=args.k, min_score=0.0, ctx=2048, max_tokens=384,
            context_chars=3600, no_stream=args.no_stream,
        )
        cmd_rag_ask(ns)
        return

    ns = argparse.Namespace(
        prompt=text.split(), model=args.model or "gemma3:q4",
        system=args.system, temperature=0.3, ctx=2048, max_tokens=512,
        think=False, no_stream=args.no_stream, stats=args.stats, format=None,
    )
    cmd_chat(ns)


# ---------------------------------------------------------------------------
# Agente interactivo: bucle REPL con herramientas y confirmaciones
# ---------------------------------------------------------------------------

AGENT_BLOCKED = {
    ".git", ".ollama-rag", ".ollama-sessions", ".ollama-agent-undo",
    ".ollama-cloud-key", ".nvidia-api-key", ".gemini-api-key",
    ".env", "id_ed25519", "id_ed25519.pub",
}
AGENT_TOOL_NAMES = ("answer", "read", "list", "grep", "rag", "write", "edit", "bash")


def agent_schema():
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string",
                        "description": "Razonamiento breve en español, una frase"},
            "tool": {"type": "string", "enum": list(AGENT_TOOL_NAMES)},
            "args": {"type": "object",
                     "description": "Argumentos de la herramienta; vacío para answer"},
            "answer": {"type": "string",
                       "description": "Respuesta final, solo cuando tool=answer"},
        },
        "required": ["tool"],
    }


def agent_system(root):
    return (
        "Eres un agente de programación en una terminal. Trabajas SOLO dentro de "
        f"{root}. Devuelve SOLO un objeto JSON, sin markdown ni texto fuera del JSON, "
        "con esta forma: {\"thought\":\"frase breve\",\"tool\":\"...\",\"args\":{...},\"answer\":\"...\"}. "
        "Elige tool=answer para responder; para write, edit o bash el usuario deberá "
        "confirmar antes de ejecutar. Usa read/list/grep/rag para investigar antes de "
        "modificar nada. Args por herramienta: "
        "read {path, start?, lines?}; list {path?}; grep {pattern, path?, max?}; "
        "rag {query, k?}; write {path, content}; edit {path, old, new}; "
        "bash {command}. No propongas rutas absolutas ni que salgan de la carpeta de trabajo."
    )


def agent_resolve(root, value, must_exist=False):
    if not isinstance(value, str) or not value.strip():
        raise SystemExit("La herramienta necesita una ruta ('path').")
    p = Path(value)
    if p.is_absolute() or ".." in p.parts:
        raise SystemExit(f"Ruta no permitida fuera del área de trabajo: {value}")
    target = (root / p).resolve()
    if target != root and root not in target.parents:
        raise SystemExit(f"Ruta no permitida fuera del área de trabajo: {value}")
    if any(part in AGENT_BLOCKED for part in target.relative_to(root).parts):
        raise SystemExit(f"Ruta bloqueada (secreto o control interno): {value}")
    if must_exist and not target.exists():
        raise SystemExit(f"No existe: {target}")
    return target


def agent_read(root, a):
    path = agent_resolve(root, a.get("path"), must_exist=True)
    if not path.is_file():
        raise SystemExit(f"No es un archivo: {path}")
    lines = path.read_text(errors="replace").splitlines()
    start = max(1, int(a.get("start", 1)))
    end = min(len(lines), start + max(1, int(a.get("lines", 200))) - 1)
    body = "\n".join(f"{i:>5} | {lines[i-1]}" for i in range(start, end + 1))
    return f"{path.relative_to(root)}:{start}-{end} de {len(lines)}\n{body}"


def agent_list(root, a):
    path = agent_resolve(root, a.get("path", "."), must_exist=True)
    if not path.is_dir():
        raise SystemExit(f"No es una carpeta: {path}")
    rows = []
    for item in sorted(path.iterdir(), key=lambda x: x.name.lower()):
        if item.name in AGENT_BLOCKED or item.name.startswith(".ollama-"):
            continue
        kind = "dir" if item.is_dir() else "file"
        size = "" if item.is_dir() else str(item.stat().st_size)
        rows.append(f"{kind:<4} {size:>9} {item.name}")
    return f"{path.relative_to(root)}\n" + "\n".join(rows[:100])


def agent_grep(root, a):
    pattern = a.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        raise SystemExit("La herramienta grep necesita 'pattern'.")
    regex = re.compile(pattern, re.IGNORECASE)
    start = agent_resolve(root, a.get("path", "."), must_exist=True)
    files = [start] if start.is_file() else start.rglob("*")
    hits = []
    for path in files:
        if not path.is_file() or path.stat().st_size > 1_000_000:
            continue
        if any(part in AGENT_BLOCKED for part in path.relative_to(root).parts):
            continue
        try:
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{path.relative_to(root)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= int(a.get("max", 30)):
                        return "\n".join(hits)
        except OSError:
            continue
    return "\n".join(hits) if hits else "(sin coincidencias)"


def agent_rag(root, a):
    query = a.get("query")
    if not isinstance(query, str) or not query.strip():
        raise SystemExit("La herramienta rag necesita 'query'.")
    data = rag_load()
    if not data["chunks"]:
        raise SystemExit("Índice RAG vacío. Usa: ocli rag index ARCHIVO|DIR")
    model = data.get("embedding_model") or "qwen:gpu"
    vector = rag_embed([query], model)[0]
    idf = rag_idf(data["chunks"])
    scored = []
    for chunk in data["chunks"]:
        score = rag_cosine(vector, chunk["vector"]) + 0.5 * rag_lexical(query, chunk, idf)
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    picked = []
    for score, chunk in scored:
        if any(chunk["path"] == c["path"]
               and abs(chunk["start"] - c["start"]) < data.get("chunk_chars", 900) // 2
               for _, c in picked):
            continue
        picked.append((score, chunk))
        if len(picked) >= int(a.get("k", 3)):
            break
    blocks = [f"[{i}] {c['path']}\n{c['text'][:1200]}" for i, (_, c) in enumerate(picked, 1)]
    return "\n\n".join(blocks) if blocks else "(sin resultados en el índice)"


def agent_confirm(args, what):
    if args.yes:
        return True
    try:
        answer = input(f"¿Permitir {what}? [s/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("s", "si", "y", "yes")


def agent_backup(path):
    UNDO_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}"
    backup = UNDO_DIR / f"{stamp}-{re.sub(r'[^a-zA-Z0-9_.-]+', '-', path.name)[:60]}"
    shutil.copy2(path, backup)
    return backup


def agent_write(root, a, args, undo):
    path = agent_resolve(root, a.get("path"))
    content = a.get("content")
    if not isinstance(content, str):
        raise SystemExit("La herramienta write necesita 'content' como texto.")
    if len(content) > 20_000:
        raise SystemExit("Contenido demasiado grande (máx. 20000 caracteres por llamada).")
    existed = path.exists()
    if existed:
        old = path.read_text(errors="replace")
        if old == content:
            return "Sin cambios: el archivo ya tiene ese contenido."
    if not agent_confirm(args, f"escribir {path.relative_to(root)}"):
        return "Cancelado por el usuario."
    backup = agent_backup(path) if existed else None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    undo.append((path, backup))
    return f"Escrito {path.relative_to(root)} ({len(content)} chars)."


def agent_edit(root, a, args, undo):
    path = agent_resolve(root, a.get("path"), must_exist=True)
    old, new = a.get("old"), a.get("new")
    if not isinstance(old, str) or not isinstance(new, str) or not old:
        raise SystemExit("La herramienta edit necesita 'old' y 'new' como texto.")
    text = path.read_text(errors="replace")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"El texto a reemplazar aparece {count} veces; amplía 'old' para que sea único.")
    preview = f"{old[:120]!r} -> {new[:120]!r}"
    if not agent_confirm(args, f"editar {path.relative_to(root)} ({preview})"):
        return "Cancelado por el usuario."
    undo.append((path, agent_backup(path)))
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return f"Editado {path.relative_to(root)}."


def agent_bash(root, a, args):
    command = a.get("command")
    if not isinstance(command, str) or not command.strip():
        raise SystemExit("La herramienta bash necesita 'command'.")
    command = command.strip().strip("`")
    first = command.split()[0]
    if not shutil.which(first):
        raise SystemExit(f"'{first}' no es un comando instalado.")
    danger = act_is_dangerous(command)
    if danger and not args.allow_dangerous:
        raise SystemExit("Comando bloqueado por peligroso. Usa --allow-dangerous si estás seguro.")
    flag = " [PELIGROSO]" if danger else ""
    if not agent_confirm(args, f"ejecutar '{command[:120]}'{flag}"):
        return "Cancelado por el usuario."
    rc = run_visible(command, timeout=args.bash_timeout)
    return f"Comando terminado con código {rc}."


def agent_undo(undo):
    if not undo:
        return "No hay cambios para deshacer."
    path, backup = undo.pop()
    if backup and Path(backup).is_file():
        shutil.copy2(backup, path)
        return f"Restaurado {path} desde copia de seguridad."
    if path.exists():
        path.unlink()
        return f"Eliminado el archivo creado {path}."
    return f"No había copia de {path}."


def agent_run_tool(tool, a, args, root, undo):
    a = a if isinstance(a, dict) else {}
    if tool == "read":
        return agent_read(root, a)
    if tool == "list":
        return agent_list(root, a)
    if tool == "grep":
        return agent_grep(root, a)
    if tool == "rag":
        return agent_rag(root, a)
    if tool == "write":
        return agent_write(root, a, args, undo)
    if tool == "edit":
        return agent_edit(root, a, args, undo)
    if tool == "bash":
        return agent_bash(root, a, args)
    raise SystemExit(f"Herramienta no admitida: {tool}")


def agent_call(messages, args):
    payload = {
        "model": args.model,
        "stream": False,
        "think": False,
        "messages": messages,
        "options": {
            "temperature": args.temperature,
            "num_ctx": args.ctx,
            "num_predict": args.max_tokens,
        },
    }
    if NO_SCHEMA and not args.no_schema:
        raise SystemExit(
            f"{PROVIDER} no admite JSON Schema aquí: usa 'ocli agent' (local) u "
            "'ogemini agent', o añade --no-schema para parsear JSON del texto."
        )
    if not (NO_SCHEMA and args.no_schema):
        payload["format"] = agent_schema()
    r = api("/api/chat", payload)
    if r.get("error"):
        raise SystemExit("ERROR: " + str(r["error"]))
    content = r.get("message", {}).get("content", "")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = None
        else:
            data = None
        if not isinstance(data, dict):
            raise SystemExit("El agente no devolvió un JSON válido: " + repr(content[:300]))
    if data.get("tool") not in AGENT_TOOL_NAMES:
        raise SystemExit(f"El agente eligió una herramienta desconocida: {data.get('tool')!r}")
    return data, r


def cmd_agent(args):
    if NO_SCHEMA and not args.no_schema:
        raise SystemExit(
            f"El modo agente usa JSON Schema y {PROVIDER} no lo admite aquí. "
            "Usa 'oagent' (local), 'ogemini agent' o añade --no-schema."
        )
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"La carpeta de trabajo no existe: {root}")

    history = []
    session = getattr(args, "session", None)
    if session and not args.new_session:
        saved = session_load(session)
        history = [m for m in saved["messages"] if m.get("role") != "system"]
        if history:
            print(f"Sesión '{session}': {len(history)} mensajes recuperados.")
    undo = []

    print(f"AGENTE | modelo: {args.model} | proveedor: {PROVIDER} | raíz: {root}")
    print("Escribe una petición o /help. /exit para salir. Las escrituras y "
          "comandos piden confirmación salvo --yes.\n")

    while True:
        try:
            line = input("ag> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in ("/exit", "/salir", "/quit", "salir", "exit", "quit"):
            break
        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            value = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/help", "/?"):
                print("/help  /model NAME  /provider  /session NAME  /clear  /root PATH")
                print("/tools  /rag PREGUNTA  /bash COMANDO  /undo  /yes  /exit")
            elif cmd == "/model" and value:
                args.model = value
                print(f"Modelo: {args.model}")
            elif cmd == "/provider":
                print(f"Proveedor actual: {PROVIDER} ({HOST}). Para cambiar, "
                      "sal y usa el prefijo ocli/ocloud/onvidia/ogemini.")
            elif cmd == "/session" and value:
                if session:
                    session_save(session, args.model, history)
                session = value
                args.session = session
                history = [m for m in session_load(session)["messages"]
                           if m.get("role") != "system"]
                print(f"Sesión: {session} ({len(history)} mensajes)")
            elif cmd == "/clear":
                history = []
                print("Historial limpiado en esta sesión.")
            elif cmd == "/root" and value:
                candidate = Path(value).expanduser().resolve()
                if candidate.is_dir():
                    root = candidate
                    print(f"Raíz: {root}")
                else:
                    print("No es una carpeta.")
            elif cmd == "/tools":
                print(", ".join(AGENT_TOOL_NAMES))
            elif cmd == "/rag" and value:
                print(agent_rag(root, {"query": value}))
            elif cmd == "/bash" and value:
                print(agent_bash(root, {"command": value}, args))
            elif cmd == "/undo":
                print(agent_undo(undo))
            elif cmd == "/yes":
                args.yes = not args.yes
                print(f"Confirmaciones: {'desactivadas' if args.yes else 'activadas'}")
            else:
                print("Comando no reconocido. Usa /help.")
            continue

        history.append({"role": "user", "content": line})
        for _ in range(max(1, args.steps)):
            messages = [{"role": "system", "content": agent_system(root)}] + history
            decision, stats = agent_call(messages, args)
            thought = decision.get("thought")
            if thought:
                print(f"  · {thought}")
            tool = decision.get("tool")
            if tool == "answer":
                answer = str(decision.get("answer", ""))
                print(answer)
                history.append({"role": "assistant", "content": answer})
                break
            print(f"[{tool}]", flush=True)
            result = agent_run_tool(tool, decision.get("args", {}), args, root, undo)
            print(result[:4000])
            history.append({"role": "assistant",
                            "content": json.dumps(decision, ensure_ascii=False)})
            history.append({"role": "user", "content": f"RESULTADO de {tool}:\n{result[:6000]}"})
        else:
            print(f"Se alcanzó el límite de {args.steps} pasos. Reformula o usa /clear.")
        history = history[-max(4, args.history):]
        if session:
            session_save(session, args.model, history)


def build_parser():
    p = argparse.ArgumentParser(
        prog="ollama-cli.py",
        description="CLI personal para los modelos locales (iGPU Carrizo).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Ejemplos:
  ollama-cli.py chat "hola"
  ollama-cli.py chat -m gemma3:gpu --stats "explica esto"
  ollama-cli.py act --dry-run "muestra la fecha"
  ollama-cli.py act --list
  ollama-cli.py act --free --dry-run "muestra los procesos que más CPU usan"
  ollama-cli.py bench gemma3:q4 gemma3:gpu
  ollama-cli.py status
  ollama-cli.py doctor
  ollama-cli.py embed -m qwen:gpu "texto"
  ollama-cli.py rag index /home/onion/ollama-kit.md
  ollama-cli.py rag ask "cómo funciona el modo act" -k 3
  ollama-cli.py rag list
  ollama-cli.py run "muestra la fecha"              # enrutador automático
  ollama-cli.py run --intent rag "qué dice el manual del RAG"
  ollama-cli.py watch --count 5 --interval 1        # monitor en vivo
  ollama-cli.py chat --session proyecto "hola"      # conversación persistente
  ollama-cli.py session list
  ollama-cli.py prompt add resumen --text 'Resume {{args}}'
  ollama-cli.py prompt use resumen "este texto largo"
  ollama-cli.py bench --history
  ollama-cli.py code review ~/ollama-cli.py --dry-run
  ollama-cli.py code tests modulo.py
  ollama-cli.py code commit
  ollama-cli.py agent                          # REPL con herramientas y permisos
  ollama-cli.py agent --root ~/ollama-local-kit --session trabajo
  ollama-cli.py --debug act "muestra la fecha"
  ollama-cli.py debug --full
  ollama-cli.py import gemma3:q2 /home/onion/Documentos/ModelsIA/Gemma-...-Q2_k.gguf
""",
    )
    p.add_argument("--debug", action="store_true",
                   help="trazas y peticiones en stderr; también OLLAMA_KIT_DEBUG=1")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat", aliases=["ask"], help="chatear (sin prompt = interactivo)")
    c.add_argument("prompt", nargs="*")
    c.add_argument("-m", "--model", default="gemma3:q4")
    c.add_argument("--system", help="mensaje de sistema")
    c.add_argument("-t", "--temperature", type=float, default=0.3)
    c.add_argument("--ctx", type=int, default=2048)
    c.add_argument("--max-tokens", type=int, default=512)
    c.add_argument("--think", action="store_true", help="permitir <think>")
    c.add_argument("--no-stream", action="store_true")
    c.add_argument("--stats", action="store_true")
    c.add_argument("--format", help="JSON Schema de salida")
    c.add_argument("--session", help="guardar/recuperar la conversación con este nombre")
    c.add_argument("--new-session", action="store_true",
                   help="empezar la sesión desde cero, borrando la anterior")
    c.set_defaults(func=cmd_chat)

    se = sub.add_parser("session", help="gestionar conversaciones guardadas")
    se.add_argument("action", choices=["list", "show", "clear"], nargs="?", default="list")
    se.add_argument("name", nargs="?")
    se.add_argument("--all", action="store_true", help="borrar todas las sesiones")
    se.add_argument("--yes", action="store_true", help="no pedir confirmación al borrar")
    se.set_defaults(func=cmd_session)

    pr = sub.add_parser("prompt", help="biblioteca de prompts reutilizables")
    pr.add_argument("action", choices=["list", "show", "add", "rm", "use"], nargs="?", default="list")
    pr.add_argument("name", nargs="?")
    pr.add_argument("prompt_args", nargs="*", help="valores para {{1}}, {{2}} o {{args}}")
    pr.add_argument("--text", help="texto del prompt al añadir")
    pr.add_argument("--desc", default="", help="descripción corta")
    pr.add_argument("-m", "--model", help="modelo a usar (override)")
    pr.add_argument("--system", help="mensaje de sistema (override)")
    pr.add_argument("--var", nargs="*", default=[], metavar="CLAVE=VALOR",
                    help="sustituir {{clave}} en el texto")
    pr.add_argument("--session", help="guardar la respuesta en una sesión")
    pr.add_argument("--no-stream", action="store_true")
    pr.add_argument("--stats", action="store_true")
    pr.set_defaults(func=cmd_prompt)

    cd = sub.add_parser("code", help="review/tests/explain/commit con modelos locales")
    cd.add_argument("action", choices=list(CODE_ACTIONS))
    cd.add_argument("paths", nargs="*", help="archivos, texto o ruta del repo en commit")
    cd.add_argument("-m", "--model")
    cd.add_argument("--system")
    cd.add_argument("--from", "--start", dest="start", type=int, default=1,
                    help="primera línea a leer (default 1)")
    cd.add_argument("--lines", type=int, default=120, help="líneas por archivo o diff")
    cd.add_argument("--stdin", action="store_true", help="leer el código de stdin")
    cd.add_argument("--ctx", type=int, default=2048)
    cd.add_argument("--max-tokens", type=int)
    cd.add_argument("--no-schema", action="store_true", help="salida libre sin JSON")
    cd.add_argument("--dry-run", action="store_true", help="mostrar el prompt sin llamar al modelo")
    cd.add_argument("--no-stream", action="store_true")
    cd.add_argument("--stats", action="store_true")
    cd.add_argument("--session")
    cd.set_defaults(func=cmd_code)

    ag = sub.add_parser("agent", help="REPL de agente con herramientas y confirmaciones")
    ag.add_argument("-m", "--model", default="nemotron3:gpu",
                    help="modelo del bucle (default: nemotron3:gpu)")
    ag.add_argument("--root", default=".", help="carpeta de trabajo (default: la actual)")
    ag.add_argument("--session", help="guardar/recuperar la conversación")
    ag.add_argument("--new-session", action="store_true", help="empezar de cero")
    ag.add_argument("--steps", type=int, default=8, help="máximo de herramientas por petición")
    ag.add_argument("--history", type=int, default=20, help="mensajes conservados en memoria")
    ag.add_argument("--ctx", type=int, default=2048)
    ag.add_argument("--max-tokens", type=int, default=512)
    ag.add_argument("-t", "--temperature", type=float, default=0.2)
    ag.add_argument("--bash-timeout", type=int, default=60,
                    help="segundos por comando de la herramienta bash")
    ag.add_argument("--yes", action="store_true",
                    help="no pedir confirmación para write/edit/bash")
    ag.add_argument("--allow-dangerous", action="store_true",
                    help="permitir comandos peligrosos (sigue pidiendo confirmación)")
    ag.add_argument("--no-schema", action="store_true",
                    help="parsear JSON del texto cuando el proveedor no admite format")
    ag.set_defaults(func=cmd_agent)

    a = sub.add_parser("act", help="ejecutar acción en lenguaje natural")
    a.add_argument("text", nargs="*")
    a.add_argument("-m", "--model", default=None,
                   help="por defecto: gemma3:q4 si el catálogo es corto, nemotron3:gpu si es largo o --free")
    a.add_argument("--dry-run", action="store_true")
    a.add_argument("--list", action="store_true")
    a.add_argument("--free", action="store_true",
                   help="el modelo propone cualquier comando; se pide confirmación")
    a.add_argument("--yes", action="store_true", help="confirmar automáticamente en modo libre")
    a.add_argument("--allow-dangerous", action="store_true",
                   help="permitir comandos marcados como peligrosos (sigue pidiendo confirmación)")
    a.add_argument("--catalog", help="ruta a otro catálogo JSON")
    a.set_defaults(func=cmd_act)

    u = sub.add_parser("run", aliases=["do"], help="enrutador: decide entre act, rag y chat")
    u.add_argument("text", nargs="+")
    u.add_argument("--intent", choices=["act", "rag", "chat"], help="saltar la clasificación")
    u.add_argument("-m", "--model", default=None, help="modelo para chat o respuesta RAG")
    u.add_argument("--act-model", default=None, help="modelo para la acción")
    u.add_argument("-k", "--top-k", dest="k", type=int, default=3)
    u.add_argument("--dry-run", action="store_true")
    u.add_argument("--no-stream", action="store_true")
    u.add_argument("--stats", action="store_true")
    u.add_argument("--system", help="mensaje de sistema solo para chat")
    u.add_argument("--catalog", help="catálogo alternativo para act")
    u.set_defaults(func=cmd_run)

    b = sub.add_parser("bench", help="medir tokens/s")
    b.add_argument("models", nargs="*")
    b.add_argument("--tokens", type=int, default=30)
    b.add_argument("--history", nargs="?", const=15, type=int, metavar="N",
                   help="mostrar los últimos N resultados guardados")
    b.add_argument("--no-save", action="store_true", help="no añadir estos resultados al historial")
    b.set_defaults(func=cmd_bench)

    s = sub.add_parser("status", help="estado de servicio, GPU y sensores")
    s.set_defaults(func=cmd_status)

    w = sub.add_parser("watch", help="monitor en vivo (Ctrl+C para salir)")
    w.add_argument("--interval", type=float, default=2.0, help="segundos entre muestras")
    w.add_argument("--count", type=int, default=0, help="número de muestras (0 = infinito)")
    w.add_argument("--no-clear", dest="clear", action="store_false",
                   help="no limpiar la pantalla entre muestras")
    w.set_defaults(clear=True, func=cmd_watch)

    d = sub.add_parser("doctor", help="comprobar toda la configuración")
    d.set_defaults(func=cmd_doctor)

    dbg = sub.add_parser("debug", help="ver el registro de errores y eventos")
    dbg.add_argument("-n", "--lines", type=int, default=15)
    dbg.add_argument("--full", action="store_true", help="mostrar tracebacks completos")
    dbg.add_argument("--clear", action="store_true", help="borrar el registro")
    dbg.set_defaults(func=cmd_debug)

    l = sub.add_parser("list", help="modelos instalados y su uso recomendado")
    l.set_defaults(func=cmd_list)

    sh_ = sub.add_parser("show", help="detalle de un modelo")
    sh_.add_argument("model")
    sh_.set_defaults(func=cmd_show)

    st = sub.add_parser("stop", help="descargar un modelo de memoria")
    st.add_argument("model", nargs="?")
    st.add_argument("--all", action="store_true", help="descargar todos los modelos cargados")
    st.set_defaults(func=cmd_stop)

    im = sub.add_parser("import", help="importar un GGUF local")
    im.add_argument("tag")
    im.add_argument("gguf")
    im.add_argument("--gpu", type=int, default=27)
    im.add_argument("--force", action="store_true")
    im.set_defaults(func=cmd_import)

    rm = sub.add_parser("rm", help="borrar un tag")
    rm.add_argument("model")
    rm.add_argument("--yes", action="store_true")
    rm.set_defaults(func=cmd_rm)

    e = sub.add_parser("embed", help="generar embeddings")
    e.add_argument("text", nargs="*")
    e.add_argument("--file")
    e.add_argument("-m", "--model", default="qwen:gpu")
    e.add_argument("--preview", type=int, default=5)
    e.set_defaults(func=cmd_embed)

    r = sub.add_parser("rag", help="indexar archivos y preguntar sobre ellos")
    rs = r.add_subparsers(dest="rag_cmd", required=True)

    ri = rs.add_parser("index", help="indexar archivos o directorios")
    ri.add_argument("paths", nargs="+")
    ri.add_argument("--embed-model", default="qwen:gpu")
    ri.add_argument("--chunk-chars", type=int, default=900)
    ri.add_argument("--overlap", type=int, default=150)
    ri.add_argument("--batch", type=int, default=32)
    ri.add_argument("--max-files", type=int, default=120)
    ri.add_argument("--max-mb", type=float, default=2)
    ri.set_defaults(func=cmd_rag_index)

    rq = rs.add_parser("ask", help="recuperar fragmentos y responder")
    rq.add_argument("question", nargs="+")
    rq.add_argument("-m", "--model", default="nemotron3:gpu",
                    help="nemotron3:gpu sintetiza mejor; gemma3:q4 tiende a copiar fragmentos")
    rq.add_argument("--embed-model", default="qwen:gpu")
    rq.add_argument("-k", "--top-k", dest="k", type=int, default=3)
    rq.add_argument("--min-score", type=float, default=0.0)
    rq.add_argument("--ctx", type=int, default=2048)
    rq.add_argument("--max-tokens", type=int, default=384)
    rq.add_argument("--context-chars", type=int, default=3600)
    rq.add_argument("--no-stream", action="store_true")
    rq.set_defaults(func=cmd_rag_ask)

    rl = rs.add_parser("list", help="ver el índice actual")
    rl.set_defaults(func=cmd_rag_list)

    rc = rs.add_parser("clear", help="borrar el índice")
    rc.add_argument("--yes", action="store_true")
    rc.set_defaults(func=cmd_rag_clear)

    cf = sub.add_parser("config", help="configuración efectiva y atajos")
    cf.set_defaults(func=cmd_config)

    return p


CODE_ACTIONS = {
    "review": {
        "model": "nemotron3:gpu",
        "max_tokens": 640,
        "system": (
            "Eres un revisor de código experimentado. Sé concreto y breve. "
            "No inventes líneas ni archivos que no aparezcan en el código mostrado."
        ),
        "template": (
            "Revisa el código siguiente. Busca bugs, vulnerabilidades, problemas de "
            "concurrencia y código muerto. Devuelve solo los hallazgos relevantes con "
            "una sugerencia de corrección.\n\nCÓDIGO:\n{code}"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "problemas": {"type": "array", "items": {
                    "type": "object",
                    "properties": {
                        "archivo": {"type": "string"},
                        "linea": {"type": "string"},
                        "severidad": {"type": "string",
                                      "enum": ["alta", "media", "baja"]},
                        "sugerencia": {"type": "string"},
                    },
                    "required": ["severidad", "sugerencia"],
                }},
                "resumen": {"type": "string"},
            },
            "required": ["problemas", "resumen"],
        },
    },
    "tests": {
        "model": "gemma3:q4",
        "max_tokens": 640,
        "system": "Escribes tests unitarios claros y minimalistas en pytest.",
        "template": (
            "Escribe tests unitarios en pytest para el código siguiente. Usa "
            "monkeypatch/mock si hay dependencias externas. Incluye casos límite.\n\n"
            "CÓDIGO:\n{code}"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "tests": {"type": "string"},
                "casos": {"type": "array", "items": {"type": "string"}},
                "notas": {"type": "string"},
            },
            "required": ["tests"],
        },
    },
    "explain": {
        "model": "gemma3:q4",
        "max_tokens": 512,
        "system": "Explicas código con palabras sencillas y sin rodeos.",
        "template": (
            "Explica qué hace el código siguiente, para qué sirve y en qué orden "
            "se ejecuta. Después propón 3 mejoras concretas.\n\nCÓDIGO:\n{code}"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "explicacion": {"type": "string"},
                "mejoras": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["explicacion", "mejoras"],
        },
    },
    "commit": {
        "model": "nemotron3:gpu",
        "max_tokens": 256,
        "system": "Redactas mensajes de commit en formato Conventional Commits.",
        "template": (
            "Redacta UN mensaje de commit para este diff. Usa tipo, alcance opcional, "
            "un imperativo corto en español y un cuerpo solo si aporta contexto.\n\n"
            "DIFF:\n{code}"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "tipo": {"type": "string",
                         "enum": ["feat", "fix", "docs", "refactor", "test",
                                  "chore", "perf"]},
                "alcance": {"type": "string"},
                "mensaje": {"type": "string"},
                "cuerpo": {"type": "string"},
            },
            "required": ["tipo", "mensaje"],
        },
    },
}


def cmd_code(args):
    if NO_SCHEMA and not args.no_schema:
        raise SystemExit(
            f"{PROVIDER} no soporta JSON Schema en esta ruta. Añade --no-schema "
            "o usa 'ocode' (local) para salida estructurada."
        )
    info = CODE_ACTIONS[args.action]
    chunks = []

    if args.action == "commit":
        repo = args.paths[0] if args.paths else "/home/onion/ollama-local-kit"
        target = shlex.quote(str(Path(repo).expanduser()))
        diff = sh(f"git -C {target} diff --unified=1", timeout=30)
        stat = sh(f"git -C {target} diff --stat", timeout=30)
        if not diff.strip():
            raise SystemExit(f"No hay cambios sin confirmar en {repo}")
        selected = diff.splitlines()[:args.lines]
        text = "git diff --stat:\n" + stat + "\n\ngit diff:\n" + "\n".join(selected)
        chunks.append((f"git diff de {repo}", text))
    else:
        if args.stdin:
            chunks.append(("stdin", sys.stdin.read()))
        literal = []
        for raw in args.paths:
            p = Path(raw).expanduser()
            if not p.is_file():
                literal.append(raw)
                continue
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            start = max(args.start - 1, 0)
            selected = lines[start:start + args.lines]
            end = start + len(selected)
            label = f"{p} (líneas {start + 1}-{end} de {len(lines)})"
            if end < len(lines):
                label += f" — {len(lines) - end} líneas fuera; usa --start/--lines u orag"
            chunks.append((label, "\n".join(selected)))
        if literal:
            chunks.append(("texto", " ".join(literal)))
        if not chunks:
            raise SystemExit(
                "Indica un archivo, texto entre comillas o --stdin. "
                "Ej: ocli code review ~/ollama-cli.py"
            )

    code = "\n\n".join(f"--- {label} ---\n{text}" for label, text in chunks)
    prompt = info["template"].format(code=code)
    nlines = sum(text.count("\n") + 1 for _, text in chunks)
    model = args.model or info["model"]
    print(f"[code] {args.action} | {len(chunks)} bloque(s) | {nlines} líneas | "
          f"{len(code)} caracteres | modelo {model}")
    if len(code) > 7000:
        print(f"Aviso: ~{len(code) // 4} tokens de código; el contexto es "
              f"{args.ctx}. Reduce --lines o usa orag index.", file=sys.stderr)

    if args.dry_run:
        print("\n--- SYSTEM ---\n" + (args.system or info["system"]))
        print("\n--- PROMPT ---\n" + prompt)
        print("\n--- FORMAT ---\n" + json.dumps(
            None if args.no_schema else info["schema"], ensure_ascii=False, indent=2))
        return

    ns = argparse.Namespace(
        prompt=[prompt],
        model=model,
        system=args.system or info["system"],
        temperature=0.2,
        ctx=args.ctx,
        max_tokens=args.max_tokens or info["max_tokens"],
        think=False,
        no_stream=args.no_stream,
        stats=args.stats,
        format=None if args.no_schema else json.dumps(info["schema"]),
        session=args.session,
        new_session=False,
    )
    cmd_chat(ns)


def cmd_debug(args):
    if args.clear:
        if DEBUG_LOG.exists():
            DEBUG_LOG.unlink()
            print(f"Borrado: {DEBUG_LOG}")
        else:
            print("No había log de debug.")
        return
    if not DEBUG_LOG.exists():
        print(f"Sin eventos en {DEBUG_LOG}")
        return
    rows = []
    for line in DEBUG_LOG.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows = rows[-args.lines:]
    if args.full:
        for row in rows:
            print(json.dumps(row, ensure_ascii=False, indent=2))
        return
    print(f"{'HORA':<20} {'EVENTO':<16} DETALLE")
    print("-" * 78)
    for row in rows:
        detail = " ".join(f"{k}={v}" for k, v in row.items()
                          if k not in ("time", "event"))
        print(f"{row.get('time', ''):<20} {row.get('event', ''):<16} {detail[:150]}")
    print(f"\n{len(rows)} evento(s) de {DEBUG_LOG} | --full para ver tracebacks")


def main():
    global DEBUG
    parser = build_parser()
    args = parser.parse_args()
    DEBUG = getattr(args, "debug", False) or bool(os.environ.get("OLLAMA_KIT_DEBUG"))
    if LOCAL_ONLY:
        for field in ("model", "models", "embed_model"):
            value = getattr(args, field, None)
            if isinstance(value, str):
                setattr(args, field, free_model(value))
            elif isinstance(value, list):
                setattr(args, field, [free_model(v) for v in value])
    try:
        args.func(args)
    except SystemExit:
        raise
    except Exception as exc:
        tb = traceback.format_exc()
        debug_log("error", command=getattr(args, "cmd", "?"),
                  error=str(exc), traceback=tb)
        print(f"ERROR: {exc}", file=sys.stderr)
        print(f"Detalle en {DEBUG_LOG} (ocli debug --full)", file=sys.stderr)
        if DEBUG:
            print(tb, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
