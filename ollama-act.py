#!/usr/bin/env python3
# Ejecutor de acciones con gemma3:q4 + structured output (JSON Schema).
# El modelo genera solo 6-7 tokens: {"command":"..."}. Nada de prosa ni <think>.
import json, os, subprocess, sys, urllib.request

MODEL = "gemma3:q4"
HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
URL = HOST + "/api/chat"

ALLOWED = {
    "date": ["date"],
    "whoami": ["whoami"],
    "pwd": ["pwd"],
    "free -h": ["free", "-h"],
    "uptime": ["uptime"],
    "ls": ["ls", "-la"],
}
FORMAT = {
    "type": "object",
    "properties": {"command": {"type": "string", "enum": list(ALLOWED)}},
    "required": ["command"],
}
SYSTEM = (
    "Eres un ejecutor. Responde SOLO con JSON: {\"command\":\"...\"}. "
    "Elige el comando según esta lista: "
    "date=fecha o hora actual; whoami=usuario actual; "
    "pwd=directorio actual o dónde estoy; "
    "free -h=memoria RAM libre y usada; "
    "uptime=tiempo que lleva encendido el equipo o carga del sistema; "
    "ls=listar archivos del directorio actual."
)

def ask(text):
    payload = {
        "model": MODEL,
        "stream": False,
        "think": False,
        "format": FORMAT,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": text},
        ],
        "options": {"temperature": 0, "num_ctx": 2048, "num_predict": 16},
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    r = json.load(urllib.request.urlopen(req, timeout=120))
    return json.loads(r["message"]["content"])["command"], r["eval_count"]

def main():
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    if not args:
        print("Uso: ollama-act.py [--dry-run] \"instrucción en lenguaje natural\"")
        sys.exit(2)
    cmd, tokens = ask(" ".join(args))
    print(f"comando: {cmd}  ({tokens} tokens)")
    if "--dry-run" in sys.argv:
        sys.exit(0)
    subprocess.run(ALLOWED[cmd], check=True)

if __name__ == "__main__":
    main()
