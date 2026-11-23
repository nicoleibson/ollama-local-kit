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
ENV_KEYS = ("OLLAMA_KIT_PROVIDER", "OLLAMA_KIT_LOCAL", "OLLAMA_API_KEY",
            "NVIDIA_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST",
            "OLLAMA_FLOW_DIR", "OLLAMA_RAG_DIR")


def load():
    for key in ENV_KEYS:
        os.environ.pop(key, None)
    name = "cli_flow_probe_" + str(abs(hash(object())))[:8]
    spec = importlib.util.spec_from_file_location(name, CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.FLOW_DIR = tempdir("oflow-")
    return mod


def tempdir(prefix):
    """Crea el temporal de la prueba y lo borra al salir, aunque falle."""
    path = Path(tempfile.mkdtemp(prefix=prefix, dir="/tmp/opencode"))
    atexit.register(shutil.rmtree, path, True)
    return path


m = load()

root = tempdir("oflow-docs-")
doc = root / "contrato.md"
doc.write_text(
    "La cláusula de rescisión PALABRA_UNO permite terminar el contrato.\n\n"
    "El precio del producto X PALABRA_DOS es cien euros.\n\n"
    "La confidencialidad PALABRA_TRES obliga a no divulgar.\n\n"
    "Otra cláusula de rescisión PALABRA_CUATRO aparece al final.\n",
    encoding="utf-8",
)

CHUNKS = m.flow_collect([doc], 100, 20)
assert len(CHUNKS) == 4, [(c["index"], c["text"]) for c in CHUNKS]
for chunk in CHUNKS:
    assert "\n\n" not in chunk["text"], chunk
    assert any(chunk["text"].startswith(p) for p in
               ("La cláusula", "El precio", "La confidencialidad", "Otra cláusula")), chunk
assert all(len(c["id"]) == 12 for c in CHUNKS)
print("OK fragmentación por párrafos (4 fragmentos, IDs estables)")


def make_args(argv):
    args = m.build_parser().parse_args(argv)
    if getattr(args, "flow_cmd", None) is not None:
        args.model = "gemma3:gpu"
    return args


def run(argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        m.cmd_flow(make_args(argv))
    return out.getvalue()


# --- map + reanudación -------------------------------------------------------
calls = []


def fake_api(path, payload=None, **kwargs):
    calls.append({"path": path, "payload": payload, "kwargs": kwargs})
    return {"message": {"content": "RESUMEN-" + str(len(calls))}, "eval_count": 1}


m.api = fake_api

base = ["flow", "map", str(doc), "--prompt", "Resume breve",
        "--chunk-chars", "100", "--overlap", "20", "--out", "mapa"]
out = run(base + ["--limit", "2"])
job = m.FLOW_DIR / "mapa"
rows = m.flow_rows(job)
assert len(rows) == 2, rows
assert "Reanudando" not in out
assert len(calls) == 2, len(calls)
assert (job / "report.md").exists()
assert "RESUMEN-1" in (job / "report.md").read_text()

out = run(base)
rows = m.flow_rows(job)
assert len(rows) == 4, rows
assert "Reanudando" in out and "2 fragmentos ya procesados" in out, out
assert len(calls) == 4, len(calls)

sample = calls[0]["payload"]
assert sample["stream"] is False and "think" not in sample
assert sample["options"]["temperature"] == 0.0
assert sample["options"]["num_ctx"] == 2048
assert sample["options"]["num_predict"] == 256
assert "RESUMEN" not in sample["messages"][-1]["content"]
assert "PALABRA_UNO" in sample["messages"][-1]["content"]
print("OK map: estado en disco, reanudación y payload sin memoria")

# Un cambio de configuración no debe mezclarse con el estado anterior.
try:
    run(["flow", "map", str(doc), "--prompt", "OTRA TAREA", "--out", "mapa"])
except SystemExit as exc:
    assert "Usa --restart" in str(exc), str(exc)
else:
    raise AssertionError("Debe rechazar un estado con otra firma")

out = run(base + ["--restart", "--limit", "1"])
assert len(m.flow_rows(job)) == 1
assert "Reanudando" not in out
print("OK map: firma de configuración y --restart")

# --- slide ------------------------------------------------------------------
calls.clear()
base = ["flow", "slide", str(doc), "--prompt", "Continúa el resumen",
        "--chunk-chars", "100", "--overlap", "20", "--out", "deslizante",
        "--carry", "60"]
run(base + ["--limit", "2"])
carry_path = m.FLOW_DIR / "deslizante" / "carry.txt"
carry = carry_path.read_text()
assert 0 < len(carry) <= 60, repr(carry)
assert "RESUMEN-2" in carry and "RESUMEN-1" in carry, carry
assert "RESUMEN PARCIAL" in calls[1]["payload"]["messages"][-1]["content"]
assert "PALABRA_DOS" in calls[1]["payload"]["messages"][-1]["content"]
assert len(m.flow_rows(m.FLOW_DIR / "deslizante")) == 2
print("OK slide: carry acotado y realimentado al siguiente fragmento")

# --- classify ---------------------------------------------------------------
calls.clear()


def fake_classify(path, payload=None, **kwargs):
    calls.append({"path": path, "payload": payload})
    return {"message": {"content": json.dumps({
        "matches": [{"label": "rescision", "evidence": "cláusula de rescisión"}],
    })}, "eval_count": 1}


m.api = fake_classify
base = ["flow", "classify", str(doc), "--labels", "rescision,confidencialidad",
        "--chunk-chars", "100", "--overlap", "20", "--out", "clasifica"]
run(base)
job = m.FLOW_DIR / "clasifica"
assert len(m.flow_rows(job)) == 4
schema = calls[0]["payload"]["format"]
assert schema["required"] == ["matches"]
labels = schema["properties"]["matches"]["items"]["properties"]["label"]["enum"]
assert labels == ["rescision", "confidencialidad"], labels
report = (job / "report.md").read_text()
assert "## rescision (4)" in report and "## confidencialidad (0)" in report, report
assert "(sin coincidencias)" in report
print("OK classify: JSON Schema con enum e informe agrupado")

m.api = lambda path, payload=None, **kwargs: {
    "message": {"content": "esto no es JSON"}}
run(base + ["--restart"])
assert "Fragmentos sin JSON válido" in (job / "report.md").read_text()
assert all("raw" in r["output"] for r in m.flow_rows(job))
print("OK classify: degrade sin JSON")

# --- ask (RAG efímero, sin índice global) -----------------------------------
def fake_embed(texts, model):
    return [[1.0, 0.0] if "rescisión" in t.lower() else [0.0, 1.0]
            for t in texts]


def fake_answer(path, payload=None, **kwargs):
    user = payload["messages"][-1]["content"]
    marker = "PALABRA_UNO" if "PALABRA_UNO" in user else "PALABRA_CUATRO"
    return {"message": {"content": "Sí, aparece en " + marker}}


m.rag_embed = fake_embed
m.api = fake_answer
rag_index = m.RAG_INDEX
before = ((rag_index.read_bytes(), rag_index.stat().st_mtime_ns)
          if rag_index.exists() else None)

out = run(["flow", "ask", "--path", str(doc), "--chunk-chars", "100",
           "--overlap", "20", "-k", "2",
           "¿hay cláusula de rescisión?"])
assert "Fuentes:" in out, out
assert "PALABRA_UNO" in out or "PALABRA_CUATRO" in out, out
assert "PALABRA_DOS" not in out, out
after = ((rag_index.read_bytes(), rag_index.stat().st_mtime_ns)
         if rag_index.exists() else None)
assert after == before, "oflow ask no debe escribir el índice global"
assert not (m.FLOW_DIR / "ask").exists(), "ask no debe crear estado"
print("OK ask: recupera por similitud y no toca el índice global")

# --- bloqueo de proveedores externos ---------------------------------------
m.EXTERNAL = True
try:
    m.cmd_flow(make_args(["flow", "map", str(doc), "--prompt", "x"]))
except SystemExit as exc:
    assert "ofree flow" in str(exc), str(exc)
else:
    raise AssertionError("oflow debe bloquear proveedores externos")
m.EXTERNAL = False
print("OK oflow rechaza proveedores externos")

# --- defaults por proveedor -------------------------------------------------
saved_argv = sys.argv
captured = {}
m.cmd_flow = lambda a: captured.update(cmd=a.flow_cmd, model=a.model)
for alias in ("flow", "oflow"):
    sys.argv = ["ollama-cli.py", alias, "map", str(doc), "--prompt", "x"]
    m.main()
    assert captured == {"cmd": "map", "model": "gemma3:gpu"}, (alias, captured)
sys.argv = saved_argv
print("OK defaults locales para flow y oflow")

for key in ENV_KEYS:
    os.environ.pop(key, None)
print("FLUJOS CONTEXTO CORTO: TODO OK")
