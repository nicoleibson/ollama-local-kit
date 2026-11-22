import importlib.util
import io
import contextlib
import os
import sys

CLI = "/home/onion/ollama-cli.py"

BASE_ENV = ("OLLAMA_KIT_PROVIDER", "OLLAMA_KIT_LOCAL", "OLLAMA_API_KEY",
            "NVIDIA_API_KEY", "GEMINI_API_KEY", "OLLAMA_HOST")


def load(**env):
    for key in BASE_ENV:
        os.environ.pop(key, None)
    os.environ.update(env)
    name = "cli_default_" + str(abs(hash(tuple(sorted(env.items())))))[:8]
    spec = importlib.util.spec_from_file_location(name, CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolved(mod, argv, fake_main):
    args = mod.build_parser().parse_args(argv)
    args.func = fake_main

    class FakeParser:
        def parse_args(self):
            return args

    saved = mod.build_parser
    mod.build_parser = lambda: FakeParser()
    mod.main()
    mod.build_parser = saved
    return args


def capture(mod, argv):
    result = {}
    resolved(mod, argv, lambda a: result.update(model=a.model))
    return result["model"]


NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

nv = load(OLLAMA_KIT_PROVIDER="nvidia", NVIDIA_API_KEY="k")
assert nv.PROVIDER_DEFAULT_MODELS["nvidia"] == NVIDIA_MODEL
assert capture(nv, ["chat", "hola"]) == NVIDIA_MODEL
assert capture(nv, ["ask", "hola"]) == NVIDIA_MODEL
assert capture(nv, ["chat", "-m", "nvidia/nemotron-3-ultra-550b-a55b", "hola"]) \
    == "nvidia/nemotron-3-ultra-550b-a55b"
assert capture(nv, ["agent"]) == NVIDIA_MODEL
assert capture(nv, ["rag", "ask", "hola"]) == NVIDIA_MODEL
assert capture(nv, ["code", "review", "x.py"]) == NVIDIA_MODEL
assert capture(nv, ["run", "muestra la fecha"]) == NVIDIA_MODEL
assert capture(nv, ["act", "muestra la fecha"]) == NVIDIA_MODEL
assert capture(nv, ["prompt", "use", "resumen", "texto"]) == NVIDIA_MODEL

out = io.StringIO()
with contextlib.redirect_stdout(out):
    nv.cmd_config(nv.build_parser().parse_args(["config"]))
assert f"Modelo por defecto: {NVIDIA_MODEL}" in out.getvalue(), out.getvalue()

loc = load()
assert capture(loc, ["chat", "hola"]) == "gemma3:gpu"
assert capture(loc, ["ask", "hola"]) == "gemma3:gpu"
assert capture(loc, ["agent"]) == "gemma3:gpu"
assert capture(loc, ["rag", "ask", "hola"]) == "gemma3:gpu"
assert capture(loc, ["agent", "--reasoning"]) == "nemotron3:gpu"
assert capture(loc, ["rag", "ask", "--reasoning", "hola"]) == "nemotron3:gpu"
# Los comandos que resuelven su modelo internamente no deben verse afectados.
loc2 = load()
args = loc2.build_parser().parse_args(["act", "muestra la fecha"])
assert args.model is None, args.model
args = loc2.build_parser().parse_args(["run", "muestra la fecha"])
assert args.model is None, args.model

free = load(OLLAMA_KIT_LOCAL="1", OLLAMA_API_KEY="k",
            OLLAMA_HOST="https://ollama.com")
assert capture(free, ["chat", "hola"]) == "gemma3:gpu"
assert capture(free, ["chat", "-m", "glm-5.3-flash", "hola"]) == "nemotron3:gpu"

for key in BASE_ENV:
    os.environ.pop(key, None)

print("DEFAULT NVIDIA: TODO OK")
