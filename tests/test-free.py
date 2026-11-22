import importlib.util
import os
import sys

CLI = "/home/onion/ollama-cli.py"

# Simula un entorno "cloud" con key y host remoto; el modo gratuito debe ignorarlos.
os.environ.update({
    "OLLAMA_KIT_LOCAL": "1",
    "OLLAMA_KIT_PROVIDER": "ollama",
    "OLLAMA_API_KEY": "clave-falsa",
    "OLLAMA_HOST": "https://ollama.com",
})

spec = importlib.util.spec_from_file_location("cli_free_probe", CLI)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

assert m.LOCAL_ONLY is True
assert m.HOST == "http://127.0.0.1:11434", m.HOST
assert m.API_KEY == "", m.API_KEY
assert m.CLOUD_MODE is False and m.EXTERNAL is False and m.NO_SCHEMA is False
assert m.free_model("glm-5.3-flash") == "nemotron3:gpu"
assert m.free_model("gemma4:31b") == "gemma4:cpu"
assert m.free_model("qwen:gpu") == "qwen:gpu"

detail = m.http_detail(402, '{"error":"Add usage credits or subscribe"}')
assert "no son gratis" in detail and "ofree" in detail, detail

# main() debe reescribir args.model antes de ejecutar el subcomando.
captured = {}


def fake_main_func(args):
    captured["model"] = args.model


sys.argv = ["ollama-cli.py", "chat", "-m", "glm-5.3-flash", "hola"]
args = m.build_parser().parse_args(sys.argv[1:])
args.func = fake_main_func


class FakeParser:
    def parse_args(self):
        return args


m.build_parser = lambda: FakeParser()
# Ejecutamos solo la lógica de main sin volumen de red.
m.main()
assert captured["model"] == "nemotron3:gpu", captured

# El header no debe llevar Authorization aunque haya key de falsa.
assert "Authorization" not in m.api_headers(), m.api_headers()

print("MODO GRATUITO: TODO OK")
