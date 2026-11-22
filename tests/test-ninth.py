import importlib.util
import json
import os
import sys

CLI = "/home/onion/ollama-cli.py"


class FakeResponse:
    def __init__(self, data=b"", lines=None, status=200):
        self._data = data
        self._lines = lines or []
        self.status = status

    def read(self):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self._lines)


def make_module(**env):
    for key in ("OLLAMA_KIT_PROVIDER", "OLLAMA_API_KEY", "NVIDIA_API_KEY",
                "GEMINI_API_KEY", "OLLAMA_HOST"):
        os.environ.pop(key, None)
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location("cli_probe", CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install_fake(mod, seen):
    def fake(req, timeout=None):
        seen["url"] = req.full_url
        seen["headers"] = dict(req.header_items())
        seen["body"] = json.loads(req.data.decode()) if req.data else None
        url = req.full_url
        if url.endswith("/models") or url.endswith("/api/tags"):
            return FakeResponse(b'{"data":[{"id":"modelo-x"}]}')
        if url.endswith("/embeddings") or url.endswith("/api/embed"):
            return FakeResponse(b'{"data":[{"embedding":[0.1,0.2]}]}')
        if url.endswith("/chat/completions"):
            body = seen["body"] or {}
            if body.get("stream"):
                lines = [
                    b'data: {"choices":[{"delta":{"content":"Hol"}}]}\n\n',
                    b'data: {"choices":[{"delta":{"content":"a"}}],"usage":{"completion_tokens":2,"prompt_tokens":5}}\n\n',
                    b"data: [DONE]\n\n",
                ]
                return FakeResponse(b"", lines)
            return FakeResponse(
                b'{"choices":[{"message":{"content":"Respuesta"}}],'
                b'"usage":{"completion_tokens":3,"prompt_tokens":7}}'
            )
        if url.endswith("/api/chat"):
            lines = [
                b'{"message":{"content":"Hol"}}\n',
                b'{"message":{"content":"a"},"done":true,"eval_count":2}\n',
            ]
            return FakeResponse(b'{"message":{"content":"Respuesta"},"eval_count":3}', lines)
        raise AssertionError("URL inesperada: " + url)

    mod.urllib.request.urlopen = fake


def expect_exit(fn, needle):
    try:
        fn()
    except SystemExit as exc:
        assert needle in str(exc), (needle, str(exc))
        return str(exc)
    raise AssertionError("Se esperaba SystemExit con: " + needle)


cases = [
    ("local", {}, "ollama", "http://127.0.0.1:11434", False, False, None),
    ("cloud", {"OLLAMA_API_KEY": "k-cloud"}, "ollama", "https://ollama.com", True, True, "Bearer k-cloud"),
    ("nvidia", {"OLLAMA_KIT_PROVIDER": "nvidia", "NVIDIA_API_KEY": "k-nv"},
     "nvidia", "https://integrate.api.nvidia.com/v1", True, True, "Bearer k-nv"),
    ("gemini", {"OLLAMA_KIT_PROVIDER": "gemini", "GEMINI_API_KEY": "k-ge"},
     "gemini", "https://generativelanguage.googleapis.com/v1beta/openai", True, False, "Bearer k-ge"),
]

for name, env, provider, host, external, no_schema, auth in cases:
    mod = make_module(**env)
    seen = {}
    install_fake(mod, seen)
    assert mod.PROVIDER == provider, (name, mod.PROVIDER)
    assert mod.HOST == host, (name, mod.HOST)
    assert mod.EXTERNAL == external, (name, mod.EXTERNAL)
    assert mod.NO_SCHEMA == no_schema, (name, mod.NO_SCHEMA)
    assert seen == {}, name

    # Catálogo y normalización de respuesta.
    tags = mod.api("/api/tags")
    assert seen["url"] == host + ("/models" if external else "/api/tags"), seen
    assert seen["headers"].get("Authorization") == auth, seen
    if external:
        assert tags["models"][0]["name"] == "modelo-x", tags

    payload = {
        "model": "m", "stream": False,
        "messages": [{"role": "user", "content": "hola"}],
        "options": {"temperature": 0.2, "num_predict": 40},
        "think": True,
        "format": {"type": "object"},
    }
    answer = mod.api("/api/chat", payload)
    if external:
        assert seen["url"].endswith("/chat/completions"), seen
        assert answer["message"]["content"] == "Respuesta", answer
        assert answer["eval_count"] == 3, answer
        if provider == "nvidia":
            assert seen["body"]["reasoning_effort"] == "medium", seen["body"]
            assert "response_format" not in seen["body"], seen["body"]
        if provider == "gemini":
            assert seen["body"]["response_format"]["type"] == "json_schema", seen["body"]
        if provider == "ollama":
            assert "response_format" not in seen["body"], seen["body"]
    else:
        assert answer["message"]["content"] == "Respuesta", answer

    # Streaming SSE traducido a forma Ollama.
    stream_payload = dict(payload, stream=True)
    chunks = list(mod.api_stream("/api/chat", stream_payload))
    assert chunks[0]["message"]["content"] == "Hol", chunks
    assert chunks[1]["message"]["content"] == "a", chunks
    assert chunks[-1]["done"] is True, chunks
    if external:
        assert chunks[-1]["eval_count"] == 2, chunks
        assert chunks[-1]["prompt_eval_count"] == 5, chunks

    # Embeddings: Gemini sí, NVIDIA lo bloquea sin llamar a la red.
    if provider == "nvidia":
        msg = expect_exit(lambda: mod.api("/api/embed", {"model": "m", "input": "x"}),
                          "no expone /embeddings")
        assert "qwen:gpu" in msg, msg
    else:
        emb = mod.api("/api/embed", {"model": "m", "input": "x"})
        expected = "/embeddings" if external else "/api/embed"
        assert seen["url"].endswith(expected), seen
        if external:
            assert emb["embeddings"] == [[0.1, 0.2]], emb

    # Guards de act/code según JSON Schema.
    class Arg:
        no_schema = False
        action = "explain"
        paths = ["x"]

    if no_schema:
        expect_exit(lambda: mod.cmd_act(Arg()), "JSON Schema")
        expect_exit(lambda: mod.cmd_code(Arg()), "--no-schema")
        Arg.no_schema = True
        assert mod.NO_SCHEMA and mod.cmd_code.__name__ == "cmd_code"

    print(f"OK {name}: PROVIDER={provider} EXTERNAL={external} NO_SCHEMA={no_schema}")

print("PRUEBAS SIN RED: TODO OK")
