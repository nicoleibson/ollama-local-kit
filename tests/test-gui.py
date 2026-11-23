#!/usr/bin/env python3
"""Prueba de la GUI sin inferencias ni subprocess reales."""
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

GUI = Path(__file__).resolve().parents[1] / "ollama-kit-gui.py"
spec = importlib.util.spec_from_file_location("kit_gui", GUI)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# 1. Entorno local forzado.
with patch.dict(os.environ, {
    "OLLAMA_HOST": "https://ollama.com",
    "OLLAMA_API_KEY": "secreto",
    "NVIDIA_API_KEY": "secreto",
    "GEMINI_API_KEY": "secreto",
    "OLLAMA_KIT_PROVIDER": "gemini",
}, clear=False):
    env = mod.local_env()
assert env["OLLAMA_KIT_LOCAL"] == "1", env
assert env["OLLAMA_KIT_TASK_MODE"] == "1", env
for key in mod.CLEAR_ENV:
    assert key not in env, key
assert env["HOME"] == os.environ["HOME"]
print("ENTORNO LOCAL: OK")

# 2. argv de la CLI.
argv = mod.build_cli_argv("status", "-n", 20)
assert argv[:2] == [sys.executable, str(mod.CLI_PATH)], argv
assert argv[2:] == ["status", "-n", "20"], argv
print("ARGV CLI: OK")

# 3. Parseo del inventario.
LIST = """TAG                  DESCRIPCIÓN                        RENDIMIENTO  USO
----------------------------------------------------------------------------------------
gemma3:gpu           Gemma 3 1B Q6_K                    7,8 tok/s    principal: más calidad
nemotron3:gpu        Nemotron-H 4B IQ2_M                2,6 tok/s    razonamiento complejo
qwen:gpu             Qwen3 0.6B Base F16                embedding    solo /api/embed
"""
rows = mod.parse_model_rows(LIST)
assert len(rows) == 3, rows
assert rows[0][0] == "gemma3:gpu" and rows[0][2] == "7,8 tok/s", rows[0]
assert rows[-1][3] == "solo /api/embed", rows[-1]
print("PARSEO MODELOS: OK")

# 4. Catálogo permitido y catálogo completo.
with tempfile.TemporaryDirectory() as tmp:
    catalog = Path(tmp) / "commands.json"
    catalog.write_text(json.dumps({
        "_comment": "x",
        "date": {"cmd": "date", "desc": "fecha"},
        "ollama_bench": {"cmd": "ocli bench", "desc": "inferencia"},
        "git_log": {"cmd": "git -C /home/onion/ollama-local-kit log --oneline -5",
                    "desc": "commits"},
    }))
    with patch.object(mod, "CATALOG_PATH", catalog):
        entries = mod.catalog_entries()
        names = mod.all_catalog_names()
    assert [e["key"] for e in entries] == ["date", "git_log"], entries
    assert names == ["date", "git_log", "ollama_bench"], names
print("CATÁLOGOS: OK")

# 5. Comandos de chat, flujo, RAG y acciones.
chat = mod.build_chat_argv("gemma3:gpu", "hola", "proyecto", True, "breve", True)
assert "--no-stream" in chat and "--session" in chat, chat
assert chat[chat.index("--session") + 1] == "proyecto", chat
assert "--new-session" in chat and "--stats" in chat, chat
assert chat[-1] == "hola", chat

flow = mod.build_flow_argv("classify", "/tmp/x.md", labels="rojo, azul", limit=3,
                           out="informe", restart=True, model="gemma3:gpu")
assert flow[2:4] == ["flow", "classify"], flow
assert flow[flow.index("--labels") + 1:flow.index("--labels") + 3] == ["rojo", "azul"], flow
assert "--restart" in flow and "--out" in flow, flow
assert not any("nvidia" in part.lower() or "gemini" in part.lower() for part in flow), flow

rag = mod.build_rag_ask_argv("qué dice", "nemotron3:gpu", k=4, min_score=0.2,
                             context_chars=2000)
assert rag[2:4] == ["rag", "ask"] and "--no-stream" in rag, rag
assert rag[rag.index("-k") + 1] == "4" and rag[rag.index("--min-score") + 1] == "0.2", rag

act = mod.build_act_argv("muestra la fecha", dry_run=True)
assert "--dry-run" in act and "--free" not in act, act
assert "--yes" not in act and "--allow-dangerous" not in act, act

run = mod.build_run_argv("hola", "chat", dry_run=True)
assert run[run.index("--intent") + 1] == "chat" and "--no-stream" in run, run

code = mod.build_code_argv("review", ["/tmp/x.py"], start=10, lines=20, dry_run=True)
assert code[code.index("--from") + 1] == "10", code
assert code[code.index("--lines") + 1] == "20" and "--dry-run" in code, code

agent = mod.build_agent_argv("di LISTO", "/tmp", "gemma3:gpu", "trabajo", 3)
assert agent[2:4] == ["agent", "--prompt"] and agent[4] == "di LISTO", agent
assert "--steps" in agent and agent[agent.index("--steps") + 1] == "3", agent
assert "--yes" not in agent and "--allow-dangerous" not in agent, agent
assert "--reasoning" not in agent
assert "--reasoning" in mod.build_agent_argv("di LISTO", reasoning=True)

prompt_add = mod.build_prompt_argv("add", "resumen", "Resume {{args}}", "texto",
                                   "gemma3:gpu")
assert prompt_add[2:4] == ["prompt", "add"] and "--text" in prompt_add, prompt_add
prompt_use = mod.build_prompt_argv("use", "resumen", prompt_args=["hola"],
                                   variables=["clave=valor"])
assert "--no-stream" in prompt_use and "hola" in prompt_use, prompt_use
assert prompt_use[prompt_use.index("--var") + 1] == "clave=valor", prompt_use

assert mod.parse_act_choice("modelo: x | comando: date -> date (4 tokens)") == "date"
assert mod.parse_act_choice("sin comando") == ""
print("ARGV CHAT/FLUJO/RAG/ACCIONES/AGENTE/PROMPTS: OK")

# 6. Instanciación real de Tkinter y simulación de acciones.
calls = {"cli": [], "run": []}

def fake_cli(args, timeout=180):
    args = list(args)
    calls["cli"].append(args)
    if args and args[0] == "list":
        return LIST
    if args and args[0] == "config":
        return "PROVEEDOR: ollama\nMODO: LOCAL\n"
    if args[:2] == ["act", "--list"]:
        return "Catálogo: /home/onion/ollama-commands.json\n  date   date\n"
    return "SALIDA SIMULADA: " + " ".join(map(str, args)) + "\n"

def fake_run(argv=None, command=None, timeout=180):
    calls["run"].append({"argv": list(argv) if argv else None, "command": command})
    if command == "date":
        return "fecha simulada\n"
    if argv and "act" in argv and "--dry-run" in argv:
        return "modelo: gemma3:gpu | comando: date -> date (4 tokens)\n(dry-run)\n"
    return "SALIDA SIMULADA LOCAL\n"

def pump(app, seconds=0.7):
    end = time.time() + seconds
    while time.time() < end or app.worker.busy:
        app.update()
        time.sleep(0.01)

answers = iter([])
def fake_yesno(title, message, **kwargs):
    return next(answers, True)

def fake_info(*args, **kwargs):
    return None

fake_session_rows = lambda: [{
    "name": "prueba", "msgs": 2, "model": "gemma3:gpu",
    "updated": "2026-11-22 16:00:00",
}]
fake_flow_rows = lambda: [{
    "name": "informe", "mode": "map", "model": "gemma3:gpu", "done": 1,
    "prompt": "Resume", "created": "2026-11-22 16:00:00", "job": "/tmp/informe",
}]

fake_prompts = {
    "resumen": {"text": "Resume {{args}}", "desc": "resumen breve", "model": "gemma3:gpu"},
    "clasifica": {"text": "Clasifica {{args}}", "desc": "etiquetas", "model": ""},
}
fake_bench = [
    {"time": "2026-11-22 16:00", "model": "gemma3:gpu", "tokens": 30,
     "seconds": 4.0, "tps": 7.5, "vram": 1.6},
    {"time": "2026-11-22 16:01", "model": "nemotron3:gpu", "tokens": 30,
     "seconds": 10.0, "tps": 3.0, "vram": 2.1},
]

with patch.object(mod, "cli_text", fake_cli), \
     patch.object(mod, "run_process_text", fake_run), \
     patch.object(mod, "session_rows", fake_session_rows), \
     patch.object(mod, "flow_rows", fake_flow_rows), \
     patch.object(mod, "load_prompts", lambda: fake_prompts), \
     patch.object(mod, "bench_rows", lambda limit=15: fake_bench), \
     patch.object(mod.messagebox, "askyesno", fake_yesno), \
     patch.object(mod.messagebox, "showinfo", fake_info), \
     patch.object(mod.messagebox, "showerror", fake_info):
    app = mod.KitGUI()
    app.withdraw()
    pump(app, 2.0)

    assert len(app.notebook.tabs()) == 12, len(app.notebook.tabs())
    assert len(app.acciones_tab.inner.tabs()) == 3
    assert len(app.prompts_tab.tree.get_children()) == 2
    assert len(app.bench_tab.tree.get_children()) == 2
    assert app.model_names == ["gemma3:gpu", "nemotron3:gpu", "qwen:gpu"], app.model_names
    assert app.chat_tab.model_var.get() == "gemma3:gpu"
    assert len(app.modelos_tab.tree.get_children()) == 3
    assert len(app.sesiones_tab.tree.get_children()) == 1
    assert len(app.flujos_tab.tree.get_children()) == 1
    assert "MODO: LOCAL" in app.resumen_tab.config_out.text.get("1.0", "end")
    assert len(app.acciones_tab.act_catalog["values"]) == 53

    # Chat: usa el modelo local y --no-stream.
    app.chat_tab.prompt.insert("1.0", "hola gui")
    app.chat_tab.send()
    pump(app)
    assert any(c["argv"] and "chat" in c["argv"] and "--no-stream" in c["argv"]
               and c["argv"][-1] == "hola gui" for c in calls["run"]), calls["run"]

    # RAG: pregunta local sin inferencia real.
    app.rag_tab.question_var.set("qué dice el manual")
    app.rag_tab.ask()
    pump(app)
    assert any(c["argv"] and "rag" in c["argv"] and "ask" in c["argv"]
               and "--no-stream" in c["argv"] for c in calls["run"])

    # Comandos: cancelar no ejecuta; aceptar sí ejecuta la entrada permitida.
    item = app.comandos_tab.tree.get_children()[0]
    app.comandos_tab.tree.focus(item)
    app.comandos_tab.tree.selection_set(item)
    entry = app.comandos_tab.selected()
    assert entry is not None
    answers = iter([False])
    before = len(calls["run"])
    app.comandos_tab.run()
    pump(app, 0.2)
    assert len(calls["run"]) == before, "cancelación no respetada"

    answers = iter([True])
    app.comandos_tab.run()
    pump(app)
    assert any(c["command"] == entry["cmd"] for c in calls["run"]), entry["cmd"]

    # Borrar sesión: confirmación y comando exacto.
    item = app.sesiones_tab.tree.get_children()[0]
    app.sesiones_tab.tree.focus(item)
    app.sesiones_tab.tree.selection_set(item)
    answers = iter([True])
    app.sesiones_tab.clear()
    pump(app)
    assert ["session", "clear", "prueba", "--yes"] in calls["cli"], calls["cli"]

    # Act: simular, validar lista blanca y ejecutar solo la entrada segura.
    app.acciones_tab.act_text.set("muestra la fecha")
    app.acciones_tab.act_model.set("automático")
    app.acciones_tab.act_simulate()
    pump(app)
    assert app.acciones_tab.last_action_key == "date", app.acciones_tab.last_action_key
    answers = iter([True])
    app.acciones_tab.act_execute()
    pump(app)
    assert any(c["command"] == "date" for c in calls["run"])

    # Run: automático no ejecuta acciones por seguridad.
    app.acciones_tab.run_text.set("muestra la fecha")
    app.acciones_tab.run_intent.set("automático")
    before = len(calls["run"])
    app.acciones_tab.run_execute()
    pump(app, 0.2)
    assert len(calls["run"]) == before, "run automático ejecutó una acción"

    # Code: preparar prompt sin llamar al modelo de verdad.
    app.acciones_tab.code_action.set("review")
    app.acciones_tab.code_path.set("/home/onion/ollama-cli.py")
    app.acciones_tab.code_prepare()
    pump(app)
    assert any(c["argv"] and "code" in c["argv"] and "--dry-run" in c["argv"]
               for c in calls["run"])

    # Agente: tarea única, sin --yes ni --allow-dangerous.
    app.agente_tab.prompt.set("di LISTO")
    app.agente_tab.root_var.set("/tmp")
    app.agente_tab.steps_var.set("3")
    app.agente_tab.simulate()
    assert "--prompt" in app.agente_tab.output.text.get("1.0", "end")
    answers = iter([True])
    app.agente_tab.execute()
    pump(app)
    agent_call = [c for c in calls["run"] if c["argv"] and "agent" in c["argv"]][-1]
    assert "--prompt" in agent_call["argv"] and "--yes" not in agent_call["argv"]
    assert "--allow-dangerous" not in agent_call["argv"], agent_call

    # Prompts: guardar y usar con confirmación.
    app.prompts_tab.name_var.set("nuevo")
    app.prompts_tab.text.insert("1.0", "Resume {{args}}")
    app.prompts_tab.add()
    pump(app)
    assert any(c["argv"] and c["argv"][2:] == ["prompt", "add", "nuevo",
                                               "--text", "Resume {{args}}"]
               for c in calls["run"]), calls["run"]
    app.prompts_tab.name_var.set("resumen")
    app.prompts_tab.args_var.set("texto clave=valor")
    answers = iter([True])
    app.prompts_tab.use()
    pump(app)
    assert any(c["argv"] and c["argv"][2:4] == ["prompt", "use"]
               and "--no-stream" in c["argv"] for c in calls["run"])

    # Bench: medir locales con confirmación y refrescar el historial.
    answers = iter([True])
    app.bench_tab.measure()
    pump(app)
    assert any(c["argv"] and c["argv"][2] == "bench" for c in calls["run"]), calls["run"]
    app.bench_tab.refresh()
    assert len(app.bench_tab.tree.get_children()) == 2

    # Precisión global: la casilla alcanza a todas las pestañas por entorno.
    app.precise_var.set(True)
    app.toggle_precise()
    assert mod.PRECISE_MODE is True
    assert mod.local_env()["OLLAMA_KIT_PRECISE"] == "1"
    assert "perfil preciso" in app.status_var.get()
    assert "perfil preciso" in app.profile_var.get()
    app.precise_var.set(False)
    app.toggle_precise()
    assert "OLLAMA_KIT_PRECISE" not in mod.local_env()
    assert "perfil tareas" in app.status_var.get()

    app.destroy()
print("TKINTER Y BOTONES: OK")
print("GUI: TODO OK")
