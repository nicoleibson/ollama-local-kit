#!/usr/bin/env python3
"""Ventana Tkinter local para ver, ejecutar y chatear con el kit Ollama.

No abre servidor ni navegador. Todos los subprocess de la CLI se lanzan con
OLLAMA_KIT_LOCAL=1 y sin variables de nube/API key, de modo que no se puedan
gastar créditos por accidente. Las acciones que escriben o descargan modelos
piden confirmación. No se lanza ninguna inferencia al abrir la ventana.
"""
from __future__ import annotations

import json
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

HOME = Path.home()
CLI_PATH = Path(os.environ.get("OLLAMA_KIT_CLI", str(HOME / "ollama-cli.py")))
if not CLI_PATH.is_file():
    CLI_PATH = HOME / "ollama-local-kit" / "ollama-cli.py"
CATALOG_PATH = Path(os.environ.get("OLLAMA_COMMANDS", str(HOME / "ollama-commands.json")))
SESSIONS_DIR = Path(os.environ.get("OLLAMA_SESSIONS_DIR", str(HOME / ".ollama-sessions")))
FLOW_DIR = Path(os.environ.get("OLLAMA_FLOW_DIR", str(HOME / ".ollama-flow")))
DEBUG_LOG = HOME / ".ollama-debug.log"
PROMPTS_PATH = HOME / ".ollama-prompts.json"
BENCH_HISTORY = HOME / ".ollama-bench-history.jsonl"
DEFAULT_MODEL = "gemma3:gpu"
REASONING_MODEL = "nemotron3:gpu"

# Se eliminan antes de llamar a la CLI. OLLAMA_KIT_LOCAL basta para que el CLI
# use el servidor de esta máquina aunque existan keys guardadas en ~/.ollama-*.
CLEAR_ENV = (
    "OLLAMA_HOST",
    "OLLAMA_API_KEY",
    "NVIDIA_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "OLLAMA_KIT_PROVIDER",
    "OPENAI_BASE_URL",
)

# Solo entradas de solo lectura del catálogo. Quedan fuera bench, reindexado,
# review con modelo, borrados, stop y copias de seguridad.
SAFE_CATALOG_KEYS = frozenset({
    "date", "whoami", "pwd", "uname", "hostname", "os", "uptime", "free",
    "meminfo", "df", "df_inode", "ls", "lsblk", "lscpu", "sensors", "temp",
    "ip", "ports", "proc_cpu", "logs", "ollama_list", "ollama_ps",
    "ollama_status", "gpu", "governor", "sys_info", "cpu_info", "mem_info",
    "disk_info", "gpu_info", "net_info", "proc_info", "logs_tail",
    "ollama_report", "ollama_models", "ollama_doctor", "big_files",
    "systemd_failed", "py_compile", "shell_check", "json_check", "run_tests",
    "git_status", "git_diff", "git_log", "kit_doctor", "debug_log",
})


# ---------------------------------------------------------------------------
# Helpers comprobables sin interfaz
# ---------------------------------------------------------------------------

def local_env(extra: dict | None = None) -> dict:
    """Devuelve el entorno de los subprocess, forzando modo local."""
    env = os.environ.copy()
    for key in CLEAR_ENV:
        env.pop(key, None)
    env["OLLAMA_KIT_LOCAL"] = "1"
    env.setdefault("OLLAMA_KIT_TASK_MODE", "1")
    if extra:
        env.update(extra)
    return env


def build_cli_argv(*args: object) -> list[str]:
    """Argumentos para ejecutar la CLI con el mismo intérprete que la GUI."""
    return [sys.executable, str(CLI_PATH)] + [str(a) for a in args]


def pretty_command(command: list[str] | str) -> str:
    if isinstance(command, str):
        return command
    return " ".join(shlex.quote(part) for part in command)


def format_process(proc: subprocess.CompletedProcess, timeout: int = 0) -> str:
    chunks: list[str] = []
    if proc.stdout:
        chunks.append(proc.stdout.rstrip())
    if proc.stderr:
        chunks.append("[stderr]\n" + proc.stderr.rstrip())
    if timeout and proc.returncode == 124:
        chunks.append(f"TIMEOUT tras {timeout} s")
    elif proc.returncode:
        chunks.append(f"[código de salida {proc.returncode}]")
    if not chunks:
        chunks.append("(sin salida)")
    return "\n".join(chunks) + "\n"


def run_process_text(argv: list[str] | None = None, command: str | None = None,
                     timeout: int = 180) -> str:
    """Ejecuta argv o una orden de catálogo, siempre con entorno local."""
    if argv is None and command is None:
        raise ValueError("Falta argv o command")
    kwargs = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "env": local_env(),
        "cwd": str(HOME),
    }
    target: list[str] | str
    if command is not None:
        target = command
        kwargs.update(shell=True, executable="/bin/bash")
    else:
        target = argv or []
    try:
        proc = subprocess.run(target, **kwargs)
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", "replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", "replace")
        fake = subprocess.CompletedProcess(target, 124, stdout, stderr)
        return format_process(fake, timeout)
    return format_process(proc, timeout)


def cli_text(args: list[str], timeout: int = 180) -> str:
    return run_process_text(build_cli_argv(*args), timeout=timeout)


def parse_model_rows(text: str) -> list[list[str]]:
    """Convierte la salida de `cli list` en filas [tag, desc, vel, uso]."""
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("TAG", "---")):
            continue
        parts = re.split(r"\s{2,}", stripped, maxsplit=3)
        if not parts or ":" not in parts[0]:
            continue
        parts += [""] * (4 - len(parts))
        rows.append(parts[:4])
    return rows


def catalog_entries() -> list[dict[str, str]]:
    """Entradas permitidas del catálogo, ordenadas por clave."""
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    entries = []
    for key, value in data.items():
        if key.startswith("_") or key not in SAFE_CATALOG_KEYS or not isinstance(value, dict):
            continue
        command = str(value.get("cmd") or value.get("script") or "").strip()
        if not command:
            continue
        entries.append({
            "key": key,
            "cmd": command,
            "desc": str(value.get("desc", "")),
        })
    return sorted(entries, key=lambda item: item["key"])


def session_rows() -> list[dict[str, object]]:
    rows = []
    for path in SESSIONS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        rows.append({
            "name": str(data.get("name") or path.stem),
            "msgs": len(data.get("messages", [])),
            "model": str(data.get("model") or "?"),
            "updated": str(data.get("updated") or "?"),
        })
    return sorted(rows, key=lambda row: str(row["updated"]), reverse=True)


def flow_rows() -> list[dict[str, object]]:
    rows = []
    if not FLOW_DIR.is_dir():
        return rows
    for job in sorted((p for p in FLOW_DIR.iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True):
        state_path = job / "state.json"
        if not state_path.is_file():
            continue
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        signature = state.get("signature") if isinstance(state, dict) else {}
        signature = signature if isinstance(signature, dict) else {}
        results = job / "results.jsonl"
        done = 0
        if results.is_file():
            done = sum(1 for line in results.read_text(encoding="utf-8").splitlines()
                       if line.strip())
        prompt = str(signature.get("prompt") or "")
        if not prompt and signature.get("labels"):
            prompt = "etiquetas: " + ", ".join(map(str, signature["labels"]))
        rows.append({
            "name": job.name,
            "mode": str(state.get("mode", "?")),
            "model": str(signature.get("model") or "?"),
            "done": done,
            "prompt": prompt[:80],
            "created": str(state.get("created") or "?"),
            "job": str(job),
        })
    return rows


def build_chat_argv(model: str, prompt: str, session: str = "",
                    new_session: bool = False, system: str = "",
                    stats: bool = False) -> list[str]:
    args = ["chat", "-m", model, "--no-stream"]
    if session:
        args += ["--session", session]
    if new_session:
        args.append("--new-session")
    if system:
        args += ["--system", system]
    if stats:
        args.append("--stats")
    args.append(prompt)
    return build_cli_argv(*args)


def build_flow_argv(mode: str, path: str, prompt: str = "", labels: str = "",
                    carry: int = 300, limit: int = 0, out: str = "",
                    restart: bool = False, model: str = DEFAULT_MODEL) -> list[str]:
    args = ["flow", mode, "-m", model]
    if mode == "ask":
        args += ["--path", path, prompt or "Resume este documento"]
    else:
        args.append(path)
        if mode == "classify":
            tags = [tag.strip() for tag in re.split(r"[,\s]+", labels) if tag.strip()]
            args += ["--labels", *(tags or ["general"])]
        else:
            args += ["--prompt", prompt or "Resume este fragmento"]
        if mode == "slide":
            args += ["--carry", str(carry)]
        args += ["--limit", str(limit)]
    if out:
        args += ["--out", out]
    if restart:
        args.append("--restart")
    return build_cli_argv(*args)


def build_rag_ask_argv(question: str, model: str = "nemotron3:gpu",
                       embed_model: str = "qwen:gpu", k: int = 3,
                       min_score: float = 0.0, context_chars: int = 3600,
                       max_tokens: int = 384) -> list[str]:
    args = ["rag", "ask", question, "-m", model,
            "--embed-model", embed_model, "-k", str(k),
            "--min-score", str(min_score), "--context-chars", str(context_chars),
            "--max-tokens", str(max_tokens), "--no-stream"]
    return build_cli_argv(*args)


def build_act_argv(text: str, model: str = "", dry_run: bool = False) -> list[str]:
    """Nunca expone --free, --yes ni --allow-dangerous desde la GUI."""
    args = ["act"]
    if model:
        args += ["-m", model]
    if dry_run:
        args.append("--dry-run")
    args.append(text)
    return build_cli_argv(*args)


def build_run_argv(text: str, intent: str = "", model: str = "",
                   dry_run: bool = False) -> list[str]:
    args = ["run"]
    if intent and intent != "automático":
        args += ["--intent", intent]
    if model:
        args += ["-m", model]
    if dry_run:
        args.append("--dry-run")
    args += ["--no-stream", text]
    return build_cli_argv(*args)


def build_code_argv(action: str, paths: list[str], model: str = "",
                    start: int = 1, lines: int = 120, dry_run: bool = False,
                    no_schema: bool = False, stats: bool = False) -> list[str]:
    args = ["code", action] + list(paths)
    if model:
        args += ["-m", model]
    args += ["--from", str(start), "--lines", str(lines), "--no-stream"]
    if dry_run:
        args.append("--dry-run")
    if no_schema:
        args.append("--no-schema")
    if stats:
        args.append("--stats")
    return build_cli_argv(*args)


def parse_act_choice(text: str) -> str:
    """Extrae el identificador elegido de la salida de `act --dry-run`."""
    match = re.search(r"comando:\s*([^\s|]+)", text)
    return match.group(1).strip() if match else ""


def all_catalog_names() -> list[str]:
    """Todas las claves del catálogo, incluidas las no permitidas en Comandos."""
    try:
        data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, dict):
        return []
    return sorted(key for key in data if not key.startswith("_"))


def build_agent_argv(prompt: str, root: str = "", model: str = "",
                     session: str = "", steps: int = 4,
                     reasoning: bool = False) -> list[str]:
    """Agente de una sola tarea. Nunca expone --yes ni --allow-dangerous."""
    args = ["agent", "--prompt", prompt, "--steps", str(steps)]
    if root:
        args += ["--root", root]
    if model and model != "automático":
        args += ["-m", model]
    if session:
        args += ["--session", session]
    if reasoning:
        args.append("--reasoning")
    return build_cli_argv(*args)


def build_prompt_argv(action: str, name: str = "", text: str = "",
                      desc: str = "", model: str = "",
                      prompt_args: list[str] | None = None,
                      variables: list[str] | None = None) -> list[str]:
    args = ["prompt", action]
    if name:
        args.append(name)
    if prompt_args:
        args += list(prompt_args)
    if action == "add":
        args += ["--text", text]
        if desc:
            args += ["--desc", desc]
    if action == "use":
        args.append("--no-stream")
    if model and model != "automático":
        args += ["-m", model]
    for pair in variables or []:
        if "=" in pair:
            args += ["--var", pair]
    return build_cli_argv(*args)


def load_prompts() -> dict:
    try:
        data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def bench_rows(limit: int = 15) -> list[dict]:
    try:
        lines = BENCH_HISTORY.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines[-limit:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


# ---------------------------------------------------------------------------
# Widgets reutilizables
# ---------------------------------------------------------------------------

class OutputView:
    """Área de texto con cabeceras, copia y limpieza."""

    def __init__(self, parent: tk.Widget, height: int = 12):
        self.frame = ttk.Frame(parent)
        self.text = scrolledtext.ScrolledText(
            self.frame, wrap="word", height=height,
            font=("DejaVu Sans Mono", 10), bg="#0e1317", fg="#dfe7ec",
            insertbackground="#dfe7ec", relief="flat", padx=8, pady=6,
        )
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("head", foreground="#7fd1b9",
                                font=("DejaVu Sans Mono", 10, "bold"))
        self.text.tag_configure("error", foreground="#ff9b8f")
        bar = ttk.Frame(self.frame)
        bar.pack(fill="x")
        ttk.Button(bar, text="Copiar", command=self.copy).pack(side="left", padx=(0, 4), pady=4)
        ttk.Button(bar, text="Limpiar", command=self.clear).pack(side="left", pady=4)

    def write(self, text: str, heading: str | None = None, clear: bool = False) -> None:
        self.text.configure(state="normal")
        if clear:
            self.text.delete("1.0", "end")
        if heading:
            self.text.insert("end", f"\n=== {heading} ===\n", "head")
        if text:
            tag = "error" if text.lstrip().lower().startswith(("error", "timeout")) else None
            self.text.insert("end", text if text.endswith("\n") else text + "\n", tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def copy(self) -> None:
        content = self.text.get("1.0", "end-1c")
        if content:
            self.text.master.clipboard_clear()
            self.text.master.clipboard_append(content)


def make_tree(parent: tk.Widget, columns: list[tuple[str, str, int]],
              height: int = 10) -> tuple[ttk.Frame, ttk.Treeview]:
    frame = ttk.Frame(parent)
    names = [name for name, _title, _width in columns]
    tree = ttk.Treeview(frame, columns=names, show="headings", height=height,
                        selectmode="browse")
    for name, title, width in columns:
        tree.heading(name, text=title, anchor="w")
        tree.column(name, width=width, anchor="w", stretch=(name == names[-1]))
    scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")
    tree.tag_configure("odd", background="#151d23")
    return frame, tree


# ---------------------------------------------------------------------------
# Trabajador en segundo plano
# ---------------------------------------------------------------------------

class Worker:
    def __init__(self, app: "KitGUI"):
        self.app = app
        self.queue: queue.Queue = queue.Queue()
        self.busy = False
        self._scheduled = False

    def submit(self, label: str, fn, sink: OutputView | None = None,
               on_success=None, after=None) -> bool:
        if self.busy:
            messagebox.showinfo("Ocupado", "Hay una tarea en curso. Espera a que termine.",
                                parent=self.app)
            return False
        self.busy = True
        self.app.status_var.set("Trabajando: " + label.splitlines()[0])
        self.app.configure(cursor="watch")
        if sink:
            sink.write("", heading=label)
        thread = threading.Thread(
            target=self._target,
            args=(label, fn, sink, on_success, after),
            daemon=True,
        )
        thread.start()
        self._schedule()
        return True

    def _target(self, label, fn, sink, on_success, after) -> None:
        try:
            result = fn()
        except Exception as exc:  # se muestra en la GUI, no se pierde en un hilo
            result = f"ERROR: {exc}\n"
        self.queue.put((label, result, sink, on_success, after))

    def _schedule(self) -> None:
        if not self._scheduled:
            self._scheduled = True
            self.app.after(80, self._poll)

    def _poll(self) -> None:
        self._scheduled = False
        try:
            label, result, sink, on_success, after = self.queue.get_nowait()
        except queue.Empty:
            if self.busy:
                self._schedule()
            return
        self.busy = False
        self.app.configure(cursor="")
        self.app.status_var.set("Listo · local, perfil tareas")
        if sink:
            sink.write(str(result))
        if on_success:
            self.app.after(0, lambda: on_success(result))
        if after:
            self.app.after(1, after)
        if self.busy:
            self._schedule()


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------

class BaseTab(ttk.Frame):
    def __init__(self, app: "KitGUI", title: str):
        super().__init__(app.notebook, padding=10)
        self.app = app
        app.notebook.add(self, text=title)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)


class ResumenTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Resumen")
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for text, command in (
            ("Actualizar todo", app.refresh_all),
            ("Estado", lambda: self.run(["status"], "Estado de Ollama")),
            ("Doctor", lambda: self.run(["doctor"], "Diagnóstico")),
            ("Config", lambda: self.run(["config"], "Configuración efectiva")),
            ("RAG", lambda: self.run(["rag", "list"], "Índice RAG")),
            ("Historial bench", lambda: self.run(["bench", "--history"], "Historial de benchmarks")),
        ):
            ttk.Button(top, text=text, command=command).pack(side="left", padx=(0, 6))

        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.grid(row=1, column=0, sticky="nsew")
        self.status_out = OutputView(paned, height=24)
        self.config_out = OutputView(paned, height=24)
        paned.add(self.status_out.frame, weight=3)
        paned.add(self.config_out.frame, weight=2)

    def run(self, args: list[str], title: str) -> None:
        cmd = pretty_command(build_cli_argv(*args))
        self.app.worker.submit(title + "\n" + cmd,
                               lambda: cli_text(args, timeout=90),
                               self.status_out)

    def refresh(self, after=None) -> None:
        cmd = pretty_command(build_cli_argv("status"))
        self.app.worker.submit("Estado\n" + cmd,
                               lambda: cli_text(["status"], timeout=90),
                               self.status_out,
                               after=after)

    def refresh_config(self, after=None) -> None:
        cmd = pretty_command(build_cli_argv("config"))
        self.app.worker.submit("Configuración\n" + cmd,
                               lambda: cli_text(["config"], timeout=30),
                               self.config_out, after=after)


class ModelosTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Modelos")
        self.rowconfigure(1, weight=1)
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(top, text="Actualizar", command=self.refresh).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Ver detalle", command=self.show).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Ver cargados", command=self.loaded).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Descargar modelo", command=self.stop).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Descargar todos", command=self.stop_all).pack(side="left")

        frame, self.tree = make_tree(self, [
            ("tag", "TAG", 190), ("desc", "DESCRIPCIÓN", 260),
            ("speed", "RENDIMIENTO", 130), ("use", "USO", 430),
        ], height=9)
        frame.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", lambda _e: self.show())
        self.output = OutputView(self, height=10)
        self.output.frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.rowconfigure(2, weight=1)

    def refresh(self, after=None) -> None:
        cmd = pretty_command(build_cli_argv("list"))
        self.app.worker.submit("Inventario\n" + cmd,
                               lambda: cli_text(["list"], timeout=30),
                               on_success=self._fill, after=after)

    def _fill(self, text: str) -> None:
        rows = parse_model_rows(str(text))
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(rows):
            self.tree.insert("", "end", values=row,
                             tags=("odd",) if index % 2 else ())
        self.app.model_names = [row[0] for row in rows]
        self.app.sync_model_combos()
        self.output.write(f"{len(rows)} modelos detectados.", heading="Inventario")

    def selected(self) -> str:
        item = self.tree.focus()
        if not item:
            messagebox.showinfo("Modelos", "Selecciona un modelo.", parent=self.app)
            return ""
        return str(self.tree.item(item, "values")[0])

    def show(self) -> None:
        model = self.selected()
        if not model:
            return
        args = ["show", model]
        cmd = pretty_command(build_cli_argv(*args))
        self.app.worker.submit("Detalle " + model + "\n" + cmd,
                               lambda: cli_text(args, timeout=30), self.output)

    def loaded(self) -> None:
        cmd = "ollama ps"
        self.app.worker.submit("Modelos cargados\n" + cmd,
                               lambda: run_process_text(command="ollama ps", timeout=30),
                               self.output)

    def stop(self) -> None:
        model = self.selected()
        if not model:
            return
        cmd = pretty_command(build_cli_argv("stop", model))
        if not messagebox.askyesno("Confirmar", f"¿Descargar '{model}' de memoria?\n\n{cmd}",
                                   parent=self.app):
            return
        self.app.worker.submit("Descargando " + model + "\n" + cmd,
                               lambda: cli_text(["stop", model], timeout=60), self.output)

    def stop_all(self) -> None:
        cmd = pretty_command(build_cli_argv("stop", "--all"))
        if not messagebox.askyesno("Confirmar", "¿Descargar todos los modelos cargados?\n\n" + cmd,
                                   parent=self.app):
            return
        self.app.worker.submit("Descargando todos\n" + cmd,
                               lambda: cli_text(["stop", "--all"], timeout=60), self.output)


class ChatTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Chat")
        self.rowconfigure(1, weight=1)
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Modelo:").pack(side="left")
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.model_combo = ttk.Combobox(top, textvariable=self.model_var, width=20,
                                        state="readonly")
        self.model_combo.pack(side="left", padx=(4, 12))
        app.model_combos.append(self.model_combo)

        ttk.Label(top, text="Sesión:").pack(side="left")
        self.session_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.session_var, width=18).pack(side="left", padx=(4, 12))
        ttk.Label(top, text="System:").pack(side="left")
        self.system_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.system_var, width=32).pack(side="left", padx=(4, 12))
        self.new_var = tk.BooleanVar()
        self.stats_var = tk.BooleanVar()
        ttk.Checkbutton(top, text="Nueva", variable=self.new_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(top, text="Stats", variable=self.stats_var).pack(side="left")

        self.output = OutputView(self, height=20)
        self.output.frame.grid(row=1, column=0, sticky="nsew")

        bottom = ttk.Frame(self)
        bottom.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        bottom.columnconfigure(0, weight=1)
        self.prompt = tk.Text(bottom, height=3, wrap="word", font=("DejaVu Sans", 11),
                              bg="#10181e", fg="#e6eef3", insertbackground="#e6eef3",
                              relief="flat", padx=8, pady=6)
        self.prompt.grid(row=0, column=0, sticky="ew")
        self.prompt.bind("<Control-Return>", lambda _e: self.send())
        actions = ttk.Frame(bottom)
        actions.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        ttk.Button(actions, text="Enviar", command=self.send).pack(fill="x")
        ttk.Button(actions, text="Limpiar", command=self.output.clear).pack(fill="x", pady=(4, 0))
        ttk.Label(bottom, text="Ctrl+Enter para enviar · modelo local, sin streaming",
                  foreground="#7f8c96").grid(row=1, column=0, sticky="w", pady=(4, 0))

    def send(self) -> None:
        text = self.prompt.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Chat", "Escribe una pregunta.", parent=self.app)
            return
        model = self.model_var.get() or DEFAULT_MODEL
        cmd = build_chat_argv(model, text, self.session_var.get().strip(),
                              self.new_var.get(), self.system_var.get().strip(),
                              self.stats_var.get())
        shown = pretty_command(cmd)
        self.prompt.delete("1.0", "end")
        self.output.write(f"tú> {text}\n", heading=f"{model} · {pretty_command(cmd[:-1])} …")
        self.app.worker.submit("Chat local\n" + shown,
                               lambda: run_process_text(cmd, timeout=300),
                               self.output,
                               after=self._after_chat)

    def _after_chat(self) -> None:
        if self.session_var.get().strip():
            self.app.after(50, self.app.sessions_tab.refresh)


class SesionesTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Sesiones")
        self.rowconfigure(1, weight=1)
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for text, command in (
            ("Actualizar", self.refresh),
            ("Ver mensajes", self.show),
            ("Última", self.last),
            ("Reanudar", self.resume),
            ("Borrar", self.clear),
        ):
            ttk.Button(top, text=text, command=command).pack(side="left", padx=(0, 6))

        frame, self.tree = make_tree(self, [
            ("name", "SESIÓN", 220), ("msgs", "MSGS", 70),
            ("model", "MODELO", 320), ("updated", "ACTUALIZADA", 200),
        ], height=12)
        frame.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", lambda _e: self.show())
        self.output = OutputView(self, height=10)
        self.output.frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.rowconfigure(2, weight=1)
        self.refresh()

    def refresh(self, after=None) -> None:
        rows = session_rows()
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(rows):
            self.tree.insert("", "end", values=(
                row["name"], row["msgs"], row["model"], row["updated"]),
                tags=("odd",) if index % 2 else ())
        self.output.write(f"{len(rows)} sesiones en {SESSIONS_DIR}", heading="Sesiones")
        if after:
            self.app.after(1, after)

    def selected(self) -> str:
        item = self.tree.focus()
        if not item:
            messagebox.showinfo("Sesiones", "Selecciona una sesión.", parent=self.app)
            return ""
        return str(self.tree.item(item, "values")[0])

    def show(self) -> None:
        name = self.selected()
        if not name:
            return
        args = ["session", "show", name]
        cmd = pretty_command(build_cli_argv(*args))
        self.app.worker.submit("Mensajes de " + name + "\n" + cmd,
                               lambda: cli_text(args, timeout=30), self.output)

    def last(self) -> None:
        args = ["session", "last"]
        cmd = pretty_command(build_cli_argv(*args))
        self.app.worker.submit("Última sesión\n" + cmd,
                               lambda: cli_text(args, timeout=30), self.output)

    def resume(self) -> None:
        name = self.selected()
        if not name:
            return
        args = ["session", "resume", name]
        cmd = pretty_command(build_cli_argv(*args))
        self.app.worker.submit("Reanudar " + name + "\n" + cmd,
                               lambda: cli_text(args, timeout=30), self.output)

    def clear(self) -> None:
        name = self.selected()
        if not name:
            return
        args = ["session", "clear", name, "--yes"]
        cmd = pretty_command(build_cli_argv(*args))
        if not messagebox.askyesno("Confirmar", f"¿Borrar la sesión '{name}'?\n\n{cmd}",
                                   parent=self.app):
            return
        self.app.worker.submit("Borrando " + name + "\n" + cmd,
                               lambda: cli_text(args, timeout=30), self.output,
                               after=self.refresh)


class FlujosTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Flujos")
        self.rowconfigure(3, weight=1)
        form = ttk.Frame(self)
        form.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="Modo:").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="map")
        ttk.Combobox(form, textvariable=self.mode_var, state="readonly", width=12,
                     values=["map", "slide", "classify", "ask"]).grid(row=0, column=1, sticky="w")
        ttk.Label(form, text="Modelo:").grid(row=0, column=2, sticky="w", padx=(12, 0))
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        combo = ttk.Combobox(form, textvariable=self.model_var, width=20, state="readonly")
        combo.grid(row=0, column=3, sticky="w", padx=(4, 0))
        app.model_combos.append(combo)

        ttk.Label(form, text="Archivo:").grid(row=1, column=0, sticky="w")
        self.path_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.path_var).grid(row=1, column=1, columnspan=3,
                                                         sticky="ew", pady=3)
        ttk.Button(form, text="...", width=4, command=self.pick).grid(row=1, column=3,
                                                                      sticky="e")
        ttk.Label(form, text="Prompt / pregunta:").grid(row=2, column=0, sticky="w")
        self.prompt_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.prompt_var).grid(row=2, column=1, columnspan=3,
                                                           sticky="ew", pady=3)
        ttk.Label(form, text="Etiquetas:").grid(row=3, column=0, sticky="w")
        self.labels_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.labels_var).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Label(form, text="Carry:").grid(row=3, column=2, sticky="w", padx=(12, 0))
        self.carry_var = tk.StringVar(value="300")
        ttk.Spinbox(form, from_=0, to=2000, textvariable=self.carry_var,
                    width=7).grid(row=3, column=3, sticky="w", padx=(4, 0))
        ttk.Label(form, text="Límite:").grid(row=4, column=0, sticky="w")
        self.limit_var = tk.StringVar(value="0")
        ttk.Spinbox(form, from_=0, to=1000, textvariable=self.limit_var,
                    width=7).grid(row=4, column=1, sticky="w", pady=3)
        ttk.Label(form, text="Salida:").grid(row=4, column=2, sticky="w", padx=(12, 0))
        self.out_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.out_var, width=20).grid(row=4, column=3,
                                                                  sticky="ew", padx=(4, 0))
        self.restart_var = tk.BooleanVar()
        ttk.Checkbutton(form, text="Reiniciar estado", variable=self.restart_var).grid(
            row=5, column=0, sticky="w", pady=(4, 0))
        actions = ttk.Frame(form)
        actions.grid(row=5, column=1, columnspan=3, sticky="e")
        ttk.Button(actions, text="Actualizar", command=self.refresh).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Ver informe", command=self.report).pack(side="left", padx=(0, 6))
        ttk.Button(actions, text="Abrir carpeta", command=self.open_folder).pack(side="left",
                                                                                padx=(0, 6))
        ttk.Button(actions, text="Ejecutar flujo", command=self.run).pack(side="left")

        frame, self.tree = make_tree(self, [
            ("name", "TRABAJO", 260), ("mode", "MODO", 90),
            ("model", "MODELO", 170), ("done", "FRAG.", 70),
            ("prompt", "PROMPT", 500),
        ], height=8)
        frame.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", lambda _e: self.report())
        self.output = OutputView(self, height=10)
        self.output.frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=1)
        self.refresh()

    def pick(self) -> None:
        path = filedialog.askopenfilename(initialdir=str(HOME), title="Archivo para el flujo")
        if path:
            self.path_var.set(path)

    def refresh(self, after=None) -> None:
        rows = flow_rows()
        self.tree.delete(*self.tree.get_children())
        for index, row in enumerate(rows):
            self.tree.insert("", "end", values=(
                row["name"], row["mode"], row["model"], row["done"], row["prompt"]),
                tags=("odd",) if index % 2 else ())
        self.output.write(f"{len(rows)} trabajos en {FLOW_DIR}", heading="Flujos")
        if after:
            self.app.after(1, after)

    def selected(self) -> dict[str, object] | None:
        item = self.tree.focus()
        if not item:
            messagebox.showinfo("Flujos", "Selecciona un trabajo.", parent=self.app)
            return None
        name = str(self.tree.item(item, "values")[0])
        return next((row for row in flow_rows() if row["name"] == name), None)

    def report(self) -> None:
        row = self.selected()
        if not row:
            return
        report = Path(str(row["job"])) / "report.md"
        if not report.is_file():
            self.output.write(f"Sin informe todavía: {report}", heading="Informe")
            return
        self.output.write(report.read_text(encoding="utf-8"), heading=str(report))

    def open_folder(self) -> None:
        row = self.selected()
        if not row:
            return
        subprocess.Popen(["xdg-open", str(row["job"])], env=local_env(),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def run(self) -> None:
        mode = self.mode_var.get()
        path = self.path_var.get().strip()
        if not path or not Path(path).expanduser().is_file():
            messagebox.showerror("Flujos", "Indica un archivo existente.", parent=self.app)
            return
        if mode == "classify" and not self.labels_var.get().strip():
            messagebox.showerror("Flujos", "classify necesita etiquetas.", parent=self.app)
            return
        argv = build_flow_argv(
            mode, str(Path(path).expanduser()), self.prompt_var.get().strip(),
            self.labels_var.get().strip(), int(self.carry_var.get() or 0),
            int(self.limit_var.get() or 0), self.out_var.get().strip(),
            self.restart_var.get(), self.model_var.get() or DEFAULT_MODEL,
        )
        shown = pretty_command(argv)
        if not messagebox.askyesno(
            "Confirmar flujo local",
            "Se ejecutará un flujo local con el modelo seleccionado.\n"
            "Puede tardar varios minutos y no usa créditos.\n\n" + shown,
            parent=self.app,
        ):
            return
        self.app.worker.submit("Flujo " + mode + "\n" + shown,
                               lambda: run_process_text(argv, timeout=1800),
                               self.output, after=self.refresh)


class ComandosTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Comandos")
        self.rowconfigure(1, weight=1)
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Buscar:").pack(side="left")
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_a: self._fill())
        ttk.Entry(top, textvariable=self.search_var, width=28).pack(side="left", padx=(4, 12))
        ttk.Button(top, text="Actualizar", command=self.refresh).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Ver comando", command=self.view).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Ejecutar", command=self.run).pack(side="left")
        ttk.Label(top, text="Solo entradas de solo lectura",
                  foreground="#7f8c96").pack(side="right")

        frame, self.tree = make_tree(self, [
            ("key", "CLAVE", 190), ("cmd", "COMANDO", 560), ("desc", "DESCRIPCIÓN", 420),
        ], height=14)
        frame.grid(row=1, column=0, sticky="nsew")
        self.tree.bind("<Double-1>", lambda _e: self.view())
        self.output = OutputView(self, height=9)
        self.output.frame.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        self.rowconfigure(2, weight=1)
        self.entries = catalog_entries()
        self._fill()

    def refresh(self, after=None) -> None:
        self.entries = catalog_entries()
        self._fill()
        self.output.write(f"{len(self.entries)} comandos permitidos en {CATALOG_PATH}",
                          heading="Catálogo")
        if after:
            self.app.after(1, after)

    def _fill(self) -> None:
        needle = self.search_var.get().strip().lower()
        self.tree.delete(*self.tree.get_children())
        for index, entry in enumerate(self.entries):
            haystack = " ".join(entry.values()).lower()
            if needle and needle not in haystack:
                continue
            self.tree.insert("", "end", values=(entry["key"], entry["cmd"], entry["desc"]),
                             tags=("odd",) if index % 2 else ())

    def selected(self) -> dict[str, str] | None:
        item = self.tree.focus()
        if not item:
            messagebox.showinfo("Comandos", "Selecciona un comando.", parent=self.app)
            return None
        key = str(self.tree.item(item, "values")[0])
        return next((entry for entry in self.entries if entry["key"] == key), None)

    def view(self) -> None:
        entry = self.selected()
        if not entry:
            return
        self.output.write(f"{entry['key']}: {entry['cmd']}\n{entry['desc']}",
                          heading="Comando")

    def run(self) -> None:
        entry = self.selected()
        if not entry:
            return
        if not messagebox.askyesno(
            "Confirmar comando",
            "¿Ejecutar esta entrada de solo lectura?\n\n" + entry["cmd"],
            parent=self.app,
        ):
            return
        self.app.worker.submit("Comando " + entry["key"] + "\n" + entry["cmd"],
                               lambda: run_process_text(command=entry["cmd"], timeout=180),
                               self.output)


class RegistroTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Registro")
        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for text, command in (
            ("Debug 20", lambda: self.run(["debug", "-n", "20"], "Últimos 20 eventos")),
            ("Debug completo", lambda: self.run(["debug", "--full", "-n", "40"], "Debug completo")),
            ("Historial bench", lambda: self.run(["bench", "--history"], "Historial de benchmarks")),
            ("Actualizar", self.refresh),
            ("Limpiar log", self.clear),
        ):
            ttk.Button(top, text=text, command=command).pack(side="left", padx=(0, 6))
        self.output = OutputView(self, height=26)
        self.output.frame.grid(row=1, column=0, sticky="nsew")
        self.output.write(f"Archivo: {DEBUG_LOG}", heading="Registro")

    def run(self, args: list[str], title: str) -> None:
        cmd = pretty_command(build_cli_argv(*args))
        self.app.worker.submit(title + "\n" + cmd,
                               lambda: cli_text(args, timeout=60), self.output)

    def refresh(self, after=None) -> None:
        self.run(["debug", "-n", "20"], "Últimos 20 eventos")
        if after:
            self.app.after(1, after)

    def clear(self) -> None:
        cmd = pretty_command(build_cli_argv("debug", "--clear"))
        if not messagebox.askyesno("Confirmar", "¿Borrar el registro de debug?\n\n" + cmd,
                                   parent=self.app):
            return
        self.app.worker.submit("Borrando registro\n" + cmd,
                               lambda: cli_text(["debug", "--clear"], timeout=30),
                               self.output, after=self.refresh)


class RAGTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "RAG")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        form = ttk.Frame(self)
        form.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Pregunta:").grid(row=0, column=0, sticky="w")
        self.question_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.question_var).grid(
            row=0, column=1, columnspan=4, sticky="ew", padx=(4, 8))
        ttk.Button(form, text="Preguntar", command=self.ask).grid(row=0, column=5)

        ttk.Label(form, text="Modelo:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.model_var = tk.StringVar(value="nemotron3:gpu")
        combo = ttk.Combobox(form, textvariable=self.model_var, width=20, state="readonly")
        combo.grid(row=1, column=1, sticky="w", padx=(4, 8), pady=(6, 0))
        app.model_combos.append(combo)
        ttk.Label(form, text="Embeddings:").grid(row=1, column=2, sticky="w", pady=(6, 0))
        self.embed_var = tk.StringVar(value="qwen:gpu")
        ttk.Combobox(form, textvariable=self.embed_var, state="readonly", width=16,
                     values=["qwen:gpu", "qwen35:embed"]).grid(
            row=1, column=3, sticky="w", padx=(4, 8), pady=(6, 0))
        ttk.Label(form, text="k:").grid(row=1, column=4, sticky="w", pady=(6, 0))
        self.k_var = tk.StringVar(value="3")
        ttk.Spinbox(form, from_=1, to=12, textvariable=self.k_var, width=5).grid(
            row=1, column=5, sticky="w", pady=(6, 0))
        self.min_var = tk.StringVar(value="0.0")
        self.ctx_var = tk.StringVar(value="3600")
        ttk.Label(form, text="Min score:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.min_var, width=8).grid(row=2, column=1, sticky="w", padx=(4, 8))
        ttk.Label(form, text="Contexto:").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(form, textvariable=self.ctx_var, width=8).grid(row=2, column=3, sticky="w", padx=(4, 8))

        paths_frame = ttk.LabelFrame(self, text="Archivos para el índice", padding=6)
        paths_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        paths_frame.columnconfigure(0, weight=1)
        list_frame = ttk.Frame(paths_frame)
        list_frame.grid(row=0, column=0, sticky="nsew")
        self.paths = tk.Listbox(list_frame, height=4, selectmode="extended",
                                bg="#10181e", fg="#dfe7ec", font=("DejaVu Sans Mono", 10))
        self.paths.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(list_frame, orient="vertical",
                               command=self.paths.yview)
        scroll.pack(side="right", fill="y")
        self.paths.configure(yscrollcommand=scroll.set)
        actions = ttk.Frame(paths_frame)
        actions.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        for text, command in (
            ("Añadir archivo", self.add_files),
            ("Añadir carpeta", self.add_folder),
            ("Quitar", self.remove_paths),
            ("Indexar", self.index),
            ("Ver índice", self.refresh),
            ("Limpiar índice", self.clear),
        ):
            ttk.Button(actions, text=text, command=command).pack(fill="x", pady=1)

        self.output = OutputView(self, height=12)
        self.output.frame.grid(row=2, column=0, sticky="nsew")

    def add_files(self) -> None:
        paths = filedialog.askopenfilename(initialdir=str(HOME), multiple=True,
                                           title="Archivos para indexar")
        for path in (paths or ()):
            if path not in self.paths.get(0, "end"):
                self.paths.insert("end", path)

    def add_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=str(HOME), title="Carpeta para indexar")
        if path and path not in self.paths.get(0, "end"):
            self.paths.insert("end", path)

    def remove_paths(self) -> None:
        for index in reversed(self.paths.curselection()):
            self.paths.delete(index)

    def refresh(self, after=None) -> None:
        cmd = pretty_command(build_cli_argv("rag", "list"))
        self.app.worker.submit("Índice RAG\n" + cmd,
                               lambda: cli_text(["rag", "list"], timeout=30),
                               self.output, after=after)

    def ask(self) -> None:
        question = self.question_var.get().strip()
        if not question:
            messagebox.showinfo("RAG", "Escribe una pregunta.", parent=self.app)
            return
        argv = build_rag_ask_argv(
            question, self.model_var.get() or "nemotron3:gpu",
            self.embed_var.get() or "qwen:gpu", int(self.k_var.get() or 3),
            float(self.min_var.get() or 0), int(self.ctx_var.get() or 3600),
        )
        self.app.worker.submit("Pregunta RAG\n" + pretty_command(argv),
                               lambda: run_process_text(argv, timeout=300),
                               self.output)

    def index(self) -> None:
        paths = list(self.paths.get(0, "end"))
        if not paths:
            messagebox.showinfo("RAG", "Añade al menos un archivo o carpeta.", parent=self.app)
            return
        argv = build_cli_argv("rag", "index", *paths,
                              "--embed-model", self.embed_var.get() or "qwen:gpu")
        shown = pretty_command(argv)
        if not messagebox.askyesno("Confirmar índice",
                                   "¿Reconstruir el índice RAG con estos archivos?\n\n" + shown,
                                   parent=self.app):
            return
        self.app.worker.submit("Indexando\n" + shown,
                               lambda: run_process_text(argv, timeout=1200),
                               self.output, after=self.refresh)

    def clear(self) -> None:
        argv = build_cli_argv("rag", "clear", "--yes")
        shown = pretty_command(argv)
        if not messagebox.askyesno("Confirmar", "¿Borrar el índice RAG actual?\n\n" + shown,
                                   parent=self.app):
            return
        self.app.worker.submit("Borrando índice\n" + shown,
                               lambda: cli_text(["rag", "clear", "--yes"], timeout=30),
                               self.output, after=self.refresh)


class AccionesTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Acciones")
        self.rowconfigure(1, weight=1)
        self.last_action_key = ""
        self.last_action_cmd = ""

        self.inner = ttk.Notebook(self)
        self.inner.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        self.act_frame = ttk.Frame(self.inner, padding=8)
        self.run_frame = ttk.Frame(self.inner, padding=8)
        self.code_frame = ttk.Frame(self.inner, padding=8)
        self.inner.add(self.act_frame, text="Act")
        self.inner.add(self.run_frame, text="Run")
        self.inner.add(self.code_frame, text="Code")
        self._build_act(app)
        self._build_run(app)
        self._build_code(app)

        self.output = OutputView(self, height=13)
        self.output.frame.grid(row=1, column=0, sticky="nsew")

    @staticmethod
    def _model(frame, app, row: int, column: int) -> ttk.Combobox:
        ttk.Label(frame, text="Modelo:").grid(row=row, column=column, sticky="w")
        combo = ttk.Combobox(frame, width=20, state="readonly",
                             values=["automático", DEFAULT_MODEL, "nemotron3:gpu"])
        combo.set("automático")
        combo.grid(row=row, column=column + 1, sticky="w", padx=(4, 8))
        app.optional_model_combos.append(combo)
        return combo

    def _build_act(self, app) -> None:
        frame = self.act_frame
        ttk.Label(frame, text="Instrucción:").grid(row=0, column=0, sticky="w")
        self.act_text = tk.StringVar(value="muestra la fecha")
        ttk.Entry(frame, textvariable=self.act_text, width=46).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(4, 8))
        self.act_model = self._model(frame, app, 1, 0)

        ttk.Label(frame, text="Catálogo:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.act_catalog = ttk.Combobox(frame, state="readonly", width=24,
                                        values=all_catalog_names() or ["date"])
        self.act_catalog.set("date")
        self.act_catalog.grid(row=2, column=1, sticky="w", padx=(4, 8), pady=(6, 0))
        ttk.Button(frame, text="Usar", command=self.act_use_catalog).grid(
            row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Button(frame, text="Ver catálogo", command=self.act_list).grid(
            row=2, column=3, sticky="w", pady=(6, 0))
        ttk.Button(frame, text="Simular", command=self.act_simulate).grid(
            row=3, column=1, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Ejecutar segura", command=self.act_execute).grid(
            row=3, column=2, sticky="w", pady=(8, 0))
        ttk.Label(frame, text="Simular primero: solo se ejecutan entradas de solo lectura.",
                  foreground="#7f8c96").grid(row=4, column=1, columnspan=3, sticky="w", pady=(6, 0))

    def _build_run(self, app) -> None:
        frame = self.run_frame
        ttk.Label(frame, text="Petición:").grid(row=0, column=0, sticky="w")
        self.run_text = tk.StringVar()
        ttk.Entry(frame, textvariable=self.run_text, width=50).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(4, 8))
        ttk.Label(frame, text="Intención:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.run_intent = ttk.Combobox(frame, state="readonly", width=14,
                                       values=["automático", "act", "rag", "chat"])
        self.run_intent.set("automático")
        self.run_intent.grid(row=1, column=1, sticky="w", padx=(4, 8), pady=(6, 0))
        self.run_model = self._model(frame, app, 1, 2)
        ttk.Button(frame, text="Simular", command=self.run_simulate).grid(
            row=2, column=1, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Ejecutar chat/RAG", command=self.run_execute).grid(
            row=2, column=2, sticky="w", pady=(8, 0))
        ttk.Label(frame, text="act y automático solo se simulan: usa Acciones para elegir una entrada segura.",
                  foreground="#7f8c96").grid(row=3, column=1, columnspan=3, sticky="w", pady=(6, 0))

    def _build_code(self, app) -> None:
        frame = self.code_frame
        ttk.Label(frame, text="Acción:").grid(row=0, column=0, sticky="w")
        self.code_action = ttk.Combobox(frame, state="readonly", width=14,
                                        values=["review", "tests", "explain", "commit"])
        self.code_action.set("review")
        self.code_action.grid(row=0, column=1, sticky="w", padx=(4, 8))
        self.code_model = self._model(frame, app, 0, 2)
        ttk.Label(frame, text="Archivo:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.code_path = tk.StringVar()
        ttk.Entry(frame, textvariable=self.code_path, width=46).grid(
            row=1, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=(6, 0))
        ttk.Button(frame, text="...", width=4, command=self.code_pick).grid(
            row=1, column=3, sticky="w", pady=(6, 0))
        self.code_start = tk.StringVar(value="1")
        self.code_lines = tk.StringVar(value="120")
        ttk.Label(frame, text="Desde:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(frame, from_=1, to=9999, textvariable=self.code_start,
                    width=7).grid(row=2, column=1, sticky="w", padx=(4, 8))
        ttk.Label(frame, text="Líneas:").grid(row=2, column=2, sticky="w")
        ttk.Spinbox(frame, from_=1, to=999, textvariable=self.code_lines,
                    width=7).grid(row=2, column=3, sticky="w", padx=(4, 8))
        self.code_schema_var = tk.BooleanVar()
        self.code_stats_var = tk.BooleanVar()
        ttk.Checkbutton(frame, text="Sin JSON Schema", variable=self.code_schema_var).grid(
            row=3, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(frame, text="Stats", variable=self.code_stats_var).grid(
            row=3, column=2, sticky="w", pady=(6, 0))
        ttk.Button(frame, text="Preparar prompt", command=self.code_prepare).grid(
            row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Button(frame, text="Ejecutar", command=self.code_execute).grid(
            row=4, column=2, sticky="w", pady=(8, 0))

    def act_use_catalog(self) -> None:
        key = self.act_catalog.get()
        if key:
            self.act_text.set(key)

    def act_list(self) -> None:
        self.app.worker.submit("Catálogo de acciones\n" +
                               pretty_command(build_cli_argv("act", "--list")),
                               lambda: cli_text(["act", "--list"], timeout=30),
                               self.output)

    def act_simulate(self) -> None:
        text = self.act_text.get().strip()
        if not text:
            messagebox.showinfo("Act", "Escribe una instrucción.", parent=self.app)
            return
        argv = build_act_argv(text, self.act_model.get(), dry_run=True)
        self.app.worker.submit("Simular acción\n" + pretty_command(argv),
                               lambda: run_process_text(argv, timeout=180),
                               self.output, on_success=self._act_parsed)

    def _act_parsed(self, result) -> None:
        key = parse_act_choice(str(result))
        allowed = {entry["key"]: entry["cmd"] for entry in catalog_entries()}
        self.last_action_key = key if key in allowed else ""
        self.last_action_cmd = allowed.get(key, "")
        if self.last_action_key:
            self.output.write(f"Acción permitida: {key} -> {self.last_action_cmd}\n"
                              "Puedes ejecutarla con 'Ejecutar segura'.",
                              heading="Resultado")
        elif key:
            self.output.write(f"La acción '{key}' no está en la lista blanca de la GUI. "
                              "Ejecútala desde la terminal si procede.", heading="Resultado")

    def act_execute(self) -> None:
        if not self.last_action_key:
            messagebox.showinfo("Act", "Simula primero y elige una acción permitida.",
                                parent=self.app)
            return
        if not messagebox.askyesno("Confirmar acción",
                                   "¿Ejecutar esta entrada de solo lectura?\n\n" +
                                   self.last_action_cmd, parent=self.app):
            return
        self.app.worker.submit("Acción " + self.last_action_key + "\n" + self.last_action_cmd,
                               lambda: run_process_text(command=self.last_action_cmd,
                                                        timeout=180),
                               self.output)

    def run_simulate(self) -> None:
        text = self.run_text.get().strip()
        if not text:
            messagebox.showinfo("Run", "Escribe una petición.", parent=self.app)
            return
        argv = build_run_argv(text, self.run_intent.get(), self.run_model.get(),
                              dry_run=True)
        self.app.worker.submit("Simular enrutador\n" + pretty_command(argv),
                               lambda: run_process_text(argv, timeout=180), self.output)

    def run_execute(self) -> None:
        text = self.run_text.get().strip()
        if not text:
            messagebox.showinfo("Run", "Escribe una petición.", parent=self.app)
            return
        intent = self.run_intent.get()
        if intent in ("act", "automático"):
            messagebox.showinfo(
                "Run", "El enrutador podría elegir act. Simula primero y ejecuta "
                "solo entradas permitidas desde la pestaña Acciones.", parent=self.app)
            return
        argv = build_run_argv(text, intent, self.run_model.get(), dry_run=False)
        shown = pretty_command(argv)
        if not messagebox.askyesno("Confirmar", "¿Ejecutar esta petición local?\n\n" + shown,
                                   parent=self.app):
            return
        self.app.worker.submit("Enrutador " + intent + "\n" + shown,
                               lambda: run_process_text(argv, timeout=300), self.output)

    def code_pick(self) -> None:
        path = filedialog.askopenfilename(initialdir=str(HOME), title="Archivo de código")
        if path:
            self.code_path.set(path)

    def code_argv(self, dry_run: bool) -> list[str]:
        action = self.code_action.get()
        raw = self.code_path.get().strip()
        paths = []
        if raw:
            target = Path(raw).expanduser()
            if action != "commit" and not target.exists():
                raise SystemExit(f"No existe: {target}")
            paths = [str(target)]
        return build_code_argv(
            action, paths, self.code_model.get(),
            int(self.code_start.get() or 1), int(self.code_lines.get() or 120),
            dry_run=dry_run, no_schema=self.code_schema_var.get(),
            stats=self.code_stats_var.get(),
        )

    def code_prepare(self) -> None:
        try:
            argv = self.code_argv(dry_run=True)
        except SystemExit as exc:
            messagebox.showerror("Code", str(exc), parent=self.app)
            return
        self.app.worker.submit("Preparar prompt\n" + pretty_command(argv),
                               lambda: run_process_text(argv, timeout=120), self.output)

    def code_execute(self) -> None:
        try:
            argv = self.code_argv(dry_run=False)
        except SystemExit as exc:
            messagebox.showerror("Code", str(exc), parent=self.app)
            return
        shown = pretty_command(argv)
        if not messagebox.askyesno("Confirmar", "¿Ejecutar esta acción de código?\n\n" + shown,
                                   parent=self.app):
            return
        self.app.worker.submit("Code " + self.code_action.get() + "\n" + shown,
                               lambda: run_process_text(argv, timeout=300), self.output)


class AgenteTab(BaseTab):
    """Agente de una sola tarea: sin REPL, sin --yes y sin --allow-dangerous."""

    def __init__(self, app):
        super().__init__(app, "Agente")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        form = ttk.LabelFrame(self, text="Tarea para el agente", padding=8)
        form.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Petición:").grid(row=0, column=0, sticky="w")
        self.prompt = tk.StringVar()
        ttk.Entry(form, textvariable=self.prompt).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(4, 8))
        ttk.Label(form, text="Modelo:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.model_var = tk.StringVar(value="automático")
        combo = ttk.Combobox(form, textvariable=self.model_var, width=20, state="readonly",
                             values=["automático", DEFAULT_MODEL, REASONING_MODEL])
        combo.grid(row=1, column=1, sticky="w", padx=(4, 8), pady=(6, 0))
        app.optional_model_combos.append(combo)
        self.reasoning_var = tk.BooleanVar()
        ttk.Checkbutton(form, text="Usar respaldo (--reasoning)",
                        variable=self.reasoning_var).grid(
            row=1, column=2, sticky="w", pady=(6, 0))
        ttk.Label(form, text="Pasos:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.steps_var = tk.StringVar(value="4")
        ttk.Spinbox(form, from_=1, to=12, textvariable=self.steps_var,
                    width=6).grid(row=2, column=1, sticky="w", padx=(4, 8), pady=(6, 0))
        ttk.Label(form, text="Raíz:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.root_var = tk.StringVar(value=str(HOME))
        ttk.Entry(form, textvariable=self.root_var).grid(
            row=3, column=1, sticky="ew", padx=(4, 8), pady=(6, 0))
        ttk.Button(form, text="...", width=4, command=self.pick_root).grid(
            row=3, column=2, sticky="w", pady=(6, 0))
        ttk.Label(form, text="Sesión (opcional):").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.session_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.session_var).grid(
            row=4, column=1, sticky="ew", padx=(4, 8), pady=(6, 0))

        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Button(bar, text="Simular", command=self.simulate).pack(side="left", padx=(0, 6))
        ttk.Button(bar, text="Ejecutar tarea", command=self.execute).pack(side="left")
        ttk.Label(bar, text="  ·  razonamiento visible; escrituras y bash piden confirmación en terminal",
                  foreground="#7f8c96").pack(side="left")

        self.output = OutputView(self, height=13)
        self.output.frame.grid(row=2, column=0, sticky="nsew")

    def argv(self) -> list[str]:
        return build_agent_argv(
            self.prompt.get().strip(), self.root_var.get().strip(),
            self.model_var.get(), self.session_var.get().strip(),
            int(self.steps_var.get() or 4), self.reasoning_var.get(),
        )

    def pick_root(self) -> None:
        path = filedialog.askdirectory(initialdir=self.root_var.get() or str(HOME),
                                       title="Carpeta de trabajo del agente")
        if path:
            self.root_var.set(path)

    def simulate(self) -> None:
        if not self.prompt.get().strip():
            messagebox.showinfo("Agente", "Escribe una petición.", parent=self.app)
            return
        self.output.write(pretty_command(self.argv()), heading="Comando simulado")

    def execute(self) -> None:
        if not self.prompt.get().strip():
            messagebox.showinfo("Agente", "Escribe una petición.", parent=self.app)
            return
        argv = self.argv()
        if not messagebox.askyesno("Confirmar tarea",
                                   "¿Ejecutar esta tarea local?\n\n" + pretty_command(argv),
                                   parent=self.app):
            return
        self.app.worker.submit("Agente · tarea única\n" + pretty_command(argv),
                               lambda: run_process_text(argv, timeout=300), self.output)


class PromptsTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Prompts")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        frame, self.tree = make_tree(top, [
            ("name", "NOMBRE", 140), ("model", "MODELO", 140), ("desc", "DESCRIPCIÓN", 320),
        ], height=7)
        frame.grid(row=0, column=0, sticky="nsew")
        actions = ttk.Frame(top)
        actions.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        for text, command in (
            ("Actualizar", self.refresh),
            ("Ver", self.show),
            ("Borrar", self.remove),
        ):
            ttk.Button(actions, text=text, command=command).pack(fill="x", pady=1)

        form = ttk.LabelFrame(self, text="Añadir o usar prompt", padding=8)
        form.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="Nombre:").grid(row=0, column=0, sticky="w")
        self.name_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.name_var, width=20).grid(
            row=0, column=1, sticky="w", padx=(4, 8))
        ttk.Label(form, text="Modelo:").grid(row=0, column=2, sticky="w")
        self.model_var = tk.StringVar(value="automático")
        combo = ttk.Combobox(form, textvariable=self.model_var, width=18, state="readonly",
                             values=["automático", DEFAULT_MODEL, REASONING_MODEL])
        combo.grid(row=0, column=3, sticky="w", padx=(4, 8))
        app.optional_model_combos.append(combo)
        ttk.Label(form, text="Texto:").grid(row=1, column=0, sticky="nw", pady=(6, 0))
        self.text = scrolledtext.ScrolledText(form, height=3, wrap="word",
                                              font=("DejaVu Sans Mono", 10),
                                              bg="#10181e", fg="#dfe7ec", relief="flat")
        self.text.grid(row=1, column=1, columnspan=3, sticky="ew", padx=(4, 8), pady=(6, 0))
        ttk.Label(form, text="Descripción:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.desc_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.desc_var).grid(
            row=2, column=1, sticky="ew", padx=(4, 8), pady=(6, 0))
        ttk.Label(form, text="Args / variables:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.args_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.args_var).grid(
            row=3, column=1, columnspan=2, sticky="ew", padx=(4, 8), pady=(6, 0))
        ttk.Button(form, text="Guardar", command=self.add).grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Button(form, text="Usar", command=self.use).grid(row=4, column=2, sticky="w", pady=(8, 0))

        self.output = OutputView(self, height=9)
        self.output.frame.grid(row=2, column=0, sticky="nsew")

    def refresh(self, after=None) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, (name, item) in enumerate(sorted(load_prompts().items())):
            self.tree.insert("", "end", iid=name, values=(
                name, item.get("model") or "default", item.get("desc", "")),
                tags=("odd",) if index % 2 else ())
        if after:
            after()

    def selected(self) -> str:
        selection = self.tree.selection()
        return selection[0] if selection else ""

    def show(self) -> None:
        name = self.selected()
        if not name:
            messagebox.showinfo("Prompts", "Selecciona un prompt.", parent=self.app)
            return
        item = load_prompts().get(name, {})
        self.output.write(f"Modelo: {item.get('model') or 'default'}\n\n{item.get('text', '')}",
                          heading=name)

    def add(self) -> None:
        name = self.name_var.get().strip()
        text = self.text.get("1.0", "end-1c").strip()
        if not name or not text:
            messagebox.showinfo("Prompts", "Indica nombre y texto.", parent=self.app)
            return
        argv = build_prompt_argv("add", name, text, self.desc_var.get().strip(),
                                 self.model_var.get())
        self.app.worker.submit("Guardar prompt\n" + pretty_command(argv),
                               lambda: run_process_text(argv, timeout=30),
                               self.output, after=self.refresh)

    def remove(self) -> None:
        name = self.selected()
        if not name:
            messagebox.showinfo("Prompts", "Selecciona un prompt.", parent=self.app)
            return
        argv = build_prompt_argv("rm", name)
        if not messagebox.askyesno("Confirmar", f"¿Borrar el prompt '{name}'?\n\n" +
                                   pretty_command(argv), parent=self.app):
            return
        self.app.worker.submit("Borrar prompt\n" + pretty_command(argv),
                               lambda: run_process_text(argv, timeout=30),
                               self.output, after=self.refresh)

    def use(self) -> None:
        name = self.name_var.get().strip() or self.selected()
        if not name:
            messagebox.showinfo("Prompts", "Indica qué prompt usar.", parent=self.app)
            return
        raw = self.args_var.get().split()
        prompt_args = [part for part in raw if "=" not in part]
        variables = [part for part in raw if "=" in part]
        argv = build_prompt_argv("use", name, model=self.model_var.get(),
                                 prompt_args=prompt_args, variables=variables)
        shown = pretty_command(argv)
        if not messagebox.askyesno("Confirmar", "¿Ejecutar este prompt local?\n\n" + shown,
                                   parent=self.app):
            return
        self.app.worker.submit("Usar prompt\n" + shown,
                               lambda: run_process_text(argv, timeout=300), self.output)


class BenchTab(BaseTab):
    def __init__(self, app):
        super().__init__(app, "Bench")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(top, text="Modelos:").pack(side="left")
        self.models_var = tk.StringVar(value="gemma3:gpu nemotron3:gpu")
        ttk.Entry(top, textvariable=self.models_var, width=42).pack(
            side="left", padx=(4, 8))
        ttk.Button(top, text="Medir", command=self.measure).pack(side="left", padx=(0, 6))
        ttk.Button(top, text="Actualizar historial", command=self.refresh).pack(side="left")

        frame, self.tree = make_tree(self, [
            ("time", "FECHA", 140), ("model", "MODELO", 150),
            ("tokens", "TOKENS", 80), ("seconds", "SEG", 80),
            ("tps", "TOK/S", 80), ("vram", "VRAM", 80),
        ], height=7)
        frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        chart = ttk.LabelFrame(self, text="Rendimiento (tok/s)", padding=4)
        chart.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.canvas = tk.Canvas(chart, height=140, bg="#0e1317", highlightthickness=0)
        self.canvas.pack(fill="x", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.app.after_idle(self.draw_bars))

        self.output = OutputView(self, height=6)
        self.output.frame.grid(row=3, column=0, sticky="nsew")

    def refresh(self, after=None) -> None:
        rows = bench_rows(30)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, row in enumerate(rows):
            self.tree.insert("", "end", values=(
                row.get("time", "?"), row.get("model", "?"), row.get("tokens", 0),
                row.get("seconds", 0), row.get("tps", 0), row.get("vram", 0)),
                tags=("odd",) if index % 2 else ())
        self.rows = rows
        self.draw_bars()
        self.output.write(f"{len(rows)} medición(es) en {BENCH_HISTORY}",
                          heading="Historial")
        if after:
            after()

    def draw_bars(self) -> None:
        rows = getattr(self, "rows", [])
        canvas = self.canvas
        canvas.delete("all")
        if not rows:
            canvas.create_text(10, 16, text="Sin mediciones guardadas", anchor="w",
                               fill="#7f8c96")
            return
        width = max(canvas.winfo_width(), 420)
        top = max(float(row.get("tps") or 0) for row in rows[:10]) or 1
        for index, row in enumerate(rows[:10]):
            y = 12 + index * 13
            tps = float(row.get("tps") or 0)
            model = str(row.get("model") or "?")[:22]
            canvas.create_text(6, y, text=model, anchor="w", fill="#dfe7ec",
                               font=("DejaVu Sans Mono", 9))
            bar = (width - 190) * tps / top
            canvas.create_rectangle(170, y - 5, 170 + max(bar, 1), y + 5,
                                    fill="#2b6f5e", outline="")
            canvas.create_text(178 + max(bar, 1), y, text=f"{tps:.2f}", anchor="w",
                               fill="#7fd1b9", font=("DejaVu Sans Mono", 9))

    def measure(self) -> None:
        models = self.models_var.get().split()
        if not models:
            messagebox.showinfo("Bench", "Indica al menos un modelo.", parent=self.app)
            return
        argv = build_cli_argv("bench", *models)
        shown = pretty_command(argv)
        if not messagebox.askyesno("Confirmar",
                                   "¿Medir modelos locales?\n\n" + shown, parent=self.app):
            return
        self.app.worker.submit("Medir modelos\n" + shown,
                               lambda: run_process_text(argv, timeout=600),
                               self.output, after=self.refresh)


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------

class KitGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Kit Ollama local · ver, ejecutar y chatear")
        self.geometry("1400x860")
        self.minsize(1050, 680)
        self.configure(bg="#101418")
        self.status_var = tk.StringVar(value="Listo · local, perfil tareas")
        self.model_names: list[str] = []
        self.model_combos: list[ttk.Combobox] = []
        self.optional_model_combos: list[ttk.Combobox] = []
        self._chain: list = []
        self._chain_index = 0

        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#101418")
        style.configure("TLabel", background="#101418", foreground="#dfe7ec")
        style.configure("TButton", padding=5)
        style.configure("TNotebook", background="#101418", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 6), background="#1c252c",
                        foreground="#dfe7ec")
        style.map("TNotebook.Tab", background=[("selected", "#2b6f5e")],
                  foreground=[("selected", "#ffffff")])
        style.configure("Treeview", background="#111920", fieldbackground="#111920",
                        foreground="#dfe7ec", rowheight=25)
        style.configure("Treeview.Heading", background="#1c252c", foreground="#dfe7ec")

        head = ttk.Frame(self, padding=(10, 8))
        head.pack(fill="x")
        ttk.Label(head, text="KIT OLLAMA LOCAL",
                  font=("DejaVu Sans", 16, "bold")).pack(side="left")
        ttk.Label(head, text="  ·  local, perfil tareas, razonamiento visible",
                  foreground="#7fd1b9").pack(side="left")
        ttk.Button(head, text="Actualizar todo", command=self.refresh_all).pack(side="right")
        ttk.Label(head, textvariable=self.status_var, foreground="#9fb0bb").pack(
            side="right", padx=(0, 12))

        self.worker = Worker(self)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.resumen_tab = ResumenTab(self)
        self.modelos_tab = ModelosTab(self)
        self.chat_tab = ChatTab(self)
        self.rag_tab = RAGTab(self)
        self.acciones_tab = AccionesTab(self)
        self.agente_tab = AgenteTab(self)
        self.prompts_tab = PromptsTab(self)
        self.bench_tab = BenchTab(self)
        self.sesiones_tab = SesionesTab(self)
        self.flujos_tab = FlujosTab(self)
        self.comandos_tab = ComandosTab(self)
        self.registro_tab = RegistroTab(self)

        self.after(250, self.refresh_all)

    def sync_model_combos(self) -> None:
        values = self.model_names or [DEFAULT_MODEL]
        for combo in self.model_combos:
            current = combo.get()
            combo.configure(values=values)
            if current not in values:
                combo.set(DEFAULT_MODEL if DEFAULT_MODEL in values else values[0])
        for combo in self.optional_model_combos:
            current = combo.get()
            combo.configure(values=["automático"] + values)
            if current not in ["automático"] + values:
                combo.set("automático")

    def refresh_all(self) -> None:
        if self.worker.busy:
            return
        self._chain = [
            self.resumen_tab.refresh,
            self.resumen_tab.refresh_config,
            self.modelos_tab.refresh,
            self.rag_tab.refresh,
            self.prompts_tab.refresh,
            self.bench_tab.refresh,
            self.sesiones_tab.refresh,
            self.flujos_tab.refresh,
            self.comandos_tab.refresh,
        ]
        self._chain_index = 0
        self._chain_next()

    def _chain_next(self) -> None:
        if self._chain_index >= len(self._chain):
            self.status_var.set("Listo · local, perfil tareas")
            return
        step = self._chain[self._chain_index]
        self._chain_index += 1
        step(after=self._chain_next)


def main() -> None:
    app = KitGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
