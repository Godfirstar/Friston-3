"""Locate an already installed main runtime; never install or change models."""
from __future__ import annotations

from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

from friston3.config import PROJECT_ROOT


_REQUIRED_APIS = (
    ("numpy", ("frombuffer",)),
    ("cv2", ("imdecode", "FaceDetectorYN.create")),
    ("serial", ("Serial",)),
    ("onnxruntime", ("InferenceSession",)),
    ("funasr_onnx", ("SeacoParaformer", "SeacoParaformer.proc_hotword")),
    ("fastembed", ("TextEmbedding",)),
    ("pypinyin", ("lazy_pinyin",)),
)
_ENV_NAMES = ("photorobot-seaco", "photorobot")
_PROBE_MARKER = "FRISTON_RUNTIME_CHECK="


def _missing_here():
    """Import required APIs without constructing a model or loading weights."""
    for name, attributes in _REQUIRED_APIS:
        try:
            module = import_module(name)
            for attribute in attributes:
                value = module
                for part in attribute.split("."):
                    value = getattr(value, part)
                if not callable(value):
                    raise ImportError(f"{name}.{attribute} is not callable")
        except Exception as error:
            # find_spec alone accepts old wrappers without SeacoParaformer
            # and installed packages whose native DLLs cannot actually load.
            detail = " ".join(str(error).split())[:220]
            return [f"{name}: {type(error).__name__}: {detail}"]
    return []


def _candidate_pythons():
    """Known environment names near the active interpreter and Conda roots."""
    windows = sys.platform == "win32"
    executable = Path(sys.executable).absolute()
    prefixes = [Path(sys.prefix), Path(sys.base_prefix),
                executable.parent if windows else executable.parent.parent]
    # These are supplied by Conda itself; users do not configure new variables.
    for key in ("CONDA_PREFIX", "_CONDA_ROOT"):
        if os.environ.get(key):
            prefixes.append(Path(os.environ[key]))
    for key in ("CONDA_EXE", "CONDA_PYTHON_EXE"):
        if os.environ.get(key):
            parent = Path(os.environ[key]).parent
            if parent.name.lower() in ("scripts", "bin", "condabin"):
                parent = parent.parent
            prefixes.append(parent)

    directories = []
    for prefix in prefixes:
        if prefix.name.lower() in _ENV_NAMES or prefix.parent.name.lower() == "envs":
            directories.append(prefix.parent)  # Current environment's siblings.
        directories.append(prefix / "envs")  # A base interpreter's envs folder.
    seen = {os.path.normcase(str(executable))}
    for name in _ENV_NAMES:
        for directory in directories:
            candidate = (directory / name / ("python.exe" if windows else "bin/python")).absolute()
            key = os.path.normcase(str(candidate))
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                yield name, candidate


def interpreter_environment(candidate):
    """Give the candidate its own executable/DLL paths in the child only."""
    environment = os.environ.copy()
    prefix = candidate.parent if sys.platform == "win32" else candidate.parent.parent
    if sys.platform == "win32":
        entries = (prefix, prefix / "Library" / "mingw-w64" / "bin",
                   prefix / "Library" / "usr" / "bin", prefix / "Library" / "bin",
                   prefix / "Scripts", prefix / "bin")
    else:
        entries = (prefix / "bin",)
    environment["PATH"] = os.pathsep.join(
        [str(entry) for entry in entries] + [environment.get("PATH", "")])
    environment["CONDA_PREFIX"] = str(prefix)
    environment["CONDA_DEFAULT_ENV"] = prefix.name
    return environment


def _wait_for_main(child):
    # Both processes inherit the same console/process group. The child handles
    # Ctrl+C itself (including entering the microphone's keyboard console).
    # subprocess.call() instead kills its child when the parent's wait raises.
    while True:
        try:
            return child.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            continue
        except KeyboardInterrupt:
            # Do not forward a second interrupt, kill, terminate or restart.
            continue


def use_installed_runtime():
    """Return child exit status if delegated, or None if this runtime is ready.

    Users historically have photorobot and photorobot-seaco. They contain the
    same application path, not alternative model pipelines. Prefer the current
    interpreter; only inspect these known sibling installations if required
    APIs cannot import. No model initialization or package installation runs.
    """
    missing = _missing_here()
    if not missing:
        return None
    probe = ("import json; from friston3.runtime_bootstrap import _missing_here; "
             "print(" + repr(_PROBE_MARKER) + " + json.dumps(_missing_here(), ensure_ascii=True))")
    rejected = []
    for name, candidate in _candidate_pythons():
        environment = interpreter_environment(candidate)
        try:
            checked = subprocess.run([str(candidate), "-B", "-c", probe],
                cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15, check=False, env=environment)
            # Native packages can print startup notices before our JSON line.
            payload = next((line[len(_PROBE_MARKER):] for line in reversed(checked.stdout.splitlines())
                            if line.startswith(_PROBE_MARKER)), None)
            problems = json.loads(payload) if payload is not None else None
            if checked.returncode != 0 or problems != []:
                detail = ("; ".join(problems) if isinstance(problems, list) and problems
                          else " ".join(checked.stderr.split())[:220] or f"探测返回 {checked.returncode}")
                rejected.append(f"{name}: {detail}")
                continue
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            rejected.append(f"{name}: {type(error).__name__}")
            continue
        print(f"[启动] 自动使用已安装的 {name} 环境运行 SeACo 主程序。", flush=True)
        child = subprocess.Popen([str(candidate), "-B", str(PROJECT_ROOT / "main.py"), *sys.argv[1:]],
                                 cwd=PROJECT_ROOT, env=environment)
        return _wait_for_main(child)
    candidates = ("；候选环境未就绪：" + "；".join(rejected[:2]) if rejected
                  else "；未找到已安装的 photorobot-seaco / photorobot 候选环境")
    raise RuntimeError(
        "现有主程序环境的依赖无法使用：" + ", ".join(missing) + candidates
        + "。请修复上面指出的依赖；语音识别固定使用 SeACo。"
    )
