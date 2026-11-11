"""Locate an already installed main runtime; never install or change models."""
from __future__ import annotations

# 全部是Python标准库：importlib动态导入，pathlib描述路径，subprocess启动进程，sys读取当前解释器。
# 本模块由main.main按需调用；不是自动安装器，不下载模型，探测时只检查已有环境的导入与接口。
from importlib import import_module
import json
import os
from pathlib import Path
import subprocess
import sys

from friston3.config import PROJECT_ROOT


# 外层tuple每项为(模块名,所需API名字tuple)；“能找到包”还不够，要确认实际API可导入并可调用。
_REQUIRED_APIS = (
    ("numpy", ("frombuffer",)),
    ("cv2", ("imdecode", "FaceDetectorYN.create")),
    ("serial", ("Serial",)),
    ("onnxruntime", ("InferenceSession",)),
    ("funasr_onnx", ("SeacoParaformer", "SeacoParaformer.proc_hotword")),
    ("fastembed", ("TextEmbedding",)),
    ("pypinyin", ("lazy_pinyin",)),
)
# 搜索限定到已有的两个主程序环境名；_PROBE_MARKER用于从依赖启动日志中定位本程序的JSON结果行。
_ENV_NAMES = ("photorobot-seaco", "photorobot")
_PROBE_MARKER = "FRISTON_RUNTIME_CHECK="


def _missing_here():
    """Import required APIs without constructing a model or loading weights."""
    # use_installed_runtime与候选解释器探测脚本调用；返回[]表示通过，否则返回首个问题字符串列表。
    # name为模块名str，attributes为API路径tuple；此处实际import包，但不调用模型构造器。
    for name, attributes in _REQUIRED_APIS:
        try:
            # import_module来自标准库importlib，按字符串导入，module是模块对象，不是文字。
            module = import_module(name)
            for attribute in attributes:
                # value先指向模块，再按属性路径逐层前进，例如cv2 -> FaceDetectorYN -> create。
                value = module
                # split('.')把“类.方法”字符串拆成路径片段list；getattr(obj,name)按名字取属性。
                for part in attribute.split("."):
                    value = getattr(value, part)
                # callable只检查能否调用，不执行value；缺API/不可调用/导入DLL失败都会进入except。
                if not callable(value):
                    raise ImportError(f"{name}.{attribute} is not callable")
        except Exception as error:
            # find_spec alone accepts old wrappers without SeacoParaformer
            # and installed packages whose native DLLs cannot actually load.
            # str(error)取异常说明，split再join压平换行空白，[:220]仅截短诊断文本而非代码或模型输入。
            detail = " ".join(str(error).split())[:220]
            return [f"{name}: {type(error).__name__}: {detail}"]
    return []


def _candidate_pythons():
    """Known environment names near the active interpreter and Conda roots."""
    # use_installed_runtime遍历本生成器；每次yield返回(环境名str,python可执行文件Path)。
    # windows由sys.platform判断；executable来自当前sys.executable，与当前工作目录是不同概念。
    windows = sys.platform == "win32"
    executable = Path(sys.executable).absolute()
    # prefixes为候选环境根Path列表；Windows解释器常在环境根，Unix通常在根/bin下面。
    prefixes = [Path(sys.prefix), Path(sys.base_prefix),
                executable.parent if windows else executable.parent.parent]
    # These are supplied by Conda itself; users do not configure new variables.
    for key in ("CONDA_PREFIX", "_CONDA_ROOT"):
        # os.environ是进程环境变量映射；get缺键返回None，不要求用户自己额外设置变量。
        if os.environ.get(key):
            prefixes.append(Path(os.environ[key]))
    for key in ("CONDA_EXE", "CONDA_PYTHON_EXE"):
        if os.environ.get(key):
            # 这些变量可能直接指可执行文件；先取parent，再去掉Scripts/bin/condabin这层。
            parent = Path(os.environ[key]).parent
            if parent.name.lower() in ("scripts", "bin", "condabin"):
                parent = parent.parent
            prefixes.append(parent)

    # directories收集“环境名字应该放在哪个目录”列表；不是直接遍历硬盘查找所有python。
    directories = []
    for prefix in prefixes:
        if prefix.name.lower() in _ENV_NAMES or prefix.parent.name.lower() == "envs":
            directories.append(prefix.parent)  # Current environment's siblings.
        directories.append(prefix / "envs")  # A base interpreter's envs folder.
    # seen用set去重并排除当前解释器；normcase在Windows规范大小写/分隔符用于比较路径。
    seen = {os.path.normcase(str(executable))}
    for name in _ENV_NAMES:
        for directory in directories:
            # Path的/是拼接路径，条件表达式选择不同系统的可执行文件位置；absolute返回绝对Path。
            candidate = (directory / name / ("python.exe" if windows else "bin/python")).absolute()
            key = os.path.normcase(str(candidate))
            if key in seen:
                continue
            seen.add(key)
            if candidate.is_file():
                # yield暂停函数并交出一对值；调用方下一次迭代时从这里继续，不一次生成全部结果。
                yield name, candidate


def interpreter_environment(candidate):
    """Give the candidate its own executable/DLL paths in the child only."""
    # 本文件、arms/integration.py、stackchan_integration/voice_output.py启动子进程时调用。
    # candidate为目标解释器Path，返回环境变量dict；复制后修改，不影响当前os.environ。
    environment = os.environ.copy()
    # prefix是目标环境根；entries列该环境的可执行文件/DLL目录，让原生库优先找到配套版本。
    prefix = candidate.parent if sys.platform == "win32" else candidate.parent.parent
    if sys.platform == "win32":
        entries = (prefix, prefix / "Library" / "mingw-w64" / "bin",
                   prefix / "Library" / "usr" / "bin", prefix / "Library" / "bin",
                   prefix / "Scripts", prefix / "bin")
    else:
        entries = (prefix / "bin",)
    # os.pathsep是PATH列表分隔符（Windows ;，Unix :）；新目录放前面，原PATH仍保留在后面。
    environment["PATH"] = os.pathsep.join(
        [str(entry) for entry in entries] + [environment.get("PATH", "")])
    environment["CONDA_PREFIX"] = str(prefix)
    environment["CONDA_DEFAULT_ENV"] = prefix.name
    return environment


def _wait_for_main(child):
    # use_installed_runtime调用；child为subprocess.Popen对象，返回子进程退出码int。
    # wait(timeout)只等待已启动子进程，不重复启动；TimeoutExpired表示仍在运行，继续等待。
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
    # main.main在普通接待入口调用；当前环境可用返回None让main继续，转交子进程后返回其退出码。
    # 该函数会检查依赖并可能启动main子进程；阅读注释/语法检查不应直接调用它。
    # missing为问题list；空列表在if里是假，因此not missing表示当前环境满足所需接口。
    missing = _missing_here()
    if not missing:
        return None
    # probe是准备给候选Python -c执行的代码字符串；repr(marker)生成合法的带引号字符串字面量。
    # 内容固定，仅导入检查并打印带标记JSON；这里构造字符串还没启动子进程。
    probe = ("import json; from friston3.runtime_bootstrap import _missing_here; "
             "print(" + repr(_PROBE_MARKER) + " + json.dumps(_missing_here(), ensure_ascii=True))")
    # rejected收集候选失败说明list；每轮name/candidate来自上方生成器。
    rejected = []
    for name, candidate in _candidate_pythons():
        environment = interpreter_environment(candidate)
        try:
            # run同步执行一次探测，checked为CompletedProcess；参数list直接传程序，不经shell拼接。
            # -B禁止写pyc，-c执行probe；cwd指定项目目录；capture_output捕获输出，不把日志混进控制台。
            # text/encoding按UTF-8读文本；check=False不因非零退出码自动抛错，下面自己判断。
            checked = subprocess.run([str(candidate), "-B", "-c", probe],
                cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=15, check=False, env=environment)
            # Native packages can print startup notices before our JSON line.
            # stdout拆行后逆序找标记行；切片去标记前缀，next(...,None)找不到时给None而非报错。
            payload = next((line[len(_PROBE_MARKER):] for line in reversed(checked.stdout.splitlines())
                            if line.startswith(_PROBE_MARKER)), None)
            # json.loads将JSON文本解码成问题列表；缺标记的None与通过的[]严格区分。
            problems = json.loads(payload) if payload is not None else None
            if checked.returncode != 0 or problems != []:
                # detail优先使用结构化问题列表；否则读stderr摘要；还没有内容则显示退出码。
                detail = ("; ".join(problems) if isinstance(problems, list) and problems
                          else " ".join(checked.stderr.split())[:220] or f"探测返回 {checked.returncode}")
                rejected.append(f"{name}: {detail}")
                continue
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            rejected.append(f"{name}: {type(error).__name__}")
            continue
        print(f"[启动] 自动使用已安装的 {name} 环境运行 SeACo 主程序。", flush=True)
        # Popen启动正式main子进程后立即返回对象；这里没有capture_output，因此共享当前控制台。
        # sys.argv[1:]是原启动参数（排除脚本名），*把序列展开到参数list，保留用户传入选项。
        child = subprocess.Popen([str(candidate), "-B", str(PROJECT_ROOT / "main.py"), *sys.argv[1:]],
                                 cwd=PROJECT_ROOT, env=environment)
        return _wait_for_main(child)
    # 所有候选都不可用时才走到这里；最多显示两项失败摘要，然后抛异常由main的启动错误处理显示。
    candidates = ("；候选环境未就绪：" + "；".join(rejected[:2]) if rejected
                  else "；未找到已安装的 photorobot-seaco / photorobot 候选环境")
    raise RuntimeError(
        "现有主程序环境的依赖无法使用：" + ", ".join(missing) + candidates
        + "。请修复上面指出的依赖；语音识别固定使用 SeACo。"
    )
