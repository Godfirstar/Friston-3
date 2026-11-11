"""Explicit one-time owner-model download, separate from offline runtime.

Only the CLI setup action should call ``download_owner_models``. Importing this
module, starting dialogue, or enrolling a template never calls the network.
"""
# 本文件是显式模型准备入口：下载文件与安装Python包、构造神经网络、运行推理是不同阶段。
# 只导入此模块不会调用下面的下载函数；main的--download-owner-models分支才进入准备流程。
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

from friston3.config import MODELS_DIR, YUNET_MODEL_PATH


# 下面是当前支持的模型文件名、预期字节数与SHA256；不是运行时自动从网络决定版本。
_SFACE_NAME = "face_recognition_sface_2021dec.onnx"
_SFACE_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
_SFACE_SIZE = 38696353
_SFACE_URLS = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_recognition_sface/" + _SFACE_NAME,
    "https://huggingface.co/opencv/face_recognition_sface/resolve/main/" + _SFACE_NAME,
)
_VOICE_REPO = "iic/speech_campplus_sv_zh-cn_16k-common"
_VOICE_WEIGHT = "campplus_cn_common.bin"
_VOICE_FILES = ("config.yaml", "configuration.json", _VOICE_WEIGHT)
# Official funasr/campplus commit 0817bacbf3f87a85962e3b7bbe231ff3c606f404.
_VOICE_SHA256 = "3388cf5fd3493c9ac9c69851d8e7a8badcfb4f3dc631020c4961371646d5ada8"
_VOICE_SIZE = 28036335


def _missing_modules(specs):
    # _prepare_face/_prepare_voice调用；specs为(module导入名,pip包显示名)元组序列，返回缺失包名list。
    # 导入名与安装包名可不同，例如yaml与PyYAML；此函数只检查可发现性，不执行pip安装。
    missing = []
    for module, package in specs:
        try:
            # find_spec来自标准库importlib，检查模块能否找到；不等于真实import和模型加载已通过。
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        if not available:
            missing.append(package)
    return missing


def _digest(path):
    # _matches调用；path是Path，返回文件SHA256十六进制str，用内容识别文件是否完整。
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        # 每次lambda读取1MiB；iter(callable,哨兵)在读到空bytes b''时停止，控制峰值内存。
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(path, expected_size, expected_hash):
    # 下载/声纹包检查调用；先检查是否文件及大小，再算哈希，返回bool，不修改文件。
    # and短路：文件不存在就不继续stat/hash；OSError表示无法读取时按不匹配处理。
    try:
        return (path.is_file() and path.stat().st_size == expected_size
                and _digest(path) == expected_hash)
    except OSError:
        return False


def _download_verified(urls, destination, expected_size, expected_hash):
    """Stream into a temporary .part file; publish only a verified whole file."""
    # _prepare_face调用；urls为地址tuple，destination为Path；返回existing/downloaded或最终抛错。
    if _matches(destination, expected_size, expected_hash):
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # errors保存各下载源失败文本；enumerate从1编号，便于终端显示当前尝试第几个源。
    errors = []
    for index, url in enumerate(urls, start=1):
        # mkstemp返回已打开的整数文件描述符及路径str；.part与最终文件同目录，方便原子替换。
        descriptor, temp_name = tempfile.mkstemp(
            prefix=destination.name + ".", suffix=".part", dir=destination.parent)
        part = Path(temp_name)
        try:
            started = time.monotonic()
            # Request只构造请求对象，下面urlopen才联网；User-Agent是请求头中的客户端说明。
            request = urllib.request.Request(url, headers={"User-Agent": "Friston-3-Owner-Setup/1.0"})
            print(f"[OwnerSetup] 下载 {destination.name}（官方源 {index}/{len(urls)}）…", flush=True)
            # fdopen把描述符包装为二进制写流；双层with在正常/异常退出时关闭文件与网络响应。
            with os.fdopen(descriptor, "wb") as output:
                with urllib.request.urlopen(request, timeout=30) as response:
                    # total为已收字节数int，digest为增量哈希对象；while循环每次读一块bytes。
                    total = 0
                    digest = hashlib.sha256()
                    while True:
                        if time.monotonic() - started > 600:
                            raise TimeoutError("本次下载超过 10 分钟，请检查网络后重试")
                        block = response.read(1024 * 1024)
                        # 空bytes表示响应读完，break仅结束while，继续执行后面的完整性核对。
                        if not block:
                            break
                        total += len(block)
                        if total > expected_size:
                            raise ValueError("下载大小超过官方文件大小，未覆盖本地文件")
                        digest.update(block)
                        output.write(block)
                    # flush推出Python缓冲，fsync要求操作系统同步文件；此时仍未覆盖目标权重。
                    output.flush()
                    os.fsync(output.fileno())
            if total != expected_size or digest.hexdigest() != expected_hash:
                raise ValueError("下载文件的大小或 SHA256 与官方权重不一致")
            # 大小与哈希都符合才发布目标文件；replace替换路径，避免运行时读到半个权重。
            os.replace(part, destination)
            return "downloaded"
        except Exception as error:
            errors.append(f"源 {index}: {type(error).__name__}: {error}")
        finally:
            # finally无论下载成功、失败或return都执行；replace成功后part已不存在也允许清理。
            part.unlink(missing_ok=True)
    raise RuntimeError("；".join(errors))


def _copy_atomic(source, destination):
    # _prepare_voice把已验证的暂存文件复制到模型目录时调用；参数为两个Path，隐式返回None。
    # 先复制到同目录.part，再replace；它不负责重新校验哈希，校验在调用方整包执行。
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".part", dir=destination.parent)
    part = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_file:
            # shutil标准库按1MiB分块从输入流复制到输出流，不把整个模型文件放入内存。
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        os.replace(part, destination)
    finally:
        part.unlink(missing_ok=True)


def _voice_bundle_valid(directory):
    """Check known weights and plain configuration without importing torch."""
    # _prepare_voice调用；directory是Path，返回bool；校验权重和配置的组合，不构造CAM++网络。
    if not _matches(directory / _VOICE_WEIGHT, _VOICE_SIZE, _VOICE_SHA256):
        return False
    try:
        # yaml来自第三方PyYAML；safe_load解析配置数据，json.loads解析元数据，两者预期dict。
        import yaml
        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8-sig"))
        metadata = json.loads((directory / "configuration.json").read_text(encoding="utf-8-sig"))
        # and逐项检查模型类别、192维输出、segment级特征、16kHz前端及实际权重文件名。
        # 配置匹配并不证明依赖导入/真实音频推理成功，这些由owner_voice.py的preload/extract验证。
        return (
            isinstance(config, dict) and config.get("model") == "CAMPPlus"
            and isinstance(config.get("model_conf"), dict)
            and config["model_conf"].get("embedding_size") == 192
            and config["model_conf"].get("output_level", "segment") == "segment"
            and (config.get("frontend_conf") or {}).get("fs", 16000) == 16000
            and isinstance(metadata, dict)
            and metadata.get("file_path_metas", {}).get("init_param") == _VOICE_WEIGHT
        )
    except (ImportError, OSError, ValueError, TypeError, AttributeError):
        return False
    except Exception:
        # PyYAML parser errors have their own type; malformed local config is
        # simply an incomplete bundle, never authority to execute model code.
        return False


def _prepare_face():
    # download_owner_models选择face/all时调用；返回单组件状态dict，区分文件success与运行ready。
    target = MODELS_DIR / _SFACE_NAME
    result = {"component": "face", "path": str(target), "success": False, "ready": False}
    # missing为缺失包名list；dependency_command仅构造可显示命令，不在这里自动执行。
    missing = _missing_modules((("cv2", '"opencv-python==4.11.0.86"'),
                                ("numpy", '"numpy==1.26.4"')))
    if missing:
        result["dependency_command"] = "python -m pip install " + " ".join(missing)
    try:
        result["status"] = _download_verified(_SFACE_URLS, target, _SFACE_SIZE, _SFACE_SHA256)
        # SFace文件已核对完整才标记success；还要已有YuNet与依赖模块，ready才为True。
        result["success"] = True
        yunet_ready = YUNET_MODEL_PATH.is_file() and YUNET_MODEL_PATH.stat().st_size >= 200_000
        result["ready"] = not missing and yunet_ready
        if not yunet_ready:
            result["reason"] = f"SFace 已就绪；缺少已有摄影功能使用的 YuNet：{YUNET_MODEL_PATH}"
        elif missing:
            result["reason"] = "模型已就绪；请先安装列出的运行依赖"
        else:
            result["reason"] = "SFace 校验通过；已有 YuNet 文件和 Python 依赖可用"
    except Exception as error:
        result.update(status="error", reason=f"{type(error).__name__}: {error}")
    return result


def _prepare_voice():
    # download_owner_models选择voice/all时调用；按需准备CAM++包，返回状态dict，不运行声纹模型。
    target = MODELS_DIR / "campplus"
    result = {"component": "voice", "path": str(target), "success": False, "ready": False}
    specs = (("funasr", '"funasr==1.4.16"'), ("yaml", "PyYAML"),
             ("numpy", '"numpy==1.26.4"'))
    other_missing = _missing_modules(specs)
    torch_missing = _missing_modules((("torch", "torch"), ("torchaudio", "torchaudio")))
    missing = other_missing + torch_missing
    # valid表示现有本地包是否完整；仅不完整时还需要下载SDK依赖，否则可直接复用。
    valid = _voice_bundle_valid(target)
    download_missing = _missing_modules((("modelscope", "modelscope"), ("yaml", "PyYAML"))) if not valid else []
    # dict.fromkeys按首次出现顺序去重，list再取键；避免重复列出同一个安装包。
    packages = list(dict.fromkeys(other_missing + download_missing))
    commands = []
    # sys.version_info来自标准库sys，[:2]为(主,次版本)；元组可按字典序比较版本区间。
    # 只有torch/torchaudio都缺时才建议配套组合；已有一项时提示匹配版本，不擅自升级。
    if len(torch_missing) == 2 and (3, 10) <= sys.version_info[:2] <= (3, 12):
        commands.append(
            'python -m pip install "torch==2.5.1" "torchaudio==2.5.1" '
            '--index-url https://download.pytorch.org/whl/cpu')
    elif torch_missing:
        if len(torch_missing) == 1:
            installed = "torch" if torch_missing[0] == "torchaudio" else "torchaudio"
            result["dependency_note"] = (
                f"检测到已有 {installed}，缺少 {torch_missing[0]}；"
                "请按已有包的版本及 CPU/CUDA 构建补齐配套包，不要单独安装另一项的最新版本。")
        else:
            result["dependency_note"] = (
                f"当前 Python {sys.version_info.major}.{sys.version_info.minor} 尚无 torch/torchaudio；"
                "请安装支持该 Python 版本的同版本 CPU 配套组合，或在已有 Python 3.10–3.12 运行环境启用声纹。")
    if packages:
        commands.append("python -m pip install " + " ".join(packages))
    if commands:
        result["dependency_commands"] = commands
        result["dependency_command"] = commands[0]
    if download_missing:
        # 无下载依赖时直接报告，return后不会进入真正联网代码。
        result.update(status="missing_dependency", reason="下载所需依赖未安装；执行给出的命令后重试")
        return result
    try:
        if valid:
            result["status"] = "existing"
        else:
            # 第三方ModelScope SDK，直到明确需要下载时才导入并调用snapshot_download。
            from modelscope.hub.snapshot_download import snapshot_download
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"[OwnerSetup] 从 ModelScope 下载 {_VOICE_REPO}（仅三个必要文件）…", flush=True)
            # SDK owns its network retries and hash checks. Stage its cache and
            # partial transfers outside the final model directory, then remove
            # that temporary tree even when interrupted or download fails.
            with tempfile.TemporaryDirectory(prefix=".owner-campplus-", dir=target.parent) as stage:
                # TemporaryDirectory上下文提供目录字符串stage，退出自动删除暂存目录；Path再转路径对象。
                staged = Path(stage) / "snapshot"
                # allow_file_pattern只请求配置与权重三个文件；这是下载，不是FunASR模型实例化。
                snapshot_download(
                    model_id=_VOICE_REPO,
                    local_dir=str(staged),
                    allow_file_pattern=list(_VOICE_FILES),
                )
                if not _voice_bundle_valid(staged):
                    raise ValueError(
                        "下载的 CAM++ 配置或 SHA256 未通过校验；保留了现有模型。"
                        "请确认使用 iic/speech_campplus_sv_zh-cn_16k-common 原版权重。")
                for name in _VOICE_FILES:
                    # 整包先验证，再逐个原子复制到正式目录，目标文件不会出现半写状态。
                    _copy_atomic(staged / name, target / name)
            result["status"] = "downloaded"
        result.update(success=True, ready=not missing,
                      reason="CAM++ 权重与配置校验通过" if not missing else "模型已就绪；请先安装列出的运行依赖")
    except Exception as error:
        result.update(status="error", reason=f"{type(error).__name__}: {error}")
    return result


def download_owner_models(selection="all"):
    """Download selected official models only after an explicit setup command.

    ``success`` describes the requested model files. ``runtime_ready`` also
    requires existing YuNet and installed module names; actual imports/model
    loading are checked later by the runtime. No dependency installation runs.
    """
    # 调用方main.main的显式--download-owner-models分支；输入face/voice/all，返回总结果dict。
    # str/strip/lower把选项规范成小写文本；非法值抛ValueError，不猜用户希望下载的内容。
    selection = str(selection).strip().lower()
    if selection not in ("face", "voice", "all"):
        raise ValueError("模型下载选项只能是 face、voice 或 all")
    print("[OwnerSetup] 现在显式联网准备模型；之后对话与身份识别使用本地文件。", flush=True)
    # results为各组件结果list[dict]；循环元组中的prepare是函数对象，不是已经执行后的结果。
    results = []
    for component, prepare in (("face", _prepare_face), ("voice", _prepare_voice)):
        if selection not in ("all", component):
            continue
        # 此处()才调用_prepare_face或_prepare_voice；列表只包含用户选中的组件结果。
        result = prepare()
        results.append(result)
        print(f"[OwnerSetup] {component}: {result['status']}；{result.get('reason', '')}", flush=True)
        print(f"  路径：{result['path']}", flush=True)
        for command in result.get("dependency_commands", [result.get("dependency_command")]):
            if command:
                print(f"  安装依赖：{command}", flush=True)
        if result.get("dependency_note"):
            print(f"  依赖说明：{result['dependency_note']}", flush=True)
    # 双层生成器依次遍历组件及其命令列表，去掉None/空串，再按顺序去重为list[str]。
    commands = list(dict.fromkeys(
        command for item in results
        for command in item.get("dependency_commands", [item.get("dependency_command")])
        if command))
    # all要求所有已选择组件满足条件；文件成功和运行准备分别汇总，调用方可明确显示差别。
    return {"success": all(item["success"] for item in results),
            "runtime_ready": all(item["ready"] for item in results),
            "selection": selection, "results": results,
            "dependency_commands": commands}
