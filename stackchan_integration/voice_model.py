"""Explicit model setup outside the project tree; no network during dialogue.

  python -m stackchan_integration.voice_model install
  python -m stackchan_integration.voice_model install --archive C:\\Downloads\\kokoro-int8-multi-lang-v1_1.tar.bz2
  python -m stackchan_integration.voice_model status

The one model comes from the official sherpa-onnx release. A local archive
can be imported before an offline event; it contains the same fixed model.
"""

from __future__ import annotations
# 这是显式模型安装/status工具；对话时的voice_input/voice_output不会调用它自动下载。
# “安装资源文件”“加载TTS引擎”“对文本运行推理”分别在本文件、local_tts构造、generate中发生。

import argparse
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request

from .local_tts import BILINGUAL_MODEL_NAME, default_bilingual_dir, kokoro_missing, model_root


MODEL_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    "kokoro-int8-multi-lang-v1_1.tar.bz2"
)


def _download(destination: Path) -> None:
    # 【调用方】install且没有提供本地archive时；输入目标归档路径，边读网络边写文件，返回None。
    print("Downloading the official bilingual voice. This is only needed once.", flush=True)
    print(MODEL_URL, flush=True)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "Friston3-Voice-Setup/1"})
    # Request描述HTTP请求；真正网络连接在下面urlopen发生，不是在构造Request时。
    maximum = 700 * 1024 * 1024
    with urllib.request.urlopen(request, timeout=45) as response, destination.open("wb") as out:
        # 同一个with管理网络响应和目标文件；wb表示写二进制；异常时两者也会自动关闭。
        expected = int(response.headers.get("Content-Length", "0") or "0")
        if expected > maximum:
            raise ValueError("Unexpected archive size; download rejected")
        received, last_report = 0, -1
        while True:
            block = response.read(1024 * 1024)
            # 每次最多读1MiB到内存，不一次性把整个模型归档加载成大bytes。
            if not block:
                break
            received += len(block)
            if received > maximum:
                raise ValueError("Unexpected archive size; download rejected")
            out.write(block)
            # 将本块bytes写到文件；received累计网络字节数，检查完整性和显示进度。
            report = received // (10 * 1024 * 1024)
            # //整数除法得到已跨过几个10MiB区间，只有区间变化才打印一次进度。
            if report != last_report:
                total = f" / {expected / 1024**2:.1f} MiB" if expected else " MiB"
                print(f"Downloaded {received / 1024**2:.1f}{total}", flush=True)
                last_report = report
        if expected and received != expected:
            raise OSError(f"Incomplete download: received {received} of {expected} bytes")
        if received == 0:
            raise OSError("Empty model download")


def _extract(archive: Path, destination: Path) -> Path:
    """Extract only regular files/directories, never links or absolute paths."""
    # 【调用方】install；读取tar归档并写入暂存目录，返回经过资源完整性检查的模型目录。
    count, unpacked = 0, 0
    with tarfile.open(archive, "r:*") as bundle:
        # tarfile是标准库；r:*自动识别压缩方式；bundle可迭代出TarInfo成员元数据。
        for member in bundle:
            count += 1
            unpacked += max(0, member.size)
            if count > 10000 or unpacked > 1500 * 1024 * 1024:
                raise ValueError("Unexpected model archive contents")
            path = PurePosixPath(member.name)
            # 归档内部路径通常用/；PurePosixPath仅解析这种路径语法，不访问实际文件系统。
            parts = path.parts
            if path.is_absolute() or ".." in parts or any(":" in p or "\\" in p for p in parts):
                raise ValueError("Unsafe path in model archive")
            target = destination.joinpath(*parts)
            # *parts把路径段tuple解包成多个参数；joinpath将它们拼到指定暂存目录下。
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError("Model archive contains a link or unsupported special file")
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)
                # 只复制普通文件内容，分块转写；不执行归档里的程序，也不加载ONNX模型。
    for folder in (destination / BILINGUAL_MODEL_NAME, destination):
        if not kokoro_missing(folder):
            return folder
    raise ValueError("Archive lacks the complete Kokoro bilingual model, voices, lexicons or data folders")


def install(archive: Path | None = None) -> Path:
    # 【调用方】main的install子命令；archive=None才下载，有Path则读取用户指定本地包。
    # 返回实际可用模型目录；已有完整资源时直接复用，避免重复大权重。
    target = (model_root() / BILINGUAL_MODEL_NAME).resolve()
    if not kokoro_missing(target):
        print(f"Bilingual voice is already installed: {target}")
        return target
    existing = default_bilingual_dir().resolve()
    if existing != target and not kokoro_missing(existing):
        # The consolidation installer owns migration and deletion. Locating a
        # model here must not move a symlink target or duplicate its weights.
        print(f"已发现完整 Kokoro 模型，无需再次下载：{existing}")
        return existing
    if target.exists():
        raise FileExistsError(f"模型目录不完整：{target}。请先完成该目录的清理，再重试安装。")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Same filesystem as target makes the final rename atomic.
    with tempfile.TemporaryDirectory(prefix=".friston_voice_", dir=target.parent) as staging:
        # 在目标同一文件系统创建临时目录；with结束自动清理暂存，最终目标不在暂存目录内。
        stage = Path(staging)
        if archive is None:
            archive = stage / "voice.tar.bz2"
            _download(archive)
        else:
            archive = archive.expanduser().resolve()
            if not archive.is_file():
                raise FileNotFoundError(f"Archive not found: {archive}")
        unpacked = stage / "unpacked"
        unpacked.mkdir()
        complete = _extract(archive, unpacked)
        # 完整解包并检查资源后才准备落位；complete是解出的模型目录Path。
        if target.exists():
            raise FileExistsError(f"Destination appeared during install: {target}")
        complete.rename(target)
        # 同文件系统内重命名把完整目录一次落位；不是逐文件复制到最终目录再让运行时碰半成品。
    print(f"Installed bilingual voice: {target}")
    print("模型已就绪；直接启动 main.py，使用固定中英音源。")
    return target


def main() -> None:
    # 【入口】python -m stackchan_integration.voice_model install/status；不连接StackChan。
    parser = argparse.ArgumentParser(description="Install or locate Friston-3's offline Chinese/English voice")
    commands = parser.add_subparsers(dest="command", required=True)
    # argparse子命令：要求选择install或status，选择结果存为args.command字符串。
    setup = commands.add_parser("install", help="Explicit one-time download (or local archive import)")
    setup.add_argument("--archive", type=Path, help="Use a previously downloaded official tar.bz2, no network")
    # type=Path将命令行路径字符串变为Path；例如install --archive C:\\Downloads\\voice.tar.bz2。
    commands.add_parser("status", help="Check files only; does not synthesize or open the device")
    args = parser.parse_args()
    # 实际解析在这里；--help显示帮助并退出，之后的下载/检查分支不会执行。
    try:
        if args.command == "install":
            install(args.archive)
        else:
            folder = default_bilingual_dir()
            missing = kokoro_missing(folder)
            # status只检查文件/目录存在且非空，不加载模型、更不代表合成质量已经验证。
            print(f"Bilingual model: {folder}")
            print("Ready (model files found)." if not missing else "Not installed or incomplete.")
            if missing:
                print("Run: python -m stackchan_integration.voice_model install")
    except (OSError, ValueError, tarfile.TarError, urllib.error.URLError) as exc:
        parser.exit(1, f"Voice setup failed: {exc}\nIf downloading is unavailable, use --archive with the official tar.bz2.\n")
        # 退出码1表示失败，便于shell识别；异常内容作为用户可读信息输出。


if __name__ == "__main__":
    # 只有作为模块入口运行才执行main；import提供函数，不自动安装。
    main()
