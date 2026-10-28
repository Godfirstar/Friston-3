"""Remove confirmed obsolete project assets after the retained models validate.

This is the v2 installation step, not a model selector or a background cleaner.
It never deletes shared model caches, unrelated Ollama models or identity data.
"""
from __future__ import annotations

import errno
import os
from pathlib import Path
import shutil
import stat
import tempfile

from friston3.config import PROJECT_ROOT, MODELS_DIR


OBSOLETE_FILES = (
    "friston3/knowledge/rerank_lore.py",
    "README_肩上搭档_v1.md",
    "README_语音动作修复_v1_1.md",
    "README_语音自述音量_v1_2.md",
)


def _reparse(info):
    # Path.is_junction() is unavailable in the existing Python 3.10/3.11
    # environments. lstat exposes junctions and other reparse points there too.
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    )


def _linked(path):
    try:
        return _reparse(path.lstat())
    except FileNotFoundError:
        return False


def _owned(path: Path, root: Path):
    """Only ordinary descendants of an explicitly named application folder."""
    path, root = path.absolute(), root.absolute()
    if ".." in path.parts or ".." in root.parts:
        return False
    if path == root or not path.is_relative_to(root):
        return False
    # Check above root as well: Friston2/tts can be an ordinary directory
    # underneath a Friston2 junction pointing into another project.
    for current in (path, *path.parents):
        if _linked(current):
            return False
    return True


def _walk_error(error):
    raise error


def _tree_shape(folder: Path):
    """Read a complete ordinary tree; never silently skip unreadable entries."""
    info = folder.lstat()
    if _reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"保留链接或非常规目录：{folder}")
    shape = {}
    for parent, dirs, files in os.walk(folder, followlinks=False, onerror=_walk_error):
        for name in dirs + files:
            entry = Path(parent) / name
            info = entry.lstat()
            if _reparse(info):
                raise RuntimeError(f"保留含链接的模型目录：{folder}")
            if stat.S_ISDIR(info.st_mode):
                shape[entry.relative_to(folder)] = ("dir", 0)
            elif stat.S_ISREG(info.st_mode):
                shape[entry.relative_to(folder)] = ("file", info.st_size)
            else:
                raise RuntimeError(f"保留含非常规文件的模型目录：{folder}")
    return shape


def _same_tree(left: Path, right: Path):
    """Require identical names and bytes, including any extra user files."""
    shape = _tree_shape(left)
    if shape != _tree_shape(right):
        return False
    for name, (kind, _size) in shape.items():
        if kind == "dir":
            continue
        with (left / name).open("rb") as a, (right / name).open("rb") as b:
            while True:
                chunk = a.read(1024 * 1024)
                if chunk != b.read(1024 * 1024):
                    return False
                if not chunk:
                    break
    return True


def _delete(path: Path, root: Path, removed, skipped):
    try:
        if not path.exists() and not _linked(path):
            return
        if not _owned(path, root):
            skipped.append(f"保留链接或范围外路径：{path}")
            return
        if path.is_dir():
            _tree_shape(path)
            shutil.rmtree(path)
        else:
            path.unlink()
        removed.append(str(path))
    except (OSError, RuntimeError) as error:
        skipped.append(f"未删除 {path}：{error}")


def _migrate_model(source: Path, target: Path, complete, source_root: Path, target_root: Path,
                   migrated, skipped):
    """Return False when migration is blocked, so duplicate cleanup can wait."""
    if source == target:
        return True
    committed = False
    try:
        if not source.exists() and not _linked(source):
            return True
        if not _owned(source, source_root) or not _owned(target, target_root):
            skipped.append(f"保留链接或范围外的模型位置：{source} -> {target}")
            return False
        if not source.is_dir() or not complete(source):
            return True
        if target.exists() or _linked(target):
            if target.is_dir() and complete(target):
                return True
            skipped.append(f"规范目录已存在但不完整，未覆盖；继续保留原模型：{target}；{source}")
            return False
        _tree_shape(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            source.rename(target)
        except OSError as error:
            if error.errno != errno.EXDEV and getattr(error, "winerror", None) != 17:
                raise
            # Cross-drive copies never write directly into the runtime's
            # canonical path. Interrupted copies remain unselected staging
            # folders; source removal starts only after the complete commit.
            with tempfile.TemporaryDirectory(prefix=".friston3-migrate-", dir=target.parent) as work:
                staged = Path(work) / "model"
                shutil.copytree(source, staged, symlinks=True)
                if not complete(staged) or not _same_tree(source, staged):
                    raise RuntimeError("迁移暂存副本未通过完整内容检查")
                if target.exists() or _linked(target):
                    raise FileExistsError(f"规范目录在迁移期间已出现，未覆盖：{target}")
                staged.rename(target)
                committed = True
            if not _owned(source, source_root):
                raise RuntimeError(f"源目录在迁移期间已变化，未删除：{source}")
            _tree_shape(source)
            shutil.rmtree(source)
        migrated.append(f"{source} -> {target}")
        return True
    except (OSError, RuntimeError) as error:
        status = "规范模型已提交，旧副本清理未完成" if committed else "迁移未提交，保留原模型"
        skipped.append(f"{status}：{source} -> {target}：{error}")
        return False


def _remove_duplicate(source: Path, target: Path, complete, source_root: Path, target_root: Path,
                      removed, skipped):
    if source == target:
        return
    try:
        if not source.exists() and not _linked(source):
            return
        if not _owned(source, source_root) or not _owned(target, target_root):
            skipped.append(f"保留链接或范围外的同型号副本：{source}")
            return
        if not source.is_dir() or not complete(source) or not complete(target):
            skipped.append(f"同型号目录未通过完整性检查，未做去重：{source}")
            return
        if not _same_tree(source, target):
            skipped.append(f"同型号副本的文件或内容不同，保留供核对：{source}")
            return
        _delete(source, source_root, removed, skipped)
    except (OSError, RuntimeError) as error:
        skipped.append(f"同型号副本未完成核对，未删除 {source}：{error}")


def consolidate():
    """Perform only the explicit v2 removal list, without loading any model."""
    from stackchan_integration.local_tts import (
        BILINGUAL_MODEL_NAME, default_bilingual_dir, kokoro_missing, model_root,
    )
    from stackchan_integration.voice_input import seaco_missing, seaco_model_dir

    removed, migrated, skipped = [], [], []
    root = PROJECT_ROOT.absolute()
    for name in OBSOLETE_FILES:
        _delete(root / name, root, removed, skipped)
    cache = root / "friston3" / "knowledge" / "__pycache__"
    if cache.is_dir() and not _linked(cache):
        for path in cache.glob("rerank_lore.*.pyc"):
            _delete(path, root, removed, skipped)

    seaco_target = MODELS_DIR / "seaco_paraformer"
    seaco_migration_ok = _migrate_model(
        MODELS_DIR / "seaco", seaco_target, lambda p: not seaco_missing(p),
        MODELS_DIR, MODELS_DIR, migrated, skipped,
    )
    seaco = seaco_model_dir()
    seaco_ready = not seaco_missing(seaco)
    if seaco_ready:
        _delete(MODELS_DIR / "sensevoice", MODELS_DIR, removed, skipped)
        for folder in (MODELS_DIR / "seaco_paraformer", MODELS_DIR / "seaco"):
            if not seaco_missing(folder):
                for name in ("model.onnx", "model_eb.onnx"):
                    _delete(folder / name, MODELS_DIR, removed, skipped)
        if seaco_migration_ok and not seaco_missing(seaco_target):
            _remove_duplicate(
                MODELS_DIR / "seaco", seaco_target, lambda p: not seaco_missing(p),
                MODELS_DIR, MODELS_DIR, removed, skipped,
            )
    else:
        skipped.append("SeACo 量化模型尚不完整，本次保留旧 ASR 文件，主程序仍只使用 SeACo。")

    app_data = os.environ.get("LOCALAPPDATA")
    tts_roots = [Path(app_data) / name / "tts" for name in ("Friston3", "Friston2")] if app_data else [
        Path.home() / name / "tts" for name in ("Friston3Models", "Friston2Models")]
    target = model_root() / BILINGUAL_MODEL_NAME
    source = default_bilingual_dir()
    source_owner = next((r for r in [*tts_roots, MODELS_DIR] if source.is_relative_to(r)), None)
    tts_migration_ok = source == target
    if source_owner is not None:
        tts_migration_ok = _migrate_model(
            source, target, lambda p: not kokoro_missing(p), source_owner,
            model_root(), migrated, skipped,
        )
    retained_tts = default_bilingual_dir()
    tts_ready = not kokoro_missing(retained_tts)
    if tts_ready:
        for tts_root in tts_roots:
            for name in ("vits-zh-hf-fanchen-wnj", "kokoro-multi-lang-v1_1"):
                _delete(tts_root / name, tts_root, removed, skipped)
        # Strip the known unused fp32 file from each complete same-model
        # location before comparing any duplicate packages byte for byte.
        for owner in [*tts_roots, MODELS_DIR]:
            folder = owner / BILINGUAL_MODEL_NAME
            if not kokoro_missing(folder):
                _delete(folder / "model.onnx", owner, removed, skipped)
        if tts_migration_ok and not kokoro_missing(target):
            for owner in [*tts_roots, MODELS_DIR]:
                _remove_duplicate(
                    owner / BILINGUAL_MODEL_NAME, target, lambda p: not kokoro_missing(p),
                    owner, model_root(), removed, skipped,
                )
    else:
        skipped.append("Kokoro int8 模型尚不完整，本次保留旧 TTS 文件，主程序仍只使用 Kokoro。")

    for item in migrated:
        print(f"[收拢] 已迁移同一模型：{item}")
    for item in removed:
        print(f"[收拢] 已删除：{item}")
    for item in skipped:
        print(f"[收拢] {item}")
    print(f"[收拢] 删除 {len(removed)} 项，迁移 {len(migrated)} 项。"
          "保留知识库、语义索引、身份模板及各自必需的模型组件。")
    return {"removed": removed, "migrated": migrated, "skipped": skipped,
            "seaco_ready": seaco_ready, "tts_ready": tts_ready}


if __name__ == "__main__":
    consolidate()
