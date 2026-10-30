"""Small shared helpers: atomic file writes, JSON/YAML, hashing, time."""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("model_trainer")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, Mapping):
        return dict(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def atomic_write_bytes(path: str | os.PathLike[str], data: bytes) -> None:
    """Write to a temporary file in the same folder, then rename: a crash never leaves half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _replace(src: str | Path, dst: Path) -> None:
    # Windows refuses to replace a file another process has open for a moment (e.g. the GUI reading
    # state.json): retry briefly instead of failing a training run over it
    for i in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if i == 19:
                raise
            time.sleep(0.05)


def atomic_copy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp")
    shutil.copyfile(src, tmp)
    with open(tmp, "rb+") as fh:
        os.fsync(fh.fileno())
    _replace(tmp, dst)


def write_text(path: str | os.PathLike[str], text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def write_json(path: str | os.PathLike[str], data: Any) -> None:
    write_text(path, json.dumps(data, indent=2, default=_json_default))


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        if default is not None:
            return default
        raise


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=_json_default) + "\n")
        fh.flush()


def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(json.JSONDecodeError):  # a line cut by a power loss is skipped
            out.append(json.loads(line))
    return out


def write_yaml(path: str | os.PathLike[str], data: Any) -> None:
    write_text(path, yaml.safe_dump(_plain(data), sort_keys=False, allow_unicode=True))


def read_yaml(path: str | os.PathLike[str]) -> Any:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _plain(o: Any) -> Any:
    if isinstance(o, Mapping):
        return {str(k) if not isinstance(k, int) else k: _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, Path):
        return o.as_posix()
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return o.item()
    return o


def sha1_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def link_or_copy(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Hard link (no extra disk space) when possible, otherwise copy."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def free_disk_gb(path: str | os.PathLike[str]) -> float:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def gpu_status() -> list[dict[str, Any]]:
    """Name, memory, load and temperature of each NVIDIA GPU (empty without nvidia-smi)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    fields = "name,memory.used,memory.total,utilization.gpu,temperature.gpu,driver_version"
    try:
        out = subprocess.run(
            [exe, f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for i, line in enumerate(out.strip().splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        def num(s: str) -> float | None:
            try:
                return float(s)
            except ValueError:
                return None

        gpus.append(
            {
                "index": i,
                "name": parts[0],
                "memory_used_mb": num(parts[1]),
                "memory_total_mb": num(parts[2]),
                "load_pct": num(parts[3]),
                "temperature_c": num(parts[4]),
                "driver": parts[5],
            }
        )
    return gpus


@contextlib.contextmanager
def keep_awake(reason: str = "training") -> Iterator[bool]:
    """Stop the computer from sleeping inside the block (Windows sleep kills long runs)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        ok = bool(kernel32.SetThreadExecutionState(0x80000000 | 0x00000001))
        if ok:
            log.info("keeping the computer awake while %s", reason)
        try:
            yield ok
        finally:
            kernel32.SetThreadExecutionState(0x80000000)
        return
    cmd = None
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        cmd = ["caffeinate", "-i", "-w", str(os.getpid())]
    elif shutil.which("systemd-inhibit"):
        cmd = ["systemd-inhibit", f"--why={reason}", "--what=sleep", "sleep", "infinity"]
    proc = None
    if cmd:
        with contextlib.suppress(OSError):
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        yield proc is not None
    finally:
        if proc is not None:
            proc.terminate()


def trust_system_certificates() -> None:
    """Use the operating system's certificate store for HTTPS (downloads of base models, fonts).

    Python's `requests` ships its own certificate list; behind antivirus HTTPS scanning or a company
    proxy only the system store trusts the connection, and model downloads fail with
    CERTIFICATE_VERIFY_FAILED. truststore fixes that for every library in the process.
    """
    with contextlib.suppress(Exception):
        import truststore

        truststore.inject_into_ssl()


def human_time(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "-"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {s:02d} s"
    return f"{s} s"
