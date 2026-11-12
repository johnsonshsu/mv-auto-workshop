"""硬體偵測：顯示卡、系統記憶體、磁碟空間、FFmpeg。"""
from __future__ import annotations

import ctypes
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# 不要讓子程序跳出主控台視窗
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass
class GpuInfo:
    name: str
    vram_gb: float
    driver: str


def _run(args: list[str], timeout: float = 10.0) -> str | None:
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout if r.returncode == 0 else None


def probe_gpus() -> list[GpuInfo]:
    """以 nvidia-smi 查詢；沒有 NVIDIA 顯示卡或驅動時回傳空清單。"""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits"])
    gpus = []
    for line in (out or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            try:
                gpus.append(GpuInfo(parts[0], float(parts[1]) / 1024, parts[2]))
            except ValueError:
                continue
    return gpus


def probe_ram_gb() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    status = MemoryStatus()
    status.dwLength = ctypes.sizeof(MemoryStatus)
    try:
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except AttributeError:      # 非 Windows（只在測試環境會發生）
        return 0.0
    return status.ullTotalPhys / (1 << 30)


def probe_disk_free_gb(path: Path) -> float | None:
    """path 不存在時往上找到存在的上層資料夾；整個磁碟不存在則回傳 None。"""
    path = Path(path)
    for candidate in [path, *path.parents]:
        if candidate.exists():
            return shutil.disk_usage(candidate).free / (1 << 30)
    return None


def probe_ffmpeg() -> str | None:
    """回傳 FFmpeg 版本字串；找不到時回傳 None。"""
    out = _run(["ffmpeg", "-version"])
    if not out:
        return None
    first = out.splitlines()[0]          # ffmpeg version 9.0.2-full_build-... Copyright ...
    parts = first.split()
    return parts[2] if len(parts) >= 3 else first
