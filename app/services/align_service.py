"""啟動對齊子程序並回報進度。對齊模型在獨立程序執行，結束即釋放 VRAM。"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

from app.core.models import UserFacingError
from app.pipeline.stage_base import Reporter
from app.services.comfy_client import ComfyCancelled
from app.storage.app_config import AppConfig

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def worker_python_and_script(config: AppConfig) -> tuple[str, Path]:
    """對齊子程序要用的 Python 與腳本。

    從原始碼執行時就用目前的 Python。打包成執行檔後沒有 Python 可用（torch 太大，沒有包進去），
    所以用設定檔的 align.python，或執行檔旁邊由 setup_align_env.ps1 建立的 align-env。
    """
    frozen = getattr(sys, "frozen", False)
    script = config.root / "tools" / "align_worker.py" if frozen else Path(__file__).with_name("align_worker.py")
    python = config.get("align.python") or ("" if frozen else sys.executable)
    if not python:
        bundled = config.root / "align-env" / "Scripts" / "python.exe"
        if not bundled.exists():
            raise UserFacingError(
                "找不到字幕對齊用的 Python 環境",
                f"請在 {config.root} 執行 setup_align_env.ps1（會安裝 torch 與 stable-ts，約 3 GB），"
                "或在 config\\default.toml 的 align.python 填入已安裝 stable-ts 的 python.exe")
        python = str(bundled)
    if not script.exists():
        raise UserFacingError(f"找不到對齊程式 {script}", "請重新安裝 MV 自動工坊")
    return python, script


def run_alignment(config: AppConfig, vocals: Path, lyrics_txt: Path, output: Path, language: str,
                  reporter: Reporter, cancel: threading.Event | None = None) -> list[dict]:
    """回傳每行的對齊結果（start、end、words）。"""
    python, script = worker_python_and_script(config)
    command = [python, "-X", "utf8", str(script), str(vocals), str(lyrics_txt), str(output),
               "--model", config.get("align.model", "large-v3"), "--language", language]
    try:
        process = subprocess.Popen(command, cwd=str(config.root), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
    except OSError as e:
        raise UserFacingError(f"無法啟動對齊程式：{e}", "請確認 config 的 align.python 指向可用的 Python")

    stderr_lines: list[str] = []

    def read_stdout() -> None:
        for line in process.stdout:
            if line.startswith("PROGRESS "):
                _, fraction, text = line.strip().split(" ", 2)
                reporter.progress(float(fraction), text)

    def read_stderr() -> None:
        for line in process.stderr:
            stderr_lines.append(line.rstrip())

    readers = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for reader in readers:
        reader.start()
    deadline = time.monotonic() + float(config.get("align.timeout", 3600))
    while process.poll() is None:
        if cancel is not None and cancel.is_set():
            process.kill()
            raise ComfyCancelled()
        if time.monotonic() > deadline:
            process.kill()
            raise UserFacingError("字幕對齊逾時", "首次使用需要下載約 3 GB 的 Whisper 模型，請確認網路後再試一次")
        time.sleep(0.2)
    for reader in readers:
        reader.join(timeout=2)

    if process.returncode != 0:
        detail = next((line for line in reversed(stderr_lines) if line.startswith("ALIGN_ERROR")),
                      stderr_lines[-1] if stderr_lines else f"結束碼 {process.returncode}")
        text = "\n".join(stderr_lines).lower()
        if "out of memory" in text:
            raise UserFacingError("字幕對齊時顯示記憶體不足", "請關閉其他佔用顯示卡的程式後再試")
        if "no module named" in text:
            raise UserFacingError(f"對齊程式缺少套件：{detail}",
                                  "請在專案資料夾執行 .\\.venv\\Scripts\\python.exe -m pip install stable-ts")
        raise UserFacingError(f"字幕對齊失敗：{detail.replace('ALIGN_ERROR ', '')}", "詳細內容已寫在記錄區")
    try:
        return json.loads(output.read_text(encoding="utf-8"))["lines"]
    except (OSError, ValueError, KeyError) as e:
        raise UserFacingError(f"對齊程式沒有產生可讀的結果：{e}", "請再試一次")
