"""階段的共同介面。每個階段只讀寫專案資料夾內的檔案，因此都能單獨重跑。"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from app.core.models import Problem
from app.services.comfy_client import ComfyCancelled
from app.storage.app_config import AppConfig
from app.storage.project_store import Project


class PauseRequested(ComfyCancelled):
    """使用者按了暫停，階段在安全的地方（例如兩段影片之間）停下。"""


@dataclass
class Reporter:
    """階段回報進度與記錄的管道；介面層把這些接到 Qt 訊號。"""
    on_log: Callable[[str], None] = lambda text: None
    on_progress: Callable[[float, str], None] = lambda fraction, text: None

    def log(self, text: str) -> None:
        self.on_log(text)

    def progress(self, fraction: float, text: str = "") -> None:
        self.on_progress(max(0.0, min(1.0, fraction)), text)


@dataclass
class StageContext:
    config: AppConfig
    project: Project | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    # 暫停：目前這一段（或這個階段）跑完後停下，不中斷正在進行的生成
    pause: threading.Event = field(default_factory=threading.Event)


@dataclass
class StageResult:
    ok: bool
    warnings: list[str] = field(default_factory=list)
    message: str = ""
    # 給流程編排層參考的數值，例如字幕階段回報的未對齊比例
    data: dict = field(default_factory=dict)


class Stage(Protocol):
    id: str

    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        """需要的輸入檔是否齊全。"""

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        """從頭執行。"""

    def resume(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        """從中斷處接續；沒有中間狀態的階段等同 run。"""

    def outputs(self, ctx: StageContext) -> list[Path]:
        """這個階段的產出檔。"""
