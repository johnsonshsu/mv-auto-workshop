"""背景工作：長任務一律在 QThread 執行，介面不凍結。"""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

from app.core.models import UserFacingError
from app.pipeline.stage_base import Reporter
from app.services.comfy_client import ComfyCancelled


class TaskWorker(QThread):
    """執行 task(reporter, cancel)；結果與錯誤以訊號送回主執行緒。"""

    log = Signal(str)
    progress = Signal(float, str)
    succeeded = Signal(object)
    failed = Signal(str, str)        # 訊息、建議動作
    cancelled = Signal()

    def __init__(self, task: Callable[[Reporter, threading.Event], Any], parent=None):
        super().__init__(parent)
        self._task = task
        self.cancel_event = threading.Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        reporter = Reporter(on_log=self.log.emit, on_progress=self.progress.emit)
        try:
            result = self._task(reporter, self.cancel_event)
        except ComfyCancelled:
            self.cancelled.emit()
        except UserFacingError as e:
            self.failed.emit(e.message, e.fix)
        except Exception as e:      # 未預期的錯誤也要讓使用者看到，不可無聲失敗
            self.log.emit(traceback.format_exc())
            self.failed.emit(f"發生未預期的錯誤：{type(e).__name__}: {e}", "詳細內容已寫在記錄區")
        else:
            self.succeeded.emit(result)
