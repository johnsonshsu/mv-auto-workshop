"""執行頁：七階段總進度、目前階段的細部進度、即時記錄、取消。"""
from __future__ import annotations

import shutil
import threading
import time

from PySide6.QtCore import QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QVBoxLayout, QWidget)

from app.core.models import (DONE, FAILED, PENDING, RUNNING, STAGE_IDS, STAGE_LABELS, STALE, STATUS_LABELS)
from app.pipeline import runner as R
from app.pipeline.stage_base import Reporter, StageContext
from app.storage.app_config import AppConfig
from app.storage.project_store import Project

_MARKS = {PENDING: "○", RUNNING: "▶", DONE: "✓", FAILED: "✗", STALE: "↻"}
_COLORS = {PENDING: "#888888", RUNNING: "#2f6fd0", DONE: "#2e9d5b", FAILED: "#d0453a", STALE: "#c98a00"}


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


class RunWorker(QThread):
    log = Signal(str)
    progress = Signal(float, str)
    stage_changed = Signal(str, str)
    done = Signal(object)            # RunOutcome

    def __init__(self, config: AppConfig, project: Project, parent=None):
        super().__init__(parent)
        self.cancel_event = threading.Event()
        self.pause_event = threading.Event()
        self._ctx = StageContext(config, project, self.cancel_event, self.pause_event)

    def run(self) -> None:
        reporter = Reporter(on_log=self.log.emit, on_progress=self.progress.emit)
        outcome = R.PipelineRunner(self._ctx).run(reporter, self.stage_changed.emit)
        self.done.emit(outcome)


class RunPage(QWidget):
    run_finished = Signal(object)    # RunOutcome
    stage_done = Signal(str)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.project: Project | None = None
        self._worker: RunWorker | None = None
        self._stage_started = 0.0

        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-size: 15px; font-weight: 600;")

        self._rows: dict[str, tuple[QLabel, QLabel]] = {}
        grid = QGridLayout()
        for row, stage_id in enumerate(STAGE_IDS):
            name, detail = QLabel(""), QLabel("")
            detail.setWordWrap(True)
            grid.addWidget(name, row, 0)
            grid.addWidget(detail, row, 1)
            self._rows[stage_id] = (name, detail)
        grid.setColumnMinimumWidth(0, 170)
        grid.setColumnStretch(1, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_text = QLabel("")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(5000)
        self.log_box.setPlaceholderText("記錄")

        self.start_button = QPushButton("開始")
        self.start_button.setMinimumHeight(34)
        self.start_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel)
        self.pause_button = QPushButton("暫停")
        self.pause_button.setToolTip("目前這一段影片（或這個階段）跑完後停下；之後按「繼續」接著跑")
        self.pause_button.clicked.connect(self.pause)
        self.revideo_button = QPushButton("重新生成影片")
        self.revideo_button.setToolTip("刪除已生成的各段影片，用同樣的照片與歌曲重新生成")
        self.revideo_button.clicked.connect(self._regenerate_video)
        self.resong_button = QPushButton("重新生成歌曲")
        self.resong_button.setToolTip("保留歌詞，換一個 seed 重新生成歌曲，之後的階段會跟著重做")
        self.resong_button.clicked.connect(self._regenerate_song)
        self.song_button = QPushButton("播放歌曲")
        self.song_button.clicked.connect(self._open_song)
        self.folder_button = QPushButton("開啟專案資料夾")
        self.folder_button.clicked.connect(self._open_folder)
        buttons = QHBoxLayout()
        for b in (self.start_button, self.pause_button, self.cancel_button, self.song_button, self.resong_button,
                  self.revideo_button, self.folder_button):
            buttons.addWidget(b)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.headline)
        layout.addLayout(grid)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_text)
        layout.addWidget(self.log_box, 1)
        layout.addLayout(buttons)
        self.set_project(None)

    # ---- 專案 ----
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.log_box.clear()
        self.headline.setText("" if project else "請先在左側選擇或建立專案")
        self.refresh()

    def refresh(self) -> None:
        """依專案目前的狀態更新畫面。"""
        project = self.project
        for stage_id, (name, detail) in self._rows.items():
            status = project.status(stage_id) if project else PENDING
            entry = project.stage(stage_id) if project else {}
            number = STAGE_IDS.index(stage_id) + 1
            name.setText(f"{_MARKS[status]} {number} {STAGE_LABELS[stage_id]}")
            name.setStyleSheet(f"color: {_COLORS[status]};" + ("font-weight: 600;" if status == RUNNING else ""))
            parts = []
            if status in (DONE, FAILED) and entry.get("seconds") is not None:
                parts.append(_clock(entry["seconds"]))
            if status in (FAILED, STALE):
                parts.append(entry.get("error") or STATUS_LABELS[status])
            parts.extend(f"⚠ {w}" for w in entry.get("warnings", []))
            detail.setText("　".join(parts))
        busy = self._worker is not None
        unfinished = project.first_unfinished() if project else None
        started = bool(project) and any(project.status(s) != PENDING for s in STAGE_IDS[1:])
        self.start_button.setText("繼續" if started else "開始")
        self.start_button.setEnabled(bool(project) and not busy and unfinished is not None)
        self.cancel_button.setVisible(busy)
        self.pause_button.setEnabled(True)
        has_song = bool(project) and project.path("audio", "song.flac").exists()
        self.song_button.setVisible(has_song)
        self.resong_button.setVisible(has_song)
        self.resong_button.setEnabled(not busy)
        self.pause_button.setVisible(busy)
        has_video = bool(project) and any(project.path("segments").glob("seg_*.mp4"))
        self.revideo_button.setVisible(has_video)
        self.revideo_button.setEnabled(not busy)
        self.folder_button.setEnabled(bool(project))
        if not busy:
            self.progress.setValue(0)
            self.progress_text.setText("")

    def is_busy(self) -> bool:
        return self._worker is not None

    # ---- 執行 ----
    def start(self) -> None:
        if self.project is None or self._worker is not None:
            return
        worker = RunWorker(self.config, self.project, self)
        worker.log.connect(self._log)
        worker.progress.connect(self._on_progress)
        worker.stage_changed.connect(self._on_stage)
        worker.done.connect(self._on_done)
        worker.finished.connect(self._on_thread_finished)
        self._worker = worker
        self.headline.setText("執行中…")
        self.headline.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.refresh()
        worker.start()

    def _regenerate_song(self) -> None:
        if self.project is None or self._worker is not None:
            return
        self.project.data["song_takes"] = []
        self.project.stage("s3_song")["status"] = STALE
        self.project.mark_stale_after("s3_song")
        self.project.save()
        self.start()

    def _regenerate_video(self) -> None:
        if self.project is None or self._worker is not None:
            return
        if QMessageBox.question(self, "重新生成影片", "已生成的各段影片會被刪除，整支影片要重新生成。確定嗎？") \
                != QMessageBox.StandardButton.Yes:
            return
        shutil.rmtree(self.project.path("segments"), ignore_errors=True)
        self.project.path("analysis", "segment_plan.json").unlink(missing_ok=True)
        self.project.stage("s5_video")["status"] = STALE
        self.project.stage("s5_video")["segments"] = []
        self.project.mark_stale_after("s5_video")
        self.project.save()
        self.start()

    def pause(self) -> None:
        if self._worker is not None:
            self._log("已要求暫停：目前這一段跑完後會停下")
            self._worker.pause_event.set()
            self.pause_button.setEnabled(False)

    def cancel(self) -> None:
        if self._worker is not None:
            self._log("取消中…（正在進行的 LLM 請求會等它回應後才停）")
            self._worker.cancel_event.set()

    def _on_stage(self, stage_id: str, status: str) -> None:
        if status == RUNNING:
            self._stage_started = time.monotonic()
        elif status == DONE:
            self.stage_done.emit(stage_id)
        self.refresh()

    def _on_progress(self, fraction: float, text: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        elapsed = _clock(time.monotonic() - self._stage_started) if self._stage_started else ""
        self.progress_text.setText(f"{text}　已執行 {elapsed}" if elapsed else text)

    def _on_done(self, outcome) -> None:
        kind = outcome.kind
        text = {
            R.FINISHED: "✓ 全部完成",
            R.PAUSED: f"⏸ {outcome.message}",
            R.STOPPED: f"✓ 目前可用的階段都完成了。{outcome.message}",
            R.BLOCKED: f"✗ 無法繼續：{outcome.message}",
            R.ERROR: f"✗ {STAGE_LABELS.get(outcome.stage_id, '')}失敗：{outcome.message}",
            R.CANCELLED: ("已暫停，按「繼續」接著跑" if outcome.message == "已暫停"
                          else "已取消，可按「繼續」從中斷的階段接著跑"),
        }[kind]
        if outcome.fix:
            text += f"\n→ {outcome.fix}"
        color = {R.FINISHED: _COLORS[DONE], R.STOPPED: _COLORS[DONE], R.PAUSED: _COLORS[RUNNING],
                 R.CANCELLED: _COLORS[PENDING]}.get(kind, _COLORS[FAILED])
        self.headline.setText(text)
        self.headline.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {color};")
        self._log(text)
        self._outcome = outcome

    def _on_thread_finished(self) -> None:
        self._worker = None
        self.refresh()
        outcome = getattr(self, "_outcome", None)
        if outcome is not None:
            self._outcome = None
            self.run_finished.emit(outcome)

    def _log(self, text: str) -> None:
        self.log_box.appendPlainText(f"{time.strftime('%H:%M:%S')} {text}")

    def _open_song(self) -> None:
        if self.project:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.path("audio", "song.flac"))))

    def _open_folder(self) -> None:
        if self.project:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.folder)))

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.cancel_event.set()
            self._worker.wait(8000)
