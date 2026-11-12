"""完成頁：影片預覽、開啟輸出資料夾、各階段耗時與警告摘要、授權提醒、重新套用字幕樣式。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSlider, QTableWidget,
                               QTableWidgetItem, QVBoxLayout, QWidget)

from app.core.models import DONE, STAGE_IDS, STAGE_LABELS, STATUS_LABELS
from app.pipeline.report import write_report
from app.pipeline.stages.s7_export import LICENSE_NOTICE
from app.storage.app_config import AppConfig
from app.storage.project_store import Project
from app.ui.subtitle_style_dialog import edit_subtitle_style


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


class DonePage(QWidget):
    restyle_requested = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.project: Project | None = None

        self.headline = QLabel("")
        self.headline.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.video = QVideoWidget()
        self.video.setMinimumHeight(300)
        self.audio = QAudioOutput(self)
        self.player = QMediaPlayer(self)
        self.player.setVideoOutput(self.video)
        self.player.setAudioOutput(self.audio)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(lambda d: self.slider.setRange(0, d))
        self.player.playbackStateChanged.connect(self._on_state)

        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self._toggle)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.player.setPosition)
        self.time_label = QLabel("0:00 / 0:00")
        controls = QHBoxLayout()
        controls.addWidget(self.play_button)
        controls.addWidget(self.slider, 1)
        controls.addWidget(self.time_label)

        self.folder_button = QPushButton("開啟輸出資料夾")
        self.folder_button.clicked.connect(self._open_folder)
        self.style_button = QPushButton("字幕樣式…")
        self.style_button.clicked.connect(self._edit_style)
        actions = QHBoxLayout()
        actions.addWidget(self.folder_button)
        actions.addWidget(self.style_button)
        self.report_button = QPushButton("匯出報告…")
        self.report_button.setToolTip("把各階段的執行方式、耗時與總時間存成 Markdown 或 HTML")
        self.report_button.clicked.connect(self._export_report)
        actions.addWidget(self.report_button)
        actions.addStretch(1)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["階段", "耗時", "警告"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setColumnWidth(0, 130)
        self.table.setColumnWidth(1, 80)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setMaximumHeight(250)
        self.license = QLabel(f"⚠ {LICENSE_NOTICE}")
        self.license.setWordWrap(True)
        self.license.setStyleSheet("color: #c98a00;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.headline)
        layout.addWidget(self.video, 1)
        layout.addLayout(controls)
        layout.addLayout(actions)
        layout.addWidget(self.table)
        layout.addWidget(self.license)
        self.set_project(None)

    # ---- 專案 ----
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        self.release()
        project = self.project
        final = project.folder / "final_mv.mp4" if project else None
        finished = bool(project) and project.status("s7_export") == DONE and final.exists()
        for widget in (self.play_button, self.slider, self.style_button):
            widget.setEnabled(finished)
        self.folder_button.setEnabled(bool(project))
        self.report_button.setEnabled(bool(project))
        self.table.setRowCount(0)
        if not project:
            self.headline.setText("請先在左側選擇或建立專案")
            return
        total = 0.0
        for stage_id in STAGE_IDS:
            entry = project.stage(stage_id)
            row = self.table.rowCount()
            self.table.insertRow(row)
            seconds = entry.get("seconds")
            total += seconds or 0
            status = entry.get("status", "pending")
            self.table.setItem(row, 0, QTableWidgetItem(STAGE_LABELS[stage_id]))
            self.table.setItem(row, 1, QTableWidgetItem(
                _clock(seconds) if seconds is not None and status == DONE else STATUS_LABELS.get(status, "")))
            warnings = QTableWidgetItem("；".join(entry.get("warnings", [])))
            warnings.setToolTip("\n".join(entry.get("warnings", [])))
            self.table.setItem(row, 2, warnings)
        if finished:
            self.headline.setText(f"✓ 《{project.name}》完成，總耗時 {_clock(total)}")
            self.player.setSource(QUrl.fromLocalFile(str(final)))
            self.player.pause()                       # 載入並顯示第一幀
        else:
            self.headline.setText("這個專案還沒有完成；到「執行」頁按「開始」或「繼續」")

    def release(self) -> None:
        """放開影片檔。重新燒字幕會取代 final_mv.mp4，播放器開著檔案時 Windows 不允許取代。"""
        self.player.stop()
        self.player.setSource(QUrl())

    # ---- 播放 ----
    def _toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _on_state(self, state) -> None:
        self.play_button.setText("暫停" if state == QMediaPlayer.PlaybackState.PlayingState else "播放")

    def _on_position(self, position: int) -> None:
        if not self.slider.isSliderDown():
            self.slider.setValue(position)
        self.time_label.setText(f"{_clock(position / 1000)} / {_clock(self.player.duration() / 1000)}")

    # ---- 動作 ----
    def _open_folder(self) -> None:
        if self.project:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.folder)))

    def _edit_style(self) -> None:
        if self.project is None:
            return
        if not edit_subtitle_style(self.config, self.project, self, "套用並重新燒字幕"):
            return
        self.release()
        self.restyle_requested.emit()

    def _export_report(self) -> None:
        if self.project is None:
            return
        filters = {"HTML 網頁 (*.html)": "html", "Markdown 文件 (*.md)": "md"}
        chosen, used = QFileDialog.getSaveFileName(self, "匯出執行結果報告", str(self.project.folder / "report.html"),
                                                   ";;".join(filters))
        if not chosen:
            return
        kind = "md" if chosen.lower().endswith(".md") else "html" if chosen.lower().endswith((".html", ".htm")) \
            else filters.get(used, "html")
        dest = Path(chosen) if Path(chosen).suffix else Path(f"{chosen}.{kind}")
        try:
            write_report(self.config, self.project, kind, dest)
        except OSError as e:
            QMessageBox.warning(self, "匯出報告", f"無法寫入 {dest}：{e}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(dest)))

    def shutdown(self) -> None:
        self.release()
