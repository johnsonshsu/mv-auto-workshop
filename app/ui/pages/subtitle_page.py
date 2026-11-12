"""字幕校對頁：人聲波形加字幕時間軸，標示低信心的行，可拖曳調整起訖時間並試聽。"""
from __future__ import annotations

import numpy as np
import soundfile
from PySide6.QtCore import QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from app.core import subtitle_edit as E
from app.core.models import DONE, UserFacingError
from app.storage.app_config import AppConfig
from app.storage.project_store import Project

_FLAG_COLORS = {"red": "#d0453a", "yellow": "#c98a00", "": "#2f6fd0"}
_EDGE_PIXELS = 6


def _stamp(seconds: float) -> str:
    return f"{int(seconds // 60)}:{seconds % 60:05.2f}"


class Timeline(QWidget):
    """波形與字幕方塊。拖曳方塊邊緣調整起訖，拖曳方塊中間整則移動，點一下選取。"""
    selected = Signal(int)
    changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cues: list[E.Cue] = []
        self.duration = 0.0
        self.low_confidence = 0.4
        self.scale = 60.0                      # 每秒幾個像素
        self.current = -1
        self.playhead = -1.0
        self._peaks = np.zeros(0)
        self._samples: np.ndarray | None = None
        self._rate = 1
        self._drag: tuple[int, str, float] | None = None       # （哪一則、start/end/move、按下時的秒數）
        self.setMinimumHeight(170)
        self.setMouseTracking(True)

    def load(self, cues: list[E.Cue], duration: float, samples: np.ndarray | None, rate: int, low: float) -> None:
        self.cues, self.duration, self.low_confidence = cues, duration, low
        self._samples, self._rate = samples, max(rate, 1)
        self.current = -1
        self.set_scale(self.scale)

    def set_scale(self, scale: float) -> None:
        self.scale = scale
        width = max(int(self.duration * scale) + 1, 200)
        self.setFixedWidth(width)
        # 每個像素一格，取該格內的最大振幅
        if self._samples is not None and len(self._samples):
            edges = np.linspace(0, len(self._samples), width + 1).astype(int)
            self._peaks = np.maximum.reduceat(np.abs(self._samples), edges[:-1])[:width]
            top = float(self._peaks.max()) or 1.0
            self._peaks = self._peaks / top
        else:
            self._peaks = np.zeros(0)
        self.update()

    def set_current(self, index: int) -> None:
        self.current = index
        self.update()

    def set_playhead(self, seconds: float) -> None:
        self.playhead = seconds
        self.update()

    # ---- 繪製 ----
    def _cue_rect(self, cue: E.Cue) -> QRectF:
        return QRectF(cue.start * self.scale, 96, max((cue.end - cue.start) * self.scale, 2), 46)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        middle, half = 46, 40
        painter.setPen(QPen(QColor("#8aa0b8")))
        first, last = max(event.rect().left(), 0), min(event.rect().right() + 1, len(self._peaks))
        for x in range(first, last):
            height = float(self._peaks[x]) * half
            painter.drawLine(x, int(middle - height), x, int(middle + height))
        # 時間刻度：每 5 秒一條
        painter.setPen(QPen(QColor("#999999")))
        for second in range(0, int(self.duration) + 1, 5):
            x = int(second * self.scale)
            painter.drawLine(x, 150, x, 158)
            painter.drawText(x + 3, 166, f"{second // 60}:{second % 60:02d}")
        for index, cue in enumerate(self.cues):
            rect = self._cue_rect(cue)
            color = QColor(_FLAG_COLORS[cue.flag(self.low_confidence)])
            fill = QColor(color)
            fill.setAlpha(120 if index == self.current else 60)
            painter.fillRect(rect, fill)
            painter.setPen(QPen(color, 2 if index == self.current else 1))
            painter.drawRect(rect)
            painter.setPen(self.palette().text().color())
            painter.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                             f"{cue.id} {cue.text}")
        if self.playhead >= 0:
            painter.setPen(QPen(QColor("#d0453a"), 1))
            x = int(self.playhead * self.scale)
            painter.drawLine(x, 0, x, 146)

    # ---- 滑鼠 ----
    def _hit(self, x: float, y: float) -> tuple[int, str] | None:
        for index, cue in enumerate(self.cues):
            rect = self._cue_rect(cue)
            if rect.top() <= y <= rect.bottom() and rect.left() - _EDGE_PIXELS <= x <= rect.right() + _EDGE_PIXELS:
                if abs(x - rect.left()) <= _EDGE_PIXELS:
                    return index, "start"
                if abs(x - rect.right()) <= _EDGE_PIXELS:
                    return index, "end"
                return index, "move"
        return None

    def mousePressEvent(self, event) -> None:
        hit = self._hit(event.position().x(), event.position().y())
        if hit:
            self._drag = (hit[0], hit[1], event.position().x() / self.scale)
            self.selected.emit(hit[0])

    def mouseMoveEvent(self, event) -> None:
        x, seconds = event.position().x(), event.position().x() / self.scale
        if self._drag is None:
            hit = self._hit(x, event.position().y())
            self.setCursor(Qt.CursorShape.SizeHorCursor if hit and hit[1] != "move"
                           else Qt.CursorShape.OpenHandCursor if hit else Qt.CursorShape.ArrowCursor)
            return
        index, mode, origin = self._drag
        if mode == "move":
            moved = E.shift(self.cues, index, seconds - origin, self.duration)
            self._drag = (index, mode, origin + moved)
        else:
            E.move_edge(self.cues, index, mode, seconds, self.duration)
        self.changed.emit(index)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._drag = None


class SubtitlePage(QWidget):
    saved = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.project: Project | None = None
        self.cues: list[E.Cue] = []
        self.duration = 0.0
        self._dirty = False
        self._stop_at = -1.0

        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.timeline = Timeline()
        self.timeline.selected.connect(self._select)
        self.timeline.changed.connect(self._on_changed)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.timeline)
        self.scroll.setFixedHeight(200)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.zoom = QSlider(Qt.Orientation.Horizontal)
        self.zoom.setRange(15, 240)
        self.zoom.setValue(60)
        self.zoom.setFixedWidth(180)
        self.zoom.valueChanged.connect(lambda value: self.timeline.set_scale(float(value)))

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "開始", "結束", "狀態", "歌詞"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for column, width in enumerate((40, 80, 80, 110)):
            self.table.setColumnWidth(column, width)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.currentCellChanged.connect(lambda row, *_: self._select(row))

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.player.positionChanged.connect(self._on_position)

        self.play_line = QPushButton("試聽這一行")
        self.play_line.clicked.connect(self._play_line)
        self.stop_button = QPushButton("停止")
        self.stop_button.clicked.connect(self.player.pause)
        nudges = []
        for label, edge, delta in (("開始 −0.05", "start", -0.05), ("開始 +0.05", "start", 0.05),
                                   ("結束 −0.05", "end", -0.05), ("結束 +0.05", "end", 0.05)):
            button = QPushButton(label)
            button.clicked.connect(lambda _=False, e=edge, d=delta: self._nudge(e, d))
            nudges.append(button)
        self.source = QPushButton("聽完整歌曲")
        self.source.setCheckable(True)
        self.source.setToolTip("預設試聽分離後的人聲，比較容易聽出每一句的起訖；按下後改聽完整歌曲")
        self.source.toggled.connect(self._load_audio)
        self.save_button = QPushButton("儲存並重新燒字幕")
        self.save_button.clicked.connect(self.save)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        top = QHBoxLayout()
        top.addWidget(QLabel('<span style="color:#d0453a">■ 時間是內插的</span>　'
                             '<span style="color:#c98a00">■ 對齊信心偏低</span>　'
                             '拖曳方塊邊緣調整起訖，拖曳中間整則移動'), 1)
        top.addWidget(QLabel("縮放"))
        top.addWidget(self.zoom)
        controls = QHBoxLayout()
        for widget in (self.play_line, self.stop_button, self.source, *nudges):
            controls.addWidget(widget)
        controls.addStretch(1)
        controls.addWidget(self.save_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.headline)
        layout.addLayout(top)
        layout.addWidget(self.scroll)
        layout.addLayout(controls)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.status)
        self.set_project(None)

    # ---- 專案 ----
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.reload()

    def reload(self) -> None:
        self.release()
        self.cues, self.duration, self._dirty = [], 0.0, False
        self.status.setText("")
        project = self.project
        ready = bool(project) and project.status("s4_subtitle") == DONE
        if ready:
            try:
                self.cues, self.duration = E.load_cues(project.folder)
            except UserFacingError:
                ready = False
        low = float(self.config.get("align.low_confidence", 0.4))
        samples, rate = None, 1
        if ready:
            vocals = project.path("audio", "vocals.wav")
            if vocals.exists():
                data, rate = soundfile.read(str(vocals), dtype="float32", always_2d=True)
                samples = data.mean(axis=1)
            self._load_audio(self.source.isChecked())
        self.timeline.load(self.cues, self.duration, samples, rate, low)
        self._fill_table(low)
        self.setEnabled(ready)
        if not project:
            self.headline.setText("請先在左側選擇或建立專案")
        elif not ready:
            self.headline.setText("字幕對齊完成後，才能在這裡校對字幕")
        else:
            red = sum(1 for c in self.cues if c.flag(low) == "red")
            yellow = sum(1 for c in self.cues if c.flag(low) == "yellow")
            self.headline.setText(f"共 {len(self.cues)} 則字幕"
                                  + (f"，{red} 則時間是內插的" if red else "")
                                  + (f"，{yellow} 則對齊信心偏低" if yellow else "")
                                  + ("，沒有需要特別注意的" if not red and not yellow else ""))

    def _fill_table(self, low: float) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.cues))
        labels = {"red": "內插", "yellow": "信心偏低", "": ""}
        for row, cue in enumerate(self.cues):
            flag = cue.flag(low)
            state = "手動調整" if cue.method == "manual" else labels[flag] or f"{cue.confidence:.2f}"
            for column, text in enumerate((str(cue.id), _stamp(cue.start), _stamp(cue.end), state, cue.text)):
                item = QTableWidgetItem(text)
                if flag and column == 3:
                    item.setForeground(QColor(_FLAG_COLORS[flag]))
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)

    def _refresh_row(self, row: int) -> None:
        cue = self.cues[row]
        self.table.item(row, 1).setText(_stamp(cue.start))
        self.table.item(row, 2).setText(_stamp(cue.end))
        if cue.edited:
            self.table.item(row, 3).setText("已修改")

    # ---- 選取與調整 ----
    def _current(self) -> int:
        return self.timeline.current

    def _select(self, row: int) -> None:
        if not 0 <= row < len(self.cues) or row == self.timeline.current:
            return
        self.timeline.set_current(row)
        self.table.blockSignals(True)
        self.table.selectRow(row)
        self.table.blockSignals(False)
        cue = self.cues[row]
        self.scroll.ensureVisible(int((cue.start + cue.end) / 2 * self.timeline.scale), 100, 200, 0)

    def _on_changed(self, row: int) -> None:
        self._dirty = True
        self._refresh_row(row)
        self.status.setText("有尚未儲存的修改")

    def _nudge(self, edge: str, delta: float) -> None:
        row = self._current()
        if row < 0:
            self.status.setText("請先選取一則字幕")
            return
        cue = self.cues[row]
        E.move_edge(self.cues, row, edge, (cue.start if edge == "start" else cue.end) + delta, self.duration)
        self.timeline.update()
        self._on_changed(row)

    # ---- 試聽 ----
    def _load_audio(self, full_song: bool = False) -> None:
        if self.project is None:
            return
        path = self.project.path("audio", "song.flac" if full_song else "vocals.wav")
        self.player.setSource(QUrl.fromLocalFile(str(path)) if path.exists() else QUrl())

    def _play_line(self) -> None:
        row = self._current()
        if row < 0:
            self.status.setText("請先選取一則字幕")
            return
        cue = self.cues[row]
        self._stop_at = cue.end + 0.15
        self.player.setPosition(int(max(cue.start - 0.15, 0) * 1000))
        self.player.play()

    def _on_position(self, position: int) -> None:
        seconds = position / 1000
        self.timeline.set_playhead(seconds)
        if 0 <= self._stop_at <= seconds:
            self._stop_at = -1.0
            QTimer.singleShot(0, self.player.pause)

    # ---- 儲存 ----
    def has_unsaved_changes(self) -> bool:
        return self._dirty

    def save(self) -> None:
        if self.project is None or not self.cues:
            return
        if not self._dirty:
            self.status.setText("沒有修改")
            return
        try:
            E.save_cues(self.project.folder, self.cues)
        except UserFacingError as e:
            self.status.setText(f"✗ {e}")
            return
        self._dirty = False
        self.project.mark_stale_after("s5_video")       # 只有字幕合成與匯出需要重做
        self._fill_table(float(self.config.get("align.low_confidence", 0.4)))
        self.status.setText("✓ 已儲存字幕時間")
        self.saved.emit()

    def release(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())

    def shutdown(self) -> None:
        self.release()
