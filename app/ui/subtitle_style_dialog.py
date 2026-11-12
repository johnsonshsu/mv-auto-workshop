"""字幕樣式視窗：字型、大小、位置、顏色、描邊、陰影或底框，改動時即時預覽。

樣式只影響字幕合成，存進專案後只需要重跑階段 6、7。
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QCheckBox, QColorDialog, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout)

from app.core.models import UserFacingError
from app.pipeline.stages.s6_compose import (PREVIEW_SIZE, font_dirs, render_preview, resolve_font,
                                            subtitle_style)
from app.services import fonts
from app.storage.app_config import AppConfig
from app.storage.project_store import Project

POSITIONS = (("置底", "bottom"), ("置頂", "top"), ("畫面正中", "middle"))
BACKGROUNDS = (("無（只有描邊）", "none"), ("陰影", "shadow"), ("半透明底框", "box"))
SAMPLE_TEXT = "秋天的風吹過安靜的街道"


class ColorButton(QPushButton):
    changed = Signal()

    def __init__(self, color: str):
        super().__init__()
        self.color = color
        self.clicked.connect(self._pick)
        self._refresh()

    def set_color(self, color: str) -> None:
        self.color = color
        self._refresh()

    def _refresh(self) -> None:
        self.setText(self.color.upper())
        text = "#000000" if QColor(self.color).lightness() > 128 else "#FFFFFF"
        self.setStyleSheet(f"background: {self.color}; color: {text};")

    def _pick(self) -> None:
        chosen = QColorDialog.getColor(QColor(self.color), self, "選擇顏色")
        if chosen.isValid():
            self.set_color(chosen.name())
            self.changed.emit()


def first_lyric_line(project: Project) -> str:
    path = project.path("lyrics", "lyrics.txt")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.strip().startswith("["):
                return line.strip()
    return SAMPLE_TEXT


class SubtitleStyleDialog(QDialog):
    def __init__(self, config: AppConfig, project: Project, parent=None, accept_text: str = "套用"):
        super().__init__(parent)
        self.config, self.project = config, project
        self.setWindowTitle("字幕樣式")
        style = subtitle_style(config, project)
        self._work_dir = Path(tempfile.mkdtemp(prefix="mv_subtitle_preview_"))
        self._loading = True

        self.preset = QComboBox()
        self.preset.addItem("套用範本…", None)
        for preset in config.get("subtitle.presets", []):
            self.preset.addItem(preset["name"], preset)
        self.preset.activated.connect(self._apply_preset)

        # 字型：這台電腦上所有含繁體中文字的字型，加上設定檔列出的（沒安裝的會標示）
        self.font = QComboBox()
        self.font.setMaxVisibleItems(20)
        user_dirs = font_dirs()[1:]
        names = fonts.usable_for_subtitles(fonts.chinese_fonts(), user_dirs[0] if user_dirs else None)
        for name in config.get("subtitle.font_files", {}):
            if name not in names:
                names.append(name)
        if style["font"] not in names:
            names.append(style["font"])
        for name in names:
            self.font.addItem(name if resolve_font(config, name)[0] else f"{name}（未安裝）", name)
        self.font.setCurrentIndex(max(self.font.findData(style["font"]), 0))

        self.size = self._spin(style["size_percent"], 2.0, 20.0, 0.5)
        self.size_hint = QLabel("")
        self.size_hint.setStyleSheet("color: #888888;")
        self.position = self._choice(POSITIONS, style.get("position"))
        self.margin = self._spin(style["margin_percent"], 0.0, 40.0, 0.5)
        self.color = ColorButton(style["color"])
        self.bold = QCheckBox("粗體")
        self.bold.setChecked(bool(style.get("bold")))
        self.italic = QCheckBox("斜體")
        self.italic.setChecked(bool(style.get("italic")))
        self.outline = self._spin(style["outline_percent"], 0.0, 3.0, 0.05)
        self.outline_color = ColorButton(style["outline_color"])
        self.background = self._choice(BACKGROUNDS, style.get("background"))
        self.back_color = ColorButton(style.get("back_color") or "#000000")
        self.back_opacity = QSpinBox()
        self.back_opacity.setRange(0, 100)
        self.back_opacity.setSingleStep(5)
        self.back_opacity.setSuffix(" %")
        self.back_opacity.setValue(int(style.get("back_opacity", 60)))
        self.sample = QLineEdit(first_lyric_line(project))

        def row(*widgets) -> QHBoxLayout:
            box = QHBoxLayout()
            for widget in widgets:
                box.addWidget(widget)
            box.addStretch(1)
            return box

        form = QFormLayout()
        form.addRow("範本", self.preset)
        form.addRow("字型", self.font)
        form.addRow("大小（佔畫面高度 %）", row(self.size, self.size_hint))
        form.addRow("字體", row(self.bold, self.italic))
        form.addRow("文字顏色", self.color)
        form.addRow("位置", self.position)
        form.addRow("邊距（佔畫面高度 %）", self.margin)
        form.addRow("描邊顏色", self.outline_color)
        form.addRow("描邊粗細（佔畫面高度 %）", self.outline)
        form.addRow("文字背後", self.background)
        form.addRow("陰影／底框", row(self.back_color, QLabel("不透明度"), self.back_opacity))
        form.addRow("預覽文字", self.sample)

        self.preview = QLabel("")
        self.preview.setFixedSize(PREVIEW_SIZE[0] * 5 // 8, PREVIEW_SIZE[1] * 5 // 8)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet("background: #202020; color: #cccccc;")
        self.message = QLabel("套用後只會重新燒字幕，不會重新生成歌曲或影片。")
        self.message.setStyleSheet("color: #888888;")
        self.message.setWordWrap(True)
        right = QVBoxLayout()
        right.addWidget(self.preview)
        right.addWidget(self.message)
        right.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(accept_text)
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        body = QHBoxLayout()
        body.addLayout(form)
        body.addLayout(right)
        layout = QVBoxLayout(self)
        layout.addLayout(body)
        layout.addWidget(buttons)

        # 改動後稍等一下再重畫預覽，連續調整數字時不會每一步都跑一次 FFmpeg
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._render)
        for combo in (self.font, self.position, self.background):
            combo.currentIndexChanged.connect(self._changed)
        for spin in (self.size, self.margin, self.outline, self.back_opacity):
            spin.valueChanged.connect(self._changed)
        for check in (self.bold, self.italic):
            check.toggled.connect(self._changed)
        for button in (self.color, self.outline_color, self.back_color):
            button.changed.connect(self._changed)
        self.sample.textChanged.connect(self._changed)
        self.finished.connect(lambda _: shutil.rmtree(self._work_dir, ignore_errors=True))
        self._loading = False
        self._changed()
        self._render()

    @staticmethod
    def _spin(value: float, low: float, high: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setValue(float(value))
        return spin

    @staticmethod
    def _choice(options, current) -> QComboBox:
        combo = QComboBox()
        for label, value in options:
            combo.addItem(label, value)
        combo.setCurrentIndex(max(combo.findData(current), 0))
        return combo

    def style(self) -> dict:
        return {"font": self.font.currentData(), "size_percent": self.size.value(),
                "margin_percent": self.margin.value(), "outline_percent": self.outline.value(),
                "color": self.color.color, "outline_color": self.outline_color.color,
                "bold": self.bold.isChecked(), "italic": self.italic.isChecked(),
                "position": self.position.currentData(), "background": self.background.currentData(),
                "back_color": self.back_color.color, "back_opacity": self.back_opacity.value()}

    def _apply_preset(self, index: int) -> None:
        preset = self.preset.itemData(index)
        if not preset:
            return
        self._loading = True
        for key, spin in (("size_percent", self.size), ("margin_percent", self.margin),
                          ("outline_percent", self.outline), ("back_opacity", self.back_opacity)):
            if key in preset:
                spin.setValue(preset[key])
        for key, button in (("color", self.color), ("outline_color", self.outline_color),
                            ("back_color", self.back_color)):
            if key in preset:
                button.set_color(preset[key])
        for key, check in (("bold", self.bold), ("italic", self.italic)):
            if key in preset:
                check.setChecked(bool(preset[key]))
        for key, combo in (("font", self.font), ("position", self.position), ("background", self.background)):
            if key in preset and combo.findData(preset[key]) >= 0:
                combo.setCurrentIndex(combo.findData(preset[key]))
        self.preset.setCurrentIndex(0)
        self._loading = False
        self._changed()

    def _changed(self, *_) -> None:
        if self._loading:
            return
        percent = self.size.value()
        self.size_hint.setText(f"480p 約 {percent * 4.8:.0f} 像素、720p 約 {percent * 7.2:.0f} 像素")
        self.margin.setEnabled(self.position.currentData() != "middle")
        has_back = self.background.currentData() != "none"
        self.back_color.setEnabled(has_back)
        self.back_opacity.setEnabled(has_back)
        self._timer.start()

    def _render(self) -> None:
        try:
            image = render_preview(self.config, self.project, self.style(),
                                   self.sample.text().strip() or SAMPLE_TEXT, self._work_dir)
        except UserFacingError as e:
            self.preview.setPixmap(QPixmap())
            self.preview.setText(f"無法產生預覽：{e.message}")
            return
        self.preview.setPixmap(QPixmap(str(image)).scaled(
            self.preview.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))


def edit_subtitle_style(config: AppConfig, project: Project, parent=None, accept_text: str = "套用") -> bool:
    """開啟字幕樣式視窗；按下套用就存進專案並把字幕合成、匯出標成需要重做，回傳是否有套用。"""
    dialog = SubtitleStyleDialog(config, project, parent, accept_text)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False
    project.data.setdefault("params", {})["subtitle_style"] = dialog.style()
    project.mark_stale_after("s5_video")     # 只有字幕合成與匯出需要重做
    project.save()
    return True
