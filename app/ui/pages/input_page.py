"""輸入頁：四項輸入（音樂內容、曲風、歌詞來源與範本、人像照片）與製作參數。"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import soundfile
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
                               QInputDialog, QLabel, QListWidget, QMessageBox, QPlainTextEdit, QPushButton,
                               QRadioButton, QSpinBox, QVBoxLayout, QWidget)

from app.core.lyric_templates import NEW_TEMPLATE_TEXT, describe
from app.core.models import UserFacingError
from app.pipeline.common import duration_mode, style_presets, voice_options
from app.services.engines import JOIN_ANCHOR, JOIN_PHOTO, JOIN_PREV, make_engine
from app.pipeline.stages.s6_compose import subtitle_style
from app.ui.subtitle_style_dialog import edit_subtitle_style
from app.pipeline.stages.s2_lyrics import template_store
from app.storage.app_config import AppConfig
from app.storage.project_store import Project

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 改了這些輸入，哪個階段之後的產出就過期
JOIN_KEYS = ("join_mode", "join_every", "anchor_strength")
JOIN_LABELS = ((JOIN_ANCHOR, "以上一段的最後一幀起始，段尾回到照片的樣子"),
               (JOIN_PHOTO, "定期改以原始照片起始（畫面會跳接）"),
               (JOIN_PREV, "一律以上一段的最後一幀起始（不回到照片）"))
_FALLBACK_JOIN = {"join_mode": JOIN_PHOTO, "join_every": 4, "anchor_strength": 0.8}

_STALE_AFTER = {
    "s1_env": ("content", "style", "language", "lyrics_source", "lyric_template"),
    "s2_lyrics": ("vocal_type", "voice_character", "duration_mode", "target_seconds"),
    "s4_subtitle": ("photo", "engine", "resolution", *JOIN_KEYS),
}


def segment_hint(config: AppConfig, project: Project | None, engine_id: str) -> str:
    """這個引擎每段最長幾秒、整首歌大約分成幾段；讓使用者據此決定分段銜接的方式。

    切點要落在換氣或段落交界處，所以實際的段會比上限短，段數只能給範圍；
    專案已經用這個引擎分過段的話，直接顯示實際的段數。
    """
    settings = config.get(f"engines.{engine_id}", {})
    longest = float(settings.get("max_segment_seconds", 0) or 0)
    if longest <= 0:
        return ""
    text = f"每段最長 {longest:g} 秒"
    duration, known = 150.0, False
    if project is not None:
        song = project.path("audio", "song.flac")
        if song.exists():
            try:
                info = soundfile.info(str(song))
                duration, known = info.frames / info.samplerate, True
            except (RuntimeError, OSError):
                pass
        plan_file = project.path("analysis", "segment_plan.json")
        if known and plan_file.exists():
            try:
                plan = json.loads(plan_file.read_text(encoding="utf-8"))
                if plan.get("engine") == engine_id and abs(plan.get("duration", 0) - duration) < 0.5:
                    count = len(plan["segments"])
                    return (f"{text}。這首歌 {duration:.0f} 秒，上次用這個引擎分成 {count} 段"
                            f"（平均每段 {duration / count:.1f} 秒），有 {count - 1} 個交界。")
            except (ValueError, KeyError, ZeroDivisionError):
                pass
    # 實測各引擎的平均段長大約是上限的 65% 到 85%
    low, high = max(math.ceil(duration / longest), 1), max(math.ceil(duration / (longest * 0.65)), 1)
    count = f"{low} 段" if low == high else f"{low} 到 {high} 段"
    subject = f"這首歌 {duration:.0f} 秒" if known else "以 150 秒的歌估算"
    return f"{text}（切點落在換氣或段落交界處，實際會短一些）。{subject}，大約分成 {count}。"


def _about(seconds: float) -> str:
    minutes = max(round(seconds / 60), 1)
    return f"約 {minutes} 分鐘" if minutes < 60 else f"約 {minutes // 60} 小時 {minutes % 60:02d} 分"


def time_hint(config: AppConfig, project: Project | None, engine_id: str, resolution: str,
              sample_seconds: float = 30.0) -> str:
    """這個引擎生成影片大概要多久：以 30 秒的影片為標準，有歌曲時再加上整首的時間。

    專案已經用同一個引擎、同一個尺寸生成過，就用它自己的實測速度；否則用設定檔裡的參考值。
    只算對嘴影片這個階段，不含歌詞、歌曲、字幕（合計約幾分鐘）。
    """
    settings = config.get(f"engines.{engine_id}", {})
    speed, source = settings.get("speed", {}).get(resolution), "依 RTX 5090 Laptop 的實測值估算"
    duration = None
    if project is not None:
        song = project.path("audio", "song.flac")
        if song.exists():
            try:
                info = soundfile.info(str(song))
                duration = info.frames / info.samplerate
            except (RuntimeError, OSError):
                pass
        try:
            plan = json.loads(project.path("analysis", "segment_plan.json").read_text(encoding="utf-8"))
            size = (settings.get("resolutions") or config.get("video.resolutions", {})).get(resolution, {})
            done = [g for g in project.stage("s5_video").get("segments", []) if g.get("seconds") and g.get("frames")]
            if plan.get("engine") == engine_id and plan.get("width") == size.get("width") and done:
                speed = sum(g["seconds"] for g in done) / (sum(g["frames"] for g in done) / float(plan["fps"]))
                source = "依這個專案上次生成的實際速度估算"
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            pass
    if not speed:
        return f"預估時間：這個引擎的 {resolution} 沒有實測過，無法估算。"
    text = f"預估時間：30 秒的影片{_about(speed * sample_seconds)}"
    if duration:
        text += f"；這首歌 {duration:.0f} 秒{_about(speed * duration)}"
    return f"{text}（只算對嘴影片階段，{source}）。"


class PhotoDrop(QLabel):
    """可拖放或點選的照片區。"""
    chosen = Signal(str)

    def __init__(self):
        super().__init__("把人像照片拖到這裡\n或點一下選擇檔案")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumSize(240, 240)
        self.setStyleSheet("QLabel { border: 2px dashed #999999; border-radius: 8px; color: #888888; }")

    def show_photo(self, path: Path | None) -> None:
        pixmap = QPixmap(str(path)) if path and path.exists() else QPixmap()
        if pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText("把人像照片拖到這裡\n或點一下選擇檔案")
        else:
            self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                         Qt.TransformationMode.SmoothTransformation))

    def mousePressEvent(self, event) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇人像照片", "", "圖片 (*.png *.jpg *.jpeg *.webp *.bmp)")
        if path:
            self.chosen.emit(path)

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls and Path(urls[0].toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        self.chosen.emit(event.mimeData().urls()[0].toLocalFile())


class TemplateManager(QDialog):
    """歌詞範本管理：新增、編輯、刪除。範本是 TOML 文字，儲存時會驗證。"""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.store = template_store(config)
        self.setWindowTitle("歌詞範本管理")
        self.resize(820, 520)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._show)
        self.editor = QPlainTextEdit()
        self.editor.setStyleSheet("font-family: Consolas, 'Microsoft JhengHei UI';")
        self.note = QLabel("")
        self.note.setWordWrap(True)
        new_button, save_button, delete_button = QPushButton("新增"), QPushButton("儲存"), QPushButton("刪除")
        new_button.clicked.connect(self._new)
        save_button.clicked.connect(self._save)
        delete_button.clicked.connect(self._delete)
        buttons = QHBoxLayout()
        for b in (new_button, save_button, delete_button):
            buttons.addWidget(b)
        buttons.addStretch(1)
        right = QVBoxLayout()
        right.addWidget(self.editor, 1)
        right.addWidget(self.note)
        right.addLayout(buttons)
        layout = QHBoxLayout(self)
        layout.addWidget(self.list, 1)
        layout.addLayout(right, 2)
        self._ids: list[str] = []
        self._reload()

    def _reload(self, select: str | None = None) -> None:
        templates = self.store.list()
        self._ids = [t["id"] for t in templates]
        self.list.clear()
        for t in templates:
            self.list.addItem(f"{t['name']}{'' if t['is_user'] else '（內建）'}")
        if self._ids:
            self.list.setCurrentRow(self._ids.index(select) if select in self._ids else 0)

    def _current(self) -> str | None:
        row = self.list.currentRow()
        return self._ids[row] if 0 <= row < len(self._ids) else None

    def _show(self, row: int) -> None:
        template_id = self._current()
        if template_id:
            self.editor.setPlainText(self.store.read_text(template_id))
            self.note.setText(f"代號：{template_id}")

    def _new(self) -> None:
        template_id, ok = QInputDialog.getText(self, "新增範本", "範本代號（英數、底線或中文）：")
        if ok and template_id.strip():
            try:
                self.store.save(template_id.strip(), NEW_TEMPLATE_TEXT)
            except UserFacingError as e:
                QMessageBox.warning(self, "新增範本", str(e))
                return
            self._reload(template_id.strip())

    def _save(self) -> None:
        template_id = self._current()
        if not template_id:
            return
        try:
            template = self.store.save(template_id, self.editor.toPlainText())
        except UserFacingError as e:
            QMessageBox.warning(self, "儲存範本", str(e))
            return
        self._reload(template_id)
        self.note.setText(f"已儲存：{describe(template)}")

    def _delete(self) -> None:
        template_id = self._current()
        if not template_id:
            return
        try:
            self.store.delete(template_id)
        except UserFacingError as e:
            QMessageBox.information(self, "刪除範本", str(e))
            return
        self._reload()


class InputPage(QWidget):
    start_requested = Signal()
    reset_requested = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.project: Project | None = None
        self._photo: str = ""

        self.content = QPlainTextEdit()
        self.content.setPlaceholderText("描述歌曲的主題、故事與情緒，例如：夏天夜晚騎車穿過城市，想起一個已經離開的人…")
        self.content.setMaximumHeight(120)
        self.style = QComboBox()
        self.style.setEditable(True)
        self.style.addItems([p["name"] for p in style_presets(config)])
        self.language = QComboBox()
        for code, info in config.get("languages", {}).items():
            self.language.addItem(info.get("label", code), code)
        if self.language.count() == 0:
            self.language.addItem("國語", "zh-TW")
        self.voice = QComboBox()
        self.voice.setToolTip("聲線會寫進 YuE2 的風格描述；選「自動」則由 AI 依歌曲內容決定")
        for option in voice_options(config):
            self.voice.addItem(option["label"], option["value"])
        self.vocal_female, self.vocal_male = QRadioButton("女聲"), QRadioButton("男聲")
        self._vocal_group = QButtonGroup(self)
        self._vocal_group.addButton(self.vocal_female)
        self._vocal_group.addButton(self.vocal_male)
        self.length_auto = QRadioButton("自動（由 AI 決定，唱完為止）")
        self.length_auto.setToolTip("歌曲會唱完整份歌詞後自己結束，長度依歌詞多寡與速度而定")
        self.length_max = QRadioButton("設定上限")
        self.length_max.setToolTip("歌曲不會超過這個長度。AI 會先把歌詞結構縮到上限內唱得完；"
                                   "唱完時還不到上限，就以實際長度為準")
        self._length_group = QButtonGroup(self)
        self._length_group.addButton(self.length_auto)
        self._length_group.addButton(self.length_max)
        self.length_max.toggled.connect(self._update_enabled)
        self.target = QSpinBox()
        self.target.setRange(30, 600)
        self.target.setPrefix("最多 ")
        self.target.setSuffix(" 秒")
        self.source_ai, self.source_manual = QRadioButton("AI 產生"), QRadioButton("自己輸入")
        self._source_group = QButtonGroup(self)
        self._source_group.addButton(self.source_ai)
        self._source_group.addButton(self.source_manual)
        self.source_ai.toggled.connect(self._update_enabled)
        self.template = QComboBox()
        manage = QPushButton("管理範本…")
        manage.clicked.connect(self._manage_templates)
        self.engine = QComboBox()
        for engine_id, engine in config.get("engines", {}).items():
            self.engine.addItem(engine["label"], engine_id)
        self.engine.activated.connect(lambda _: self._load_join(self._engine_join(self.engine.currentData())[0]))
        self.engine.setToolTip("速度與成品特性差很多，選擇前請先看說明文件的「如何選擇對嘴引擎」")
        self._join_cache: dict[str, tuple[dict, bool]] = {}
        self.join_mode = QComboBox()
        for value, label in JOIN_LABELS:
            self.join_mode.addItem(label, value)
        self.join_mode.setToolTip("整首歌要分成好幾段生成再接起來。這裡決定每一段從哪個畫面開始，以及怎麼防止人物越接越不像")
        self.join_mode.currentIndexChanged.connect(self._update_enabled)
        self.join_every = QSpinBox()
        self.join_every.setRange(1, 20)
        self.join_every.setPrefix("每 ")
        self.join_every.setSuffix(" 段回到照片一次")
        self.join_every.setToolTip("數字越大動作越自由、回到照片的次數越少，但人物比較容易走樣")
        self.join_strength = QDoubleSpinBox()
        self.join_strength.setRange(0.3, 1.0)
        self.join_strength.setSingleStep(0.05)
        self.join_strength.setDecimals(2)
        self.join_strength.setPrefix("貼近照片的程度 ")
        self.join_strength.setToolTip("1.0 是段尾完全貼齊照片；調低只是靠近照片，回歸的痕跡較淡，但人物可能慢慢走樣")
        self.join_note = QLabel("")
        self.join_note.setStyleSheet("color: #888888;")
        self.join_note.setWordWrap(True)
        self.engine_note = QLabel("")
        self.engine_note.setStyleSheet("color: #888888;")
        self.engine_note.setWordWrap(True)
        self.resolution = QComboBox()
        for name in config.get("video.resolutions", {}):
            self.resolution.addItem(name, name)
        self.resolution.currentIndexChanged.connect(self._update_engine_note)
        self.mode_auto, self.mode_step = QRadioButton("全自動"), QRadioButton("逐步確認")
        self.mode_step.setToolTip("在歌詞產生後、歌曲生成後、字幕產生後各停一次，讓你檢查、修改或重新產生")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self.mode_auto)
        self._mode_group.addButton(self.mode_step)

        self.photo = PhotoDrop()
        self.photo.chosen.connect(self._choose_photo)
        self.photo_note = QLabel("")
        self.photo_note.setWordWrap(True)
        self.photo_note.setStyleSheet("color: #888888;")

        def row(*widgets) -> QHBoxLayout:
            box = QHBoxLayout()
            for w in widgets:
                box.addWidget(w)
            box.addStretch(1)
            return box

        form = QFormLayout()
        form.addRow("音樂內容", self.content)
        form.addRow("曲風", self.style)
        form.addRow("歌曲語言", self.language)
        form.addRow("人聲類型", row(self.vocal_female, self.vocal_male))
        form.addRow("聲線", self.voice)
        form.addRow("歌曲長度", row(self.length_auto, self.length_max, self.target))
        form.addRow("歌詞來源", row(self.source_ai, self.source_manual))
        form.addRow("歌詞範本", row(self.template, manage))
        form.addRow("對嘴引擎", self.engine)
        form.addRow("", self.engine_note)
        form.addRow("解析度", self.resolution)
        form.addRow("分段銜接", self.join_mode)
        form.addRow("", row(self.join_every, self.join_strength))
        form.addRow("", self.join_note)
        form.addRow("執行模式", row(self.mode_auto, self.mode_step))
        self.subtitle_button = QPushButton("字幕樣式…")
        self.subtitle_button.setToolTip("字型、大小、位置、顏色、描邊、陰影或底框；隨時可以改，改完只會重新燒字幕")
        self.subtitle_button.clicked.connect(self._edit_subtitle_style)
        self.subtitle_summary = QLabel("")
        self.subtitle_summary.setStyleSheet("color: #888888;")
        form.addRow("字幕", row(self.subtitle_button, self.subtitle_summary))

        side = QVBoxLayout()
        side.addWidget(QLabel("人像照片（站立或坐在麥克風前）"))
        side.addWidget(self.photo)
        side.addWidget(self.photo_note)
        side.addStretch(1)

        top = QHBoxLayout()
        left = QVBoxLayout()
        left.addLayout(form)
        left.addStretch(1)          # 視窗拉高時多出來的空間留在表單下方，不要把最後一列撐開
        top.addLayout(left, 3)
        top.addLayout(side, 2)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.start_button = QPushButton("開始製作")
        self.start_button.setMinimumHeight(36)
        self.start_button.clicked.connect(self._start)
        self.reset_button = QPushButton("重新開始…")
        self.reset_button.setToolTip("把這個專案的設定回復成預設值，或連同已產生的歌詞、歌曲、影片全部清除")
        self.reset_button.clicked.connect(self.reset_requested.emit)
        bottom = QHBoxLayout()
        bottom.addWidget(self.reset_button)
        bottom.addWidget(self.message, 1)
        bottom.addWidget(self.start_button)

        layout = QVBoxLayout(self)
        layout.addLayout(top, 1)
        layout.addLayout(bottom)
        self.set_project(None)

    # ---- 專案 ----
    def _reload_templates(self, select: str) -> None:
        self.template.clear()
        for t in template_store(self.config).list():
            self.template.addItem(f"{t['name']}（{describe(t)}）", t["id"])
        index = self.template.findData(select)
        self.template.setCurrentIndex(index if index >= 0 else 0)

    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.setEnabled(project is not None)
        self.message.setText("")
        inputs = project.data.get("inputs", {}) if project else {}
        cfg = self.config
        self.content.setPlainText(inputs.get("content", ""))
        self.style.setCurrentText(inputs.get("style", self.style.itemText(0)))
        (self.vocal_male if inputs.get("vocal_type") == "male" else self.vocal_female).setChecked(True)
        self.language.setCurrentIndex(max(self.language.findData(inputs.get("language", cfg.get("song.language"))), 0))
        self.voice.setCurrentIndex(max(self.voice.findData(inputs.get("voice_character", "")), 0))
        mode = duration_mode(inputs, cfg) if project else cfg.get("song.duration_mode", "auto")
        (self.length_max if mode == "max" else self.length_auto).setChecked(True)
        self.target.setValue(int(inputs.get("target_seconds", cfg.get("song.target_seconds", 150))))
        (self.source_manual if inputs.get("lyrics_source") == "manual" else self.source_ai).setChecked(True)
        self._reload_templates(inputs.get("lyric_template", "verse_chorus_bridge"))
        self.engine.setCurrentIndex(max(self.engine.findData(inputs.get("engine", cfg.get("video.default_engine"))), 0))
        self.resolution.setCurrentIndex(
            max(self.resolution.findData(inputs.get("resolution", cfg.get("video.default_resolution"))), 0))
        self._load_join({**self._engine_join(self.engine.currentData())[0],
                         **{k: inputs[k] for k in JOIN_KEYS if inputs.get(k) not in (None, "")}})
        step = bool(project) and project.data.get("mode") == "step"
        (self.mode_step if step else self.mode_auto).setChecked(True)
        self._photo = inputs.get("photo", "")
        self._show_photo()
        self._update_enabled()
        self._show_subtitle_summary()

    def _show_subtitle_summary(self) -> None:
        if self.project is None:
            self.subtitle_summary.setText("")
            return
        style = subtitle_style(self.config, self.project)
        self.subtitle_summary.setText(f"{style['font']}，大小 {style['size_percent']:g}%")

    def _edit_subtitle_style(self) -> None:
        if self.project is not None and edit_subtitle_style(self.config, self.project, self):
            self._show_subtitle_summary()

    def _show_photo(self) -> None:
        path = self.project.path(self._photo) if self.project and self._photo else None
        self.photo.show_photo(path)
        self.photo_note.setText("" if path and path.exists() else "尚未選擇照片；對嘴影片階段需要它。")

    def _update_enabled(self, *_) -> None:
        self.template.setEnabled(self.source_ai.isChecked())
        self.target.setEnabled(self.length_max.isChecked())
        mode = self.join_mode.currentData()
        self.join_every.setEnabled(mode != JOIN_PREV)
        self.join_strength.setEnabled(mode == JOIN_ANCHOR)
        self.join_note.setText({
            JOIN_ANCHOR: "畫面連續不跳接；人物在換氣處平順地回到照片的樣子，所以會週期性出現同一個表情。",
            JOIN_PHOTO: "回到照片的那一刻畫面會直接跳接，像換鏡頭；只發生在換氣或段落交界處。",
            # 引擎自己有說明的（例如人物不會走樣的 Wan 2.2 S2V）優先用它的
            JOIN_PREV: self.config.get(f"engines.{self.engine.currentData()}.join_note", "")
            or "畫面連續，也不會回到照片；段數多時（LTX-2.3 約二十幾段）後段的人物很可能已經不像本人。",
        }.get(mode, ""))

    # ---- 分段銜接 ----
    def _engine_join(self, engine_id: str) -> tuple[dict, bool]:
        """（這個引擎預設的銜接方式, 是否支援段尾回到照片）。"""
        if engine_id not in self._join_cache:
            try:
                engine = make_engine(self.config, engine_id)
                self._join_cache[engine_id] = (engine.default_join(), engine.supports_anchor)
            except (UserFacingError, KeyError, OSError):
                self._join_cache[engine_id] = (dict(_FALLBACK_JOIN), False)
        return self._join_cache[engine_id]

    def _load_join(self, join: dict) -> None:
        """把銜接設定放到畫面上；引擎不支援段尾回到照片時停用那個選項。"""
        supported = self._engine_join(self.engine.currentData())[1]
        anchor_item = self.join_mode.model().item(self.join_mode.findData(JOIN_ANCHOR))
        anchor_item.setEnabled(supported)
        anchor_item.setToolTip("" if supported else "這個引擎沒有段尾回到照片用的工作流")
        mode = join.get("join_mode")
        if mode == JOIN_ANCHOR and not supported:
            mode = JOIN_PHOTO
        self.join_mode.setCurrentIndex(max(self.join_mode.findData(mode), 0))
        self.join_every.setValue(int(join.get("join_every", 1)))
        self.join_strength.setValue(float(join.get("anchor_strength", 0.8)))
        self._update_enabled()
        self._update_engine_note()

    def _update_engine_note(self, *_) -> None:
        engine_id = self.engine.currentData()
        self.engine_note.setText("\n".join(filter(None, [
            segment_hint(self.config, self.project, engine_id),
            time_hint(self.config, self.project, engine_id, self.resolution.currentData())])))

    def _manage_templates(self) -> None:
        TemplateManager(self.config, self).exec()
        self._reload_templates(self.template.currentData() or "")

    def _choose_photo(self, source: str) -> None:
        if self.project is None:
            return
        src = Path(source)
        if src.suffix.lower() not in IMAGE_SUFFIXES:
            self.message.setText("這不是支援的圖片格式（png、jpg、webp、bmp）")
            return
        # 複製進專案；原圖不動
        for old in self.project.path("input").glob("photo_original.*"):
            old.unlink()
        dest = self.project.path("input", f"photo_original{src.suffix.lower()}")
        try:
            shutil.copyfile(src, dest)
        except OSError as e:
            self.message.setText(f"無法複製照片：{e}")
            return
        self._photo = f"input/{dest.name}"
        self._show_photo()
        self.save()

    # ---- 儲存與開始 ----
    def collect(self) -> dict:
        return {
            "content": self.content.toPlainText().strip(),
            "style": self.style.currentText().strip(),
            "language": self.language.currentData(),
            "vocal_type": "male" if self.vocal_male.isChecked() else "female",
            "voice_character": self.voice.currentData() or "",
            "duration_mode": "max" if self.length_max.isChecked() else "auto",
            "target_seconds": self.target.value(),
            "lyrics_source": "manual" if self.source_manual.isChecked() else "ai",
            "lyric_template": self.template.currentData() or "verse_chorus_bridge",
            "photo": self._photo,
            "engine": self.engine.currentData(),
            "resolution": self.resolution.currentData(),
            "join_mode": self.join_mode.currentData(),
            "join_every": self.join_every.value(),
            "anchor_strength": round(self.join_strength.value(), 2),
        }

    def save(self) -> list[str]:
        """把畫面上的值寫回專案；回傳因此過期的階段。"""
        if self.project is None:
            return []
        old = self.project.data.get("inputs", {})
        new = {**old, **self.collect()}
        # 舊專案沒有銜接欄位：把引擎的預設值當成原本的值，改了才會重新生成影片
        old = {**{k: v for k, v in self._engine_join(old.get("engine", ""))[0].items() if "engine" in old}, **old}
        stale: list[str] = []
        for after, keys in _STALE_AFTER.items():
            if any(old.get(k) != new.get(k) for k in keys if k in old):
                stale = self.project.mark_stale_after(after)
                break
        self.project.data["inputs"] = new
        self.project.data["mode"] = "step" if self.mode_step.isChecked() else "auto"
        self.project.save()
        return stale

    def _start(self) -> None:
        inputs = self.collect()
        if inputs["lyrics_source"] == "ai" and not inputs["content"]:
            self.message.setText("請先填寫音樂內容")
            return
        if not inputs["style"]:
            self.message.setText("請先選擇或輸入曲風")
            return
        self.save()
        self.message.setText("")
        self.start_requested.emit()
