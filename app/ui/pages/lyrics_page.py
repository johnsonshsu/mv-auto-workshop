"""歌詞頁：歌詞編輯器（段落以色塊區分），可整首或單段重新產生。"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout,
                               QWidget)

from app.core import lyrics as L
from app.core.models import DONE, UserFacingError
from app.pipeline.common import duration_mode, style_hint
from app.pipeline.stages.s2_lyrics import (generate_lyrics, load_lyrics, regenerate_section, save_lyrics,
                                           template_store)
from app.services.llm_client import make_llm
from app.storage.app_config import AppConfig
from app.storage.project_store import Project
from app.ui.workers import TaskWorker

SECTION_COLORS = {
    "intro": "#7a7a7a",
    "verse": "#2f6fd0",
    "prechorus": "#1f9d8f",
    "chorus": "#d9771f",
    "bridge": "#8a4fd0",
    "outro": "#7a7a7a",
}


class LyricsHighlighter(QSyntaxHighlighter):
    """段落標記用該段的顏色加粗，段落內的歌詞加上同色的淡底。"""

    def highlightBlock(self, text: str) -> None:
        line = text.strip()
        kind = L.marker_type(line) if line else None
        state = self.previousBlockState()
        fmt = QTextCharFormat()
        if kind:                                      # 看得懂的段落標記
            state = L.SECTION_TYPES.index(kind)
            fmt.setForeground(QColor(SECTION_COLORS[kind]))
            fmt.setFontWeight(QFont.Weight.Bold)
            self.setFormat(0, len(text), fmt)
        elif kind == "":                              # 看不懂的標記
            fmt.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)
            fmt.setUnderlineColor(QColor("#d0453a"))
            self.setFormat(0, len(text), fmt)
        elif line and state >= 0:
            tint = QColor(SECTION_COLORS[L.SECTION_TYPES[state]])
            tint.setAlpha(34)
            fmt.setBackground(tint)
            self.setFormat(0, len(text), fmt)
        self.setCurrentBlockState(state)


class LyricsPage(QWidget):
    lyrics_saved = Signal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.project: Project | None = None
        self._worker: TaskWorker | None = None
        self._saved_text = ""

        self.title = QLabel("")
        self.title.setStyleSheet("font-size: 15px; font-weight: 600;")
        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(
            "在這裡貼上或編輯歌詞。用 [主歌]、[副歌]、[過渡]（或 [Verse]、[Chorus]、[Bridge]）標示段落，一行一句。")
        self.editor.setStyleSheet("font-size: 15px;")
        self._highlighter = LyricsHighlighter(self.editor.document())
        self.editor.textChanged.connect(self._refresh_info)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setMinimumWidth(260)
        self.info.setAlignment(Qt.AlignmentFlag.AlignTop)
        legend = "　".join(f'<span style="color:{SECTION_COLORS[t]}">■ {L.SECTION_LABELS[t]}</span>'
                          for t in ("verse", "prechorus", "chorus", "bridge"))
        self.legend = QLabel(legend)

        self.save_button = QPushButton("儲存歌詞")
        self.save_button.clicked.connect(self.save)
        self.regen_all = QPushButton("整首重新產生")
        self.regen_all.clicked.connect(self._regenerate_all)
        self.regen_section = QPushButton("重新產生游標所在的段落")
        self.regen_section.clicked.connect(self._regenerate_section)
        self.to_trad = QPushButton("簡體轉繁體")
        self.to_trad.clicked.connect(self._to_traditional)
        self.status = QLabel("")
        self.status.setWordWrap(True)

        side = QVBoxLayout()
        side.addWidget(self.legend)
        side.addWidget(self.info, 1)
        body = QHBoxLayout()
        body.addWidget(self.editor, 3)
        body.addLayout(side, 1)
        buttons = QHBoxLayout()
        for b in (self.save_button, self.regen_all, self.regen_section, self.to_trad):
            buttons.addWidget(b)
        buttons.addStretch(1)
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addLayout(body, 1)
        layout.addLayout(buttons)
        layout.addWidget(self.status)
        self.set_project(None)

    # ---- 專案 ----
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.setEnabled(project is not None)
        self.status.setText("")
        self.reload()

    def reload(self) -> None:
        """從專案重新載入歌詞（歌詞階段跑完後呼叫）。"""
        lyrics = load_lyrics(self.project) if self.project else None
        if lyrics:
            text = L.to_editor_text(lyrics)
        else:
            text = (self.project.data.get("inputs", {}).get("manual_lyrics", "") if self.project else "")
        self.title.setText(f"《{lyrics['title']}》" if lyrics and lyrics.get("title") else "歌詞")
        self._saved_text = text
        self.editor.setPlainText(text)
        self._refresh_info()

    def is_busy(self) -> bool:
        return self._worker is not None

    def has_unsaved_changes(self) -> bool:
        return self.editor.toPlainText().strip() != self._saved_text.strip()

    # ---- 即時檢查 ----
    def _parse(self) -> dict:
        title = self.title.text().strip("《》") if self.title.text().startswith("《") else ""
        language = self.project.data.get("inputs", {}).get("language", "zh-TW") if self.project else "zh-TW"
        return L.parse_lyrics_text(self.editor.toPlainText(), title or (self.project.name if self.project else ""),
                                   language)

    def _refresh_info(self) -> None:
        if not self.editor.toPlainText().strip():
            self.info.setText("還沒有歌詞。")
            return
        try:
            lyrics = self._parse()
        except L.LyricsFormatError as e:
            self.info.setText("<b style='color:#d0453a'>格式有誤</b><br>" + "<br>".join(e.problems))
            return
        inputs = self.project.data.get("inputs", {}) if self.project else {}
        bpm = L.bpm_from_text(style_hint(self.config, str(inputs.get("style", ""))))
        estimate = L.lyrics_seconds(lyrics, bpm)
        rows = [f"段落：{len(lyrics['sections'])}", f"歌詞行數：{len(L.lyric_lines(lyrics))}",
                f"唱完估計需要：約 {estimate:.0f} 秒"]
        if duration_mode(inputs, self.config) == "max":
            limit = float(inputs.get("target_seconds", 0) or 0)
            if limit and estimate > limit:
                rows.append(f"<span style='color:#d0453a'>超過上限 {limit:.0f} 秒，歌曲可能唱不完；"
                            "請減少歌詞、提高上限，或把長度改成「自動」</span>")
            elif limit:
                rows.append(f"上限 {limit:.0f} 秒內唱得完")
        else:
            rows.append("長度由 AI 決定，唱完為止")
        long_lines = [line for line in L.lyric_lines(lyrics) if L.count_chars(line) > 14]
        if long_lines:
            rows.append(f"<span style='color:#c98a00'>有 {len(long_lines)} 行超過 14 字，可能不好唱</span>")
        code = self.project.data.get("inputs", {}).get("language", "zh-TW") if self.project else "zh-TW"
        chinese = self.config.get("languages", {}).get(code, {}).get("chinese", True)
        simplified = L.find_simplified("".join(L.lyric_lines(lyrics))) if chinese else []
        if simplified:
            rows.append(f"<span style='color:#c98a00'>含有簡體字：{'、'.join(simplified)}"
                        "（字幕會照這裡的文字顯示）</span>")
        if self.has_unsaved_changes():
            rows.append("<span style='color:#c98a00'>有尚未儲存的修改</span>")
        self.info.setText("<br>".join(rows))

    # ---- 儲存 ----
    def save(self) -> bool:
        if self.project is None:
            return False
        try:
            lyrics = self._parse()
        except L.LyricsFormatError as e:
            self.status.setText("✗ 無法儲存：" + "；".join(e.problems))
            return False
        changed = save_lyrics(self.project, lyrics)
        inputs = self.project.data.setdefault("inputs", {})
        if inputs.get("lyrics_source") == "manual":
            inputs["manual_lyrics"] = self.editor.toPlainText()
        # 使用者親手確認過的歌詞視為歌詞階段已完成
        self.project.set_status("s2_lyrics", DONE)
        self._saved_text = self.editor.toPlainText()
        self.status.setText("✓ 已儲存" + ("；歌曲與之後的階段需要重新產生" if changed else ""))
        self._refresh_info()
        self.lyrics_saved.emit()
        return True

    def _to_traditional(self) -> None:
        text = self.editor.toPlainText()
        converted = L.to_traditional(text)
        if converted == text:
            self.status.setText("沒有需要轉換的簡體字")
        else:
            self.editor.setPlainText(converted)
            self.status.setText("已轉成繁體，確認後請按「儲存歌詞」")

    # ---- 重新產生 ----
    def _cursor_section(self) -> int:
        """游標所在的段落編號（從 0 起）；游標在第一個標記之前回傳 -1。"""
        block = self.editor.textCursor().block()
        index = -1
        current = self.editor.document().firstBlock()
        while current.isValid():
            if L.marker_type(current.text().strip()):
                index += 1
            if current == block:
                break
            current = current.next()
        return index

    def _regenerate_all(self) -> None:
        if self.project is None or self._worker is not None:
            return
        inputs = dict(self.project.data.get("inputs", {}))
        if not inputs.get("content"):
            self.status.setText("✗ 請先到輸入頁填寫音樂內容")
            return
        if self.editor.toPlainText().strip() and QMessageBox.question(
                self, "整首重新產生", "目前的歌詞會被取代，確定要重新產生嗎？") != QMessageBox.StandardButton.Yes:
            return
        config = self.config

        def task(reporter, cancel):
            template = template_store(config).get(inputs.get("lyric_template", "verse_chorus_bridge"))
            llm = make_llm(config)
            try:
                return generate_lyrics(llm, template, inputs, config, reporter)
            finally:
                llm.unload()

        self._start(task, "產生歌詞中，本機模型第一次載入約需一分鐘…")

    def _regenerate_section(self) -> None:
        if self.project is None or self._worker is not None:
            return
        try:
            lyrics = self._parse()
        except L.LyricsFormatError as e:
            self.status.setText("✗ 請先修正格式：" + "；".join(e.problems))
            return
        index = self._cursor_section()
        if index < 0:
            self.status.setText("✗ 請先把游標放在要重新產生的段落裡")
            return
        inputs = dict(self.project.data.get("inputs", {}))
        config = self.config

        def task(reporter, cancel):
            try:
                template = template_store(config).get(inputs.get("lyric_template", ""))
            except UserFacingError:
                template = None
            llm = make_llm(config)
            try:
                return regenerate_section(llm, lyrics, index, inputs, config, template, reporter)
            finally:
                llm.unload()

        label = L.SECTION_LABELS[lyrics["sections"][index]["type"]]
        self._start(task, f"重新產生第 {index + 1} 段（{label}）中…")

    def _start(self, task, message: str) -> None:
        worker = TaskWorker(task, self)
        worker.log.connect(self.status.setText)
        worker.succeeded.connect(self._generated)
        worker.failed.connect(lambda m, fix: self.status.setText(f"✗ {m}" + (f"\n→ {fix}" if fix else "")))
        worker.finished.connect(self._finished)
        self._worker = worker
        self._set_busy(True)
        self.status.setText(message)
        worker.start()

    def _generated(self, lyrics: dict) -> None:
        self.title.setText(f"《{lyrics['title']}》" if lyrics.get("title") else "歌詞")
        self.editor.setPlainText(L.to_editor_text(lyrics))
        self.save()

    def _finished(self) -> None:
        self._worker = None
        self._set_busy(False)

    def _set_busy(self, busy: bool) -> None:
        for widget in (self.save_button, self.regen_all, self.regen_section):
            widget.setEnabled(not busy)
        self.editor.setReadOnly(busy)

    def shutdown(self) -> None:
        if self._worker is not None:
            self._worker.wait(5000)
