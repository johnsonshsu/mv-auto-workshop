"""主視窗：左側專案清單，右側精靈六頁。"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                               QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QSplitter,
                               QStackedWidget, QVBoxLayout, QWidget)

from app.core.models import DONE, FAILED, STAGE_IDS, STALE, UserFacingError
from app.pipeline.common import default_inputs
from app.storage import secret_store
from app.storage.app_config import AppConfig
from app.storage.project_store import Project, ProjectStore
from app.ui.pages.done_page import DonePage
from app.ui.pages.env_page import EnvPage
from app.ui.pages.input_page import InputPage
from app.ui.pages.lyrics_page import LyricsPage
from app.ui.pages.run_page import RunPage
from app.ui.pages.subtitle_page import SubtitlePage

# 精靈頁面：（標題、預計完成的開發階段）
PAGES = [("環境檢查", ""), ("輸入", ""), ("歌詞", ""), ("執行", ""), ("字幕校對", ""), ("完成", "")]


def _project_mark(project: Project) -> str:
    statuses = [project.status(s) for s in STAGE_IDS]
    if all(s == DONE for s in statuses):
        return "✓"
    if FAILED in statuses or STALE in statuses:
        return "⚠"
    return "●"


class SettingsDialog(QDialog):
    """設定（M1 範圍：ComfyUI 位址與專案資料夾；其餘項目隨各開發階段加入）。"""

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.setWindowTitle("設定")
        self.setMinimumWidth(520)

        self.comfy_url = QLineEdit(config.get("comfyui.last_url", ""))
        self.comfy_url.setPlaceholderText("留白＝自動偵測（先試 8000，再試 8188）")
        self.projects_root = QLineEdit(str(config.get("paths.projects_root")))
        browse = QPushButton("瀏覽…")
        browse.clicked.connect(self._browse)
        root_row = QHBoxLayout()
        root_row.addWidget(self.projects_root)
        root_row.addWidget(browse)

        self.provider = QComboBox()
        for value, label in (("ollama", "本機 Ollama"), ("claude", "Claude API"),
                             ("openai_compatible", "OpenAI 相容端點")):
            self.provider.addItem(label, value)
        self.provider.setCurrentIndex(max(self.provider.findData(config.get("llm.provider")), 0))
        self.ollama_text = QLineEdit(config.get("llm.ollama_text_model", ""))
        self.ollama_vision = QLineEdit(config.get("llm.ollama_vision_model", ""))
        self.claude_model = QLineEdit(config.get("llm.claude_model", ""))
        self.claude_key = self._key_field("claude_api_key")
        self.openai_url = QLineEdit(config.get("llm.openai_base_url", ""))
        self.openai_model = QLineEdit(config.get("llm.openai_model", ""))
        self.openai_key = self._key_field("openai_api_key")

        form = QFormLayout()
        form.addRow("ComfyUI 位址", self.comfy_url)
        form.addRow("專案資料夾", root_row)
        form.addRow("LLM 供應商", self.provider)
        form.addRow("Ollama 文字模型", self.ollama_text)
        form.addRow("Ollama 視覺模型", self.ollama_vision)
        form.addRow("Claude 模型", self.claude_model)
        form.addRow("Claude API 金鑰", self.claude_key)
        form.addRow("OpenAI 相容端點", self.openai_url)
        form.addRow("OpenAI 相容模型", self.openai_model)
        form.addRow("OpenAI 相容金鑰", self.openai_key)
        note = QLabel("金鑰存在 Windows 認證管理員，不會寫進專案檔或記錄。欄位留白表示不變更。")
        note.setStyleSheet("color: #888888;")
        form.addRow("", note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("儲存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "選擇專案資料夾", self.projects_root.text())
        if folder:
            self.projects_root.setText(folder)

    def _save(self) -> None:
        root = self.projects_root.text().strip()
        if not root:
            QMessageBox.warning(self, "設定", "專案資料夾不可空白")
            return
        self.config.set("comfyui.last_url", self.comfy_url.text().strip().rstrip("/"))
        self.config.set("paths.projects_root", root)
        self.config.set("llm.provider", self.provider.currentData())
        for path, field in (("llm.ollama_text_model", self.ollama_text),
                            ("llm.ollama_vision_model", self.ollama_vision),
                            ("llm.claude_model", self.claude_model),
                            ("llm.openai_base_url", self.openai_url),
                            ("llm.openai_model", self.openai_model)):
            self.config.set(path, field.text().strip())
        for name, field in (("claude_api_key", self.claude_key), ("openai_api_key", self.openai_key)):
            if field.text().strip():
                secret_store.set_secret(name, field.text().strip())
        self.accept()

    @staticmethod
    def _key_field(name: str) -> QLineEdit:
        field = QLineEdit()
        field.setEchoMode(QLineEdit.EchoMode.Password)
        field.setPlaceholderText("已設定（留白不變更）" if secret_store.get_secret(name) else "尚未設定")
        return field


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.store = ProjectStore(Path(config.get("paths.projects_root")))
        self.current: Project | None = None
        self.setWindowTitle("MV 自動工坊")
        self.resize(1180, 760)

        # ---- 左側：專案清單 ----
        self.project_list = QListWidget()
        self.project_list.currentItemChanged.connect(self._on_project_selected)
        new_button = QPushButton("＋ 新專案")
        new_button.clicked.connect(self._new_project)
        folder_button = QPushButton("開啟資料夾")
        folder_button.clicked.connect(self._open_folder)
        settings_button = QPushButton("⚙ 設定")
        settings_button.clicked.connect(self._open_settings)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("專案"))
        left_layout.addWidget(new_button)
        left_layout.addWidget(self.project_list, 1)
        left_layout.addWidget(folder_button)
        left_layout.addWidget(settings_button)

        # ---- 右側：步驟列＋頁面 ----
        self.stack = QStackedWidget()
        self.step_group = QButtonGroup(self)
        step_bar = QHBoxLayout()
        self.env_page = EnvPage(config)
        self.input_page = InputPage(config)
        self.lyrics_page = LyricsPage(config)
        self.run_page = RunPage(config)
        self.done_page = DonePage(config)
        self.subtitle_page = SubtitlePage(config)
        self.subtitle_page.saved.connect(self._on_subtitles_saved)
        pages = {0: self.env_page, 1: self.input_page, 2: self.lyrics_page, 3: self.run_page,
                 4: self.subtitle_page, 5: self.done_page}
        self.done_page.restyle_requested.connect(self._start_run)
        self.input_page.start_requested.connect(self._start_run)
        self.input_page.reset_requested.connect(self._reset_project)
        self.lyrics_page.lyrics_saved.connect(self._on_lyrics_saved)
        self.run_page.stage_done.connect(self._on_stage_done)
        self.run_page.run_finished.connect(self._on_run_finished)
        for index, (title, milestone) in enumerate(PAGES):
            button = QPushButton(f"{'①②③④⑤⑥'[index]} {title}")
            button.setCheckable(True)
            button.setStyleSheet(
                "QPushButton { border: none; border-bottom: 2px solid transparent; padding: 6px 12px;"
                " color: palette(text); background: transparent; }"
                "QPushButton:checked { color: palette(highlight); font-weight: 600;"
                " border-bottom: 2px solid palette(highlight); }")
            self.step_group.addButton(button, index)
            step_bar.addWidget(button)
            self.stack.addWidget(pages.get(index) or self._placeholder(title, milestone))
        step_bar.addStretch(1)
        self.step_group.idClicked.connect(self._go_to)
        self.step_group.button(0).setChecked(True)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addLayout(step_bar)
        right_layout.addWidget(self.stack, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([230, 950])
        self.setCentralWidget(splitter)
        self.statusBar().showMessage(f"專案資料夾：{self.store.root}")

        self._reload_projects()

    @staticmethod
    def _placeholder(title: str, milestone: str) -> QWidget:
        label = QLabel(f"「{title}」頁面將在開發階段 {milestone} 加入")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #888888; font-size: 15px;")
        return label

    # ---- 頁面切換與流程 ----
    def _is_busy(self) -> bool:
        return self.env_page.is_busy() or self.run_page.is_busy() or self.lyrics_page.is_busy()

    def _go_to(self, index: int) -> None:
        # 離開輸入頁時把畫面上的值存回專案
        if self.stack.currentWidget() is self.input_page and self.current and not self.run_page.is_busy():
            self.input_page.save()
            self.run_page.refresh()
        # 完成頁的播放器只在看完成頁時開著檔案，離開就放開，免得擋到之後要取代影片的階段
        if self.stack.widget(index) is self.done_page:
            if not self.run_page.is_busy():
                self.done_page.reload()
        else:
            self.done_page.release()
        self.stack.setCurrentIndex(index)
        self.step_group.button(index).setChecked(True)

    def _start_run(self) -> None:
        if self.lyrics_page.has_unsaved_changes() and not self.lyrics_page.save():
            self._go_to(2)
            return
        self.done_page.release()        # 放開 final_mv.mp4，後面的階段可能要取代它
        self._go_to(3)
        self.run_page.start()

    def _on_lyrics_saved(self) -> None:
        self.run_page.refresh()
        self._refresh_marks()

    def _on_subtitles_saved(self) -> None:
        """字幕時間改了：影片已經生成的話，直接重新燒字幕。"""
        self.run_page.refresh()
        self._refresh_marks()
        if self.current and self.current.status("s5_video") == DONE and not self.run_page.is_busy():
            self._start_run()

    def _on_stage_done(self, stage_id: str) -> None:
        if stage_id == "s4_subtitle":
            self.subtitle_page.reload()
        if stage_id == "s2_lyrics":
            self.lyrics_page.reload()

    def _on_run_finished(self, outcome) -> None:
        from app.pipeline import runner as R
        self._refresh_marks()
        self.done_page.reload()
        if outcome.kind == R.FINISHED:
            self._go_to(5)
            return
        # 逐步確認：歌詞產生後帶到歌詞頁；缺歌詞時也帶到歌詞頁
        if (outcome.kind == R.PAUSED and outcome.stage_id == "s2_lyrics") or \
                (outcome.kind == R.BLOCKED and outcome.stage_id == "s2_lyrics"
                 and self.current and self.current.data.get("inputs", {}).get("lyrics_source") == "manual"):
            self._go_to(2)
            self.statusBar().showMessage(outcome.message, 10000)
        elif outcome.kind == R.PAUSED and outcome.stage_id == "s4_subtitle":
            self._go_to(4)
            self.statusBar().showMessage(outcome.message, 10000)
        elif outcome.kind == R.BLOCKED and outcome.stage_id == "s1_env":
            self.env_page.set_project(self.current)
            self._go_to(0)

    def _refresh_marks(self) -> None:
        for row in range(self.project_list.count()):
            item = self.project_list.item(row)
            if self.current and item.data(Qt.ItemDataRole.UserRole) == str(self.current.folder):
                item.setText(f"{_project_mark(self.current)}  {self.current.name}")

    # ---- 專案 ----
    def _reload_projects(self, select: str | None = None) -> None:
        self.project_list.blockSignals(True)
        self.project_list.clear()
        for project in self.store.list_projects():
            item = QListWidgetItem(f"{_project_mark(project)}  {project.name}")
            item.setData(Qt.ItemDataRole.UserRole, str(project.folder))
            self.project_list.addItem(item)
            if project.name == select:
                self.project_list.setCurrentItem(item)
        self.project_list.blockSignals(False)
        if select is None and self.project_list.count():
            self.project_list.setCurrentRow(0)
        else:
            self._on_project_selected(self.project_list.currentItem(), None)

    def _on_project_selected(self, item: QListWidgetItem | None, previous: QListWidgetItem | None) -> None:
        if self._is_busy() and previous is not None:
            # 背景工作屬於目前的專案，跑完才能切換
            self.project_list.blockSignals(True)
            self.project_list.setCurrentItem(previous)
            self.project_list.blockSignals(False)
            self.statusBar().showMessage("目前專案還有工作在執行，請等它完成或先取消", 5000)
            return
        if item is None:
            self.current = None
        else:
            try:
                self.current = self.store.load(Path(item.data(Qt.ItemDataRole.UserRole)))
            except UserFacingError as e:
                QMessageBox.warning(self, "開啟專案", str(e))
                self.current = None
        # 四個頁面共用同一個 Project 物件，狀態才會一致
        for page in (self.env_page, self.input_page, self.lyrics_page, self.run_page, self.subtitle_page,
                     self.done_page):
            page.set_project(self.current)
        if self.current is not None:
            interrupted = self.current.interrupted_stages()
            if interrupted:
                self.statusBar().showMessage("這個專案上次執行時被中斷，可從中斷處續跑", 8000)

    def _reset_project(self) -> None:
        """重新開始設計：只回復預設值，或連同已產生的檔案全部清除。"""
        project = self.current
        if project is None:
            return
        if self._is_busy():
            self.statusBar().showMessage("目前專案還有工作在執行，請等它完成或先取消", 5000)
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("重新開始")
        box.setText(f"要怎麼重新開始《{project.name}》？")
        box.setInformativeText(
            "只回復預設值：音樂內容、曲風、語言、聲線、長度、引擎、解析度、執行模式與字幕樣式回到預設值。"
            "照片和已經產生的歌詞、歌曲、影片會留在資料夾裡，但會標成已過期，下次製作時重新產生。\n\n"
            "全部清除：除了回復預設值，還會刪除照片以及這個專案已經產生的所有檔案"
            "（歌詞、歌曲、字幕、影片、記錄），無法復原。")
        keep = box.addButton("只回復預設值", QMessageBox.ButtonRole.AcceptRole)
        clear = box.addButton("全部清除", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        chosen = box.clickedButton()
        if chosen not in (keep, clear):
            return
        if chosen is clear and QMessageBox.warning(
                self, "全部清除", f"將刪除這個資料夾裡除了專案檔以外的所有內容，無法復原：\n{project.folder}\n\n確定要刪除嗎？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes:
            return
        # 放開播放器開著的檔案，Windows 才刪得掉
        self.done_page.release()
        self.subtitle_page.release()
        try:
            project.reset(default_inputs(self.config), clear_files=chosen is clear)
        except OSError as e:
            QMessageBox.warning(self, "重新開始", f"有檔案無法刪除：{e}\n請關閉正在使用這些檔案的程式後再試一次。"
                                              "設定已回復預設值，尚未刪除的檔案仍留在資料夾裡。")
            project.save()
        for page in (self.env_page, self.input_page, self.lyrics_page, self.run_page, self.subtitle_page,
                     self.done_page):
            page.set_project(project)
        self._refresh_marks()
        self.statusBar().showMessage("已全部清除，可以重新開始設計" if chosen is clear else "已回復成預設值", 8000)

    def _new_project(self) -> None:
        if self._is_busy():
            self.statusBar().showMessage("目前專案還有工作在執行，請等它完成或先取消", 5000)
            return
        name, ok = QInputDialog.getText(self, "新專案", "專案名稱（會成為資料夾名稱）：")
        if not ok:
            return
        try:
            project = self.store.create(name, default_inputs(self.config))
        except UserFacingError as e:
            QMessageBox.warning(self, "新專案", str(e))
            return
        except OSError as e:
            QMessageBox.warning(self, "新專案", f"無法建立專案資料夾：{e}\n請到設定確認專案資料夾的位置")
            return
        self._reload_projects(select=project.name)

    def _open_folder(self) -> None:
        target = self.current.folder if self.current else self.store.root
        if not target.exists():
            self.statusBar().showMessage(f"資料夾還不存在：{target}（建立第一個專案時會自動建立）", 5000)
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))

    def _open_settings(self) -> None:
        if self._is_busy():
            self.statusBar().showMessage("目前專案還有工作在執行，請等它完成或先取消", 5000)
            return
        if SettingsDialog(self.config, self).exec() == QDialog.DialogCode.Accepted:
            self.store = ProjectStore(Path(self.config.get("paths.projects_root")))
            self.statusBar().showMessage(f"專案資料夾：{self.store.root}")
            self._reload_projects()

    def closeEvent(self, event) -> None:
        for page in (self.env_page, self.lyrics_page, self.run_page, self.subtitle_page, self.done_page):
            page.shutdown()
        super().closeEvent(event)
