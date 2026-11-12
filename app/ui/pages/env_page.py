"""環境檢查頁：階段 1 的檢查清單，以及確認 ComfyUI 連線的測試工作流。"""
from __future__ import annotations

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QProgressBar, QPushButton,
                               QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from app.core.models import DEGRADE, FAIL, PASS
from app.pipeline.stage_base import StageContext
from app.pipeline.stages.s1_env import EnvStage, load_report
from app.pipeline.test_run import run_yue2_test
from app.storage.app_config import AppConfig
from app.storage.project_store import Project
from app.ui.workers import TaskWorker

_MARKS = {PASS: "✓", DEGRADE: "⚠", FAIL: "✗"}
_COLORS = {PASS: "#2e9d5b", DEGRADE: "#c98a00", FAIL: "#d0453a"}
_LEVEL_TEXT = {PASS: "通過", DEGRADE: "需降規", FAIL: "不通過"}


class EnvPage(QWidget):
    report_ready = Signal(dict)

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.project: Project | None = None
        self.report: dict | None = None
        self._worker: TaskWorker | None = None

        self.summary = QLabel("請先在左側選擇或建立專案")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size: 15px; font-weight: 600;")

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["項目", "結果", "說明"])
        self.tree.setRootIsDecorated(False)
        self.tree.setWordWrap(True)
        self.tree.setColumnWidth(0, 190)
        self.tree.setColumnWidth(1, 80)

        self.recheck_button = QPushButton("重新檢查")
        self.recheck_button.clicked.connect(self.run_check)
        self.test_button = QPushButton("送出測試工作流")
        self.test_button.setToolTip("用 YuE2 產生約 10 秒的音樂，確認送出、等待、下載整條路是通的")
        self.test_button.clicked.connect(self.run_test)
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self._cancel)
        self.cancel_button.hide()
        self.open_button = QPushButton("開啟測試音檔")
        self.open_button.clicked.connect(self._open_test_file)
        self.open_button.hide()
        self._test_file = None

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress_text = QLabel("")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setPlaceholderText("記錄")

        buttons = QHBoxLayout()
        for b in (self.recheck_button, self.test_button, self.cancel_button, self.open_button):
            buttons.addWidget(b)
        buttons.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addWidget(self.summary)
        layout.addWidget(self.tree, 3)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.progress_text)
        layout.addWidget(self.log_box, 1)
        self._set_busy(False)

    # ---- 專案切換 ----
    def set_project(self, project: Project | None) -> None:
        self.project = project
        self.report = None
        self.tree.clear()
        self.log_box.clear()
        self.open_button.hide()
        self._set_busy(False)
        if project is None:
            self.summary.setText("請先在左側選擇或建立專案")
            return
        saved = load_report(project.folder)
        if saved:
            self._show_report(saved)
        else:
            self.run_check()

    # ---- 環境檢查 ----
    def run_check(self) -> None:
        if self.project is None or self._worker is not None:
            return
        project, stage = self.project, EnvStage()

        def task(reporter, cancel):
            stage.run(StageContext(self.config, project, cancel), reporter)
            return stage.report

        self.summary.setText("檢查中…")
        self._start(task, self._show_report)

    def _show_report(self, report: dict) -> None:
        self.report = report
        self.tree.clear()
        for item in report["items"]:
            level = item["level"]
            row = QTreeWidgetItem([item["label"], f"{_MARKS[level]} {_LEVEL_TEXT[level]}", item["detail"]])
            row.setForeground(1, QBrush(QColor(_COLORS[level])))
            row.setToolTip(2, item["detail"])
            self.tree.addTopLevelItem(row)
            for line in item["fix"].splitlines():
                fix = QTreeWidgetItem(["", "", f"→ {line}"])
                fix.setForeground(2, QBrush(QColor("#888888")))
                fix.setToolTip(2, line)
                row.addChild(fix)
            row.setExpanded(True)
        usable = [self.config.get(f"engines.{e}.label", e)
                  for e, lv in report.get("engines", {}).items() if lv != FAIL]
        if report["overall"] == FAIL:
            failed = "、".join(i["label"] for i in report["items"] if i["level"] == FAIL)
            self.summary.setText(f"✗ 環境檢查不通過，無法開始製作。請先處理：{failed}")
            self.summary.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {_COLORS[FAIL]};")
        elif report["overall"] == DEGRADE:
            self.summary.setText(f"⚠ 可以開始製作，但有項目需降規。可用的對嘴引擎：{'、'.join(usable)}")
            self.summary.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {_COLORS[DEGRADE]};")
        else:
            self.summary.setText(f"✓ 環境檢查通過。可用的對嘴引擎：{'、'.join(usable)}")
            self.summary.setStyleSheet(f"font-size: 15px; font-weight: 600; color: {_COLORS[PASS]};")
        self._set_busy(False)
        self.report_ready.emit(report)

    # ---- 測試工作流 ----
    def run_test(self) -> None:
        if self.project is None or self._worker is not None or not self.report:
            return
        url, dest = self.report.get("comfy_url"), self.project.logs_dir

        def task(reporter, cancel):
            return run_yue2_test(self.config, url, dest, reporter, cancel)

        self.open_button.hide()
        self._start(task, self._test_done)

    def _test_done(self, path) -> None:
        self._log(f"✓ 測試工作流完成：{path}")
        self._test_file = path
        self.open_button.show()

    def _open_test_file(self) -> None:
        if self._test_file is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._test_file)))

    # ---- 共用 ----
    def _start(self, task, on_success) -> None:
        worker = TaskWorker(task, self)
        worker.log.connect(self._log)
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(on_success)
        worker.failed.connect(self._on_failed)
        worker.cancelled.connect(lambda: self._log("已取消"))
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self._set_busy(True)
        worker.start()

    def _on_finished(self) -> None:
        self._worker = None
        self._set_busy(False)

    def _cancel(self) -> None:
        if self._worker is not None:
            self._log("取消中…")
            self._worker.cancel()

    def _on_failed(self, message: str, fix: str) -> None:
        self._log(f"✗ {message}")
        if fix:
            self._log(f"   → {fix}")
        if self.report is None:
            self.summary.setText(f"✗ {message}")

    def _on_progress(self, fraction: float, text: str) -> None:
        self.progress.setValue(int(fraction * 1000))
        self.progress_text.setText(text)

    def _log(self, text: str) -> None:
        self.log_box.appendPlainText(text)

    def _set_busy(self, busy: bool) -> None:
        has_project = self.project is not None
        comfy_ok = bool(self.report and self.report.get("comfy_url"))
        self.recheck_button.setEnabled(has_project and not busy)
        self.test_button.setEnabled(has_project and comfy_ok and not busy)
        self.cancel_button.setVisible(busy)
        if not busy:
            self.progress.setValue(0)
            self.progress_text.setText("")

    def is_busy(self) -> bool:
        return self._worker is not None

    def shutdown(self) -> None:
        """關閉視窗前：取消並等待背景工作結束。"""
        if self._worker is not None:
            self._worker.cancel()
            self._worker.wait(5000)
