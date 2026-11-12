"""MV 自動工坊：程式進入點。執行方式：python -m app.main"""
from __future__ import annotations

import json
import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from app.storage.app_config import AppConfig
from app.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("MV 自動工坊")
    app.setFont(QFont("Microsoft JhengHei UI", 10))
    config = AppConfig()
    window = MainWindow(config)
    if "--selftest" in sys.argv:
        # 打包後的自我檢查：確認設定、工作流範本與主視窗都載得起來，把結果寫到指定的檔案後結束
        from app.pipeline.runner import default_stages
        from app.services.engines import ENGINES, make_engine
        import soundfile
        from app.core.lyrics import to_traditional
        from app.core.lyric_templates import LyricTemplateStore
        from app.storage import secret_store
        report = {"root": str(config.root), "stages": list(default_stages()),
                  "engines": [make_engine(config, name).label for name in ENGINES],
                  "pages": window.stack.count(),
                  "opencc": to_traditional("后视镜"),
                  "soundfile": soundfile.__libsndfile_version__,
                  "keyring": secret_store.get_secret("selftest-not-set") is None,
                  "lyric_templates": len(LyricTemplateStore(config.root / "config" / "lyric_templates",
                                                            config.root / "none").list())}
        target = sys.argv[sys.argv.index("--selftest") + 1]
        with open(target, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False)
        return 0
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
