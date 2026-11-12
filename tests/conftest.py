from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.storage.app_config import AppConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(tmp_path) -> AppConfig:
    """使用專案內的預設設定，但使用者設定寫到暫存資料夾，不碰真正的設定檔。"""
    cfg = AppConfig(root=REPO, user_path=tmp_path / "settings.json")
    cfg.set("paths.projects_root", str(tmp_path / "projects"))
    return cfg


@pytest.fixture
def yue2_workflow() -> dict:
    return json.loads((REPO / "workflows" / "yue2_text_to_music.api.json").read_text(encoding="utf-8"))
