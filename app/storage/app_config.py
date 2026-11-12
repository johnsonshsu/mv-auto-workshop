"""設定檔：config/default.toml 為預設值，使用者設定（JSON）覆蓋同名項目。"""
from __future__ import annotations

import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any


def app_root() -> Path:
    """專案根目錄；打包成執行檔後為執行檔所在資料夾。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def user_settings_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "MVAutoWorkshop" / "settings.json"


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class AppConfig:
    """以點號路徑存取設定，例如 cfg.get("comfyui.ports")。"""

    def __init__(self, root: Path | None = None, user_path: Path | None = None):
        self.root = root or app_root()
        self.user_path = user_path or user_settings_path()
        with open(self.root / "config" / "default.toml", "rb") as f:
            self._defaults: dict = tomllib.load(f)
        with open(self.root / "config" / "thresholds.toml", "rb") as f:
            self.thresholds: dict = tomllib.load(f)
        self._user: dict = {}
        if self.user_path.exists():
            try:
                self._user = json.loads(self.user_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # 使用者設定損毀時退回預設值，不讓程式無法啟動
                self._user = {}
        self._data = _deep_merge(self._defaults, self._user)

    @property
    def workflows_dir(self) -> Path:
        return self.root / "workflows"

    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        """寫入使用者設定並立即存檔。"""
        parts = path.split(".")
        node = self._user
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
        self._data = _deep_merge(self._defaults, self._user)
        self.user_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.user_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._user, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.user_path)
