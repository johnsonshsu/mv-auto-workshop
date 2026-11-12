"""專案資料夾與 project.json。所有狀態都在專案資料夾內，整個搬走也能繼續。"""
from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.models import DONE, FAILED, PENDING, RUNNING, STAGE_IDS, STALE, UserFacingError

SCHEMA_VERSION = 1
PROJECT_FILE = "project.json"
SUBDIRS = ["input", "lyrics", "audio", "analysis", "segments", "output", "logs"]

_INVALID_NAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class Project:
    def __init__(self, folder: Path, data: dict):
        self.folder = Path(folder)
        self.data = data

    # ---- 路徑 ----
    @property
    def name(self) -> str:
        return self.data.get("name", self.folder.name)

    def path(self, *parts: str) -> Path:
        return self.folder.joinpath(*parts)

    @property
    def logs_dir(self) -> Path:
        return self.folder / "logs"

    # ---- 存檔 ----
    def save(self) -> None:
        """先寫暫存檔再取代，避免中途斷電損毀專案。"""
        target = self.folder / PROJECT_FILE
        tmp = self.folder / (PROJECT_FILE + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)

    # ---- 階段狀態 ----
    def stage(self, stage_id: str) -> dict:
        return self.data.setdefault("stages", {}).setdefault(stage_id, {"status": PENDING})

    def status(self, stage_id: str) -> str:
        return self.stage(stage_id).get("status", PENDING)

    def set_status(self, stage_id: str, status: str, **extra: Any) -> None:
        entry = self.stage(stage_id)
        entry["status"] = status
        if status == RUNNING:
            entry["started_at"] = _now()
        entry.update(extra)
        self.save()

    def add_warning(self, stage_id: str, text: str) -> None:
        self.stage(stage_id).setdefault("warnings", []).append(text)
        self.save()

    def mark_stale_after(self, stage_id: str) -> list[str]:
        """上游產出被修改：把 stage_id 之後已完成或失敗的階段標為已過期。"""
        changed = []
        index = STAGE_IDS.index(stage_id)
        for later in STAGE_IDS[index + 1:]:
            if self.status(later) in (DONE, FAILED, RUNNING):
                self.stage(later)["status"] = STALE
                changed.append(later)
        if changed:
            self.save()
        return changed

    def reset(self, defaults: dict, clear_files: bool) -> None:
        """重新開始設計：輸入、執行模式、字幕樣式等參數回到預設值。

        clear_files=False：保留照片與已產生的檔案，但它們不再對應新的設定，所以把做過的階段標為已過期。
        clear_files=True：連同照片與所有已產生的檔案一起刪除，階段全部回到待執行；只留下專案本身。
        """
        photo = self.data.get("inputs", {}).get("photo", "")
        self.data["inputs"] = dict(defaults)
        self.data["mode"] = "auto"
        self.data["params"] = {}
        if clear_files:
            for entry in self.folder.iterdir():
                if entry.name in (PROJECT_FILE, PROJECT_FILE + ".tmp"):
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
            for sub in SUBDIRS:
                (self.folder / sub).mkdir(exist_ok=True)
            self.data["seeds"] = {}
            self.data["stages"] = {stage_id: {"status": PENDING} for stage_id in STAGE_IDS}
            self.data["song_takes"] = []
        else:
            if photo:
                self.data["inputs"]["photo"] = photo
            self.mark_stale_after("s1_env")
        self.save()

    def first_unfinished(self) -> str | None:
        """續跑起點：第一個不是「完成」的階段。"""
        for stage_id in STAGE_IDS:
            if self.status(stage_id) != DONE:
                return stage_id
        return None

    def interrupted_stages(self) -> list[str]:
        """開啟專案時仍是「執行中」的階段，代表上次被中斷。"""
        return [s for s in STAGE_IDS if self.status(s) == RUNNING]


class ProjectStore:
    def __init__(self, root: Path):
        self.root = Path(root)

    def list_projects(self) -> list[Project]:
        if not self.root.exists():
            return []
        projects = []
        for folder in sorted(self.root.iterdir()):
            if (folder / PROJECT_FILE).is_file():
                try:
                    projects.append(self.load(folder))
                except UserFacingError:
                    continue
        return projects

    def create(self, name: str, defaults: dict | None = None) -> Project:
        name = name.strip()
        if not name:
            raise UserFacingError("專案名稱不可空白", "請輸入名稱")
        if _INVALID_NAME.search(name) or name.endswith((".", " ")):
            raise UserFacingError(
                f"專案名稱「{name}」含有不能用在資料夾名稱的字元", '請避免 < > : " / \\ | ? *')
        folder = self.root / name
        if folder.exists():
            raise UserFacingError(f"已經有名為「{name}」的專案或資料夾", "請換一個名稱")
        folder.mkdir(parents=True)
        for sub in SUBDIRS:
            (folder / sub).mkdir()
        data = {
            "schema_version": SCHEMA_VERSION,
            "name": name,
            "created_at": _now(),
            "mode": "auto",
            "inputs": dict(defaults or {}),
            "params": {},
            "seeds": {},
            "stages": {stage_id: {"status": PENDING} for stage_id in STAGE_IDS},
        }
        project = Project(folder, data)
        project.save()
        return project

    def load(self, folder: Path) -> Project:
        folder = Path(folder)
        file = folder / PROJECT_FILE
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise UserFacingError(f"找不到專案檔：{file}", "請確認資料夾是否為 MV 自動工坊的專案")
        except json.JSONDecodeError as e:
            raise UserFacingError(f"專案檔損毀：{file}（{e}）", "可從備份還原 project.json")
        if data.get("schema_version", 0) > SCHEMA_VERSION:
            raise UserFacingError("這個專案是由較新版本的 APP 建立的", "請更新 MV 自動工坊")
        for sub in SUBDIRS:
            (folder / sub).mkdir(exist_ok=True)
        return Project(folder, data)
