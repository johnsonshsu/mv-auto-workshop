"""共用的常數與資料型別。"""
from __future__ import annotations

from dataclasses import asdict, dataclass

# 階段狀態
PENDING = "pending"    # 待執行
RUNNING = "running"    # 執行中
DONE = "done"          # 完成
FAILED = "failed"      # 失敗
STALE = "stale"        # 已過期（上游產出被修改）

STATUS_LABELS = {
    PENDING: "待執行",
    RUNNING: "執行中",
    DONE: "完成",
    FAILED: "失敗",
    STALE: "已過期",
}

# 七個階段，依執行順序
STAGE_IDS = ["s1_env", "s2_lyrics", "s3_song", "s4_subtitle", "s5_video", "s6_compose", "s7_export"]

STAGE_LABELS = {
    "s1_env": "環境偵測",
    "s2_lyrics": "歌詞產生",
    "s3_song": "歌曲生成",
    "s4_subtitle": "字幕對齊",
    "s5_video": "對嘴影片",
    "s6_compose": "字幕合成",
    "s7_export": "完成匯出",
}

# 環境檢查的判定等級
PASS = "pass"          # 通過
DEGRADE = "degrade"    # 可執行但需降規
FAIL = "fail"          # 不通過


@dataclass
class CheckItem:
    """環境檢查的一個項目。"""
    id: str
    label: str
    level: str
    detail: str = ""
    fix: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Problem:
    """階段輸入檢查發現的問題。"""
    message: str
    fix: str = ""


class UserFacingError(Exception):
    """要直接顯示給使用者的錯誤：message 說明發生什麼事，fix 是建議動作。"""

    def __init__(self, message: str, fix: str = ""):
        super().__init__(message)
        self.message = message
        self.fix = fix

    def __str__(self) -> str:
        return f"{self.message}（{self.fix}）" if self.fix else self.message
