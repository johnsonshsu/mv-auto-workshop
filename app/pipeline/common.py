"""各階段共用的小工具。"""
from __future__ import annotations

import tomllib
from pathlib import Path

from app.pipeline.stage_base import StageContext
from app.pipeline.stages.s1_env import load_report
from app.services.comfy_client import ComfyClient, ComfyConnectionError
from app.storage.app_config import AppConfig


def load_prompt(config: AppConfig, name: str, **values: object) -> str:
    """讀取 prompts/<name>.md 並填入變數。用取代而非 str.format，提示詞裡的 JSON 大括號才不會出錯。"""
    text = (config.root / "prompts" / f"{name}.md").read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace("{" + key + "}", str(value))
    return text


def comfy_url(ctx: StageContext) -> str:
    """ComfyUI 位址：優先用這個專案環境檢查時的位址，否則重新偵測。"""
    report = load_report(ctx.project.folder) if ctx.project else None
    url = (report or {}).get("comfy_url")
    cfg = ctx.config
    found = ComfyClient.detect(cfg.get("comfyui.host"), cfg.get("comfyui.ports"),
                               url or cfg.get("comfyui.last_url", ""), cfg.get("comfyui.connect_timeout", 3.0))
    if not found:
        raise ComfyConnectionError()
    return found


def make_comfy(ctx: StageContext) -> ComfyClient:
    return ComfyClient(comfy_url(ctx), ws_poll_seconds=ctx.config.get("comfyui.ws_poll_seconds", 5.0))


def style_presets(config: AppConfig) -> list[dict]:
    path = Path(config.root) / "config" / "style_presets.toml"
    with open(path, "rb") as f:
        return tomllib.load(f).get("presets", [])


def style_hint(config: AppConfig, style: str) -> str:
    for preset in style_presets(config):
        if preset["name"] == style:
            return preset.get("hint", "")
    return ""


_DEFAULT_LANGUAGE = {"label": "國語", "yue2": "Mandarin", "whisper": "zh", "rule": "國語。全部使用繁體中文",
                     "unit": "字", "length_factor": 1.0, "chinese": True}


def language_info(config: AppConfig, code: str) -> dict:
    """某個歌曲語言的設定；不認得的代碼當成國語。"""
    info = config.get("languages", {}).get(code)
    return {**_DEFAULT_LANGUAGE, **info, "code": code} if info else {**_DEFAULT_LANGUAGE, "code": "zh-TW"}


def voice_options(config: AppConfig) -> list[dict]:
    return config.get("voices", []) or [{"label": "自動（依歌曲內容決定）", "value": ""}]


def default_inputs(config: AppConfig) -> dict:
    """新專案（或重設後的專案）的輸入預設值。"""
    return {
        "language": config.get("song.language"),
        "duration_mode": config.get("song.duration_mode", "auto"),
        "target_seconds": config.get("song.target_seconds"),
        "engine": config.get("video.default_engine"),
        "resolution": config.get("video.default_resolution"),
    }


def duration_mode(inputs: dict, config: AppConfig) -> str:
    """歌曲長度模式：auto（由 AI 決定，唱完為止）或 max（不超過設定的上限）。

    舊專案沒有這個欄位但有設定秒數時，沿用原本「有上限」的行為。
    """
    mode = inputs.get("duration_mode")
    if mode in ("auto", "max"):
        return mode
    return "max" if "target_seconds" in inputs else config.get("song.duration_mode", "auto")
