"""歌詞範本：定義段落順序、每段行數、每行字數範圍、押韻與寫作風格提示。

內建範本在 config/lyric_templates/*.toml；使用者新增或修改的存在使用者設定資料夾，
同名（同 id）時使用者的版本優先。範本就是 TOML 文字檔，可直接閱讀與編輯。
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from app.core.lyrics import SECTION_LABELS, SECTION_TYPES
from app.core.models import UserFacingError

_ID = re.compile(r"^[A-Za-z0-9_\-一-鿿]+$")

NEW_TEMPLATE_TEXT = '''name = "我的範本"
style_hint = "口語、畫面感強"
rhyme = "每段偶數行押韻"

[[sections]]
type = "verse"          # intro、verse、prechorus、chorus、bridge、outro
lines = 4               # 0 代表純音樂段落
chars_min = 7
chars_max = 11

[[sections]]
type = "chorus"
lines = 4
chars_min = 6
chars_max = 10

[[sections]]
type = "chorus"
repeat_of = 2           # 與第 2 段的歌詞完全相同
'''


def parse_template(text: str, template_id: str = "") -> dict:
    """解析並驗證範本文字；有問題時指出是哪一段、哪個欄位。"""
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise UserFacingError(f"範本不是合法的 TOML：{e}", "請檢查引號與等號")
    problems = []
    if not str(data.get("name", "")).strip():
        problems.append("缺少 name（範本名稱）")
    sections = data.get("sections")
    if not isinstance(sections, list) or not sections:
        problems.append("至少要有一個 [[sections]] 段落")
        sections = []
    normalized = []
    for number, section in enumerate(sections, start=1):
        kind = section.get("type")
        if kind not in SECTION_TYPES:
            problems.append(f"第 {number} 段：type 必須是 {'、'.join(SECTION_TYPES)} 其中之一")
            # 放一個佔位，後面段落的 repeat_of 編號才對得上
            normalized.append({"type": "verse", "lines": 0, "chars_min": 0, "chars_max": 0, "repeat_of": 0})
            continue
        repeat_of = section.get("repeat_of", 0)
        if repeat_of:
            if not isinstance(repeat_of, int) or not 1 <= repeat_of < number:
                problems.append(f"第 {number} 段：repeat_of 必須是前面某一段的編號（1 到 {number - 1}）")
            elif normalized[repeat_of - 1]["repeat_of"]:
                problems.append(f"第 {number} 段：repeat_of 不能指向另一個重複段落")
            elif normalized[repeat_of - 1]["lines"] == 0:
                problems.append(f"第 {number} 段：repeat_of 指向的第 {repeat_of} 段沒有歌詞")
            else:
                source = normalized[repeat_of - 1]
                normalized.append({**source, "type": kind, "repeat_of": repeat_of})
                continue
            normalized.append({"type": kind, "lines": 0, "chars_min": 0, "chars_max": 0, "repeat_of": 0})
            continue
        lines = section.get("lines", 0)
        low, high = section.get("chars_min", 0), section.get("chars_max", 0)
        if not isinstance(lines, int) or lines < 0 or lines > 16:
            problems.append(f"第 {number} 段：lines 必須是 0 到 16 的整數")
            lines = 0
        if lines and not (isinstance(low, int) and isinstance(high, int) and 1 <= low <= high <= 30):
            problems.append(f"第 {number} 段：chars_min、chars_max 必須是 1 到 30 的整數，且 chars_min ≤ chars_max")
        normalized.append({"type": kind, "lines": lines, "chars_min": low, "chars_max": high, "repeat_of": 0})
    if not problems and not any(s["lines"] for s in normalized):
        problems.append("至少要有一個有歌詞的段落（lines 大於 0）")
    if problems:
        raise UserFacingError("歌詞範本有誤：" + "；".join(problems), "請修正後再儲存")
    return {
        "id": template_id,
        "name": str(data["name"]).strip(),
        "style_hint": str(data.get("style_hint", "")).strip(),
        "rhyme": str(data.get("rhyme", "")).strip(),
        "sections": normalized,
    }


def total_lines(template: dict) -> int:
    return sum(s["lines"] for s in template["sections"])


def describe(template: dict) -> str:
    """一行摘要，例如「主歌–副歌–主歌–副歌–過渡–副歌 · 24 行」。"""
    order = "–".join(SECTION_LABELS[s["type"]] for s in template["sections"])
    return f"{order} · {total_lines(template)} 行"


class LyricTemplateStore:
    def __init__(self, builtin_dir: Path, user_dir: Path):
        self.builtin_dir = Path(builtin_dir)
        self.user_dir = Path(user_dir)

    def _files(self) -> dict[str, tuple[Path, bool]]:
        """id → （檔案、是否為使用者範本）；使用者的版本蓋過內建。"""
        found: dict[str, tuple[Path, bool]] = {}
        for folder, is_user in ((self.builtin_dir, False), (self.user_dir, True)):
            if folder.is_dir():
                for file in sorted(folder.glob("*.toml")):
                    found[file.stem] = (file, is_user)
        return found

    def list(self) -> list[dict]:
        """所有可用的範本；損毀的範本檔略過。"""
        templates = []
        for template_id, (file, is_user) in self._files().items():
            try:
                template = parse_template(file.read_text(encoding="utf-8"), template_id)
            except (UserFacingError, OSError):
                continue
            template["is_user"] = is_user
            templates.append(template)
        return templates

    def get(self, template_id: str) -> dict:
        entry = self._files().get(template_id)
        if entry is None:
            raise UserFacingError(f"找不到歌詞範本「{template_id}」", "請在輸入頁重新選擇歌詞範本")
        template = parse_template(entry[0].read_text(encoding="utf-8"), template_id)
        template["is_user"] = entry[1]
        return template

    def read_text(self, template_id: str) -> str:
        entry = self._files().get(template_id)
        if entry is None:
            raise UserFacingError(f"找不到歌詞範本「{template_id}」")
        return entry[0].read_text(encoding="utf-8")

    def save(self, template_id: str, text: str) -> dict:
        """驗證後存成使用者範本；修改內建範本會存成同 id 的使用者版本。"""
        template_id = template_id.strip()
        if not _ID.match(template_id):
            raise UserFacingError("範本代號只能用中英文、數字、底線與連字號", "例如 my_ballad")
        template = parse_template(text, template_id)
        self.user_dir.mkdir(parents=True, exist_ok=True)
        (self.user_dir / f"{template_id}.toml").write_text(text, encoding="utf-8")
        return template

    def delete(self, template_id: str) -> None:
        """只能刪除使用者範本；若它蓋過內建範本，刪除後內建版本會回來。"""
        file = self.user_dir / f"{template_id}.toml"
        if not file.exists():
            raise UserFacingError("內建範本不能刪除", "可以另存為新的範本再修改")
        file.unlink()
