"""字幕校對用的資料與規則：載入字幕與對齊資訊、限制可調整的範圍、存回 SRT。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.core import srt_builder as S
from app.core.models import UserFacingError

MIN_CUE_SECONDS = 0.2      # 一則字幕最短多久
GAP = 0.02                 # 相鄰兩則之間至少留的間隔


@dataclass
class Cue:
    id: int
    text: str
    start: float
    end: float
    confidence: float
    method: str             # aligned | interpolated | manual
    edited: bool = False

    def flag(self, low_confidence: float) -> str:
        """需要注意的程度：red＝時間是內插的；yellow＝對齊信心偏低；空字串＝正常。"""
        if self.method == S.INTERPOLATED:
            return "red"
        if self.method == S.ALIGNED and self.confidence < low_confidence:
            return "yellow"
        return ""


def load_cues(project_folder: Path) -> tuple[list[Cue], float]:
    """由 lyrics.srt（時間與文字）與 alignment_report.json（信心值）組出字幕清單，回傳（清單、歌曲長度）。"""
    folder = Path(project_folder)
    try:
        cues = S.parse_srt((folder / "lyrics" / "lyrics.srt").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UserFacingError("還沒有字幕檔", "請先完成字幕對齊階段")
    report: dict = {}
    try:
        report = json.loads((folder / "analysis" / "alignment_report.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        pass
    details = {line["id"]: line for line in report.get("lines", [])}
    result = []
    for number, (start, end, text) in enumerate(cues, start=1):
        info = details.get(number, {})
        result.append(Cue(number, text, start, end, float(info.get("confidence", 1.0)),
                          info.get("method", S.ALIGNED)))
    duration = float(report.get("duration") or (result[-1].end if result else 0.0))
    return result, duration


def limits(cues: list[Cue], index: int, duration: float) -> tuple[float, float]:
    """第 index 則字幕可以佔用的時間範圍：不能跨過前後兩則。"""
    low = cues[index - 1].end + GAP if index > 0 else 0.0
    high = cues[index + 1].start - GAP if index + 1 < len(cues) else duration
    return low, high


def move_edge(cues: list[Cue], index: int, edge: str, value: float, duration: float) -> float:
    """把第 index 則的起點或終點移到 value，自動限制在合法範圍內，回傳實際設定的值。"""
    cue = cues[index]
    low, high = limits(cues, index, duration)
    if edge == "start":
        value = round(min(max(value, low), cue.end - MIN_CUE_SECONDS), 3)
        changed, cue.start = value != cue.start, value
    else:
        value = round(max(min(value, high), cue.start + MIN_CUE_SECONDS), 3)
        changed, cue.end = value != cue.end, value
    if changed:
        cue.edited = True
    return value


def shift(cues: list[Cue], index: int, delta: float, duration: float) -> float:
    """整則前後移動 delta 秒（長度不變），回傳實際移動的量。"""
    cue = cues[index]
    low, high = limits(cues, index, duration)
    delta = round(min(max(delta, low - cue.start), high - cue.end), 3)
    if delta:
        cue.start, cue.end, cue.edited = round(cue.start + delta, 3), round(cue.end + delta, 3), True
    return delta


def save_cues(project_folder: Path, cues: list[Cue]) -> None:
    """寫回 lyrics.srt，並在對齊報告記下哪些行是手動調整的。字幕文字不會被改動。"""
    folder = Path(project_folder)
    for a, b in zip(cues, cues[1:]):
        if a.end > b.start:
            raise UserFacingError(f"第 {a.id} 則與第 {b.id} 則字幕的時間重疊", "請調整其中一則的時間")
    (folder / "lyrics" / "lyrics.srt").write_text(
        S.to_srt([(c.start, c.end, c.text) for c in cues]), encoding="utf-8")
    report_path = folder / "analysis" / "alignment_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return
    edited = {c.id for c in cues if c.edited}
    for line in report.get("lines", []):
        if line["id"] in edited:
            line["method"] = "manual"
    for cue in cues:
        if cue.edited:
            cue.method, cue.edited = "manual", False
    report["manual_count"] = sum(1 for line in report.get("lines", []) if line.get("method") == "manual")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
