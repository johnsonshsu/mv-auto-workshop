"""段落時間表（sections.json）：每個段落的起訖時間、是否為純音樂，以及人聲靜音的間隙。

階段 5 用它決定影片的切點。
"""
from __future__ import annotations

import numpy as np

from app.core.srt_builder import Line


def detect_silences(samples: np.ndarray, samplerate: int, threshold_db: float = -40.0,
                    min_seconds: float = 0.4, frame_seconds: float = 0.02) -> list[dict]:
    """找出人聲音軌上 RMS 低於門檻且持續夠久的區間。samples 可為單聲道或多聲道。"""
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    frame = max(int(samplerate * frame_seconds), 1)
    count = len(mono) // frame
    if count == 0:
        return []
    frames = mono[:count * frame].reshape(count, frame).astype(np.float64)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    quiet = rms < 10 ** (threshold_db / 20)
    silences, start = [], None
    for index, is_quiet in enumerate(list(quiet) + [False]):
        if is_quiet and start is None:
            start = index
        elif not is_quiet and start is not None:
            begin, end = start * frame / samplerate, index * frame / samplerate
            if end - begin >= min_seconds:
                silences.append({"start": round(begin, 3), "end": round(end, 3)})
            start = None
    return silences


def build_sections(lyrics: dict, lines: list[Line], duration: float, silences: list[dict]) -> dict:
    """依歌詞段落與各行時間算出段落邊界。

    有歌詞的相鄰段落，邊界取兩者之間空檔的中點；純音樂段落佔滿前後兩段人聲之間的空檔。
    第一段從 0 開始，最後一段到歌曲結束。
    """
    # 每個段落的人聲範圍（純音樂段為 None）
    cores: list[tuple[float, float] | None] = []
    line_ids: list[list[int]] = []
    cursor = 0
    for section in lyrics.get("sections", []):
        count = len(section["lines"])
        own = lines[cursor:cursor + count]
        cursor += count
        line_ids.append([line.id for line in own])
        cores.append((own[0].start, own[-1].end) if own else None)

    def previous_end(index: int) -> float:
        for j in range(index - 1, -1, -1):
            if cores[j]:
                return cores[j][1]
        return 0.0

    def next_start(index: int) -> float:
        for j in range(index + 1, len(cores)):
            if cores[j]:
                return cores[j][0]
        return duration

    sections = []
    for index, section in enumerate(lyrics.get("sections", [])):
        core = cores[index]
        if core is None:                                   # 純音樂：前一段人聲結束到下一段人聲開始
            start, end = previous_end(index), next_start(index)
        else:
            before_instrumental = index > 0 and cores[index - 1] is None
            after_instrumental = index + 1 < len(cores) and cores[index + 1] is None
            start = core[0] if before_instrumental else (previous_end(index) + core[0]) / 2
            end = core[1] if after_instrumental else (core[1] + next_start(index)) / 2
            if not any(cores[:index]):                     # 前面沒有任何人聲段
                start = core[0] if before_instrumental else 0.0
            if not any(cores[index + 1:]):                 # 後面沒有任何人聲段
                end = core[1] if after_instrumental else duration
        sections.append({"index": index, "type": section["type"], "start": round(max(start, 0.0), 3),
                         "end": round(min(max(end, start), duration), 3),
                         "instrumental": core is None, "line_ids": line_ids[index]})
    return {"duration": round(duration, 3), "sections": sections, "silences": silences}
