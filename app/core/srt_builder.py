"""把對齊結果變成字幕：判定哪些行沒對上、內插時間、套用顯示規則、輸出 SRT。

字幕文字一律取自輸入歌詞（lyrics.txt），對齊模型的辨識文字不會進到字幕。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from app.core.lyrics import count_chars

ALIGNED = "aligned"
INTERPOLATED = "interpolated"


@dataclass
class Line:
    id: int                 # 從 1 起，就是字幕編號
    text: str
    start: float
    end: float
    confidence: float       # 0–1；內插的行為 0
    method: str             # aligned | interpolated

    def to_dict(self) -> dict:
        return asdict(self)


def trim_outlier_words(words: list[dict], max_gap: float) -> list[dict]:
    """去掉離群的字，回傳最主要的一群。

    對齊模型有時會把一行的頭一個字或最後一個字放到間奏的另一頭，讓整行被拉長數十秒。
    把字依「相鄰間隔超過 max_gap 秒」切成幾群，取字數最多的一群（同數取信心值總和較高的）。
    """
    timed = [w for w in words if w.get("start") is not None and w.get("end") is not None]
    if not timed:
        return []
    groups, current = [], [timed[0]]
    for word in timed[1:]:
        if word["start"] - current[-1]["end"] > max_gap:
            groups.append(current)
            current = [word]
        else:
            current.append(word)
    groups.append(current)
    return max(groups, key=lambda g: (len(g), sum(w.get("probability") or 0.0 for w in g)))


def assess(raw: list[dict], texts: list[str], duration: float, min_confidence: float = 0.15,
           min_seconds_per_char: float = 0.04, max_word_gap: float = 2.0) -> list[Line | None]:
    """逐行判定對齊是否可信；不可信的行回傳 None，之後由 interpolate 補上。

    不可信的情況：沒有時間、信心值太低、長度短到不可能唱完、或時間倒退。
    """
    result: list[Line | None] = []
    last_end = 0.0
    for index, text in enumerate(texts):
        item = raw[index] if index < len(raw) else {}
        start, end = item.get("start"), item.get("end")
        words = trim_outlier_words(item.get("words", []), max_word_gap)
        if words and start is not None and end is not None:
            start, end = max(start, words[0]["start"]), min(end, words[-1]["end"])
        probabilities = [w["probability"] for w in words if w.get("probability") is not None]
        confidence = sum(probabilities) / len(probabilities) if probabilities else 0.0
        ok = (start is not None and end is not None
              and 0 <= start < end <= duration + 0.5
              and confidence >= min_confidence
              and end - start >= max(count_chars(text), 1) * min_seconds_per_char
              and start >= last_end - 0.05)
        if ok:
            line = Line(index + 1, text, float(start), min(float(end), duration), round(confidence, 3), ALIGNED)
            last_end = line.end
            result.append(line)
        else:
            result.append(None)
    return result


def interpolate(lines: list[Line | None], texts: list[str], duration: float, gap: float = 0.05) -> list[Line]:
    """未對齊的行：在前後已對齊的行之間，依字數比例分配時間。

    空檔遠大於這幾行唱完需要的時間時（例如後面接著一段間奏），不把字幕拉長填滿整個空檔，
    而是緊接在上一行後面，用其他行的平均唱速估計長度。
    """
    result: list[Line] = [line for line in lines]          # type: ignore[misc]
    paces = sorted((line.end - line.start) / max(count_chars(line.text), 1) for line in lines if line is not None)
    pace = paces[len(paces) // 2] if paces else 0.4        # 已對齊各行「每個字幾秒」的中位數
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        run_end = index
        while run_end < len(result) and result[run_end] is None:
            run_end += 1
        left = result[index - 1].end + gap if index > 0 else 0.0
        right = result[run_end].start - gap if run_end < len(result) else duration
        right = max(right, left)
        weights = [max(count_chars(texts[i]), 1) for i in range(index, run_end)]
        span, total, cursor = right - left, sum(weights), left
        expected = total * pace + gap * len(weights)
        if paces and span > expected * 1.5:            # 完全沒有對上的行可參考時，維持平均分配
            span = expected
            if index == 0 and run_end < len(result):
                # 歌曲開頭沒對上的行：前面通常是前奏，所以靠著後面那一行往前排，而不是從 0 秒開始
                cursor = left = right - expected
        for offset, weight in enumerate(weights):
            length = span * weight / total
            # 每行之間留一點空隙；空間不足時（span 為 0）起訖相同，交給 build_cues 處理
            end = cursor + max(length - gap, 0.0)
            result[index + offset] = Line(index + offset + 1, texts[index + offset],
                                          round(cursor, 3), round(end, 3), 0.0, INTERPOLATED)
            cursor += length
        index = run_end
    return result


def build_cues(lines: list[Line], tail: float = 0.3, min_duration: float = 0.8, gap: float = 0.05,
               duration: float | None = None) -> list[tuple[float, float, str]]:
    """套用顯示規則：結尾可延長至多 tail 秒但不與下一則重疊；每則至少顯示 min_duration 秒。"""
    cues = []
    for index, line in enumerate(lines):
        limit = lines[index + 1].start - gap if index + 1 < len(lines) else (duration or float("inf"))
        end = min(line.end + tail, limit)
        if end - line.start < min_duration:
            end = min(line.start + min_duration, limit)
        # 下一行緊接著開始、連最短時間都放不下時，至少不要出現結束早於開始的字幕
        end = max(end, min(line.start + 0.1, limit + gap))
        cues.append((round(line.start, 3), round(end, 3), line.text))
    return cues


def format_time(seconds: float) -> str:
    millis = max(int(round(seconds * 1000)), 0)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def to_srt(cues: list[tuple[float, float, str]]) -> str:
    blocks = [f"{number}\n{format_time(start)} --> {format_time(end)}\n{text}\n"
              for number, (start, end, text) in enumerate(cues, start=1)]
    return "\n".join(blocks)


def parse_srt(text: str) -> list[tuple[float, float, str]]:
    """讀回 SRT（測試與字幕校對用）。"""
    def seconds(stamp: str) -> float:
        clock, millis = stamp.strip().split(",")
        hours, minutes, secs = (int(part) for part in clock.split(":"))
        return hours * 3600 + minutes * 60 + secs + int(millis) / 1000

    cues = []
    for block in text.replace("\r\n", "\n").strip().split("\n\n"):
        rows = block.strip().split("\n")
        if len(rows) >= 3 and "-->" in rows[1]:
            start, end = rows[1].split("-->")
            cues.append((seconds(start), seconds(end), "\n".join(rows[2:])))
    return cues
