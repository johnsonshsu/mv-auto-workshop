"""影片分段與影格計算。

原則：所有影格邊界都從歌曲起點以累計時間換算（F = round(t × fps)），
不是把各段長度四捨五入後相加，所以各段影格數的總和恆等於 round(歌曲長度 × fps)，誤差不會累積。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

PHOTO = "photo"      # 以原始照片為起始影像
PREV = "prev"        # 以上一段的最後一幀為起始影像
MOTION = "motion"    # 以原始照片為參考、上一段的尾端為動作參考（模型自己接續，不會重複影格）


@dataclass
class Segment:
    index: int           # 從 1 起
    t_start: float
    t_end: float
    f_start: int         # 這一段在整支影片中的起始影格（含）
    f_end: int           # 結束影格（不含）
    frames: int          # 接合時要保留的影格數 = f_end − f_start
    gen_frames: int      # 實際請模型生成的影格數（符合模型限制，且 ≥ 需要的數量）
    init: str            # photo | prev | motion
    forced_cut: bool     # 起點不是段落邊界或靜音，而是強制切的
    end_anchor: bool = False   # 結尾以原始照片引導：這一段結束時回到照片的樣子，下一段照常接續

    @property
    def drop_head(self) -> int:
        """接合時要丟掉的開頭影格：延續段的第 1 幀是上一段最後一幀的重複。"""
        return 1 if self.init == PREV else 0

    @property
    def audio_start_frame(self) -> int:
        """送給模型的音訊從哪個影格的時間開始；延續段往前一幀，對齊重複的那一幀。"""
        return self.f_start - self.drop_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


def total_frames(duration: float, fps: float) -> int:
    return round(duration * fps)


def valid_frames(need: int, step: int = 1, offset: int = 0) -> int:
    """符合模型限制（step × k + offset）且不小於 need 的最小影格數。"""
    if step <= 1:
        return max(need, 1)
    k = max(-(-(need - offset) // step), 0)          # 向上取整
    return step * k + offset


def choose_cuts(duration: float, section_bounds: list[float], silences: list[dict], max_seg: float,
                min_seg: float, quietest: Callable[[float, float], float] | None = None
                ) -> tuple[list[float], set[float]]:
    """決定切點，回傳（切點時間清單，含 0 與歌曲結尾；其中哪些是強制切的）。

    候選切點是段落邊界與人聲靜音區間的中點。由左而右，每次在不超過 max_seg 的範圍內
    挑最靠後的候選點；範圍後半段有段落邊界時優先用段落邊界。完全沒有候選點（一句長音
    超過上限）時，在範圍的後 40% 找人聲最小聲的位置強制切。
    """
    bounds = sorted({round(t, 3) for t in section_bounds if 0 < t < duration})
    gaps = sorted({round((s["start"] + s["end"]) / 2, 3) for s in silences
                   if 0 < (s["start"] + s["end"]) / 2 < duration})
    cuts, forced, position = [0.0], set(), 0.0
    while duration - position > max_seg + 1e-9:
        low, high = position + min_seg, position + max_seg
        in_bounds = [t for t in bounds if low <= t <= high]
        in_gaps = [t for t in gaps if low <= t <= high]
        late_bounds = [t for t in in_bounds if t >= position + max_seg * 0.5]
        if late_bounds:
            cut = late_bounds[-1]
        elif in_bounds or in_gaps:
            cut = max(in_bounds + in_gaps)
        elif any(s["start"] < high and s["end"] > low for s in silences):
            # 範圍內有一段很長的靜音（例如前奏、間奏），中點卻不在範圍內：切在靜音裡最靠後的位置
            cut = round(max(min(high, s["end"]) for s in silences if s["start"] < high and s["end"] > low), 3)
        else:
            start = position + max_seg * 0.6
            cut = round(quietest(start, high), 3) if quietest else high
            cut = min(max(cut, start), high)
            forced.add(cut)
        cuts.append(cut)
        position = cut
    # 最後一段太短：併入前一段；併不下（超過上限）就把前一個切點移到兩段的中間
    if len(cuts) > 1 and duration - cuts[-1] < min_seg:
        last = cuts.pop()
        forced.discard(last)
        if duration - cuts[-1] > max_seg:
            middle = round((cuts[-1] + duration) / 2, 3)
            cuts.append(middle)
            forced.add(middle)
    cuts.append(duration)
    return cuts, forced


def build_plan(cuts: list[float], forced: set[float], duration: float, fps: float,
               frame_step: int = 1, frame_offset: int = 0, reset_every: int = 4,
               section_bounds: list[float] | None = None, end_anchor: bool = False,
               continuation: str = PREV) -> list[Segment]:
    """把切點變成分段計畫。

    延續段以上一段最後一幀起始；每 reset_every 段校正一次，抑制人物越來越不像的累積誤差。
    校正的方式有兩種：
    - end_anchor=False：下一段改回以原始照片起始。畫面會跳接，像換鏡頭。
    - end_anchor=True：讓前一段的結尾回到原始照片的樣子，下一段照常接續，畫面不會跳。
    兩種都只放在自然的切點（段落邊界或人聲靜音處）；遇到在人聲中強制切開的切點，
    就延後到下一個自然的切點（唱到一半被拉回照片的表情會很怪）。
    continuation 是延續段的起始方式：PREV（上一段最後一幀）或 MOTION（以上一段的尾端當動作參考）。
    """
    n_total = total_frames(duration, fps)
    edges = [round(t * fps) for t in cuts]
    edges[0], edges[-1] = 0, n_total
    segments: list[Segment] = []
    since_reset = 0
    for i in range(len(cuts) - 1):
        frames = edges[i + 1] - edges[i]
        if frames <= 0:
            continue
        natural = cuts[i] not in forced
        if not segments:
            init, since_reset = PHOTO, 1
        elif reset_every > 0 and since_reset >= reset_every and natural:
            if end_anchor:
                segments[-1].end_anchor = True
                init, since_reset = continuation, 1
            else:
                init, since_reset = PHOTO, 1
        else:
            init, since_reset = continuation, since_reset + 1
        need = frames + (1 if init == PREV else 0)
        segments.append(Segment(len(segments) + 1, round(cuts[i], 3), round(cuts[i + 1], 3), edges[i], edges[i + 1],
                                frames, valid_frames(need, frame_step, frame_offset), init, cuts[i] in forced))
    return segments


def plan_segments(duration: float, fps: float, sections: dict, max_seg: float, min_seg: float,
                  frame_step: int = 1, frame_offset: int = 0, reset_every: int = 4,
                  quietest: Callable[[float, float], float] | None = None,
                  end_anchor: bool = False, continuation: str = PREV) -> list[Segment]:
    """由 sections.json 的內容直接產生分段計畫。"""
    bounds = [s["start"] for s in sections.get("sections", [])] + [s["end"] for s in sections.get("sections", [])]
    cuts, forced = choose_cuts(duration, bounds, sections.get("silences", []), max_seg, min_seg, quietest)
    return build_plan(cuts, forced, duration, fps, frame_step, frame_offset, reset_every, bounds, end_anchor,
                      continuation)
