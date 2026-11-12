"""測試共用的假物件。"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile

from app.services.llm_client import LLMError, LLMFormatError


def flac_bytes(seconds: float, samplerate: int = 8000) -> bytes:
    """產生指定長度的靜音 FLAC。"""
    buffer = io.BytesIO()
    soundfile.write(buffer, np.zeros((int(seconds * samplerate), 2), dtype="float32"), samplerate, format="FLAC")
    return buffer.getvalue()


class FakeLLM:
    """依序回傳預先排好的回應；回應可以是 dict，或要丟出的例外。"""
    label = "假 LLM"
    supports_vision = False

    def __init__(self, responses: list, events: list | None = None):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.unloaded = 0
        self.events = events if events is not None else []

    def chat_json(self, system: str, user: str, schema: dict, image: Path | None = None) -> dict:
        self.calls.append({"system": system, "user": user, "schema": schema, "image": image})
        self.events.append("llm_call")
        if not self.responses:
            raise LLMError("假 LLM 沒有更多回應")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def unload(self) -> None:
        self.unloaded += 1
        self.events.append("llm_unload")


BAD_JSON = LLMFormatError("LLM 回傳的內容不是 JSON 物件")

# 符合內建範本 verse_chorus_bridge 的四個段落（主歌 4、副歌 4、主歌 4、過渡 2）
GOOD_SECTIONS = {
    "title": "夏夜的風",
    "sections": [
        {"lines": ["巷口的燈一盞一盞亮起", "引擎聲劃破沉默的空氣", "後照鏡裡沒有你的蹤跡", "只剩晚風陪我穿過這裡"]},
        {"lines": ["夏夜的風吹散了回憶", "城市的光閃爍像星星", "我學會一個人往前去", "把想念留在風裡"]},
        {"lines": ["便利商店還亮著的招牌", "你愛喝的汽水我還在買", "紅燈前忽然想起你的愛", "綠燈亮起我選擇不等待"]},
        {"lines": ["也許有一天再次相遇", "我會微笑說聲謝謝你"]},
    ],
}

STYLE_FEATURES = {"vocal_character": "warm, breathy", "genre": "nostalgic city pop", "bpm": 108,
                  "instruments": "electric piano, funky bass, crisp drums", "production": "polished, clear diction"}
