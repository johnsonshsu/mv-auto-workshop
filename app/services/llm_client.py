"""LLM 用戶端：產生歌詞與提示詞。三種供應商實作同一個介面，都要求回傳符合 JSON Schema 的物件。

- Ollama（本機，預設）
- Claude（Anthropic 官方 SDK；未安裝 anthropic 套件時會提示）
- OpenAI 相容端點
"""
from __future__ import annotations

import base64
import json
import mimetypes
import re
from pathlib import Path
from typing import Protocol

import httpx

from app.core.models import UserFacingError
from app.storage import secret_store
from app.storage.app_config import AppConfig


class LLMError(UserFacingError):
    pass


class LLMFormatError(LLMError):
    """LLM 有回應，但內容不是要求的 JSON。可以重試。"""


class LLMClient(Protocol):
    label: str
    supports_vision: bool

    def chat_json(self, system: str, user: str, schema: dict, image: Path | None = None) -> dict:
        """送出一次對話，回傳符合 schema 的 JSON 物件。"""

    def unload(self) -> None:
        """釋放模型佔用的 VRAM（只有本機供應商需要）。"""


def extract_json(text: str) -> dict:
    """從回應文字取出 JSON 物件；容許前後有說明文字或 ``` 圍欄。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fenced:
        candidates.insert(0, fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise LLMFormatError("LLM 回傳的內容不是 JSON 物件", f"回應開頭：{text[:80]}")


def _image_b64(path: Path) -> tuple[str, str]:
    media_type = mimetypes.guess_type(path.name)[0] or "image/png"
    return media_type, base64.b64encode(path.read_bytes()).decode("ascii")


class OllamaClient:
    supports_vision = True

    def __init__(self, base_url: str, text_model: str, vision_model: str,
                 keep_alive: str = "5m", timeout: float = 300.0, transport: httpx.BaseTransport | None = None):
        self.base_url = base_url.rstrip("/")
        self.text_model = text_model
        self.vision_model = vision_model
        self.keep_alive = keep_alive
        self.label = f"Ollama（{text_model}）"
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)
        self._used: set[str] = set()

    def chat_json(self, system: str, user: str, schema: dict, image: Path | None = None) -> dict:
        model = self.vision_model if image is not None else self.text_model
        message: dict = {"role": "user", "content": user}
        if image is not None:
            message["images"] = [_image_b64(image)[1]]
        body = {
            "model": model,
            "stream": False,
            "think": False,
            "format": schema,
            "keep_alive": self.keep_alive,
            "messages": [{"role": "system", "content": system}, message],
        }
        try:
            r = self._http.post("/api/chat", json=body)
        except httpx.TimeoutException:
            raise LLMError("Ollama 回應逾時", "可在設定換用較小的模型，或稍後再試")
        except httpx.TransportError as e:
            raise LLMError(f"無法連線到 Ollama（{self.base_url}）：{e}", "請確認 Ollama 已啟動，或在設定改用雲端 API")
        self._used.add(model)
        if r.status_code == 404:
            raise LLMError(f"Ollama 沒有模型「{model}」", f"請執行 ollama pull {model}，或在設定改用已安裝的模型")
        if r.status_code != 200:
            raise LLMError(f"Ollama 回應錯誤（HTTP {r.status_code}）：{r.text[:200]}")
        return extract_json(r.json().get("message", {}).get("content", ""))

    def unload(self) -> None:
        for model in list(self._used):
            try:
                self._http.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=30.0)
            except httpx.HTTPError:
                pass        # 卸載失敗不影響流程；模型逾時後也會自行卸載
        self._used.clear()


class OpenAICompatClient:
    supports_vision = True

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 300.0,
                 transport: httpx.BaseTransport | None = None):
        if not model:
            raise LLMError("尚未設定 OpenAI 相容端點的模型名稱", "請到設定頁填寫")
        if not api_key:
            raise LLMError("尚未設定 OpenAI 相容端點的金鑰", "請到設定頁填寫，金鑰會存在 Windows 認證管理員")
        self.model = model
        self.label = f"OpenAI 相容（{model}）"
        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport,
                                  headers={"Authorization": f"Bearer {api_key}"})

    def chat_json(self, system: str, user: str, schema: dict, image: Path | None = None) -> dict:
        content: object = user
        if image is not None:
            media_type, data = _image_b64(image)
            content = [{"type": "text", "text": user},
                       {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{data}"}}]
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        }
        try:
            r = self._http.post("/chat/completions", json=body)
        except httpx.TimeoutException:
            raise LLMError("LLM 端點回應逾時", "請稍後再試")
        except httpx.TransportError as e:
            raise LLMError(f"無法連線到 LLM 端點：{e}", "請檢查網路與設定頁的端點位址")
        if r.status_code in (401, 403):
            raise LLMError("LLM 端點拒絕了金鑰", "請到設定頁確認金鑰是否正確")
        if r.status_code != 200:
            raise LLMError(f"LLM 端點回應錯誤（HTTP {r.status_code}）：{r.text[:200]}")
        return extract_json(r.json()["choices"][0]["message"].get("content") or "")

    def unload(self) -> None:
        pass


def _strict_schema(schema: dict) -> dict:
    """Claude 的結構化輸出要求每個 object 都標明 additionalProperties: false。"""
    result = dict(schema)
    if result.get("type") == "object":
        result["additionalProperties"] = False
        result["properties"] = {k: _strict_schema(v) for k, v in result.get("properties", {}).items()}
    if "items" in result:
        result["items"] = _strict_schema(result["items"])
    return result


class ClaudeClient:
    supports_vision = True

    def __init__(self, model: str, api_key: str, timeout: float = 300.0):
        try:
            import anthropic
        except ImportError:
            raise LLMError("使用 Claude API 需要 anthropic 套件，目前尚未安裝",
                           "請在專案資料夾執行 .\\.venv\\Scripts\\python.exe -m pip install anthropic")
        if not api_key:
            raise LLMError("尚未設定 Claude API 金鑰", "請到設定頁填寫，金鑰會存在 Windows 認證管理員")
        self._anthropic = anthropic
        self.model = model
        self.label = f"Claude（{model}）"
        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout)

    def chat_json(self, system: str, user: str, schema: dict, image: Path | None = None) -> dict:
        anthropic = self._anthropic
        content: list[dict] = []
        if image is not None:
            media_type, data = _image_b64(image)
            content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        content.append({"type": "text", "text": user})
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"effort": "medium",
                               "format": {"type": "json_schema", "schema": _strict_schema(schema)}},
            )
        except anthropic.AuthenticationError:
            raise LLMError("Claude API 拒絕了金鑰", "請到設定頁確認金鑰是否正確")
        except anthropic.RateLimitError:
            raise LLMError("Claude API 目前達到用量上限", "請稍後再試")
        except anthropic.APIStatusError as e:
            raise LLMError(f"Claude API 回應錯誤（HTTP {e.status_code}）：{e.message}")
        except anthropic.APIConnectionError as e:
            raise LLMError(f"無法連線到 Claude API：{e}", "請檢查網路連線")
        if response.stop_reason == "refusal":
            raise LLMError("Claude 拒絕了這個請求", "請調整音樂內容的描述後再試")
        if response.stop_reason == "max_tokens":
            raise LLMFormatError("Claude 的回應在完成前被截斷")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return extract_json(text)

    def unload(self) -> None:
        pass


def make_llm(config: AppConfig) -> LLMClient:
    provider = config.get("llm.provider", "ollama")
    timeout = float(config.get("llm.timeout", 300))
    if provider == "ollama":
        return OllamaClient(config.get("llm.ollama_url"), config.get("llm.ollama_text_model"),
                            config.get("llm.ollama_vision_model"), config.get("llm.ollama_keep_alive", "5m"),
                            timeout)
    if provider == "claude":
        return ClaudeClient(config.get("llm.claude_model"), secret_store.get_secret("claude_api_key") or "", timeout)
    if provider == "openai_compatible":
        return OpenAICompatClient(config.get("llm.openai_base_url"), config.get("llm.openai_model"),
                                  secret_store.get_secret("openai_api_key") or "", timeout)
    raise LLMError(f"不認得的 LLM 供應商「{provider}」", "請到設定頁重新選擇")
