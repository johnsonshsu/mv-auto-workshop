"""ComfyUI 用戶端：只透過 HTTP 與 WebSocket API 溝通，不碰 ComfyUI 的安裝內容。"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import httpx
import websocket

from app.core.models import UserFacingError


class ComfyConnectionError(UserFacingError):
    """ComfyUI 未啟動或中途斷線。"""

    def __init__(self, detail: str = ""):
        super().__init__(
            "無法連線到 ComfyUI" + (f"：{detail}" if detail else ""),
            "請啟動 ComfyUI Desktop，確認設定頁的位址正確後再試一次")


class ComfyWorkflowError(UserFacingError):
    """工作流被拒絕或執行時出錯；帶有出錯的節點與 ComfyUI 回傳的訊息。"""

    def __init__(self, message: str, node_id: str = "", node_type: str = "", raw: dict | None = None):
        where = f"節點 {node_id}（{node_type}）" if node_id else "工作流"
        super().__init__(f"{where}執行失敗：{message}", "實際送出的工作流已存在專案的 logs 資料夾，可用來重現問題")
        self.node_id = node_id
        self.node_type = node_type
        self.raw = raw or {}

    @property
    def is_oom(self) -> bool:
        text = json.dumps(self.raw, ensure_ascii=False).lower() + self.message.lower()
        return "out of memory" in text or "outofmemory" in text or "allocation on device" in text


class ComfyCancelled(Exception):
    """使用者取消。"""


@dataclass
class Progress:
    """執行進度；value/max 是目前節點的步數。"""
    node_id: str = ""
    node_type: str = ""
    value: int = 0
    max: int = 0


ProgressCallback = Callable[[Progress], None]


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 30.0, ws_poll_seconds: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.client_id = uuid.uuid4().hex
        self.ws_poll_seconds = ws_poll_seconds
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._object_info: dict | None = None
        # 本 APP 送出的 prompt_id；取消時只動這些
        self.own_prompt_ids: set[str] = set()
        self._lock = threading.Lock()

    # ---- 連線 ----
    @classmethod
    def detect(cls, host: str, ports: list[int], last_url: str = "", timeout: float = 3.0) -> str | None:
        """依序嘗試上次成功的位址與各連接埠，回傳第一個有回應的位址。"""
        candidates = ([last_url] if last_url else []) + [f"http://{host}:{p}" for p in ports]
        seen = set()
        for url in candidates:
            url = url.rstrip("/")
            if url in seen:
                continue
            seen.add(url)
            try:
                r = httpx.get(f"{url}/system_stats", timeout=timeout)
                if r.status_code == 200 and "system" in r.json():
                    return url
            except (httpx.HTTPError, ValueError):
                continue
        return None

    def close(self) -> None:
        self._http.close()

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            return self._http.request(method, path, **kwargs)
        except httpx.TransportError as e:
            raise ComfyConnectionError(str(e)) from e

    def _get_json(self, path: str, **kwargs) -> dict:
        r = self._request("GET", path, **kwargs)
        r.raise_for_status()
        return r.json()

    # ---- 查詢 ----
    def system_stats(self) -> dict:
        return self._get_json("/system_stats")

    def object_info(self, refresh: bool = False) -> dict:
        if self._object_info is None or refresh:
            self._object_info = self._get_json("/object_info", timeout=120.0)
        return self._object_info

    def queue(self) -> dict:
        return self._get_json("/queue")

    def history(self, prompt_id: str) -> dict | None:
        return self._get_json(f"/history/{prompt_id}").get(prompt_id)

    # ---- 送出 ----
    def queue_prompt(self, workflow: dict, log_dir: Path | None = None, label: str = "workflow") -> str:
        """送出工作流並回傳 prompt_id。送出前把實際內容存到 log_dir。"""
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            target, extra = log_dir / f"{stamp}_{label}.json", 1
            while target.exists():          # 同一秒內送出多次時不要互相覆蓋
                extra += 1
                target = log_dir / f"{stamp}-{extra}_{label}.json"
            target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
        r = self._request("POST", "/prompt", json={"prompt": workflow, "client_id": self.client_id})
        if r.status_code != 200:
            raise self._prompt_rejected(r)
        prompt_id = r.json()["prompt_id"]
        with self._lock:
            self.own_prompt_ids.add(prompt_id)
        return prompt_id

    @staticmethod
    def _prompt_rejected(r: httpx.Response) -> ComfyWorkflowError:
        try:
            body = r.json()
        except ValueError:
            return ComfyWorkflowError(f"ComfyUI 回應 HTTP {r.status_code}：{r.text[:300]}")
        # 驗證失敗時 node_errors 以節點 ID 為鍵，列出每個欄位的問題
        for node_id, info in (body.get("node_errors") or {}).items():
            details = "；".join(
                f"{e.get('message', '')} {e.get('details', '')}".strip() for e in info.get("errors", []))
            return ComfyWorkflowError(details or "欄位驗證失敗", node_id, info.get("class_type", ""), body)
        error = body.get("error")
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        return ComfyWorkflowError(message, raw=body)

    def run(self, workflow: dict, on_progress: ProgressCallback | None = None,
            cancel: threading.Event | None = None, log_dir: Path | None = None,
            label: str = "workflow") -> dict:
        """送出工作流並等到執行結束，回傳 /history 的該筆記錄。

        先開 WebSocket 再送出，避免漏掉訊息；WebSocket 連不上或中斷時改為輪詢 /history。
        """
        ws = self._open_ws()
        try:
            prompt_id = self.queue_prompt(workflow, log_dir, label)
            return self._wait(prompt_id, workflow, ws, on_progress, cancel)
        finally:
            if ws is not None:
                try:
                    ws.shutdown()       # 直接關閉；close() 會等對方回應關閉訊框，最長 3 秒
                except Exception:
                    pass

    def _open_ws(self):
        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        try:
            return websocket.create_connection(
                f"{scheme}://{parsed.netloc}/ws?clientId={self.client_id}", timeout=self.ws_poll_seconds)
        except Exception:
            return None

    def _wait(self, prompt_id: str, workflow: dict, ws, on_progress, cancel) -> dict:
        progress = Progress()
        while True:
            if cancel is not None and cancel.is_set():
                self.cancel_own()
                raise ComfyCancelled()
            message = None
            if ws is not None:
                try:
                    raw = ws.recv()
                    if isinstance(raw, str):          # 二進位訊框是預覽圖，略過
                        message = json.loads(raw)
                except websocket.WebSocketTimeoutException:
                    pass
                except Exception:
                    ws = None                          # 斷線：之後改用輪詢
            else:
                time.sleep(min(self.ws_poll_seconds, 1.0))

            if message is not None:
                data = message.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue                           # 別人的工作
                kind = message.get("type")
                if kind == "execution_error":
                    raise ComfyWorkflowError(
                        f"{data.get('exception_type', '')} {data.get('exception_message', '')}".strip(),
                        str(data.get("node_id", "")), data.get("node_type", ""), data)
                if kind == "execution_interrupted":
                    raise ComfyCancelled()
                if kind == "executing" and data.get("node") is not None:
                    node_id = str(data["node"])
                    progress = Progress(node_id, workflow.get(node_id, {}).get("class_type", ""))
                    if on_progress:
                        on_progress(progress)
                elif kind == "progress":
                    progress.value, progress.max = int(data.get("value", 0)), int(data.get("max", 0))
                    if on_progress:
                        on_progress(progress)
                # 只有結束訊號才往下查 history：execution_success，或 executing 且 node 為 None
                finished = kind == "execution_success" or (kind == "executing" and data.get("node") is None)
                if not finished:
                    continue

            # 沒有訊息（逾時或輪詢）或收到結束訊號：以 /history 為準
            entry = self.history(prompt_id)
            if entry is None:
                continue
            status = entry.get("status") or {}
            if status.get("status_str") == "error":
                raise self._error_from_history(entry)
            if status.get("completed") or entry.get("outputs"):
                return entry

    @staticmethod
    def _error_from_history(entry: dict) -> Exception:
        for name, data in (entry.get("status") or {}).get("messages", []):
            if name == "execution_error":
                return ComfyWorkflowError(
                    f"{data.get('exception_type', '')} {data.get('exception_message', '')}".strip(),
                    str(data.get("node_id", "")), data.get("node_type", ""), data)
            if name == "execution_interrupted":
                return ComfyCancelled()
        return ComfyWorkflowError("ComfyUI 回報執行失敗，但沒有提供細節", raw=entry)

    # ---- 檔案 ----
    @staticmethod
    def output_files(entry: dict, node_id: str) -> list[dict]:
        """取出某個輸出節點產生的檔案資訊（filename、subfolder、type）。"""
        files = []
        for value in (entry.get("outputs", {}).get(str(node_id)) or {}).values():
            if isinstance(value, list):
                files.extend(v for v in value if isinstance(v, dict) and "filename" in v)
        return files

    def download(self, file_info: dict, dest: Path) -> Path:
        params = {"filename": file_info["filename"], "subfolder": file_info.get("subfolder", ""),
                  "type": file_info.get("type", "output")}
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        try:
            with self._http.stream("GET", "/view", params=params, timeout=300.0) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_bytes(1 << 20):
                        f.write(chunk)
        except httpx.TransportError as e:
            raise ComfyConnectionError(str(e)) from e
        tmp.replace(dest)
        return dest

    def upload_file(self, path: Path, subfolder: str, name: str | None = None) -> str:
        """上傳照片或音訊到 ComfyUI 的 input 資料夾，回傳 Load 節點要填的名稱。"""
        with open(path, "rb") as f:
            r = self._request(
                "POST", "/upload/image",
                files={"image": (name or path.name, f, "application/octet-stream")},
                data={"subfolder": subfolder, "type": "input", "overwrite": "true"}, timeout=300.0)
        if r.status_code != 200:
            raise UserFacingError(f"上傳 {path.name} 到 ComfyUI 失敗（HTTP {r.status_code}）", r.text[:200])
        info = r.json()
        return f"{info['subfolder']}/{info['name']}" if info.get("subfolder") else info["name"]

    # ---- 資源與取消 ----
    def free(self) -> None:
        """請 ComfyUI 卸載模型並釋放記憶體。"""
        self._request("POST", "/free", json={"unload_models": True, "free_memory": True})

    def cancel_own(self) -> None:
        """只取消本 APP 送出的工作：移除佇列中的項目；執行中的是自己的才中斷。"""
        with self._lock:
            own = set(self.own_prompt_ids)
        if not own:
            return
        try:
            queue = self.queue()
        except (UserFacingError, httpx.HTTPError):
            return
        pending = [item[1] for item in queue.get("queue_pending", []) if item[1] in own]
        if pending:
            self._request("POST", "/queue", json={"delete": pending})
        running = [item[1] for item in queue.get("queue_running", []) if item[1] in own]
        for prompt_id in running:
            self._request("POST", "/interrupt", json={"prompt_id": prompt_id})
