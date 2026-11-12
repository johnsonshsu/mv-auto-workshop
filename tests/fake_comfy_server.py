"""模擬 ComfyUI 伺服器：只用標準函式庫，供測試使用。

支援 /system_stats、/object_info、/prompt、/ws、/history、/view、/upload/image、
/free、/queue、/interrupt。工作流中若有 class_type 為 "FailNode" 的節點，執行時會回報錯誤。
"""
from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
FAKE_FILE_BYTES = b"FAKE-MEDIA-BYTES"


def object_info_for(workflow: dict, combos: dict[tuple[str, str], list[str]] | None = None) -> dict:
    """由 API 格式工作流產生對應的 /object_info：每個欄位都存在，combos 指定下拉選項。"""
    combos = combos or {}
    info: dict = {}
    for node in workflow.values():
        required = info.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
        for field in node["inputs"]:
            options = combos.get((node["class_type"], field))
            required[field] = [options, {}] if options is not None else ["ANY", {}]
    return info


class FakeComfyServer:
    def __init__(self, object_info: dict | None = None, version: str = "0.44.2",
                 step_delay: float = 0.01, enable_ws: bool = True):
        self.object_info = object_info or {}
        self.version = version
        self.step_delay = step_delay
        self.enable_ws = enable_ws
        self.history: dict[str, dict] = {}
        self.prompts: dict[str, dict] = {}
        self.running: str | None = None
        self.foreign_pending: list[str] = []      # 模擬使用者自己排的工作
        self.deleted: list[str] = []
        self.interrupted: list[str] = []
        self.uploads: list[dict] = []
        self.free_calls = 0
        # /view 回傳的內容：file_queue 有東西時依序取出，否則回傳 file_bytes
        self.file_bytes = FAKE_FILE_BYTES
        self.file_queue: list[bytes] = []
        # 接下來幾次執行要回報的錯誤訊息（每次執行取出一個；空字串表示這次正常）
        self.fail_next: list[str] = []
        self._ws_clients: list = []
        self._lock = threading.Lock()
        self._interrupt = threading.Event()

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):       # 測試時不輸出存取記錄
                pass

            def _json(self, code: int, body) -> None:
                data = json.dumps(body).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> bytes:
                return self.rfile.read(int(self.headers.get("Content-Length", 0)))

            def do_GET(self):
                url = urlparse(self.path)
                if url.path == "/system_stats":
                    self._json(200, {"system": {"comfyui_version": server.version}, "devices": []})
                elif url.path == "/object_info":
                    self._json(200, server.object_info)
                elif url.path == "/queue":
                    running = [[0, server.running, {}]] if server.running else []
                    pending = [[i + 1, pid, {}] for i, pid in enumerate(server.foreign_pending)]
                    self._json(200, {"queue_running": running, "queue_pending": pending})
                elif url.path.startswith("/history/"):
                    prompt_id = url.path.rsplit("/", 1)[1]
                    entry = server.history.get(prompt_id)
                    self._json(200, {prompt_id: entry} if entry else {})
                elif url.path == "/view":
                    data = server.file_queue.pop(0) if server.file_queue else server.file_bytes
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                elif url.path == "/ws" and server.enable_ws:
                    self._websocket()
                else:
                    self._json(404, {"error": "not found"})

            def _websocket(self):
                key = self.headers["Sec-WebSocket-Key"]
                accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode()).digest()).decode()
                self.send_response(101)
                self.send_header("Upgrade", "websocket")
                self.send_header("Connection", "Upgrade")
                self.send_header("Sec-WebSocket-Accept", accept)
                self.end_headers()
                with server._lock:
                    server._ws_clients.append(self.wfile)
                try:
                    while self.rfile.read(1):   # 等到用戶端關閉連線
                        pass
                except OSError:
                    pass
                finally:
                    with server._lock:
                        if self.wfile in server._ws_clients:
                            server._ws_clients.remove(self.wfile)

            def do_POST(self):
                url = urlparse(self.path)
                body = self._body()
                if url.path == "/prompt":
                    payload = json.loads(body)
                    workflow = payload["prompt"]
                    errors = {
                        node_id: {"class_type": node["class_type"],
                                  "errors": [{"message": "Node type not found", "details": node["class_type"]}]}
                        for node_id, node in workflow.items()
                        if node["class_type"] not in server.object_info and node["class_type"] != "FailNode"}
                    if errors:
                        self._json(400, {"error": {"message": "Prompt outputs failed validation"},
                                         "node_errors": errors})
                        return
                    prompt_id = uuid.uuid4().hex
                    server.prompts[prompt_id] = workflow
                    threading.Thread(target=server._execute, args=(prompt_id, workflow), daemon=True).start()
                    self._json(200, {"prompt_id": prompt_id, "number": 1, "node_errors": {}})
                elif url.path == "/free":
                    server.free_calls += 1
                    self._json(200, {})
                elif url.path == "/interrupt":
                    if server.running:
                        server.interrupted.append(server.running)
                        server._interrupt.set()
                    self._json(200, {})
                elif url.path == "/queue":
                    for prompt_id in json.loads(body).get("delete", []):
                        server.deleted.append(prompt_id)
                        if prompt_id in server.foreign_pending:
                            server.foreign_pending.remove(prompt_id)
                    self._json(200, {})
                elif url.path == "/upload/image":
                    # 不解析 multipart 內容，只記錄有收到並回傳固定名稱
                    server.uploads.append({"size": len(body), "query": parse_qs(url.query)})
                    self._json(200, {"name": "uploaded.bin", "subfolder": "mv_workshop", "type": "input"})
                else:
                    self._json(404, {"error": "not found"})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    # ---- 生命週期 ----
    def start(self) -> "FakeComfyServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # ---- 執行模擬 ----
    def _send(self, kind: str, data: dict) -> None:
        payload = json.dumps({"type": kind, "data": data}).encode("utf-8")
        header = bytes([0x81, len(payload)]) if len(payload) < 126 else \
            bytes([0x81, 126]) + len(payload).to_bytes(2, "big")
        with self._lock:
            for client in list(self._ws_clients):
                try:
                    client.write(header + payload)
                    client.flush()
                except OSError:
                    self._ws_clients.remove(client)

    def _execute(self, prompt_id: str, workflow: dict) -> None:
        self.running = prompt_id
        self._interrupt.clear()
        self._send("execution_start", {"prompt_id": prompt_id})
        outputs: dict = {}
        failure = self.fail_next.pop(0) if self.fail_next else ""
        try:
            if failure:
                node_id = next(iter(workflow))
                error = {"prompt_id": prompt_id, "node_id": node_id, "node_type": workflow[node_id]["class_type"],
                         "exception_type": "RuntimeError", "exception_message": failure}
                self.history[prompt_id] = {"outputs": {}, "status": {
                    "status_str": "error", "completed": False, "messages": [["execution_error", error]]}}
                self._send("execution_error", error)
                return
            for node_id, node in workflow.items():
                self._send("executing", {"node": node_id, "prompt_id": prompt_id})
                if node["class_type"] == "FailNode":
                    error = {"prompt_id": prompt_id, "node_id": node_id, "node_type": "FailNode",
                             "exception_type": "RuntimeError",
                             "exception_message": node["inputs"].get("message", "boom")}
                    self.history[prompt_id] = {"outputs": {}, "status": {
                        "status_str": "error", "completed": False, "messages": [["execution_error", error]]}}
                    self._send("execution_error", error)
                    return
                for step in (1, 2, 3):
                    time.sleep(self.step_delay)
                    if self._interrupt.is_set():
                        self.history[prompt_id] = {"outputs": {}, "status": {
                            "status_str": "error", "completed": False,
                            "messages": [["execution_interrupted", {"prompt_id": prompt_id}]]}}
                        self._send("execution_interrupted", {"prompt_id": prompt_id})
                        return
                    self._send("progress", {"value": step, "max": 3, "prompt_id": prompt_id, "node": node_id})
                if node["class_type"] == "VHS_VideoCombine":
                    outputs[node_id] = {"gifs": [{"filename": "fake_00001.mp4", "subfolder": "mv_workshop",
                                                "type": "output", "format": "video/h264-mp4"}]}
                elif node["class_type"].startswith("Save"):
                    kind = "audio" if "Audio" in node["class_type"] else "images"
                    outputs[node_id] = {kind: [{"filename": "fake_00001_.flac", "subfolder": "mv_workshop",
                                                "type": "output"}]}
            self.history[prompt_id] = {"outputs": outputs, "status": {
                "status_str": "success", "completed": True, "messages": []}}
            self._send("executing", {"node": None, "prompt_id": prompt_id})
        finally:
            self.running = None
