from __future__ import annotations

import json
import threading
import time

import pytest

from app.pipeline.stage_base import Reporter
from app.pipeline.test_run import run_yue2_test
from app.services.comfy_client import (ComfyCancelled, ComfyClient, ComfyConnectionError,
                                       ComfyWorkflowError, Progress)
from tests.fake_comfy_server import FAKE_FILE_BYTES, FakeComfyServer, object_info_for

WORKFLOW = {
    "1": {"class_type": "Gen", "inputs": {"text": "hi"}},
    "2": {"class_type": "SaveAudio", "inputs": {"audio": ["1", 0], "filename_prefix": "mv_workshop/x"}},
}


@pytest.fixture
def server():
    with FakeComfyServer(object_info_for(WORKFLOW)) as s:
        yield s


@pytest.fixture
def client(server):
    c = ComfyClient(server.url, ws_poll_seconds=0.2)
    yield c
    c.close()


def test_run_reports_progress_over_websocket_and_downloads_output(client, server, tmp_path):
    seen: list[Progress] = []
    entry = client.run(WORKFLOW, lambda p: seen.append(Progress(p.node_id, p.node_type, p.value, p.max)),
                       log_dir=tmp_path / "logs", label="unit")
    assert [(p.node_id, p.node_type) for p in seen if p.max == 0] == [("1", "Gen"), ("2", "SaveAudio")]
    assert (3, 3) in [(p.value, p.max) for p in seen if p.node_id == "2"]

    files = client.output_files(entry, "2")
    assert files == [{"filename": "fake_00001_.flac", "subfolder": "mv_workshop", "type": "output"}]
    dest = client.download(files[0], tmp_path / "out" / "song.flac")
    assert dest.read_bytes() == FAKE_FILE_BYTES
    assert not (tmp_path / "out" / "song.flac.part").exists()

    # 實際送出的工作流有存檔，內容與送出的一致
    logged = list((tmp_path / "logs").glob("*_unit.json"))
    assert len(logged) == 1
    assert json.loads(logged[0].read_text(encoding="utf-8")) == WORKFLOW
    assert server.prompts[next(iter(client.own_prompt_ids))] == WORKFLOW


def test_run_falls_back_to_polling_when_websocket_is_unavailable(tmp_path):
    with FakeComfyServer(object_info_for(WORKFLOW), enable_ws=False) as server:
        client = ComfyClient(server.url, ws_poll_seconds=0.2)
        entry = client.run(WORKFLOW)
        client.close()
    assert client.output_files(entry, "2")


def test_rejected_workflow_names_the_node(client):
    bad = {"1": {"class_type": "NoSuchNode", "inputs": {}}}
    with pytest.raises(ComfyWorkflowError) as e:
        client.run(bad)
    assert e.value.node_id == "1"
    assert e.value.node_type == "NoSuchNode"
    assert "Node type not found" in e.value.message


@pytest.mark.parametrize("enable_ws", [True, False])
def test_execution_error_carries_node_and_comfy_message(enable_ws):
    workflow = {"1": {"class_type": "Gen", "inputs": {"text": ""}},
                "7": {"class_type": "FailNode", "inputs": {"message": "CUDA out of memory"}}}
    with FakeComfyServer(object_info_for(WORKFLOW), enable_ws=enable_ws) as server:
        client = ComfyClient(server.url, ws_poll_seconds=0.2)
        with pytest.raises(ComfyWorkflowError) as e:
            client.run(workflow)
        client.close()
    assert (e.value.node_id, e.value.node_type) == ("7", "FailNode")
    assert "CUDA out of memory" in e.value.message
    assert e.value.is_oom


def test_cancel_interrupts_own_job_and_leaves_user_queue_alone(tmp_path):
    with FakeComfyServer(object_info_for(WORKFLOW), step_delay=0.3) as server:
        server.foreign_pending.append("user-own-job")
        client = ComfyClient(server.url, ws_poll_seconds=0.1)
        cancel = threading.Event()
        threading.Timer(0.4, cancel.set).start()
        started = time.monotonic()
        with pytest.raises(ComfyCancelled):
            client.run(WORKFLOW, cancel=cancel)
        elapsed = time.monotonic() - started
        own = next(iter(client.own_prompt_ids))
        client.close()
        assert server.interrupted == [own]
        assert server.deleted == []
        assert server.foreign_pending == ["user-own-job"]
    assert elapsed < 1.5        # 沒有等整個工作流跑完（完整約 1.8 秒）


def test_cancel_does_not_interrupt_when_someone_elses_job_is_running(server, client):
    client.own_prompt_ids.add("mine-but-finished")
    server.running = "user-own-job"
    client.cancel_own()
    assert server.interrupted == []
    assert server.deleted == []


def test_detect_tries_ports_in_order_and_prefers_last_url(server):
    assert ComfyClient.detect("127.0.0.1", [1, server.port], timeout=0.5) == server.url
    assert ComfyClient.detect("127.0.0.1", [1], last_url=server.url + "/", timeout=0.5) == server.url
    assert ComfyClient.detect("127.0.0.1", [1], timeout=0.5) is None


def test_connection_refused_becomes_readable_error():
    client = ComfyClient("http://127.0.0.1:1", timeout=0.5)
    with pytest.raises(ComfyConnectionError) as e:
        client.system_stats()
    assert "啟動 ComfyUI" in e.value.fix


def test_upload_and_free(client, server, tmp_path):
    photo = tmp_path / "照片.png"
    photo.write_bytes(b"x" * 100)
    assert client.upload_file(photo, "mv_workshop") == "mv_workshop/uploaded.bin"
    assert server.uploads and server.uploads[0]["size"] > 100
    client.free()
    assert server.free_calls == 1


def test_yue2_test_run_end_to_end_against_fake_server(config, yue2_workflow, tmp_path):
    """M1 整合：範本注入 → 送出 → 等待 → 下載 → 釋放模型。"""
    logs: list[str] = []
    with FakeComfyServer(object_info_for(yue2_workflow)) as server:
        path = run_yue2_test(config, server.url, tmp_path / "logs", Reporter(on_log=logs.append))
        sent = next(iter(server.prompts.values()))
        assert server.free_calls == 1
    assert path.read_bytes() == FAKE_FILE_BYTES
    assert path.name == "test_yue2.flac"
    assert sent["25"]["inputs"]["max_duration"] == 10.0
    assert sent["10"]["inputs"]["filename_prefix"] == "mv_workshop/test"
    assert sent["24"]["inputs"]["lyrics"].startswith("[Verse]")
    assert any("已取回音檔" in line for line in logs)
