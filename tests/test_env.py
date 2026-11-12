from __future__ import annotations

import json

import pytest

from app.core.models import DEGRADE, FAIL, PASS
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s1_env import (EnvFacts, EnvStage, can_start, check_threshold, combo_options,
                                        evaluate, parse_version)
from app.services.hardware_probe import GpuInfo
from app.storage.project_store import ProjectStore
from tests.conftest import REPO
from tests.fake_comfy_server import FakeComfyServer

REQUIRES = ["yue2_text_to_music", "vocal_separation", "infinitetalk_single", "ltx23_ia2v", "wan22_s2v",
            "infinitetalk_native"]


def full_object_info() -> dict:
    """滿足所有需求清單的 /object_info：每個節點都在、每個模型都在下拉選單裡。"""
    info: dict = {}

    def add(req: dict) -> None:
        for node in req.get("nodes", []):
            info.setdefault(node, {"input": {"required": {}}})
        for model in req.get("models", []):
            required = info.setdefault(model["loader_node"], {"input": {"required": {}}})["input"]["required"]
            required.setdefault(model["loader_input"], [[], {}])[0].append(model["file"])

    for name in REQUIRES:
        req = json.loads((REPO / "workflows" / f"{name}.requires.json").read_text(encoding="utf-8"))
        for option in req.get("any_of", [req]):
            add(option)
    return info


def good_facts(**overrides) -> EnvFacts:
    facts = EnvFacts(gpus=[GpuInfo("RTX 5090 Laptop", 23.9, "617.42")], ram_gb=95.4,
                     projects_root="D:/MV專案", disk_free_gb=795.0, ffmpeg_version="9.0.2",
                     comfy_url="http://127.0.0.1:8188", comfy_version="0.44.2",
                     object_info=full_object_info())
    for key, value in overrides.items():
        setattr(facts, key, value)
    return facts


def item(report: dict, item_id: str) -> dict:
    return next(i for i in report["items"] if i["id"] == item_id)


def test_capable_machine_passes_everything(config):
    report = evaluate(good_facts(), config)
    assert [i["label"] for i in report["items"] if i["level"] != PASS] == []
    assert report["overall"] == PASS
    assert can_start(report, "infinitetalk") and can_start(report, "ltx23")


def test_nominal_24gb_card_reporting_23_9_counts_as_24():
    th = {"min_vram": 16, "recommended_vram": 24, "min_ram": 32, "recommended_ram": 64}
    assert check_threshold(th, 23.9, 95.4) == (PASS, "")


def test_weak_hardware_is_blocked_with_what_is_missing(config):
    """驗收：硬體不足的電腦在階段 1 就被擋下，並說明缺什麼。"""
    report = evaluate(good_facts(gpus=[GpuInfo("GTX 1660", 6.0, "560.00")], ram_gb=16.0), config)
    assert report["overall"] == FAIL
    assert not can_start(report, "infinitetalk")
    assert not can_start(report, "ltx23")
    assert "VRAM 6 GB 低於最低需求 8 GB" in item(report, "wf_yue2")["detail"]
    assert "VRAM 6 GB 低於最低需求 10 GB" in item(report, "engine_infinitetalk")["detail"]


def test_mid_hardware_degrades_and_one_engine_can_still_run(config):
    report = evaluate(good_facts(gpus=[GpuInfo("RTX 4070", 12.0, "560.00")], ram_gb=48.0), config)
    assert item(report, "engine_infinitetalk")["level"] == DEGRADE
    assert "480p" in item(report, "engine_infinitetalk")["detail"]
    assert item(report, "engine_ltx23")["level"] == FAIL
    assert report["overall"] == DEGRADE
    assert can_start(report, "infinitetalk")
    assert not can_start(report, "ltx23")


def test_no_gpu_no_ffmpeg_low_disk_each_fail_with_a_fix(config):
    report = evaluate(good_facts(gpus=[], ffmpeg_version=None, disk_free_gb=5.0), config)
    for item_id in ("gpu", "ffmpeg", "disk"):
        assert item(report, item_id)["level"] == FAIL
        assert item(report, item_id)["fix"]
    assert report["overall"] == FAIL
    assert not can_start(report, "infinitetalk")


def test_comfyui_not_running(config):
    report = evaluate(good_facts(comfy_url=None, comfy_version="", object_info=None), config)
    comfy = item(report, "comfyui")
    assert comfy["level"] == FAIL
    assert "8000、8188" in comfy["detail"]
    assert "啟動 ComfyUI" in comfy["fix"]
    assert item(report, "wf_yue2")["level"] == FAIL
    assert report["overall"] == FAIL


def test_comfyui_too_old(config):
    report = evaluate(good_facts(comfy_version="0.35.9"), config)
    assert item(report, "comfyui")["level"] == FAIL
    assert "0.36.0" in item(report, "comfyui")["detail"]


def test_missing_model_lists_file_folder_and_download(config):
    info = full_object_info()
    info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0].remove("yue2_3b_int8_convrot.safetensors")
    report = evaluate(good_facts(object_info=info), config)
    yue2 = item(report, "wf_yue2")
    assert yue2["level"] == FAIL
    assert "缺少模型：yue2_3b_int8_convrot.safetensors" in yue2["detail"]
    assert "models/checkpoints" in yue2["fix"]
    assert "huggingface.co" in yue2["fix"]
    assert report["overall"] == FAIL


def test_missing_custom_node_names_the_pack(config):
    info = full_object_info()
    del info["MultiTalkModelLoader"]
    report = evaluate(good_facts(object_info=info), config)
    engine = item(report, "engine_infinitetalk")
    assert engine["level"] == FAIL
    assert "缺少節點：MultiTalkModelLoader" in engine["detail"]
    assert "ComfyUI-WanVideoWrapper" in engine["fix"]
    # 另一個引擎還能用，所以整體是降規而不是不通過
    assert report["overall"] == DEGRADE
    assert can_start(report, "ltx23")
    assert not can_start(report, "infinitetalk")


def test_vocal_separation_falls_back_to_second_option(config):
    info = full_object_info()
    info["MelBandRoFormerModelLoader"]["input"]["required"]["model_name"][0].clear()
    report = evaluate(good_facts(object_info=info), config)
    separation = item(report, "wf_vocal_separation")
    assert separation["level"] == DEGRADE
    assert "改用 AudioSeparation" in separation["detail"]
    assert "MelBandRoformer_fp16.safetensors" in separation["fix"]
    assert can_start(report, "infinitetalk")


def test_vocal_separation_fails_when_no_option_is_available(config):
    info = full_object_info()
    del info["MelBandRoFormerSampler"]
    del info["AudioSeparation"]
    report = evaluate(good_facts(object_info=info), config)
    assert item(report, "wf_vocal_separation")["level"] == FAIL
    assert not can_start(report, "infinitetalk")


def test_model_pattern_matches_differently_named_file(config):
    info = full_object_info()
    info["MultiTalkModelLoader"]["input"]["required"]["model"][0][:] = [
        "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"]
    assert item(evaluate(good_facts(object_info=info), config), "engine_infinitetalk")["level"] == PASS


@pytest.mark.parametrize("spec, expected", [
    ([["a", "b"], {}], ["a", "b"]),                                  # 舊格式
    (["COMBO", {"options": ["a", "b"], "multiselect": False}], ["a", "b"]),   # 新格式
    (["STRING", {"default": ""}], []),
])
def test_combo_options_handles_both_object_info_formats(spec, expected):
    info = {"N": {"input": {"required": {"x": spec}}}}
    assert combo_options(info, "N", "x") == expected
    assert combo_options(info, "N", "missing") == []


def test_parse_version():
    assert parse_version("0.44.2") > parse_version("0.36.0")
    assert parse_version("v0.36.0") == parse_version("0.36.0-rc1") == (0, 36, 0)
    assert parse_version("0.9.10") < parse_version("0.10.0")


def test_stage_against_fake_server_writes_report_and_remembers_url(config, monkeypatch):
    from app.pipeline.stages import s1_env
    monkeypatch.setattr(s1_env.hardware_probe, "probe_gpus", lambda: [GpuInfo("RTX 5090", 23.9, "1")])
    monkeypatch.setattr(s1_env.hardware_probe, "probe_ram_gb", lambda: 95.4)
    monkeypatch.setattr(s1_env.hardware_probe, "probe_disk_free_gb", lambda path: 500.0)
    monkeypatch.setattr(s1_env.hardware_probe, "probe_ffmpeg", lambda: "9.0.2")
    project = ProjectStore(config.get("paths.projects_root")).create("歌")

    with FakeComfyServer(full_object_info()) as server:
        config.set("comfyui.ports", [1, server.port])       # 第一個連接埠沒人聽，要自動試下一個
        config.set("comfyui.connect_timeout", 0.5)
        stage = EnvStage()
        result = stage.run(StageContext(config, project), Reporter())

    assert result.ok
    saved = json.loads(project.path("env_report.json").read_text(encoding="utf-8"))
    assert saved["comfy_url"] == server.url
    assert saved["overall"] == PASS
    assert config.get("comfyui.last_url") == server.url


def test_stage_reports_failure_when_comfyui_is_down(config, monkeypatch):
    from app.pipeline.stages import s1_env
    monkeypatch.setattr(s1_env.hardware_probe, "probe_gpus", lambda: [GpuInfo("RTX 5090", 23.9, "1")])
    monkeypatch.setattr(s1_env.hardware_probe, "probe_ffmpeg", lambda: "9.0.2")
    monkeypatch.setattr(s1_env.hardware_probe, "probe_disk_free_gb", lambda path: 500.0)
    config.set("comfyui.ports", [1])
    config.set("comfyui.connect_timeout", 0.5)
    result = EnvStage().run(StageContext(config), Reporter())
    assert not result.ok
    assert "ComfyUI" in result.message


def test_every_engine_in_the_config_is_checked(config):
    report = evaluate(good_facts(), config)
    assert set(report["engines"]) == {"infinitetalk", "infinitetalk_native", "ltx23", "wan22_s2v"}
    assert all(can_start(report, engine_id) for engine_id in report["engines"])
    info = full_object_info()
    del info["WanSoundImageToVideoExtend"]                    # 延伸區塊的節點不在範本裡，但需求清單有列
    report = evaluate(good_facts(object_info=info), config)
    assert not can_start(report, "wan22_s2v") and can_start(report, "ltx23")
