from __future__ import annotations

import json

import pytest

from app.core.models import UserFacingError
from app.services.workflow_template import WorkflowTemplate
from tests.conftest import REPO
from tests.fake_comfy_server import object_info_for

WORKFLOW = {
    "1": {"class_type": "Loader", "inputs": {"name": "a.safetensors"}},
    "2": {"class_type": "Gen", "inputs": {"model": ["1", 0], "text": "", "seed": 0}},
    "3": {"class_type": "Gen", "inputs": {"model": ["1", 0], "text": "", "seed": 0}},
    "4": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": "x"}},
}
MAPPING = {
    "params": {
        "text": [{"node": "2", "input": "text"}, {"node": "3", "input": "text"}],
        "seed": [{"node": "3", "input": "seed"}],
    },
    "outputs": {"audio": {"node": "4", "kind": "audio"}},
}


def test_inject_writes_every_target_and_keeps_template_untouched():
    template = WorkflowTemplate("t", WORKFLOW, MAPPING)
    result = template.inject({"text": "你好", "seed": 42})
    assert result["2"]["inputs"]["text"] == "你好"
    assert result["3"]["inputs"]["text"] == "你好"
    assert result["3"]["inputs"]["seed"] == 42
    assert result["2"]["inputs"]["seed"] == 0                 # 沒在對應表裡的欄位不動
    assert result["2"]["inputs"]["model"] == ["1", 0]         # 連線保留
    assert WORKFLOW["2"]["inputs"]["text"] == ""              # 範本本身不被修改


def test_inject_rejects_unknown_param():
    template = WorkflowTemplate("t", WORKFLOW, MAPPING)
    with pytest.raises(UserFacingError, match="沒有定義參數：width"):
        template.inject({"width": 832})


@pytest.mark.parametrize("target, expected", [
    ({"node": "99", "input": "text"}, "不存在的節點 99"),
    ({"node": "2", "input": "prompt"}, "不存在的欄位「prompt」"),
])
def test_mapping_pointing_nowhere_is_reported(target, expected):
    with pytest.raises(UserFacingError, match=expected):
        WorkflowTemplate("t", WORKFLOW, {"params": {"text": [target]}})


def test_ui_format_workflow_is_rejected():
    with pytest.raises(UserFacingError, match="不是 API 格式"):
        WorkflowTemplate("t", {"nodes": [], "links": []}, {"params": {}})


def test_swapping_workflow_only_needs_a_new_mapping(tmp_path):
    """驗收：更換工作流範本時只需修改對應表，不必改程式。"""
    other = {"70": {"class_type": "Gen", "inputs": {"prompt_text": "", "noise_seed": 0}},
             "71": {"class_type": "SaveAudio", "inputs": {"audio": ["70", 0], "filename_prefix": ""}}}
    (tmp_path / "custom.api.json").write_text(json.dumps(other), encoding="utf-8")
    (tmp_path / "song.map.json").write_text(json.dumps({
        "template": "custom.api.json",
        "params": {"text": [{"node": "70", "input": "prompt_text"}],
                   "seed": [{"node": "70", "input": "noise_seed"}]},
        "outputs": {"audio": {"node": "71", "kind": "audio"}}}), encoding="utf-8")
    template = WorkflowTemplate.load(tmp_path, "song")
    result = template.inject({"text": "你好", "seed": 7})     # 與上面相同的參數名稱
    assert result["70"]["inputs"] == {"prompt_text": "你好", "noise_seed": 7}
    assert template.output_node("audio") == "71"


def test_shipped_yue2_template_is_consistent(yue2_workflow):
    template = WorkflowTemplate.load(REPO / "workflows", "yue2_text_to_music")
    workflow = template.inject({"style": "pop", "lyrics": "[Verse]\n啦", "seed": 5, "max_duration": 150.0,
                                "ckpt_name": "m.safetensors", "plan_mode": "full", "filename_prefix": "p/q"})
    assert workflow["24"]["inputs"]["lyrics"] == workflow["25"]["inputs"]["lyrics"] == "[Verse]\n啦"
    assert {workflow[n]["inputs"]["seed"] for n in ("24", "25", "8")} == {5}
    assert workflow["25"]["inputs"]["max_duration"] == 150.0
    assert template.check_against(object_info_for(yue2_workflow)) == []


def test_check_against_reports_missing_node_and_renamed_field(yue2_workflow):
    template = WorkflowTemplate.load(REPO / "workflows", "yue2_text_to_music")
    info = object_info_for(yue2_workflow)
    del info["VAEDecodeAudio"]
    del info["YuE2GenerateMusic"]["input"]["required"]["max_duration"]
    problems = template.check_against(info)
    assert any("VAEDecodeAudio" in p for p in problems)
    assert any("max_duration" in p for p in problems)
    assert len(problems) == 2
