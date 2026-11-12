from __future__ import annotations

import copy
import json

import httpx
import pytest

from app.core import lyrics as L
from app.core.models import DONE, STALE, UserFacingError
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s2_lyrics import (LyricsStage, generate_lyrics, load_lyrics, regenerate_section,
                                           save_lyrics, template_store)
from app.services.llm_client import (LLMError, LLMFormatError, OllamaClient, OpenAICompatClient,
                                     _strict_schema, extract_json)
from app.storage.project_store import ProjectStore
from tests.helpers import BAD_JSON, GOOD_SECTIONS, FakeLLM

INPUTS = {"content": "夏夜騎車想起離開的人", "style": "City Pop", "language": "zh-TW",
          "lyrics_source": "ai", "lyric_template": "verse_chorus_bridge", "target_seconds": 150}


@pytest.fixture
def template(config):
    return template_store(config).get("verse_chorus_bridge")


@pytest.fixture
def project(config):
    return ProjectStore(config.get("paths.projects_root")).create("歌", dict(INPUTS))


def test_generate_assembles_repeats_and_asks_only_for_unique_sections(config, template):
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS)])
    lyrics = generate_lyrics(llm, template, INPUTS, config)

    assert lyrics["title"] == "夏夜的風"
    assert [s["type"] for s in lyrics["sections"]] == ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]
    chorus = GOOD_SECTIONS["sections"][1]["lines"]
    assert lyrics["sections"][1]["lines"] == lyrics["sections"][3]["lines"] == lyrics["sections"][5]["lines"] == chorus
    assert len(L.lyric_lines(lyrics)) == 22

    prompt = llm.calls[0]["user"]
    assert "夏夜騎車想起離開的人" in prompt and "City Pop" in prompt
    assert "請寫出以下 4 個段落" in prompt
    assert "1. 主歌：4 行，每行 7–11 字" in prompt and "4. 過渡：2 行，每行 7–11 字" in prompt
    assert "主歌 → 副歌 → 主歌 → 副歌（重複） → 過渡 → 副歌（重複）" in prompt
    assert "不得引用、改寫或仿寫任何現有歌曲" in prompt
    assert '{"title": "歌名"' in prompt                       # 提示詞裡的 JSON 範例沒有被變數取代弄壞


def test_generate_strips_punctuation_instead_of_failing(config, template):
    response = copy.deepcopy(GOOD_SECTIONS)
    response["sections"][0]["lines"][0] = "巷口的燈，一盞一盞亮起。"
    lyrics = generate_lyrics(FakeLLM([response]), template, INPUTS, config)
    assert lyrics["sections"][0]["lines"][0] == "巷口的燈 一盞一盞亮起"


def test_generate_retries_with_the_specific_violations(config, template):
    wrong = copy.deepcopy(GOOD_SECTIONS)
    wrong["sections"][1]["lines"].append("多出來的一行")                   # 副歌 5 行
    wrong["sections"][3]["lines"][0] = "這一行實在是太長太長太長太長太長了啦"      # 18 字，上限 11（容許 +2）
    llm = FakeLLM([wrong, BAD_JSON, copy.deepcopy(GOOD_SECTIONS)])
    logs: list[str] = []

    lyrics = generate_lyrics(llm, template, INPUTS, config, Reporter(on_log=logs.append))

    assert len(llm.calls) == 3
    second = llm.calls[1]["user"]
    assert "第 2 段（副歌）應有 4 行，實際 5 行" in second
    assert "第 4 段第 1 行「這一行實在是太長太長太長太長太長了啦」有 18 字，應為 7–11 字" in second
    assert "多出來的一行" in second                                        # 附上上一次的輸出
    assert "不是合法的 JSON" in llm.calls[2]["user"]
    assert len(L.lyric_lines(lyrics)) == 22
    assert any("第 2 次嘗試" in line for line in logs)


def test_char_count_tolerance_accepts_slightly_long_lines(config, template):
    response = copy.deepcopy(GOOD_SECTIONS)
    response["sections"][0]["lines"][0] = "一二三四五六七八九十一二三"          # 13 字 = 上限 11 + 容許 2
    assert generate_lyrics(FakeLLM([response]), template, INPUTS, config)


def test_generate_gives_up_after_retries_and_suggests_manual_input(config, template):
    wrong = {"title": "x", "sections": [{"lines": ["只有一段"]}]}
    llm = FakeLLM([wrong, wrong, wrong, GOOD_SECTIONS])
    with pytest.raises(UserFacingError) as e:
        generate_lyrics(llm, template, INPUTS, config)
    assert len(llm.calls) == 3                                           # 1 次 + 重試 2 次
    assert "連續 3 次" in e.value.message and "sections 應有 4 個段落，實際 1 個" in e.value.message
    assert "自己輸入歌詞" in e.value.fix


def test_regenerate_section_updates_identical_repeats_and_keeps_the_rest(config, template):
    lyrics = generate_lyrics(FakeLLM([copy.deepcopy(GOOD_SECTIONS)]), template, INPUTS, config)
    new_chorus = ["風吹過無人的街", "霓虹還亮著昨夜", "我把車停在路邊", "看回憶慢慢走遠"]
    llm = FakeLLM([{"lines": new_chorus}])

    updated = regenerate_section(llm, lyrics, 1, INPUTS, config, template)

    assert [updated["sections"][i]["lines"] for i in (1, 3, 5)] == [new_chorus] * 3
    assert updated["sections"][0] == lyrics["sections"][0] and updated["sections"][4] == lyrics["sections"][4]
    assert lyrics["sections"][1]["lines"] != new_chorus                  # 原本的物件沒被改動
    prompt = llm.calls[0]["user"]
    assert "請重寫第 2 段（副歌）：剛好 4 行，每行 6–10 字" in prompt
    assert "[主歌]" in prompt and "巷口的燈一盞一盞亮起" in prompt


def test_regenerate_section_rejects_instrumental_section(config):
    lyrics = {"title": "x", "sections": [{"type": "intro", "lines": []}, {"type": "verse", "lines": ["一二三四五"]}]}
    with pytest.raises(UserFacingError, match="純音樂"):
        regenerate_section(FakeLLM([]), lyrics, 0, INPUTS, config)


def test_save_lyrics_marks_downstream_stale_only_when_content_changes(project):
    lyrics = {"title": "歌", "language": "zh-TW", "sections": [{"type": "verse", "lines": ["一", "二"]}]}
    assert save_lyrics(project, lyrics) is True
    project.set_status("s2_lyrics", DONE)
    project.set_status("s3_song", DONE)

    assert save_lyrics(project, lyrics) is False
    assert project.status("s3_song") == DONE

    lyrics["sections"][0]["lines"][1] = "三"
    assert save_lyrics(project, lyrics) is True
    assert project.status("s3_song") == STALE
    assert project.path("lyrics", "lyrics.txt").read_text(encoding="utf-8") == "一\n三\n"
    assert load_lyrics(project) == lyrics


def test_stage_ai_mode_writes_files_and_unloads_model(config, project):
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS)])
    stage = LyricsStage(lambda cfg: llm)
    ctx = StageContext(config, project)
    assert stage.check_inputs(ctx) == []
    result = stage.run(ctx, Reporter())

    assert result.ok and llm.unloaded == 1
    saved = json.loads(project.path("lyrics", "lyrics.json").read_text(encoding="utf-8"))
    assert saved["title"] == "夏夜的風" and len(saved["sections"]) == 6
    assert len(project.path("lyrics", "lyrics.txt").read_text(encoding="utf-8").splitlines()) == 22
    assert result.warnings == []                                         # 22 行估計約 129 秒，在目標 150 秒的 ±20% 內


def test_stage_unloads_model_even_when_generation_fails(config, project):
    llm = FakeLLM([LLMError("逾時")])
    with pytest.raises(LLMError):
        LyricsStage(lambda cfg: llm).run(StageContext(config, project), Reporter())
    assert llm.unloaded == 1


def test_max_mode_shrinks_the_structure_so_the_song_can_finish_within_the_cap(config, project):
    """上限 90 秒放不下六段 22 行（108 BPM 約需 136 秒）：從後面拿掉段落，並說明做了什麼。"""
    project.data["inputs"].update(duration_mode="max", target_seconds=90)
    response = {"title": "短版", "sections": GOOD_SECTIONS["sections"][:3]}       # 縮短後只需要三個不重複的段落
    llm = FakeLLM([copy.deepcopy(response)])
    result = LyricsStage(lambda cfg: llm).run(StageContext(config, project), Reporter())

    assert result.ok
    lyrics = load_lyrics(project)
    assert [s["type"] for s in lyrics["sections"]] == ["verse", "chorus", "verse"]
    assert "為了在上限 90 秒內唱完，歌詞結構縮短為「主歌 → 副歌 → 主歌」（拿掉最後 3 個段落）" in result.warnings[0]
    assert "請寫出以下 3 個段落" in llm.calls[0]["user"]
    assert L.lyrics_seconds(lyrics, 108) * 1.1 <= 90                          # 留一成餘裕後仍在上限內


def test_auto_mode_keeps_the_full_structure_whatever_the_number_says(config, project):
    project.data["inputs"].update(duration_mode="auto", target_seconds=30)
    result = LyricsStage(lambda cfg: FakeLLM([copy.deepcopy(GOOD_SECTIONS)])).run(
        StageContext(config, project), Reporter())
    assert result.ok and result.warnings == [] and len(load_lyrics(project)["sections"]) == 6


def test_manual_lyrics_too_long_for_the_cap_are_flagged_before_generating(config, project):
    lines = "\n".join(f"第{i}句歌詞在這裡" for i in range(1, 9))
    project.data["inputs"].update(lyrics_source="manual", duration_mode="max", target_seconds=45, style="抒情流行",
                                  manual_lyrics=f"[主歌]\n{lines}")
    result = LyricsStage(lambda cfg: None).run(StageContext(config, project), Reporter())
    assert result.ok and "估計需要約 69 秒，超過上限 45 秒，歌曲可能唱不完" in result.warnings[0]


def test_song_length_estimate_matches_what_was_measured():
    assert L.bpm_from_text("pop ballad, slow tempo around 72 BPM") == 72 and L.bpm_from_text("no tempo") == 96
    assert L.song_seconds(22, 6, 108) == pytest.approx(136, abs=1)            # 實機：140–151 秒
    assert L.song_seconds(8, 2, 72) == pytest.approx(76, abs=1)               # 實機：45 秒唱不完

def test_stage_manual_mode_parses_pasted_lyrics_without_calling_llm(config, project):
    project.data["inputs"].update(lyrics_source="manual", manual_lyrics="[Verse]\nline one\n[Chorus]\nline two")

    def no_llm(cfg):
        raise AssertionError("人工模式不應該呼叫 LLM")

    stage = LyricsStage(no_llm)
    result = stage.run(StageContext(config, project), Reporter())
    assert result.ok
    assert [s["type"] for s in load_lyrics(project)["sections"]] == ["verse", "chorus"]


def test_stage_manual_mode_reports_bad_format_with_line_number(config, project):
    project.data["inputs"].update(lyrics_source="manual", manual_lyrics="[主歌]\n一句\n[??]\n二句")
    with pytest.raises(L.LyricsFormatError, match="第 3 行"):
        LyricsStage(lambda cfg: None).run(StageContext(config, project), Reporter())


def test_check_inputs_explains_what_is_missing(config, project):
    stage = LyricsStage(lambda cfg: None)
    project.data["inputs"].update(content="", style="")
    assert [p.message for p in stage.check_inputs(StageContext(config, project))] == [
        "還沒有填寫音樂內容", "還沒有選擇曲風"]
    project.data["inputs"].update(lyrics_source="manual")
    assert "還沒有貼上歌詞" in stage.check_inputs(StageContext(config, project))[0].message


# ---- LLM 用戶端 ----

@pytest.mark.parametrize("text", [
    '{"a": 1}',
    '好的，以下是結果：\n```json\n{"a": 1}\n```\n希望有幫助',
    '<think>要輸出 {"b": 2} 嗎</think>{"a": 1}',
    '結果是 {"a": 1} 。',
])
def test_extract_json_tolerates_wrapping(text):
    assert extract_json(text) == {"a": 1}


@pytest.mark.parametrize("text", ["", "抱歉我無法", "[1, 2]", '{"a": '])
def test_extract_json_rejects_non_objects(text):
    with pytest.raises(LLMFormatError):
        extract_json(text)


def test_ollama_client_request_shape_errors_and_unload():
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if body["model"] == "missing":
            return httpx.Response(404, json={"error": "model not found"})
        if request.url.path == "/api/generate":
            return httpx.Response(200, json={})
        return httpx.Response(200, json={"message": {"content": '{"lines": ["一"]}'}})

    client = OllamaClient("http://ollama", "text-model", "vision-model", "5m", transport=httpx.MockTransport(handler))
    schema = {"type": "object"}
    assert client.chat_json("系統", "使用者", schema) == {"lines": ["一"]}
    path, body = requests[0]
    assert path == "/api/chat"
    assert body["model"] == "text-model" and body["format"] == schema
    assert body["think"] is False and body["stream"] is False and body["keep_alive"] == "5m"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]

    client.unload()
    assert requests[-1] == ("/api/generate", {"model": "text-model", "keep_alive": 0})
    client.unload()                                                      # 沒有用過的模型不必再卸載
    assert len(requests) == 2

    missing = OllamaClient("http://ollama", "missing", "v", transport=httpx.MockTransport(handler))
    with pytest.raises(LLMError, match="ollama pull missing"):
        missing.chat_json("s", "u", schema)

    def refuse(request):
        raise httpx.ConnectError("refused")

    down = OllamaClient("http://ollama", "m", "v", transport=httpx.MockTransport(refuse))
    with pytest.raises(LLMError) as e:
        down.chat_json("s", "u", schema)
    assert "Ollama 已啟動" in e.value.fix


def test_ollama_uses_vision_model_when_an_image_is_attached(tmp_path):
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": "{}"}})

    photo = tmp_path / "p.png"
    photo.write_bytes(b"\x89PNG")
    OllamaClient("http://o", "text", "vision", transport=httpx.MockTransport(handler)).chat_json("s", "u", {}, photo)
    assert seen["model"] == "vision" and len(seen["messages"][1]["images"]) == 1


def test_openai_compatible_client():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["Authorization"] != "Bearer good":
            return httpx.Response(401, json={})
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions" and body["model"] == "m"
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": true}'}}]})

    transport = httpx.MockTransport(handler)
    assert OpenAICompatClient("http://x/v1/", "m", "good", transport=transport).chat_json("s", "u", {}) == {"ok": True}
    with pytest.raises(LLMError, match="拒絕了金鑰"):
        OpenAICompatClient("http://x/v1", "m", "bad", transport=transport).chat_json("s", "u", {})
    with pytest.raises(LLMError, match="金鑰"):
        OpenAICompatClient("http://x/v1", "m", "")
    with pytest.raises(LLMError, match="模型名稱"):
        OpenAICompatClient("http://x/v1", "", "k")


def test_strict_schema_adds_additional_properties_at_every_object_level():
    schema = {"type": "object", "properties": {"sections": {"type": "array", "items": {
        "type": "object", "properties": {"lines": {"type": "array", "items": {"type": "string"}}}}}}}
    strict = _strict_schema(schema)
    assert strict["additionalProperties"] is False
    assert strict["properties"]["sections"]["items"]["additionalProperties"] is False
    assert "additionalProperties" not in schema                           # 不改動原本的 schema
