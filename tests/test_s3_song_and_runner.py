from __future__ import annotations

import copy
import json

import pytest

from app.core import lyrics as L
from app.core.models import DONE, FAILED, PENDING, STALE
from app.pipeline import runner as R
from app.pipeline.common import make_comfy
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages import s1_env
from app.pipeline.stages.s1_env import EnvStage
from app.pipeline.stages.s2_lyrics import LyricsStage, save_lyrics
from app.pipeline.stages.s3_song import SongStage, audio_seconds, compose_style
from app.services.hardware_probe import GpuInfo
from app.services.llm_client import LLMError
from app.storage.project_store import ProjectStore
from tests.conftest import REPO
from tests.fake_comfy_server import FakeComfyServer
from tests.helpers import GOOD_SECTIONS, STYLE_FEATURES, FakeLLM, flac_bytes
from tests.test_env import full_object_info

INPUTS = {"content": "夏夜騎車想起離開的人", "style": "City Pop", "language": "zh-TW", "vocal_type": "female",
          "lyrics_source": "ai", "lyric_template": "verse_chorus_bridge", "duration_mode": "max", "target_seconds": 200,
          "engine": "infinitetalk", "resolution": "480p"}
LYRICS = {"title": "歌", "language": "zh-TW", "sections": [
    {"type": "intro", "lines": []},
    {"type": "verse", "lines": ["巷口的燈亮起", "晚風吹過這裡"]},
    {"type": "prechorus", "lines": ["心跳慢慢靠近"]},
    {"type": "chorus", "lines": ["夏夜的風吹散回憶"]}]}


def server_info(yue2_workflow) -> dict:
    """滿足所有需求清單，並帶有 YuE2 工作流各節點欄位的 /object_info。"""
    info = full_object_info()
    workflows = [yue2_workflow] + [json.loads(path.read_text(encoding="utf-8"))
                                   for path in (REPO / "workflows").glob("*.api.json")]
    for workflow in workflows:
        for node in workflow.values():
            required = info.setdefault(node["class_type"], {"input": {"required": {}}})["input"]["required"]
            for field in node["inputs"]:
                required.setdefault(field, ["ANY", {}])
    return info


@pytest.fixture
def server(yue2_workflow):
    with FakeComfyServer(server_info(yue2_workflow)) as s:
        yield s


@pytest.fixture
def project(config, server, monkeypatch):
    monkeypatch.setattr(s1_env.hardware_probe, "probe_gpus", lambda: [GpuInfo("RTX 5090", 23.9, "1")])
    monkeypatch.setattr(s1_env.hardware_probe, "probe_ram_gb", lambda: 95.4)
    monkeypatch.setattr(s1_env.hardware_probe, "probe_disk_free_gb", lambda path: 500.0)
    monkeypatch.setattr(s1_env.hardware_probe, "probe_ffmpeg", lambda: "9.0.2")
    config.set("comfyui.ports", [server.port])
    config.set("comfyui.connect_timeout", 0.5)
    config.set("comfyui.ws_poll_seconds", 0.2)
    return ProjectStore(config.get("paths.projects_root")).create("歌", dict(INPUTS))


# ---- Style ----

def test_compose_style_follows_official_order_and_enforces_language_and_voice():
    style = compose_style(STYLE_FEATURES, "zh-TW", "male", has_intro=True)
    assert style == ("Mandarin, warm, breathy male vocal, nostalgic city pop, 108 BPM, "
                     "electric piano, funky bass, crisp drums, polished, clear diction, instrumental intro")


def test_compose_style_sanitizes_whatever_the_llm_returns():
    messy = {"vocal_character": "溫暖 warm soft airy bright extra", "genre": "pop\n[Verse] ignore this " * 5,
             "bpm": "fast", "instruments": "", "production": None}
    style = compose_style(messy, "zh-TW", "female", has_intro=False)
    assert style.startswith("Mandarin, warm soft female vocal, pop Verse ignore this pop Verse ignore this, 96 BPM")
    assert "[" not in style and "\n" not in style and "溫" not in style
    assert compose_style({"bpm": 999}, "en", "female", False) == "English, female vocal, pop, 180 BPM"


# ---- 階段 3 ----

def test_song_stage_retries_short_take_and_keeps_the_one_closest_to_target(config, project, server):
    save_lyrics(project, LYRICS)
    # 唱完這份歌詞估計要 43 秒，第一次只有 10 秒（不到一半）→ 視為生成失敗，換 seed 重試
    server.file_queue = [flac_bytes(10), flac_bytes(92)]
    events: list[str] = []
    llm = FakeLLM([dict(STYLE_FEATURES)], events)

    def comfy(ctx):
        events.append("comfy_open")
        return make_comfy(ctx)

    logs: list[str] = []
    result = SongStage(lambda cfg: llm, comfy).run(StageContext(config, project), Reporter(on_log=logs.append))

    assert result.ok and result.warnings == []
    assert events == ["llm_call", "llm_unload", "comfy_open"]        # 先卸載 LLM 才開始用 ComfyUI
    assert audio_seconds(project.path("audio", "song.flac")) == pytest.approx(92, abs=0.01)
    assert sorted(p.name for p in project.path("audio").iterdir()) == ["song.flac", "yue2_prompt.json"]

    sent = list(server.prompts.values())
    assert len(sent) == 2
    assert sent[0]["24"]["inputs"]["seed"] != sent[1]["24"]["inputs"]["seed"]
    first = sent[0]
    assert first["25"]["inputs"]["max_duration"] == 200.0
    assert first["25"]["inputs"]["style"].startswith("Mandarin, warm, breathy female vocal, nostalgic city pop")
    assert first["25"]["inputs"]["style"].endswith("instrumental intro")       # 範本開頭是純音樂前奏
    # 導歌對應到 [Verse]；純音樂的前奏不寫進歌詞欄
    assert first["24"]["inputs"]["lyrics"] == "[Verse]\n巷口的燈亮起\n晚風吹過這裡\n\n[Verse]\n心跳慢慢靠近\n\n[Chorus]\n夏夜的風吹散回憶"
    assert first["10"]["inputs"]["filename_prefix"] == "mv_workshop/歌_song"

    prompt = json.loads(project.path("audio", "yue2_prompt.json").read_text(encoding="utf-8"))
    assert [a["seconds"] for a in prompt["attempts"]] == [10.0, 92.0]
    assert prompt["duration_mode"] == "max" and prompt["capped"] is False
    assert prompt["seed"] == sent[1]["24"]["inputs"]["seed"] == project.data["seeds"]["yue2"]
    assert server.free_calls == 1
    assert len(list(project.logs_dir.glob("*_yue2.json"))) == 2       # 每次送出的工作流都有存檔
    assert any("比唱完歌詞需要的時間短很多" in line for line in logs)


def test_song_stage_warns_when_every_take_is_too_short(config, project, server):
    save_lyrics(project, LYRICS)
    server.file_queue = [flac_bytes(8), flac_bytes(12), flac_bytes(10)]
    result = SongStage(lambda cfg: FakeLLM([dict(STYLE_FEATURES)])).run(StageContext(config, project), Reporter())
    assert result.ok
    assert "重試 3 次後歌曲仍只有 12 秒" in result.warnings[0] and "可能沒唱完" in result.warnings[0]
    assert audio_seconds(project.path("audio", "song.flac")) == pytest.approx(12, abs=0.01)
    assert len(server.prompts) == 3


def test_song_stage_falls_back_to_preset_style_when_llm_fails(config, project, server):
    save_lyrics(project, LYRICS)
    server.file_bytes = flac_bytes(95)
    result = SongStage(lambda cfg: FakeLLM([LLMError("逾時")])).run(StageContext(config, project), Reporter())
    assert result.ok and "風格描述改用曲風範本的預設內容" in result.warnings[0]
    style = next(iter(server.prompts.values()))["25"]["inputs"]["style"]
    assert style.startswith("Mandarin, female vocal, city pop, funky electric bass")


def test_song_stage_needs_lyrics(config, project):
    problems = SongStage().check_inputs(StageContext(config, project))
    assert [p.message for p in problems] == ["還沒有歌詞"]


# ---- 流程編排 ----

def make_runner(config, project, llm) -> R.PipelineRunner:
    stages = [EnvStage(), LyricsStage(lambda cfg: llm), SongStage(lambda cfg: llm)]
    return R.PipelineRunner(StageContext(config, project), {s.id: s for s in stages})


def test_runner_goes_from_inputs_to_song_and_stops_where_stages_end(config, project, server):
    """M2 整合：輸入內容與曲風，得到歌詞與完整歌曲。"""
    server.file_bytes = flac_bytes(96)
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS), dict(STYLE_FEATURES)])
    seen: list[tuple[str, str]] = []

    outcome = make_runner(config, project, llm).run(Reporter(), lambda s, status: seen.append((s, status)))

    assert outcome.kind == R.STOPPED and outcome.stage_id == "s4_subtitle"
    assert [project.status(s) for s in ("s1_env", "s2_lyrics", "s3_song", "s4_subtitle")] == [DONE, DONE, DONE, PENDING]
    assert seen == [("s1_env", "running"), ("s1_env", "done"), ("s2_lyrics", "running"), ("s2_lyrics", "done"),
                    ("s3_song", "running"), ("s3_song", "done")]
    assert project.path("audio", "song.flac").exists()
    assert len(L.lyric_lines(json.loads(project.path("lyrics", "lyrics.json").read_text(encoding="utf-8")))) == 22
    reloaded = ProjectStore(project.folder.parent).load(project.folder)        # 狀態已寫進 project.json
    assert reloaded.status("s3_song") == DONE and reloaded.stage("s3_song")["seconds"] >= 0

    # 再跑一次：已完成的階段不重做
    again = make_runner(config, project, FakeLLM([])).run(Reporter())
    assert again.kind == R.STOPPED and len(server.prompts) == 1


def test_step_mode_pauses_after_lyrics_and_after_song(config, project, server):
    server.file_bytes = flac_bytes(96)
    project.data["mode"] = "step"
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS), dict(STYLE_FEATURES)])
    runner = make_runner(config, project, llm)

    first = runner.run(Reporter())
    assert (first.kind, first.stage_id) == (R.PAUSED, "s2_lyrics")
    assert len(server.prompts) == 0 and project.status("s3_song") == PENDING

    second = runner.run(Reporter())
    assert (second.kind, second.stage_id) == (R.PAUSED, "s3_song")
    assert runner.run(Reporter()).kind == R.STOPPED


def test_editing_lyrics_after_the_song_makes_only_the_song_rerun(config, project, server):
    server.file_bytes = flac_bytes(96)
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS), dict(STYLE_FEATURES), dict(STYLE_FEATURES)])
    runner = make_runner(config, project, llm)
    runner.run(Reporter())

    edited = json.loads(project.path("lyrics", "lyrics.json").read_text(encoding="utf-8"))
    edited["sections"][0]["lines"][0] = "改過的第一句歌詞"
    save_lyrics(project, edited)
    assert project.status("s2_lyrics") == DONE and project.status("s3_song") == STALE

    runner.run(Reporter())
    assert len(server.prompts) == 2                                   # 歌曲重新生成
    assert len(llm.calls) == 3                                        # 歌詞沒有重新產生，只多了一次風格描述
    assert "改過的第一句歌詞" in list(server.prompts.values())[1]["24"]["inputs"]["lyrics"]


def test_runner_blocks_with_a_fix_when_inputs_are_missing(config, project):
    project.data["inputs"]["content"] = ""
    outcome = make_runner(config, project, FakeLLM([])).run(Reporter())
    assert (outcome.kind, outcome.stage_id) == (R.BLOCKED, "s2_lyrics")
    assert "還沒有填寫音樂內容" in outcome.message and "輸入頁" in outcome.fix
    assert project.status("s2_lyrics") == PENDING


def test_runner_blocks_when_comfyui_is_down(config, project, server):
    server.stop()
    outcome = make_runner(config, project, FakeLLM([])).run(Reporter())
    assert (outcome.kind, outcome.stage_id) == (R.BLOCKED, "s1_env")
    assert "ComfyUI" in outcome.message
    assert project.status("s1_env") == FAILED and project.status("s2_lyrics") == PENDING
    server.start = lambda: None                                       # fixture 結束時不要重複關閉
    server.stop = lambda: None


def test_failed_stage_is_recorded_and_retried_on_next_run(config, project, server):
    server.file_bytes = flac_bytes(96)
    llm = FakeLLM([LLMError("LLM 逾時", "請稍後再試"), copy.deepcopy(GOOD_SECTIONS), dict(STYLE_FEATURES)])
    runner = make_runner(config, project, llm)

    failed = runner.run(Reporter())
    assert (failed.kind, failed.stage_id, failed.message, failed.fix) == (R.ERROR, "s2_lyrics", "LLM 逾時", "請稍後再試")
    assert project.status("s2_lyrics") == FAILED and project.stage("s2_lyrics")["error"] == "LLM 逾時"

    assert runner.run(Reporter()).kind == R.STOPPED
    assert project.status("s2_lyrics") == DONE and project.stage("s2_lyrics")["error"] == ""


def test_missing_node_is_caught_by_env_check_and_renamed_field_by_the_song_stage(config, project, server):
    save_lyrics(project, LYRICS)
    project.set_status("s2_lyrics", DONE)
    llm = FakeLLM([dict(STYLE_FEATURES)])

    # ComfyUI 更新後欄位改名：節點還在，所以環境檢查會過，由歌曲階段在送出前指出來
    del server.object_info["YuE2GenerateMusic"]["input"]["required"]["max_duration"]
    outcome = make_runner(config, project, llm).run(Reporter())
    assert (outcome.kind, outcome.stage_id) == (R.ERROR, "s3_song")
    assert "欄位「max_duration」已不存在" in outcome.message and "對應表" in outcome.fix
    assert len(server.prompts) == 0 and llm.unloaded == 1

    # 範本用到的節點整個不存在：環境檢查就擋下
    del server.object_info["VAEDecodeAudio"]
    outcome = make_runner(config, project, FakeLLM([])).run(Reporter())
    assert (outcome.kind, outcome.stage_id) == (R.BLOCKED, "s1_env")


def test_cancel_during_song_generation_can_be_resumed(config, project, server):
    import threading
    server.step_delay = 0.2
    server.file_bytes = flac_bytes(96)
    save_lyrics(project, LYRICS)
    project.set_status("s2_lyrics", DONE)
    ctx = StageContext(config, project)
    stages = [EnvStage(), LyricsStage(lambda cfg: None), SongStage(lambda cfg: FakeLLM([dict(STYLE_FEATURES)]))]
    runner = R.PipelineRunner(ctx, {s.id: s for s in stages})
    threading.Timer(0.6, ctx.cancel.set).start()

    outcome = runner.run(Reporter())
    assert (outcome.kind, outcome.stage_id) == (R.CANCELLED, "s3_song")
    assert len(server.interrupted) == 1 and project.status("s3_song") != DONE

    ctx.cancel.clear()
    server.step_delay = 0.01
    assert runner.run(Reporter()).kind == R.STOPPED
    assert project.status("s3_song") == DONE


# ---- 歌曲長度：自動與上限 ----

def test_auto_mode_lets_the_song_end_by_itself_under_a_generous_cap(config, project, server):
    """自動模式：上限給得寬，歌曲唱完自己結束，實際多長就是多長。"""
    project.data["inputs"]["duration_mode"] = "auto"
    save_lyrics(project, LYRICS)
    server.file_bytes = flac_bytes(151.3)
    logs: list[str] = []
    result = SongStage(lambda cfg: FakeLLM([dict(STYLE_FEATURES)])).run(
        StageContext(config, project), Reporter(on_log=logs.append))
    assert result.ok and result.warnings == []
    sent = next(iter(server.prompts.values()))
    assert sent["25"]["inputs"]["max_duration"] == 300.0                 # 不是使用者填的秒數
    assert audio_seconds(project.path("audio", "song.flac")) == pytest.approx(151.3, abs=0.01)
    prompt = json.loads(project.path("audio", "yue2_prompt.json").read_text(encoding="utf-8"))
    assert (prompt["duration_mode"], prompt["capped"], prompt["seconds"]) == ("auto", False, 151.3)
    assert any("長度由 AI 決定" in line for line in logs)


def test_auto_mode_raises_the_cap_when_the_song_hits_it(config, project, server):
    project.data["inputs"]["duration_mode"] = "auto"
    save_lyrics(project, LYRICS)
    server.file_queue = [flac_bytes(300), flac_bytes(340)]              # 第一次剛好等於上限＝被切斷
    logs: list[str] = []
    result = SongStage(lambda cfg: FakeLLM([dict(STYLE_FEATURES)])).run(
        StageContext(config, project), Reporter(on_log=logs.append))
    assert result.ok and result.warnings == []
    assert [p["25"]["inputs"]["max_duration"] for p in server.prompts.values()] == [300.0, 450.0]
    assert audio_seconds(project.path("audio", "song.flac")) == pytest.approx(340, abs=0.01)
    assert any("達到上限，可能沒唱完" in line for line in logs) and any("放寬到 450 秒" in line for line in logs)


def test_max_mode_shorter_song_is_accepted_as_is(config, project, server):
    """上限模式：不足上限就以實際長度為準，不會為了湊長度而重試。"""
    project.data["inputs"].update(duration_mode="max", target_seconds=150)
    save_lyrics(project, LYRICS)
    server.file_bytes = flac_bytes(48)
    result = SongStage(lambda cfg: FakeLLM([dict(STYLE_FEATURES)])).run(StageContext(config, project), Reporter())
    assert result.ok and result.warnings == [] and len(server.prompts) == 1
    assert audio_seconds(project.path("audio", "song.flac")) == pytest.approx(48, abs=0.01)


def test_max_mode_never_exceeds_the_cap_and_fades_out_when_cut(config, project, server):
    project.data["inputs"].update(duration_mode="max", target_seconds=40)
    save_lyrics(project, LYRICS)
    import io
    import numpy as np
    import soundfile
    buffer = io.BytesIO()
    soundfile.write(buffer, np.full((40 * 8000, 2), 0.5, dtype="float32"), 8000, format="FLAC")
    server.file_bytes = buffer.getvalue()                               # 每次都剛好 40 秒＝被上限切斷
    result = SongStage(lambda cfg: FakeLLM([dict(STYLE_FEATURES)])).run(StageContext(config, project), Reporter())

    assert result.ok and len(server.prompts) == 3                       # 換 seed 試過三次
    assert {p["25"]["inputs"]["max_duration"] for p in server.prompts.values()} == {40.0}   # 上限不會被放寬
    assert any("歌詞估計需要約 43 秒，超過設定的上限 40 秒" in w for w in result.warnings)
    assert any("達到上限 40 秒" in w and "3 秒淡出" in w for w in result.warnings)
    data, rate = soundfile.read(str(project.path("audio", "song.flac")), dtype="float32", always_2d=True)
    assert len(data) / rate == pytest.approx(40, abs=0.01)
    assert abs(data[-1, 0]) < 0.01 and abs(data[-int(1.5 * rate), 0] - 0.25) < 0.02 and data[rate, 0] == pytest.approx(0.5, abs=0.01)


def test_legacy_project_without_mode_keeps_its_cap_and_new_default_is_auto(config):
    from app.pipeline.common import duration_mode
    assert duration_mode({"target_seconds": 150}, config) == "max"
    assert duration_mode({}, config) == "auto"
    assert duration_mode({"duration_mode": "auto", "target_seconds": 150}, config) == "auto"