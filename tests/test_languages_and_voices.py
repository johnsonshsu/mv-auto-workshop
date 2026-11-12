from __future__ import annotations

import copy
import json

import pytest

from app.core import lyrics as L
from app.pipeline.common import language_info, voice_options
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s2_lyrics import (LyricsStage, check_section, generate_lyrics, localize, regenerate_section,
                                           save_lyrics, template_store)
from app.pipeline.stages.s3_song import SongStage, compose_style
from app.pipeline.stages.s4_subtitle import SubtitleStage
from app.storage.project_store import ProjectStore
from tests.helpers import GOOD_SECTIONS, STYLE_FEATURES, FakeLLM, flac_bytes
from tests.test_s3_song_and_runner import project, server, server_info  # noqa: F401  （fixture）
from tests.test_s4_subtitle import FakeAligner, raw_line, vocal_flac

ENGLISH = {"title": "Autumn Street", "sections": [
    {"lines": ["Walking down the autumn street", "Golden leaves beneath my feet",
               "Every breath is soft and sweet", "Feel the wind against my cheek"]},
    {"lines": ["Let the breeze carry me", "Where the sky is wide and free",
               "All the worries drift away", "On this quiet autumn day"]},
    {"lines": ["Coffee steam and morning light", "Shadows dancing left and right",
               "Strangers smile as they go by", "Painted clouds across the sky"]},
    {"lines": ["Nothing here I need to chase", "Only wind upon my face"]},
]}
JAPANESE = {"title": "秋の風", "sections": [
    {"lines": ["秋の街をゆっくり歩いてゆく", "落ち葉が足もとで踊りだす",
               "頬をなでる風がやさしくて", "胸の奥がふっと軽くなる"]},
    {"lines": ["風よこのまま連れていって", "空の向こうまで自由に",
               "迷いも不安も置いてゆく", "静かな秋の午後に"]},
    {"lines": ["コーヒーの香りと朝の光", "影がゆらゆら揺れている",
               "すれ違う人のほほえみが", "雲のすきまに溶けてゆく"]},
    {"lines": ["追いかけるものは何もない", "ただ風だけを感じている"]},
]}


@pytest.fixture
def template(config):
    return template_store(config).get("verse_chorus_bridge")


def inputs(language: str, **extra) -> dict:
    return {"content": "在秋天的街道上走著，迎面而來自然的風", "style": "抒情流行", "language": language,
            "lyrics_source": "ai", "lyric_template": "verse_chorus_bridge", "target_seconds": 150, **extra}


# ---- 語言設定 ----

def test_language_settings(config):
    assert list(config.get("languages")) == ["zh-TW", "en", "ja", "yue"]
    assert language_info(config, "en")["yue2"] == "English" and language_info(config, "en")["whisper"] == "en"
    assert language_info(config, "yue") == {**language_info(config, "yue"), "yue2": "Cantonese", "whisper": "yue",
                                            "chinese": True, "code": "yue"}
    assert language_info(config, "不存在")["code"] == "zh-TW"          # 不認得的代碼當成國語
    assert voice_options(config)[0]["value"] == "" and len(voice_options(config)) >= 8
    assert all(len(option["value"].replace(",", " ").split()) <= 2 for option in voice_options(config))


def test_localize_scales_line_length_per_language(config, template):
    assert localize(template, language_info(config, "zh-TW")) is template
    english = localize(template, language_info(config, "en"))["sections"]
    assert (english[0]["chars_min"], english[0]["chars_max"]) == (5, 8)        # 7–11 個中文字 → 5–8 個英文單字
    japanese = localize(template, language_info(config, "ja"))["sections"]
    assert (japanese[0]["chars_min"], japanese[0]["chars_max"]) == (10, 16)    # 假名比較佔字數
    assert english[3]["repeat_of"] == 2 and template["sections"][0]["chars_max"] == 11   # 原範本不被改動


# ---- 歌詞 ----

def test_english_lyrics_are_counted_in_words_and_not_converted(config, template):
    llm = FakeLLM([copy.deepcopy(ENGLISH)])
    lyrics = generate_lyrics(llm, template, inputs("en"), config)

    assert lyrics["language"] == "en" and lyrics["title"] == "Autumn Street"
    assert lyrics["sections"][0]["lines"][0] == "Walking down the autumn street"
    assert lyrics["sections"][1]["lines"] == lyrics["sections"][3]["lines"] == lyrics["sections"][5]["lines"]
    prompt = llm.calls[0]["user"]
    assert "【歌詞語言】English. Write natural, idiomatic, singable English lyrics" in prompt
    assert "1. 主歌：4 行，每行 5–8 個英文單字" in prompt
    assert "繁體中文" not in prompt


def test_too_long_english_line_is_reported_in_words(config, template):
    wrong = copy.deepcopy(ENGLISH)
    wrong["sections"][0]["lines"][0] = "Walking slowly down the long and winding golden autumn street tonight alone"
    llm = FakeLLM([wrong, copy.deepcopy(ENGLISH)])
    generate_lyrics(llm, template, inputs("en"), config)
    assert "有 12 個英文單字，應為 5–8 個英文單字" in llm.calls[1]["user"]


def test_japanese_lyrics_count_kana_and_kanji_and_keep_their_characters(config, template):
    llm = FakeLLM([copy.deepcopy(JAPANESE)])
    lyrics = generate_lyrics(llm, template, inputs("ja"), config)
    assert lyrics["language"] == "ja"
    # 日文漢字不可被當成簡體字轉換或擋下
    assert lyrics["sections"][0]["lines"] == JAPANESE["sections"][0]["lines"]
    assert L.count_chars("秋の街をゆっくり歩いてゆく") == 13
    assert "每行 10–16 個字（漢字或假名各算一個）" in llm.calls[0]["user"]
    assert "日本語" in llm.calls[0]["user"]


def test_simplified_check_only_applies_to_chinese_languages():
    spec = {"type": "verse", "lines": 1, "chars_min": 1, "chars_max": 30}
    cleaned, problems = check_section(["后视镜里的风"], spec, 1, 2, chinese=False)
    assert cleaned == ["后视镜里的风"] and problems == []                       # 非中文語言原樣保留
    cleaned, problems = check_section(["后视镜里的风"], spec, 1, 2, chinese=True)
    assert cleaned == ["後視鏡裡的風"] and problems == []                       # 中文語言自動轉成繁體


def test_cantonese_uses_traditional_chinese_rules(config, template):
    response = copy.deepcopy(GOOD_SECTIONS)
    response["sections"][0]["lines"][0] = "街灯一盏一盏亮起嚟"
    llm = FakeLLM([response])
    lyrics = generate_lyrics(llm, template, inputs("yue"), config)
    assert lyrics["language"] == "yue" and lyrics["sections"][0]["lines"][0] == "街燈一盞一盞亮起嚟"
    assert "粵語（廣東話）" in llm.calls[0]["user"] and "每行 7–11 字" in llm.calls[0]["user"]


def test_regenerate_section_follows_the_language_of_the_lyrics(config, template):
    lyrics = generate_lyrics(FakeLLM([copy.deepcopy(ENGLISH)]), template, inputs("en"), config)
    new = ["Carry me beyond the hills", "Where the evening slowly spills",
           "Every fear I leave behind", "Peace is all I hope to find"]
    llm = FakeLLM([{"lines": new}])
    updated = regenerate_section(llm, lyrics, 1, inputs("en"), config, template)
    assert updated["sections"][5]["lines"] == new
    assert "剛好 4 行，每行 4–7 個英文單字" in llm.calls[0]["user"] and "English" in llm.calls[0]["user"]


# ---- 風格描述 ----

def test_style_uses_the_chosen_language_and_voice_over_the_llm_suggestion():
    style = compose_style(STYLE_FEATURES, "ja", "female", False, "Japanese", "ethereal, airy")
    assert style.startswith("Japanese, ethereal, airy female vocal, nostalgic city pop, 108 BPM")
    assert "warm" not in style                                               # LLM 建議的聲線被使用者的選擇取代
    auto = compose_style(STYLE_FEATURES, "yue", "male", False, "Cantonese", "")
    assert auto.startswith("Cantonese, warm, breathy male vocal")           # 選「自動」時用 LLM 的建議
    assert compose_style(STYLE_FEATURES, "yue", "male", False).startswith("Cantonese")


@pytest.mark.parametrize("language, voice, expected", [
    ("en", "husky", "English, husky male vocal"),
    ("ja", "", "Japanese, warm, breathy male vocal"),
    ("yue", "sweet", "Cantonese, sweet male vocal"),
    ("zh-TW", "deep, rich", "Mandarin, deep, rich male vocal"),
])
def test_song_stage_sends_language_and_voice_to_yue2(config, project, server, language, voice, expected):   # noqa: F811
    project.data["inputs"].update(language=language, vocal_type="male", voice_character=voice)
    lyrics = {"title": "t", "language": language, "sections": [{"type": "verse", "lines": ["la la la", "na na na"]}]}
    save_lyrics(project, lyrics)
    server.file_bytes = flac_bytes(95)
    llm = FakeLLM([dict(STYLE_FEATURES)])
    assert SongStage(lambda cfg: llm).run(StageContext(config, project), Reporter()).ok
    sent = next(iter(server.prompts.values()))
    assert sent["25"]["inputs"]["style"].startswith(expected)
    assert f"Language the song is sung in: {expected.split(',')[0]}" in llm.calls[0]["user"]
    prompt = json.loads(project.path("audio", "yue2_prompt.json").read_text(encoding="utf-8"))
    assert prompt["style"].startswith(expected)


def test_voice_change_only_invalidates_the_song_not_the_lyrics(config, project):   # noqa: F811
    from app.core.models import DONE, STALE
    from app.ui.pages.input_page import _STALE_AFTER
    assert "voice_character" in _STALE_AFTER["s2_lyrics"] and "language" in _STALE_AFTER["s1_env"]
    for stage_id in ("s2_lyrics", "s3_song"):
        project.set_status(stage_id, DONE)
    project.mark_stale_after("s2_lyrics")
    assert project.status("s2_lyrics") == DONE and project.status("s3_song") == STALE


# ---- 字幕對齊 ----

@pytest.mark.parametrize("language, whisper", [("en", "en"), ("ja", "ja"), ("yue", "yue"), ("zh-TW", "zh")])
def test_alignment_uses_the_whisper_code_of_the_lyrics_language(config, project, server, language, whisper):   # noqa: F811
    lyrics = {"title": "t", "language": language, "sections": [{"type": "verse", "lines": ["line one", "line two"]}]}
    save_lyrics(project, lyrics)
    project.path("audio", "song.flac").write_bytes(flac_bytes(30))
    server.file_queue = [vocal_flac(), flac_bytes(30)]
    aligner = FakeAligner([[raw_line(5.0, 8.0), raw_line(8.4, 11.0)]])
    assert SubtitleStage(aligner=aligner).run(StageContext(config, project), Reporter()).ok
    assert aligner.calls[0]["language"] == whisper


def test_manual_lyrics_carry_the_selected_language(config):
    project = ProjectStore(config.get("paths.projects_root")).create("英文歌", inputs("en", lyrics_source="manual",
                                                                    manual_lyrics="[Verse]\nHello autumn wind\n[Chorus]\nCarry me away"))
    assert LyricsStage(lambda cfg: None).run(StageContext(config, project), Reporter()).ok
    saved = json.loads(project.path("lyrics", "lyrics.json").read_text(encoding="utf-8"))
    assert saved["language"] == "en" and saved["sections"][1]["lines"] == ["Carry me away"]
