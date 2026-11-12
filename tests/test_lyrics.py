from __future__ import annotations

import pytest

from app.core import lyrics as L
from app.core.lyric_templates import LyricTemplateStore, describe, parse_template, total_lines
from app.core.models import UserFacingError
from tests.conftest import REPO

TAG_MAP = {"intro": "Verse", "verse": "Verse", "prechorus": "Verse", "chorus": "Chorus",
           "bridge": "Bridge", "outro": "Outro"}


def test_parse_accepts_chinese_and_english_markers_in_many_spellings():
    text = """
    [主歌]
    巷口的燈亮起
    【副歌 2】
    夏夜的風
    [Verse 1]
    second verse line
    ［Pre-Chorus］
    往上推
    (bridge)
    過渡的一句
    橋段：
    又一句
    尾奏
    """
    lyrics = L.parse_lyrics_text(text, "歌")
    assert [s["type"] for s in lyrics["sections"]] == [
        "verse", "chorus", "verse", "prechorus", "bridge", "bridge", "outro"]
    assert lyrics["sections"][0]["lines"] == ["巷口的燈亮起"]
    assert lyrics["sections"][-1]["lines"] == []          # 沒有歌詞的尾奏＝純音樂
    assert L.lyric_lines(lyrics) == ["巷口的燈亮起", "夏夜的風", "second verse line", "往上推", "過渡的一句", "又一句"]


def test_parse_points_at_the_offending_lines():
    text = "沒有標記的一行\n[主歌]\n第一句\n[Refrain]\n第二句"
    with pytest.raises(L.LyricsFormatError) as e:
        L.parse_lyrics_text(text)
    assert e.value.problems == ["第 1 行：「沒有標記的一行」前面沒有段落標記", "第 4 行：看不懂的段落標記「[Refrain]」"]


@pytest.mark.parametrize("text", ["", "   \n\n", "[主歌]\n[副歌]"])
def test_parse_rejects_text_without_any_lyrics(text):
    with pytest.raises(L.LyricsFormatError, match="沒有任何歌詞"):
        L.parse_lyrics_text(text)


def test_a_lyric_line_that_merely_contains_a_section_word_is_not_a_marker():
    lyrics = L.parse_lyrics_text("[主歌]\n副歌響起的時候\n主歌")
    # 「主歌」單獨一行是標記，「副歌響起的時候」是歌詞
    assert lyrics["sections"][0]["lines"] == ["副歌響起的時候"]
    assert len(lyrics["sections"]) == 2


def test_editor_text_round_trips():
    lyrics = {"title": "歌", "language": "zh-TW", "sections": [
        {"type": "intro", "lines": []}, {"type": "verse", "lines": ["一", "二"]},
        {"type": "prechorus", "lines": ["三"]}, {"type": "chorus", "lines": ["四"]}]}
    assert L.parse_lyrics_text(L.to_editor_text(lyrics), "歌") == lyrics


def test_yue2_format_maps_tags_and_drops_instrumental_sections():
    lyrics = {"sections": [
        {"type": "intro", "lines": []}, {"type": "verse", "lines": ["一", "二"]},
        {"type": "prechorus", "lines": ["三"]}, {"type": "chorus", "lines": ["四"]},
        {"type": "bridge", "lines": ["五"]}, {"type": "outro", "lines": []}]}
    assert L.to_yue2(lyrics, TAG_MAP) == "[Verse]\n一\n二\n\n[Verse]\n三\n\n[Chorus]\n四\n\n[Bridge]\n五"


def test_plain_text_is_one_sung_line_per_row():
    lyrics = {"sections": [{"type": "verse", "lines": ["一", "二"]}, {"type": "outro", "lines": []},
                           {"type": "chorus", "lines": ["三"]}]}
    assert L.to_plain_text(lyrics) == "一\n二\n三\n"


@pytest.mark.parametrize("line, expected", [
    ("夏夜的風，吹散了回憶。", "夏夜的風 吹散了回憶"),
    ("oh  baby（和聲）", "oh baby"),
    ("  「走吧」！  ", "走吧"),
    ("don't stop", "don't stop"),
])
def test_clean_line(line, expected):
    assert L.clean_line(line) == expected


@pytest.mark.parametrize("line, expected", [("夏夜的風", 4), ("我 love 你", 3), ("don't stop 123", 3), ("", 0)])
def test_count_chars(line, expected):
    assert L.count_chars(line) == expected


def test_estimate_seconds():
    lyrics = {"sections": [{"type": "intro", "lines": []}, {"type": "verse", "lines": ["一"] * 4},
                           {"type": "chorus", "lines": ["二"] * 4}]}
    # 8 行 × 4.5 秒 ＋（2 個交界 ＋ 1 個純音樂段）× 6 秒
    assert L.estimate_seconds(lyrics, 4.5, 6.0) == 54.0


# ---- 範本 ----

def test_builtin_templates_are_valid_and_resolve_repeats():
    store = LyricTemplateStore(REPO / "config" / "lyric_templates", REPO / "不存在")
    templates = {t["id"]: t for t in store.list()}
    assert set(templates) == {"verse_chorus_bridge", "verse_prechorus_chorus", "short_song"}
    main = templates["verse_chorus_bridge"]
    assert [s["type"] for s in main["sections"]] == ["verse", "chorus", "verse", "chorus", "bridge", "chorus"]
    assert main["sections"][3] == {**main["sections"][1], "repeat_of": 2}     # 重複段沿用來源的行數與字數
    assert total_lines(main) == 22
    assert describe(main) == "主歌–副歌–主歌–副歌–過渡–副歌 · 22 行"


@pytest.mark.parametrize("text, expected", [
    ("name = ", "不是合法的 TOML"),
    ('[[sections]]\ntype="verse"\nlines=4\nchars_min=5\nchars_max=9', "缺少 name"),
    ('name="x"', "至少要有一個"),
    ('name="x"\n[[sections]]\ntype="hook"\nlines=4', "第 1 段：type 必須是"),
    ('name="x"\n[[sections]]\ntype="verse"\nlines=4\nchars_min=9\nchars_max=5', "chars_min ≤ chars_max"),
    ('name="x"\n[[sections]]\ntype="verse"\nrepeat_of=1', "repeat_of 必須是前面某一段"),
    ('name="x"\n[[sections]]\ntype="intro"\nlines=0\n[[sections]]\ntype="chorus"\nrepeat_of=1', "沒有歌詞"),
    ('name="x"\n[[sections]]\ntype="intro"\nlines=0', "至少要有一個有歌詞的段落"),
    ('name="x"\n[[sections]]\ntype="hook"\n[[sections]]\ntype="verse"\nrepeat_of=1', "第 1 段：type"),
])
def test_template_errors_say_which_section(text, expected):
    with pytest.raises(UserFacingError, match=expected):
        parse_template(text)


def test_user_templates_can_be_added_override_builtin_and_be_deleted(tmp_path):
    store = LyricTemplateStore(REPO / "config" / "lyric_templates", tmp_path / "user")
    text = 'name = "我的短歌"\n[[sections]]\ntype = "chorus"\nlines = 2\nchars_min = 4\nchars_max = 8\n'
    store.save("mine", text)
    store.save("short_song", text)                         # 修改內建範本＝存成同 id 的使用者版本
    assert store.get("mine")["is_user"] and store.get("mine")["name"] == "我的短歌"
    assert store.get("short_song")["name"] == "我的短歌"
    assert len(store.list()) == 4

    with pytest.raises(UserFacingError, match="範本有誤"):
        store.save("mine", 'name = "壞掉"')
    assert store.get("mine")["name"] == "我的短歌"           # 驗證失敗不會覆蓋原檔

    store.delete("short_song")                             # 刪掉使用者版本後，內建版本回來
    assert store.get("short_song")["name"].startswith("短歌")
    store.delete("mine")
    with pytest.raises(UserFacingError, match="找不到"):
        store.get("mine")
    with pytest.raises(UserFacingError, match="內建範本不能刪除"):
        store.delete("verse_chorus_bridge")
    with pytest.raises(UserFacingError, match="代號"):
        store.save("a/b", text)
