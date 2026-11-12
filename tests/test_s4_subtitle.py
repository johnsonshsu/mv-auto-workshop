from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
import soundfile

from app.core import srt_builder as S
from app.core.models import DONE, UserFacingError
from app.core.sections import build_sections, detect_silences
from app.pipeline import runner as R
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s1_env import EnvStage
from app.pipeline.stages.s2_lyrics import LyricsStage, save_lyrics
from app.pipeline.stages.s3_song import SongStage, audio_seconds
from app.pipeline.stages.s4_subtitle import SubtitleStage, pick_separation
from tests.conftest import REPO
from tests.helpers import GOOD_SECTIONS, STYLE_FEATURES, FakeLLM, flac_bytes
from tests.test_env import full_object_info
from tests.test_s3_song_and_runner import project, server, server_info  # noqa: F401  （fixture）

TEXTS = ["巷口的燈亮起", "晚風吹過這裡", "心跳慢慢靠近", "夏夜的風吹散回憶"]
LYRICS = {"title": "歌", "language": "zh-TW", "sections": [
    {"type": "intro", "lines": []},
    {"type": "verse", "lines": TEXTS[:2]},
    {"type": "prechorus", "lines": TEXTS[2:3]},
    {"type": "chorus", "lines": TEXTS[3:]}]}


def raw_line(start, end, probability=0.9):
    if start is None:
        return {"start": None, "end": None, "words": []}
    return {"start": start, "end": end, "words": [{"word": "x", "start": start, "end": end,
                                                    "probability": probability}]}


GOOD_RAW = [raw_line(5.0, 8.0), raw_line(8.4, 11.0), raw_line(14.0, 16.5), raw_line(18.0, 22.0)]


# ---- 對齊結果的判定與內插 ----

def test_assess_keeps_plausible_lines():
    lines = S.assess(GOOD_RAW, TEXTS, 30.0)
    assert [(line.id, line.text, line.start, line.end, line.method) for line in lines] == [
        (1, TEXTS[0], 5.0, 8.0, "aligned"), (2, TEXTS[1], 8.4, 11.0, "aligned"),
        (3, TEXTS[2], 14.0, 16.5, "aligned"), (4, TEXTS[3], 18.0, 22.0, "aligned")]
    assert lines[0].confidence == 0.9


@pytest.mark.parametrize("bad, why", [
    (raw_line(None, None), "沒有時間"),
    (raw_line(8.4, 11.0, probability=0.05), "信心值太低"),
    (raw_line(8.4, 8.45), "短到不可能唱完六個字"),
    (raw_line(6.0, 9.0), "時間倒退到上一行裡面"),
    (raw_line(40.0, 44.0), "超過歌曲長度"),
    ({"start": 8.4, "end": 11.0, "words": []}, "沒有任何字的信心值"),
])
def test_assess_rejects_implausible_lines(bad, why):
    raw = [GOOD_RAW[0], bad, GOOD_RAW[2], GOOD_RAW[3]]
    lines = S.assess(raw, TEXTS, 30.0)
    assert lines[1] is None, why
    assert all(lines[i] is not None for i in (0, 2, 3))


def test_assess_trims_words_the_aligner_threw_across_an_instrumental_break():
    """實機遇到的情況：第一個字被放在間奏前，整行被拉長 30 秒。"""
    def word(text, start, end, p):
        return {"word": text, "start": start, "end": end, "probability": p}

    stretched_start = {"start": 104.66, "end": 134.82, "words": [
        word("街", 104.66, 106.24, 0.30), word("燈", 132.70, 133.08, 0.07), word("拉", 133.08, 133.32, 0.38),
        word("長", 133.32, 133.62, 0.56), word("了", 133.62, 133.98, 0.99), word("影", 133.98, 134.26, 0.85),
        word("子", 134.26, 134.82, 1.00)]}
    stretched_end = {"start": 50.28, "end": 58.73, "words": [
        word("我", 50.28, 51.74, 0.10), word("輕", 51.78, 52.12, 0.92), word("輕", 52.12, 52.36, 0.99),
        word("說", 52.36, 52.72, 0.98), word("聲", 52.72, 53.46, 0.96), word("再見", 57.59, 58.73, 0.00)]}
    first, second = S.assess([stretched_end, stretched_start], ["我輕輕說聲再見", "街燈拉長了影子"], 150.0)
    assert (first.start, first.end) == (50.28, 53.46)
    assert (second.start, second.end) == (132.70, 134.82)
    assert second.confidence == pytest.approx((0.07 + 0.38 + 0.56 + 0.99 + 0.85 + 1.00) / 6, abs=0.001)
    assert S.trim_outlier_words([], 2.0) == []


def test_assess_handles_fewer_results_than_lyrics():
    lines = S.assess(GOOD_RAW[:2], TEXTS, 30.0)
    assert [line is None for line in lines] == [False, False, True, True]


def test_interpolate_splits_the_gap_by_character_count():
    texts = ["一二", "一二三四", "一二三四五六", "一二"]
    assessed = S.assess([raw_line(2.0, 4.0), raw_line(None, None), raw_line(None, None), raw_line(14.0, 16.0)],
                        texts, 20.0)
    lines = S.interpolate(assessed, texts, 20.0, gap=0.05)
    # 可用區間是 4.05–13.95（共 9.9 秒），兩行字數 4：6
    assert [line.method for line in lines] == ["aligned", "interpolated", "interpolated", "aligned"]
    assert lines[1].start == pytest.approx(4.05) and lines[1].end == pytest.approx(4.05 + 3.96 - 0.05)
    assert lines[2].start == pytest.approx(8.01) and lines[2].end == pytest.approx(13.9)
    assert lines[1].confidence == 0.0 and lines[1].text == "一二三四"
    assert all(a.end <= b.start for a, b in zip(lines, lines[1:]))


def test_interpolated_line_before_a_long_break_is_not_stretched_across_it():
    """實機遇到的情況：副歌最後一句沒對上，後面接 20 秒間奏，字幕不該顯示 20 秒。"""
    texts = ["往事隨風走", "自由在等候", "把回憶鎖進風箱"]
    assessed = S.assess([raw_line(81.94, 84.38), raw_line(None, None), raw_line(106.28, 112.02)], texts, 150.0)
    lines = S.interpolate(assessed, texts, 150.0)
    middle = lines[1]
    assert middle.method == "interpolated" and middle.start == pytest.approx(84.43)
    assert 1.5 < middle.end - middle.start < 4.5                    # 依其他行的唱速估計，約 5 個字的長度
    assert middle.end < 90


def test_unaligned_opening_lines_sit_right_before_the_first_aligned_line_not_at_zero():
    """實機遇到的情況：前四行沒對上，被排在 0–6 秒的前奏裡。"""
    texts = ["踩緊枯葉聽聲脆響", "冷風輕輕拂過臉頰", "隨風飄蕩吧", "放低所有牽掛"]
    assessed = S.assess([raw_line(None, None), raw_line(None, None), raw_line(39.88, 42.78), raw_line(42.9, 45.5)],
                        texts, 60.0)
    lines = S.interpolate(assessed, texts, 60.0)
    assert lines[0].start > 20 and lines[1].end <= 39.88            # 緊接在第一句對上的歌詞之前
    assert lines[0].end <= lines[1].start


def test_interpolate_at_the_start_and_end_of_the_song():
    texts = ["甲乙", "丙丁", "戊己"]
    lines = S.interpolate(S.assess([raw_line(None, None), raw_line(10.0, 12.0), raw_line(None, None)], texts, 30.0),
                          texts, 30.0)
    # 開頭沒對上的行靠著後面那一行（前面多半是前奏）；結尾沒對上的行緊接在前一行之後
    assert 5.0 < lines[0].start < lines[0].end <= 10.0
    assert lines[2].start == pytest.approx(12.05) and lines[2].end < 16.0


def test_interpolate_when_nothing_aligned_spreads_over_whole_song():
    texts = ["甲乙", "丙丁"]
    lines = S.interpolate([None, None], texts, 20.0)
    assert [(line.start, line.method) for line in lines] == [(0.0, "interpolated"), (10.0, "interpolated")]


# ---- SRT 規則 ----

def test_cues_extend_the_tail_but_never_overlap_the_next_line():
    lines = S.assess(GOOD_RAW, TEXTS, 30.0)
    cues = S.build_cues(lines, tail=0.3, min_duration=0.8, gap=0.05, duration=30.0)
    assert cues[0] == (5.0, 8.3, TEXTS[0])                    # 8.0 + 0.3，離下一行 8.4 還有空間
    assert cues[1] == (8.4, 11.3, TEXTS[1])
    assert cues[3] == (18.0, 22.3, TEXTS[3])

    tight = S.assess([raw_line(5.0, 8.0), raw_line(8.1, 11.0)], TEXTS[:2], 30.0)
    assert S.build_cues(tight, 0.3, 0.8, 0.05, 30.0)[0][1] == 8.05      # 只能延長到下一行前 0.05 秒


def test_cues_show_for_at_least_the_minimum_duration_when_there_is_room():
    lines = [S.Line(1, "啊", 5.0, 5.2, 0.9, "aligned"), S.Line(2, "喔", 9.0, 9.1, 0.9, "aligned"),
             S.Line(3, "欸", 9.3, 9.4, 0.9, "aligned")]
    cues = S.build_cues(lines, 0.3, 0.8, 0.05, 20.0)
    assert cues[0] == (5.0, 5.8, "啊")                         # 補到 0.8 秒
    assert cues[1] == (9.0, 9.25, "喔")                        # 下一行太近，只能到它之前
    assert cues[2] == (9.3, 10.1, "欸")
    assert all(start < end for start, end, _ in cues)
    assert all(a[1] <= b[0] for a, b in zip(cues, cues[1:]))


def test_last_cue_does_not_run_past_the_end_of_the_song():
    cues = S.build_cues([S.Line(1, "尾", 29.5, 29.9, 0.9, "aligned")], 0.3, 0.8, 0.05, duration=30.0)
    assert cues == [(29.5, 30.0, "尾")]


def test_srt_format_and_round_trip():
    cues = [(5.0, 8.3, "巷口的燈亮起"), (3661.5, 3663.007, "第二行")]
    text = S.to_srt(cues)
    assert text == ("1\n00:00:05,000 --> 00:00:08,300\n巷口的燈亮起\n\n"
                    "2\n01:01:01,500 --> 01:01:03,007\n第二行\n")
    assert S.parse_srt(text) == cues
    assert S.format_time(59.9996) == "00:01:00,000"


# ---- 靜音與段落 ----

def test_detect_silences_finds_only_long_enough_gaps():
    sr = 8000
    tone = lambda seconds: 0.3 * np.sin(2 * np.pi * 220 * np.arange(int(sr * seconds)) / sr)   # noqa: E731
    quiet = lambda seconds: np.zeros(int(sr * seconds))                                       # noqa: E731
    samples = np.concatenate([quiet(2.0), tone(3.0), quiet(0.2), tone(1.0), quiet(1.5), tone(2.0)])
    silences = detect_silences(samples, sr, threshold_db=-40.0, min_seconds=0.4)
    assert [(round(s["start"], 1), round(s["end"], 1)) for s in silences] == [(0.0, 2.0), (6.2, 7.7)]
    stereo = np.stack([samples, samples], axis=1)
    assert len(detect_silences(stereo, sr)) == 2
    assert detect_silences(np.zeros(0), sr) == []


def test_sections_cover_the_song_and_mark_instrumental_parts():
    lines = S.assess(GOOD_RAW, TEXTS, 30.0)
    data = build_sections(LYRICS, lines, 30.0, [{"start": 0.0, "end": 4.9}])
    summary = [(s["type"], s["start"], s["end"], s["instrumental"], s["line_ids"]) for s in data["sections"]]
    assert summary == [
        ("intro", 0.0, 5.0, True, []),                         # 純音樂：到第一句開始
        ("verse", 5.0, 12.5, False, [1, 2]),                   # 與導歌的邊界取 11.0 與 14.0 的中點
        ("prechorus", 12.5, 17.25, False, [3]),
        ("chorus", 17.25, 30.0, False, [4]),                   # 最後一段到歌曲結束
    ]
    assert data["duration"] == 30.0 and data["silences"] == [{"start": 0.0, "end": 4.9}]
    assert all(a["end"] == b["start"] for a, b in zip(data["sections"], data["sections"][1:]))


def test_sections_without_intro_start_at_zero():
    lyrics = {"sections": LYRICS["sections"][1:]}
    data = build_sections(lyrics, S.assess(GOOD_RAW, TEXTS, 30.0), 30.0, [])
    assert data["sections"][0]["start"] == 0.0 and data["sections"][-1]["end"] == 30.0


# ---- 人聲分離方式的選擇 ----

def separation_requires() -> dict:
    return json.loads((REPO / "workflows" / "vocal_separation.requires.json").read_text(encoding="utf-8"))


def test_pick_separation_prefers_melband_and_adapts_to_the_installed_model_file():
    info = full_object_info()
    assert pick_separation(separation_requires(), info) == (
        "vocal_separation_melband", {"model_name": "MelBandRoformer_fp16.safetensors"})
    info["MelBandRoFormerModelLoader"]["input"]["required"]["model_name"][0][:] = ["MelBandRoformer_fp32.safetensors"]
    assert pick_separation(separation_requires(), info)[1] == {"model_name": "MelBandRoformer_fp32.safetensors"}


def test_pick_separation_falls_back_to_demucs_then_fails_clearly():
    info = full_object_info()
    info["MelBandRoFormerModelLoader"]["input"]["required"]["model_name"][0].clear()
    assert pick_separation(separation_requires(), info) == ("vocal_separation_demucs", {})
    del info["AudioSeparation"]
    with pytest.raises(UserFacingError, match="沒有可用的人聲分離"):
        pick_separation(separation_requires(), info)


# ---- 階段 4 ----

def vocal_flac(duration: float = 30.0, sr: int = 8000) -> bytes:
    """有聲音的區段對應 GOOD_RAW 的四行，其餘靜音。"""
    samples = np.zeros(int(duration * sr), dtype="float32")
    for item in GOOD_RAW:
        a, b = int(item["start"] * sr), int(item["end"] * sr)
        samples[a:b] = 0.3 * np.sin(2 * np.pi * 220 * np.arange(b - a) / sr)
    buffer = io.BytesIO()
    soundfile.write(buffer, np.stack([samples, samples], axis=1), sr, format="FLAC")
    return buffer.getvalue()


class FakeAligner:
    """依序回傳預先排好的對齊結果，並記錄被呼叫時的狀況。"""

    def __init__(self, results: list, events: list | None = None):
        self.results = list(results)
        self.calls: list[dict] = []
        self.events = events if events is not None else []

    def __call__(self, config, vocals, lyrics_txt, output, language, reporter, cancel):
        self.events.append("align")
        self.calls.append({"vocals": vocals, "language": language,
                           "lyrics": lyrics_txt.read_text(encoding="utf-8"), "vocals_exists": vocals.exists()})
        reporter.progress(0.5, "對齊中")
        return self.results.pop(0)


def prepare_song(project, server_obj, seconds=30.0):
    save_lyrics(project, LYRICS)
    project.path("audio", "song.flac").write_bytes(flac_bytes(seconds))
    for stage_id in ("s2_lyrics", "s3_song"):
        project.set_status(stage_id, DONE)


def test_subtitle_stage_produces_srt_sections_and_report(config, project, server):   # noqa: F811
    prepare_song(project, server)
    server.file_queue = [vocal_flac(), flac_bytes(30)]
    aligner = FakeAligner([copy.deepcopy(GOOD_RAW)])
    stage = SubtitleStage(aligner=aligner)
    ctx = StageContext(config, project)
    assert stage.check_inputs(ctx) == []

    result = stage.run(ctx, Reporter())

    assert result.ok and result.warnings == [] and result.data == {"unaligned_ratio": 0.0, "unaligned_count": 0}
    for output in stage.outputs(ctx):
        assert output.exists(), output
    # 送出的是 MelBandRoFormer 工作流，音訊用上傳後的名稱，模型用本機有的檔名
    sent = next(iter(server.prompts.values()))
    assert sent["1"]["inputs"]["audio"] == "mv_workshop/uploaded.bin"
    assert sent["2"]["inputs"]["model_name"] == "MelBandRoformer_fp16.safetensors"
    assert sent["5"]["inputs"]["filename_prefix"] == "mv_workshop/歌_vocals"
    assert len(server.uploads) == 1 and server.free_calls == 1
    # 人聲轉成 WAV，下載的暫存檔已清掉
    assert soundfile.info(str(project.path("audio", "vocals.wav"))).format == "WAV"
    assert sorted(p.name for p in project.path("audio").iterdir()) == ["instrumental.wav", "song.flac", "vocals.wav"]
    assert aligner.calls[0]["language"] == "zh" and aligner.calls[0]["vocals_exists"]

    # 驗收：字幕文字與輸入歌詞完全一致，一行歌詞一則字幕
    cues = S.parse_srt(project.path("lyrics", "lyrics.srt").read_text(encoding="utf-8"))
    assert [text for _, _, text in cues] == TEXTS
    assert aligner.calls[0]["lyrics"].splitlines() == TEXTS
    assert [start for start, _, _ in cues] == [5.0, 8.4, 14.0, 18.0]
    assert all(a[1] <= b[0] for a, b in zip(cues, cues[1:]))

    sections = json.loads(project.path("analysis", "sections.json").read_text(encoding="utf-8"))
    assert [s["type"] for s in sections["sections"]] == ["intro", "verse", "prechorus", "chorus"]
    assert sections["sections"][0]["instrumental"] is True and sections["duration"] == 30.0
    gaps = [(round(s["start"], 1), round(s["end"], 1)) for s in sections["silences"]]
    assert gaps == [(0.0, 5.0), (8.0, 8.4), (11.0, 14.0), (16.5, 18.0), (22.0, 30.0)]

    report = json.loads(project.path("analysis", "alignment_report.json").read_text(encoding="utf-8"))
    assert report["line_count"] == 4 and report["unaligned_count"] == 0
    assert report["lines"][0] == {"id": 1, "text": TEXTS[0], "start": 5.0, "end": 8.0,
                                  "confidence": 0.9, "method": "aligned"}


def test_unaligned_and_low_confidence_lines_are_interpolated_and_reported(config, project, server):   # noqa: F811
    prepare_song(project, server)
    server.file_queue = [vocal_flac(), flac_bytes(30)]
    raw = [GOOD_RAW[0], raw_line(None, None), raw_line(14.0, 16.5, probability=0.3), GOOD_RAW[3]]
    result = SubtitleStage(aligner=FakeAligner([raw])).run(StageContext(config, project), Reporter())

    assert result.ok and result.data["unaligned_ratio"] == 0.25
    assert "有 1 行歌詞對不上人聲" in result.warnings[0] and "第 2 行" in result.warnings[0]
    assert "有 1 行的對齊信心偏低" in result.warnings[1]
    report = json.loads(project.path("analysis", "alignment_report.json").read_text(encoding="utf-8"))
    assert [line["method"] for line in report["lines"]] == ["aligned", "interpolated", "aligned", "aligned"]
    cues = S.parse_srt(project.path("lyrics", "lyrics.srt").read_text(encoding="utf-8"))
    assert [text for _, _, text in cues] == TEXTS                              # 內插的行文字仍來自歌詞
    assert 8.0 < cues[1][0] < cues[1][1] <= 14.0


def test_resume_reuses_separated_vocals_and_only_realigns(config, project, server):   # noqa: F811
    prepare_song(project, server)
    server.file_queue = [vocal_flac(), flac_bytes(30)]
    stage = SubtitleStage(aligner=FakeAligner([copy.deepcopy(GOOD_RAW), copy.deepcopy(GOOD_RAW)]))
    ctx = StageContext(config, project)
    stage.run(ctx, Reporter())
    assert len(server.prompts) == 1

    logs: list[str] = []
    assert stage.resume(ctx, Reporter(on_log=logs.append)).ok
    assert len(server.prompts) == 1 and any("直接從對齊繼續" in line for line in logs)


def test_stage_needs_song_and_lyrics(config, project):   # noqa: F811
    messages = [p.message for p in SubtitleStage().check_inputs(StageContext(config, project))]
    assert messages == ["還沒有歌曲", "還沒有歌詞"]


# ---- 歌詞沒唱完時換 seed 重試（階段 3 ↔ 4）----

BAD_RAW = [GOOD_RAW[0], raw_line(None, None), raw_line(None, None), GOOD_RAW[3]]          # 一半對不上
WORSE_RAW = [GOOD_RAW[0], raw_line(None, None), raw_line(None, None), raw_line(None, None)]


def full_runner(config, project_obj, llm, aligner):
    stages = [EnvStage(), LyricsStage(lambda cfg: llm), SongStage(lambda cfg: llm), SubtitleStage(aligner=aligner)]
    return R.PipelineRunner(StageContext(config, project_obj), {s.id: s for s in stages})


def start_from_lyrics(project_obj):
    save_lyrics(project_obj, LYRICS)
    project_obj.set_status("s2_lyrics", DONE)
    project_obj.data["inputs"]["duration_mode"] = "auto"


def test_runner_regenerates_the_song_when_too_many_lines_fail_to_align(config, project, server):   # noqa: F811
    """M3 整合：歌詞沒唱完 → 換 seed 重新生成 → 得到與歌曲同步的 SRT。"""
    start_from_lyrics(project)
    server.file_queue = [flac_bytes(28), vocal_flac(), flac_bytes(28),          # 第 1 版：歌、人聲、伴奏
                         flac_bytes(30), vocal_flac(), flac_bytes(30)]          # 第 2 版
    aligner = FakeAligner([copy.deepcopy(BAD_RAW), copy.deepcopy(GOOD_RAW)])
    logs: list[str] = []

    outcome = full_runner(config, project, FakeLLM([dict(STYLE_FEATURES)] * 2), aligner).run(
        Reporter(on_log=logs.append))

    assert outcome.kind == R.STOPPED and outcome.stage_id == "s5_video"
    assert project.status("s3_song") == DONE and project.status("s4_subtitle") == DONE
    assert len(aligner.calls) == 2 and len(server.prompts) == 4                 # 歌曲 ×2、人聲分離 ×2
    assert any("有 50% 的歌詞對不上人聲" in line and "第 2／3 版" in line for line in logs)
    assert audio_seconds(project.path("audio", "song.flac")) == pytest.approx(30, abs=0.01)
    report = json.loads(project.path("analysis", "alignment_report.json").read_text(encoding="utf-8"))
    assert report["unaligned_count"] == 0
    assert not project.path("audio", "takes").exists() and project.data["song_takes"] == []
    assert not any("歌詞對不上" in w for w in outcome.warnings)


def test_runner_keeps_the_best_take_when_every_take_is_incomplete(config, project, server):   # noqa: F811
    start_from_lyrics(project)
    server.file_queue = [flac_bytes(27), vocal_flac(), flac_bytes(27),
                         flac_bytes(28), vocal_flac(), flac_bytes(28),
                         flac_bytes(29), vocal_flac(), flac_bytes(29)]
    aligner = FakeAligner([copy.deepcopy(WORSE_RAW), copy.deepcopy(BAD_RAW), copy.deepcopy(WORSE_RAW)])

    outcome = full_runner(config, project, FakeLLM([dict(STYLE_FEATURES)] * 3), aligner).run(Reporter())

    assert outcome.kind == R.STOPPED and len(aligner.calls) == 3                # 上限 3 版
    assert "重新生成 3 版後，最好的一版仍有 50% 的歌詞對不上人聲" in outcome.warnings[-1]
    # 還原的是第 2 版（歌曲 28 秒、未對齊 2 行），歌曲與字幕報告是同一版
    assert audio_seconds(project.path("audio", "song.flac")) == pytest.approx(28, abs=0.01)
    report = json.loads(project.path("analysis", "alignment_report.json").read_text(encoding="utf-8"))
    assert report["unaligned_count"] == 2
    prompt = json.loads(project.path("audio", "yue2_prompt.json").read_text(encoding="utf-8"))
    assert prompt["seconds"] == 28.0
    assert not project.path("audio", "takes").exists()


def test_step_mode_does_not_auto_retry_but_tells_the_user(config, project, server):   # noqa: F811
    start_from_lyrics(project)
    project.data["mode"] = "step"
    project.path("audio", "song.flac").write_bytes(flac_bytes(30))
    project.set_status("s3_song", DONE)
    server.file_queue = [vocal_flac(), flac_bytes(30)]
    aligner = FakeAligner([copy.deepcopy(BAD_RAW)])

    outcome = full_runner(config, project, FakeLLM([]), aligner).run(Reporter())

    assert (outcome.kind, outcome.stage_id) == (R.PAUSED, "s4_subtitle")
    assert len(aligner.calls) == 1
    assert any("重新生成歌曲" in w for w in outcome.warnings)


def test_good_first_take_needs_no_retry_and_leaves_no_take_folder(config, project, server):   # noqa: F811
    start_from_lyrics(project)
    server.file_queue = [flac_bytes(30), vocal_flac(), flac_bytes(30)]
    aligner = FakeAligner([copy.deepcopy(GOOD_RAW)])
    outcome = full_runner(config, project, FakeLLM([dict(STYLE_FEATURES)]), aligner).run(Reporter())
    assert outcome.kind == R.STOPPED and len(aligner.calls) == 1
    assert not project.path("audio", "takes").exists()
    _ = GOOD_SECTIONS
