from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from app.core import srt_builder as S
from app.core.models import DONE, STAGE_IDS, UserFacingError
from app.pipeline import runner as R
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s1_env import EnvStage
from app.pipeline.stages.s2_lyrics import LyricsStage
from app.pipeline.stages.s3_song import SongStage
from app.pipeline.stages.s4_subtitle import SubtitleStage
from app.pipeline.stages.s5_video import VideoStage
from app.pipeline.stages.s6_compose import (ComposeStage, ass_color, build_filter, escape_filter_path,
                                            find_font_dir, render_preview, subtitle_style)
from app.services import fonts
from app.pipeline.stages.s7_export import DELIVERABLES, LICENSE_NOTICE, ExportStage
from app.services import ffmpeg_service as F
from tests.helpers import GOOD_SECTIONS, STYLE_FEATURES, FakeLLM, flac_bytes
from tests.test_s3_song_and_runner import project, server, server_info  # noqa: F401  （fixture）
from tests.test_s4_subtitle import FakeAligner, raw_line, vocal_flac
from tests.test_s5_video import PROMPT, make_mp4

STYLE = {"font": "Noto Sans TC", "size_percent": 7.5, "margin_percent": 6.0, "color": "#FFFFFF",
         "outline_color": "#000000", "outline_percent": 0.5, "bold": True}


# ---- 字幕濾鏡 ----

@pytest.mark.parametrize("value, expected", [("#FFFFFF", "&H00FFFFFF"), ("#ff8000", "&H000080FF"), ("102030", "&H00302010")])
def test_ass_color_is_bgr(value, expected):
    assert ass_color(value) == expected


def test_ass_color_rejects_garbage():
    with pytest.raises(UserFacingError, match="#RRGGBB"):
        ass_color("white")


def test_filter_uses_relative_srt_and_escapes_windows_font_path():
    text = build_filter(STYLE, "Noto Sans TC", Path(r"C:\Users\王 小明\Fonts"), "lyrics/lyrics.srt")
    assert text.startswith("subtitles=lyrics/lyrics.srt:fontsdir='C\\:/Users/王 小明/Fonts':force_style='")
    assert "FontName=Noto Sans TC" in text and "Alignment=2" in text and "Bold=-1" in text
    assert "FontSize=21.6" in text and "MarginV=17" in text           # 7.5% 與 6% × 288
    assert "PrimaryColour=&H00FFFFFF" in text and "OutlineColour=&H00000000" in text
    assert "fontsdir" not in build_filter(STYLE, "X", None, "a.srt")
    assert escape_filter_path(Path(r"D:\MV專案\歌")) == "D\\:/MV專案/歌"


def test_filter_position_italic_shadow_and_box():
    plain = build_filter(STYLE, "X", None, "a.srt")                    # 舊專案沒有新欄位：置底、無陰影
    assert "BorderStyle=1" in plain and "Shadow=0.00" in plain and "Italic=0" in plain and "Alignment=2" in plain

    top = build_filter({**STYLE, "position": "top", "italic": True, "background": "shadow",
                        "back_color": "#102030", "back_opacity": 50}, "X", None, "a.srt")
    assert "Alignment=6" in top and "Italic=-1" in top and "MarginV=17" in top
    assert "BorderStyle=1" in top and "Shadow=1.44" in top and "BackColour=&H80302010" in top

    box = build_filter({**STYLE, "position": "middle", "background": "box", "back_opacity": 100}, "X", None, "a.srt")
    assert "Alignment=10" in box and "MarginV=0" in box
    assert "BorderStyle=4" in box and "BackColour=&H00000000" in box
    assert ass_color("#FFFFFF", 0) == "&HFFFFFFFF"


def test_installed_fonts_are_listed_and_checked():
    names = fonts.chinese_fonts()
    assert names and not any(n.startswith("@") for n in names)
    assert fonts.is_installed(names[0]) and not fonts.is_installed("沒安裝的字型") and not fonts.is_installed("")
    usable = fonts.usable_for_subtitles(names)
    assert usable and set(usable) <= set(names)
    assert fonts.usable_for_subtitles(["沒安裝的字型", usable[0]]) == [usable[0]]


def test_preview_draws_the_sample_line(config, project, tmp_path):   # noqa: F811
    style = subtitle_style(config, project)
    plain = Image.open(render_preview(config, project, {**style, "color": "#606060", "outline_percent": 0,
                                                        "outline_color": "#606060"}, "風", tmp_path / "a"))
    shown = Image.open(render_preview(config, project, {**style, "position": "top"}, "晚風吹過這裡",
                                      tmp_path / "b")).convert("RGB")
    assert plain.size == shown.size == (896, 512)
    bright = [(x, y) for y in range(0, 512, 2) for x in range(0, 896, 2) if min(shown.getpixel((x, y))) > 200]
    assert bright and max(y for _, y in bright) < 256                  # 沒有照片時用灰底；白字出現在上半部


def test_font_lookup_and_project_style_override(config, project, tmp_path):   # noqa: F811
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    (fonts / "NotoSansTC-VF.ttf").write_bytes(b"")
    assert find_font_dir(config, "Noto Sans TC", [fonts]) == fonts
    assert find_font_dir(config, "Microsoft JhengHei", [fonts]) is None
    assert find_font_dir(config, "沒聽過的字型", [fonts]) is None

    assert subtitle_style(config, project)["size_percent"] == config.get("subtitle.size_percent")
    project.data["params"]["subtitle_style"] = {"size_percent": 10.0, "color": "#FFFF00"}
    style = subtitle_style(config, project)
    assert (style["size_percent"], style["color"], style["font"]) == (10.0, "#FFFF00", "Noto Sans TC")


# ---- 階段 6、7 ----

SRT = "1\n00:00:00,200 --> 00:00:01,500\n巷口的燈亮起\n\n2\n00:00:01,800 --> 00:00:03,000\n晚風吹過這裡\n"


@pytest.fixture
def composed(project, tmp_path):   # noqa: F811
    """專案資料夾名稱含中文，路徑另有空白：燒字幕最容易出錯的情況。"""
    make_mp4(tmp_path / "raw.mp4", 75, size="320x180")
    silent = tmp_path / "有 空白" / "seg_001.mp4"
    silent.parent.mkdir()
    F.trim_segment(tmp_path / "raw.mp4", silent, 0, 75, 25)
    project.path("audio", "song.flac").write_bytes(flac_bytes(3.0, 48000))
    F.concat_with_audio([silent], project.path("audio", "song.flac"), project.path("output", "video_nosub.mp4"),
                        silent.parent)
    project.path("lyrics", "lyrics.srt").write_text(SRT, encoding="utf-8")
    project.path("lyrics", "lyrics.txt").write_text("巷口的燈亮起\n晚風吹過這裡\n", encoding="utf-8")
    return project


def frame_pixels(path, index: int, tmp_path) -> Image.Image:
    out = tmp_path / f"frame_{index}.png"
    F.extract_frame(path, out, index)
    return Image.open(out).convert("RGB")


def test_compose_burns_subtitles_at_the_right_time_and_place(config, composed, tmp_path):
    stage = ComposeStage()
    ctx = StageContext(config, composed)
    assert stage.check_inputs(ctx) == []
    result = stage.run(ctx, Reporter())
    assert result.ok

    final, source = composed.path("output", "final_mv.mp4"), composed.path("output", "video_nosub.mp4")
    assert F.probe_video(final)["frames"] == F.probe_video(source)["frames"] == 75
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name", "-of", "csv=p=0",
                              str(final)], capture_output=True, text=True).stdout.split()
    assert streams == ["h264", "aac"]
    assert not composed.path("output", "final_mv.partial.mp4").exists()

    def difference(index: int) -> tuple[int, int]:
        """（畫面下方 1/4、上方 3/4）與無字幕版相差明顯的像素數。"""
        a, b = frame_pixels(final, index, tmp_path), frame_pixels(source, index, tmp_path)
        counts = [0, 0]
        for y in range(a.height):
            for x in range(a.width):
                if sum(abs(p - q) for p, q in zip(a.getpixel((x, y)), b.getpixel((x, y)))) > 120:
                    counts[0 if y >= a.height * 0.75 else 1] += 1
        return counts[0], counts[1]

    during, between = difference(20), difference(41)          # 0.8 秒有字幕；1.64 秒在兩則之間
    assert during[0] > 150 and during[1] == 0                 # 字幕只出現在畫面下方
    assert between[0] < 20                                    # 兩則字幕之間沒有字
    assert composed.stage("s6_compose")["style"]["font_used"] in ("Noto Sans TC", "Microsoft JhengHei")


def test_compose_falls_back_or_fails_clearly_when_font_is_missing(config, composed):
    composed.data["params"]["subtitle_style"] = {"font": "沒安裝的字型"}
    result = ComposeStage().run(StageContext(config, composed), Reporter())
    assert result.ok and "找不到字幕字型「沒安裝的字型」，改用「Microsoft JhengHei」" in result.warnings[0]

    config.set("subtitle.fallback_font", "也沒有的字型")
    with pytest.raises(UserFacingError) as e:
        ComposeStage().run(StageContext(config, composed), Reporter())
    assert "備用的「也沒有的字型」也沒有安裝" in e.value.message and "微軟正黑體" in e.value.fix


def test_compose_and_export_need_their_inputs(config, project):   # noqa: F811
    ctx = StageContext(config, project)
    assert [p.message for p in ComposeStage().check_inputs(ctx)] == ["還沒有對嘴影片", "還沒有字幕檔"]
    assert "final_mv.mp4" in ExportStage().check_inputs(ctx)[0].message


def test_export_places_deliverables_and_writes_summary(config, composed):
    ctx = StageContext(config, composed)
    ComposeStage().run(ctx, Reporter())
    composed.set_status("s5_video", DONE, seconds=1407.5, warnings=["照片比較小"])
    logs: list[str] = []
    assert ExportStage().run(ctx, Reporter(on_log=logs.append)).ok

    for name, source in DELIVERABLES.items():
        target = composed.folder / name
        assert target.read_bytes() == composed.path(*source.split("/")).read_bytes(), name
    assert os.path.samefile(composed.folder / "final_mv.mp4", composed.path("output", "final_mv.mp4"))   # 硬連結，不多佔空間
    summary = json.loads(composed.path("logs", "summary.json").read_text(encoding="utf-8"))
    assert summary["video"] == {"width": 320, "height": 180, "fps": 25.0, "frames": 75, "seconds": 3.0}
    assert summary["license"] == LICENSE_NOTICE and LICENSE_NOTICE in logs
    video = next(s for s in summary["stages"] if s["id"] == "s5_video")
    assert video == {"id": "s5_video", "label": "對嘴影片", "status": "done", "seconds": 1407.5,
                     "warnings": ["照片比較小"]}

    # 重新匯出（例如改了字幕樣式）會換成新的檔案
    composed.path("output", "final_mv.mp4").write_bytes(composed.path("output", "video_nosub.mp4").read_bytes())
    ExportStage().run(ctx, Reporter())
    assert (composed.folder / "final_mv.mp4").read_bytes() == composed.path("output", "video_nosub.mp4").read_bytes()


# ---- 七個階段的完整流程 ----

ALIGNED = [raw_line(0.5 + i * 1.2, 1.5 + i * 1.2) for i in range(22)]        # 22 行歌詞，落在 30 秒內


def all_stages(llm, aligner) -> dict:
    stages = [EnvStage(), LyricsStage(lambda cfg: llm), SongStage(lambda cfg: llm), SubtitleStage(aligner=aligner),
              VideoStage(llm_factory=lambda cfg: llm), ComposeStage(), ExportStage()]
    return {s.id: s for s in stages}


@pytest.fixture
def full_project(config, project, tmp_path):   # noqa: F811
    config.set("engines.infinitetalk.max_segment_seconds", 20)
    # 測試用的歌只有 30 秒（影片才不會太長）；放寬「比估計短很多就重試」的門檻
    config.set("song.min_duration_ratio", 0.1)
    Image.new("RGB", (1600, 900), "gray").save(project.path("input", "photo_original.png"))
    project.data["inputs"].update(photo="input/photo_original.png", duration_mode="auto")
    return project


def queue_media(server_obj, tmp_path) -> None:
    """依流程順序排好模擬 ComfyUI 的輸出：歌曲、人聲、伴奏、兩段影片。"""
    server_obj.file_queue = [flac_bytes(30, 48000), vocal_flac(30.0), flac_bytes(30),
                             make_mp4(tmp_path / "c1.mp4", 501, size="320x180"),
                             make_mp4(tmp_path / "c2.mp4", 501, size="320x180")]


def test_full_auto_run_from_four_inputs_to_final_mv(config, full_project, server, tmp_path):   # noqa: F811
    """驗收：給定四項輸入，全自動模式不需任何人工操作就產出 final_mv.mp4。"""
    queue_media(server, tmp_path)
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS), dict(STYLE_FEATURES), dict(PROMPT)])
    seen: list[str] = []

    outcome = R.PipelineRunner(StageContext(config, full_project), all_stages(llm, FakeAligner([copy.deepcopy(ALIGNED)]))
                               ).run(Reporter(), lambda stage_id, status: seen.append(f"{stage_id}:{status}"))

    assert outcome.kind == R.FINISHED, (outcome.stage_id, outcome.message)
    assert [full_project.status(s) for s in STAGE_IDS] == [DONE] * 7
    assert seen == [f"{s}:{status}" for s in STAGE_IDS for status in ("running", "done")]

    final = full_project.folder / "final_mv.mp4"
    info = F.probe_video(final)
    assert info["frames"] == 750                                              # 30 秒 × 25 fps
    assert abs(F.duration_of(final) - 30.0) <= 1 / 25 + 0.03                  # 與歌曲等長
    for name in DELIVERABLES:
        assert (full_project.folder / name).exists(), name
    # 字幕文字與輸入歌詞完全一致
    lyrics_lines = (full_project.folder / "lyrics.txt").read_text(encoding="utf-8").splitlines()
    cues = S.parse_srt((full_project.folder / "lyrics.srt").read_text(encoding="utf-8"))
    assert [text for _, _, text in cues] == lyrics_lines and len(lyrics_lines) == 22
    # 影片階段切成兩段（單段上限 20 秒），第二段接續第一段
    plan = json.loads(full_project.path("analysis", "segment_plan.json").read_text(encoding="utf-8"))
    assert [s["init"] for s in plan["segments"]] == ["photo", "prev"] and plan["total_frames"] == 750
    assert llm.unloaded == 3 and server.free_calls == 3                       # 每個用到模型的階段都有釋放
    assert len(server.prompts) == 4                                           # 歌曲、人聲分離、影片 ×2


def test_step_mode_pauses_three_times_then_finishes(config, full_project, server, tmp_path):   # noqa: F811
    queue_media(server, tmp_path)
    full_project.data["mode"] = "step"
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS), dict(STYLE_FEATURES), dict(PROMPT)])
    runner = R.PipelineRunner(StageContext(config, full_project), all_stages(llm, FakeAligner([copy.deepcopy(ALIGNED)])))
    outcomes = [runner.run(Reporter()) for _ in range(4)]
    assert [(o.kind, o.stage_id) for o in outcomes] == [
        (R.PAUSED, "s2_lyrics"), (R.PAUSED, "s3_song"), (R.PAUSED, "s4_subtitle"), (R.FINISHED, "")]


def test_changing_subtitle_style_reruns_only_compose_and_export(config, full_project, server, tmp_path):   # noqa: F811
    queue_media(server, tmp_path)
    llm = FakeLLM([copy.deepcopy(GOOD_SECTIONS), dict(STYLE_FEATURES), dict(PROMPT)])
    stages = all_stages(llm, FakeAligner([copy.deepcopy(ALIGNED)]))
    R.PipelineRunner(StageContext(config, full_project), stages).run(Reporter())
    video = full_project.path("output", "video_nosub.mp4")
    stamp, before = video.stat().st_mtime_ns, (full_project.folder / "final_mv.mp4").read_bytes()

    full_project.data["params"]["subtitle_style"] = {"size_percent": 12.0, "color": "#FFFF00"}
    assert full_project.mark_stale_after("s5_video") == ["s6_compose", "s7_export"]
    seen: list[str] = []
    outcome = R.PipelineRunner(StageContext(config, full_project), stages).run(
        Reporter(), lambda stage_id, status: seen.append(stage_id) if status == "running" else None)

    assert outcome.kind == R.FINISHED and seen == ["s1_env", "s6_compose", "s7_export"]
    assert video.stat().st_mtime_ns == stamp and len(server.prompts) == 4      # 影片沒有重新生成
    assert (full_project.folder / "final_mv.mp4").read_bytes() != before
    assert full_project.stage("s6_compose")["style"]["size_percent"] == 12.0
