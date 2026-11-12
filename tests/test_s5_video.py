from __future__ import annotations

import json
import random
import subprocess

import numpy as np
import pytest
import soundfile
from PIL import Image

from app.core import segmenter as G
from app.core.models import DONE, UserFacingError
from app.pipeline import runner as R
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s1_env import EnvStage
from app.pipeline.stages.s5_video import VideoStage, quietest_finder
from app.services import ffmpeg_service as F
from app.services.llm_client import LLMError
from app.services.photo import prepare_photo
from app.services.engines import SegmentRequest, make_engine
from tests.test_env import full_object_info
from tests.helpers import FakeLLM, flac_bytes
from tests.test_s3_song_and_runner import project, server, server_info  # noqa: F401  （fixture）

PROMPT = {"positive": "A woman with long dark hair wearing a white shirt sings into a studio microphone, "
                      "warm stage lighting, soft bokeh background, natural expression, gentle sway, static camera."}


# ---- 影格計算 ----

def test_segment_frames_always_sum_to_the_song_length():
    """核心性質：不論怎麼切，各段影格總和恆等於 round(歌曲長度 × fps)，誤差不累積。"""
    rng = random.Random(1234)
    for _ in range(300):
        duration = rng.uniform(5, 400)
        fps = rng.choice([24, 25, 30, 23.976])
        cuts = sorted({0.0, duration, *(round(rng.uniform(0, duration), 3) for _ in range(rng.randint(0, 40)))})
        segments = G.build_plan(cuts, set(), duration, fps, 4, 1, 4, cuts)
        assert sum(s.frames for s in segments) == round(duration * fps)
        assert segments[0].f_start == 0 and segments[-1].f_end == round(duration * fps)
        assert all(a.f_end == b.f_start for a, b in zip(segments, segments[1:]))
        assert all(s.frames > 0 and s.gen_frames >= s.frames + s.drop_head for s in segments)
        assert all((s.gen_frames - 1) % 4 == 0 for s in segments)


def test_naive_per_segment_rounding_would_drift_but_cumulative_does_not():
    # 每段 0.3333 秒：逐段四捨五入是 8 幀 × 30 = 240，實際應為 round(9.999 × 25) = 250
    cuts = [round(i * 0.3333, 4) for i in range(31)]
    segments = G.build_plan(cuts, set(), cuts[-1], 25, 1, 0, 0)
    assert sum(round((b - a) * 25) for a, b in zip(cuts, cuts[1:])) == 240
    assert sum(s.frames for s in segments) == 250


@pytest.mark.parametrize("need, expected", [(1, 1), (2, 5), (81, 81), (82, 85), (1500, 1501), (0, 1)])
def test_valid_frames_rounds_up_to_the_model_constraint(need, expected):
    assert G.valid_frames(need, 4, 1) == expected
    assert G.valid_frames(need, 1, 0) == max(need, 1)


def test_valid_frames_for_ltx_style_constraint():
    assert [G.valid_frames(n, 8, 1) for n in (1, 9, 10, 250)] == [1, 9, 17, 257]


# ---- 切點選擇 ----

def test_short_song_is_a_single_segment():
    assert G.choose_cuts(40.0, [0, 20, 40], [], max_seg=60, min_seg=3) == ([0.0, 40.0], set())


def test_cuts_prefer_section_boundaries_over_later_silences():
    silences = [{"start": 55.0, "end": 57.0}, {"start": 110.0, "end": 112.0}]
    cuts, forced = G.choose_cuts(150.0, [0, 37.4, 56.1, 75.4, 94.0, 118.3, 150], silences, 60, 3)
    assert cuts == [0.0, 56.1, 94.0, 150.0] and forced == set()
    # 段落邊界 56.1 與靜音中點 56.0 都在範圍內，選段落邊界；118.3 超出第二段的範圍，所以選 94.0


def test_cuts_fall_back_to_silence_when_no_boundary_is_in_reach():
    silences = [{"start": 20.0, "end": 22.0}, {"start": 48.0, "end": 50.0}]
    cuts, forced = G.choose_cuts(100.0, [0, 100], silences, 60, 3)
    assert cuts == [0.0, 49.0, 100.0] and forced == set()


def test_long_silence_can_be_cut_anywhere_inside_it_without_forcing():
    # 前奏 2.7–20.6 秒都沒有人聲，中點 11.6 不在第一段的範圍內：切在範圍內最靠後、仍在靜音裡的位置
    silences = [{"start": 2.74, "end": 20.56}]
    cuts, forced = G.choose_cuts(30.0, [0, 30], silences, max_seg=10, min_seg=2)
    assert cuts == [0.0, 10.0, 20.0, 30.0] and forced == set()


def test_cuts_are_forced_at_the_quietest_point_when_there_is_no_candidate():
    asked = []

    def quietest(t0, t1):
        asked.append((t0, t1))
        return 51.25

    cuts, forced = G.choose_cuts(100.0, [0, 100], [], 60, 3, quietest)
    assert cuts == [0.0, 51.25, 100.0] and forced == {51.25}
    assert asked == [(36.0, 60.0)]                      # 只在範圍的後 40% 找
    assert G.choose_cuts(100.0, [0, 100], [], 60, 3)[0] == [0.0, 60.0, 100.0]   # 沒有音量資訊時切在上限


def test_too_short_last_segment_is_merged_into_the_previous_one():
    cuts, _ = G.choose_cuts(61.0, [0, 60.0, 61.0], [], max_seg=60, min_seg=3)
    assert len(cuts) == 3 and cuts[0] == 0.0 and cuts[-1] == 61.0
    assert all(b - a >= 3 for a, b in zip(cuts, cuts[1:]))                      # 沒有短於下限的段


def test_no_segment_exceeds_the_limit():
    rng = random.Random(7)
    for _ in range(100):
        duration = rng.uniform(30, 400)
        bounds = sorted(rng.uniform(0, duration) for _ in range(8))
        silences = [{"start": t, "end": t + 1} for t in sorted(rng.uniform(0, duration - 1) for _ in range(15))]
        cuts, _ = G.choose_cuts(duration, bounds, silences, 10, 3)
        assert all(0 < b - a <= 10 + 1e-6 for a, b in zip(cuts, cuts[1:]))


def test_plan_continues_from_previous_frame_and_resets_only_at_natural_cuts():
    cuts = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    segments = G.build_plan(cuts, {40.0}, 60.0, 25, 4, 1, reset_every=2, section_bounds=[0.0, 20.0, 50.0])
    assert [s.init for s in segments] == ["photo", "prev", "photo", "prev", "prev", "photo"]
    # 第 5 段已經該重設，但 40 秒是在人聲中強制切開的，延後到下一個自然的切點（50 秒）
    assert [s.forced_cut for s in segments] == [False, False, False, False, True, False]
    first, second = segments[0], segments[1]
    assert (first.frames, first.gen_frames, first.drop_head, first.audio_start_frame) == (250, 253, 0, 0)
    # 延續段多生成 1 幀（與上一段最後一幀重複），音訊往前一幀對齊
    assert (second.frames, second.gen_frames, second.drop_head, second.audio_start_frame) == (250, 253, 1, 249)
    assert G.Segment.from_dict(json.loads(json.dumps(second.to_dict()))) == second
    assert all(s.init != "photo" or i == 0 for i, s in enumerate(
        G.build_plan(cuts, set(), 60.0, 25, 4, 1, reset_every=0, section_bounds=cuts)))   # 0＝不重設


def test_end_anchor_never_restarts_from_the_photo():
    cuts = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    segments = G.build_plan(cuts, {40.0}, 60.0, 25, 4, 1, reset_every=1, section_bounds=cuts, end_anchor=True)
    # 畫面一路接續，不跳回照片；改由每段的結尾回到照片的樣子
    assert [s.init for s in segments] == ["photo", "prev", "prev", "prev", "prev", "prev"]
    # 第 4 段結束在 40 秒（人聲中強制切開），不引導；最後一段後面沒有東西要接，也不引導
    assert [s.end_anchor for s in segments] == [True, True, True, False, True, False]
    assert sum(s.frames for s in segments) == 1500
    old = {k: v for k, v in segments[0].to_dict().items() if k != "end_anchor"}
    assert G.Segment.from_dict(old).end_anchor is False                    # 舊版的分段計畫沒有這個欄位


def test_quietest_finder_locates_the_gap():
    sr = 8000
    samples = np.ones(sr * 10, dtype="float32") * 0.5
    samples[int(6.3 * sr):int(6.6 * sr)] = 0.0
    assert quietest_finder(samples, sr)(4.0, 9.0) == pytest.approx(6.45, abs=0.11)
    assert quietest_finder(samples, sr)(9.99, 10.0) == 10.0


# ---- FFmpeg ----

def make_mp4(path, frames: int, fps: int = 25, size: str = "64x48", color: str = "") -> bytes:
    source = f"color=c={color}:s={size}:r={fps}" if color else f"testsrc=s={size}:r={fps}"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", source, "-frames:v", str(frames),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)], check=True)
    return path.read_bytes()


def test_trim_drops_head_keeps_exact_count_and_pads_when_short(tmp_path):
    make_mp4(tmp_path / "raw.mp4", 30)
    assert F.trim_segment(tmp_path / "raw.mp4", tmp_path / "a.mp4", drop_head=1, keep=20, fps=25) == 0
    info = F.probe_video(tmp_path / "a.mp4")
    assert (info["frames"], info["fps"], info["codec"], info["pix_fmt"]) == (20, 25.0, "h264", "yuv420p")
    # 來源只有 30 幀，丟 1 幀後要 35 幀：補 6 幀
    assert F.trim_segment(tmp_path / "raw.mp4", tmp_path / "b.mp4", drop_head=1, keep=35, fps=25) == 6
    assert F.probe_video(tmp_path / "b.mp4")["frames"] == 35


def test_concat_keeps_every_frame_and_replaces_audio(tmp_path):
    folder = tmp_path / "有 空白的資料夾"
    folder.mkdir()
    parts = []
    for i, frames in enumerate((25, 50, 13), start=1):
        make_mp4(tmp_path / f"raw{i}.mp4", frames)
        F.trim_segment(tmp_path / f"raw{i}.mp4", folder / f"seg_{i:03d}.mp4", 0, frames, 25)
        parts.append(folder / f"seg_{i:03d}.mp4")
    song = tmp_path / "song.flac"
    song.write_bytes(flac_bytes(88 / 25, 48000))
    out = tmp_path / "out.mp4"
    F.concat_with_audio(parts, song, out, folder)
    assert F.probe_video(out)["frames"] == 88
    streams = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate", "-of",
                              "csv=p=0", str(out)], capture_output=True, text=True).stdout.split()
    assert streams == ["h264", "aac,48000"]
    assert sorted(p.name for p in folder.iterdir()) == ["seg_001.mp4", "seg_002.mp4", "seg_003.mp4"]


def test_extract_frame_and_black_detection(tmp_path):
    make_mp4(tmp_path / "v.mp4", 10)
    F.extract_frame(tmp_path / "v.mp4", tmp_path / "last.png", 9)
    assert Image.open(tmp_path / "last.png").size == (64, 48)
    make_mp4(tmp_path / "black.mp4", 25, color="black")
    assert F.detect_black(tmp_path / "black.mp4") and not F.detect_black(tmp_path / "v.mp4")


def test_ffmpeg_failure_is_readable(tmp_path):
    with pytest.raises(UserFacingError, match="讀取影片資訊"):
        F.probe_video(tmp_path / "不存在.mp4")


# ---- 照片 ----

def test_photo_is_cropped_to_target_without_touching_the_original(tmp_path):
    source = tmp_path / "原圖.jpg"
    Image.new("RGB", (1200, 1600), "red").save(source)
    before = source.read_bytes()
    warnings = prepare_photo(source, tmp_path / "out.png", 832, 480)
    assert Image.open(tmp_path / "out.png").size == (832, 480)
    assert source.read_bytes() == before
    assert len(warnings) == 1 and "只保留約" in warnings[0]          # 直式照片裁成橫式會裁掉很多

    Image.new("RGB", (1920, 1080), "blue").save(tmp_path / "wide.png")
    assert prepare_photo(tmp_path / "wide.png", tmp_path / "out2.png", 832, 480) == []
    Image.new("RGB", (400, 225), "blue").save(tmp_path / "small.png")
    assert "小於影片" in prepare_photo(tmp_path / "small.png", tmp_path / "out3.png", 832, 480)[0]
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(UserFacingError, match="無法讀取照片"):
        prepare_photo(tmp_path / "bad.png", tmp_path / "x.png", 832, 480)


# ---- 階段 5 ----

SONG_SECONDS = 6.0


@pytest.fixture
def video_project(config, project, tmp_path):   # noqa: F811
    """6 秒的歌、3 秒處有段落邊界；單段上限 4 秒 → 兩段（75 + 75 影格）。"""
    config.set("engines.infinitetalk.max_segment_seconds", 4)
    config.set("engines.infinitetalk.min_segment_seconds", 1)
    project.path("audio", "song.flac").write_bytes(flac_bytes(SONG_SECONDS, 48000))
    sr = 8000
    soundfile.write(str(project.path("audio", "vocals.wav")),
                    (0.2 * np.sin(np.arange(int(SONG_SECONDS * sr)) / 5)).astype("float32"), sr)
    project.path("analysis", "sections.json").write_text(json.dumps({
        "duration": SONG_SECONDS,
        "sections": [{"index": 0, "type": "verse", "start": 0.0, "end": 3.0, "instrumental": False, "line_ids": [1]},
                     {"index": 1, "type": "chorus", "start": 3.0, "end": 6.0, "instrumental": False, "line_ids": [2]}],
        "silences": []}), encoding="utf-8")
    Image.new("RGB", (1600, 900), "gray").save(project.path("input", "photo_original.png"))
    project.data["inputs"]["photo"] = "input/photo_original.png"
    for stage_id in ("s2_lyrics", "s3_song", "s4_subtitle"):
        project.set_status(stage_id, DONE)
    return project


@pytest.fixture
def raw_clips(tmp_path):
    """模擬 ComfyUI 產生的兩段影片：第 1 段 77 幀、第 2 段 81 幀（都比需要的多，之後會被裁掉）。"""
    return [make_mp4(tmp_path / "raw_a.mp4", 77), make_mp4(tmp_path / "raw_b.mp4", 81)]


def stage_with(llm) -> VideoStage:
    return VideoStage(llm_factory=lambda cfg: llm)


def test_video_stage_generates_segments_and_joins_them_to_song_length(config, video_project, server, raw_clips):   # noqa: F811
    """M4 整合：得到與歌曲等長（差距不超過一個影格）的對嘴影片。"""
    server.file_queue = list(raw_clips)
    llm = FakeLLM([dict(PROMPT)])
    llm.supports_vision = True
    stage = stage_with(llm)
    ctx = StageContext(config, video_project)
    assert stage.check_inputs(ctx) == []
    logs: list[str] = []

    result = stage.run(ctx, Reporter(on_log=logs.append))

    assert result.ok and result.warnings == []
    output = video_project.path("output", "video_nosub.mp4")
    info = F.probe_video(output)
    assert info["frames"] == 150 and info["fps"] == 25.0                       # 6 秒 × 25 fps，一幀不差
    assert abs(info["frames"] / 25 - SONG_SECONDS) <= 1 / 25
    assert [F.probe_video(video_project.path("segments", f"seg_00{i}.mp4"))["frames"] for i in (1, 2)] == [75, 75]

    plan = json.loads(video_project.path("analysis", "segment_plan.json").read_text(encoding="utf-8"))
    assert plan["total_frames"] == 150 and plan["prompt"] == PROMPT["positive"]
    assert [(s["t_start"], s["t_end"], s["frames"], s["gen_frames"], s["init"]) for s in plan["segments"]] == [
        (0.0, 3.0, 75, 77, "photo"), (3.0, 6.0, 75, 77, "prev")]

    first, second = list(server.prompts.values())
    assert first["8"]["inputs"]["num_frames"] == 77 and second["8"]["inputs"]["num_frames"] == 77
    assert first["17"]["inputs"]["width"] == 832 and first["17"]["inputs"]["height"] == 480
    assert first["16"]["inputs"]["positive_prompt"] == PROMPT["positive"]
    assert first["11"]["inputs"]["blocks_to_swap"] == 20
    assert first["14"]["inputs"]["clip_name"] == "clip_vision_h.safetensors"
    assert second["18"]["inputs"]["seed"] == first["18"]["inputs"]["seed"] + 1
    assert len(server.uploads) == 4 and server.free_calls == 1                 # 每段上傳一張起始影像與一段音訊
    assert llm.unloaded == 1 and llm.calls[0]["image"].name == "photo_prepared.png"
    assert Image.open(video_project.path("input", "photo_prepared.png")).size == (832, 480)

    states = video_project.stage("s5_video")["segments"]
    assert [(s["index"], s["status"], s["frames"]) for s in states] == [(1, "done", 75), (2, "done", 75)]
    # 暫存檔已清掉，只留各段成品與最後一幀
    assert sorted(p.name for p in video_project.path("segments").iterdir()) == [
        "last_001.png", "last_002.png", "seg_001.mp4", "seg_002.mp4"]
    assert any("分成 2 段" in line for line in logs)


def test_interrupted_run_resumes_from_the_unfinished_segment(config, video_project, server, raw_clips):   # noqa: F811
    """驗收：階段 5 中途中斷，重開後從未完成的段落續跑，已完成的段落不重做。"""
    server.file_queue = list(raw_clips)
    server.step_delay = 0.05
    ctx = StageContext(config, video_project)
    stages = {s.id: s for s in (EnvStage(), stage_with(FakeLLM([dict(PROMPT)])))}

    def log(text: str) -> None:
        if "第 1／2 段完成" in text:
            ctx.cancel.set()                       # 第 1 段一完成就取消，模擬中途關閉

    first = R.PipelineRunner(ctx, stages).run(Reporter(on_log=log))
    assert (first.kind, first.stage_id) == (R.CANCELLED, "s5_video")
    seg1 = video_project.path("segments", "seg_001.mp4")
    assert seg1.exists() and not video_project.path("segments", "seg_002.mp4").exists()
    stamp = seg1.stat().st_mtime_ns
    reopened = type(video_project)(video_project.folder,
                                   json.loads((video_project.folder / "project.json").read_text(encoding="utf-8")))
    assert [s["status"] for s in reopened.stage("s5_video")["segments"]] == ["done", "pending"]

    ctx2 = StageContext(config, reopened)
    logs: list[str] = []
    second = R.PipelineRunner(ctx2, {s.id: s for s in (EnvStage(), stage_with(FakeLLM([])))}).run(
        Reporter(on_log=logs.append))

    assert second.kind == R.STOPPED and reopened.status("s5_video") == DONE
    assert seg1.stat().st_mtime_ns == stamp                                    # 第 1 段沒有重做
    assert any("第 1 段已完成，跳過" in line for line in logs) and any("沿用先前的分段計畫" in line for line in logs)
    video_prompts = [p for p in server.prompts.values() if "17" in p]
    assert len(video_prompts) == 3                                             # 第 1 段、被取消的第 2 段、重跑的第 2 段
    assert F.probe_video(reopened.path("output", "video_nosub.mp4"))["frames"] == 150


def test_changing_the_photo_discards_old_segments(config, video_project, server, raw_clips):   # noqa: F811
    server.file_queue = list(raw_clips)
    stage_with(FakeLLM([dict(PROMPT)])).run(StageContext(config, video_project), Reporter())
    Image.new("RGB", (1600, 900), "white").save(video_project.path("input", "photo_original.png"))

    server.file_queue = list(raw_clips)
    logs: list[str] = []
    stage_with(FakeLLM([dict(PROMPT)])).run(StageContext(config, video_project), Reporter(on_log=logs.append))
    assert len(server.prompts) == 4 and not any("跳過" in line for line in logs)


def test_out_of_memory_retries_the_segment_with_more_block_swap(config, video_project, server, raw_clips):   # noqa: F811
    """驗收：顯示記憶體不足時有明確處理，並記錄做了什麼調整。"""
    server.file_queue = list(raw_clips)
    server.fail_next = ["CUDA out of memory. Tried to allocate 2.00 GiB"]
    logs: list[str] = []
    result = stage_with(FakeLLM([dict(PROMPT)])).run(StageContext(config, video_project), Reporter(on_log=logs.append))
    assert result.ok
    sent = list(server.prompts.values())
    assert [p["11"]["inputs"]["blocks_to_swap"] for p in sent] == [20, 40, 20]
    assert any("顯示記憶體不足" in line and "重試這一段" in line for line in logs)
    assert server.free_calls == 2


def test_out_of_memory_twice_stops_with_advice(config, video_project, server):   # noqa: F811
    server.fail_next = ["CUDA out of memory", "CUDA out of memory"]
    with pytest.raises(UserFacingError) as e:
        stage_with(FakeLLM([dict(PROMPT)])).run(StageContext(config, video_project), Reporter())
    assert "仍然失敗" in e.value.message and "較低的解析度" in e.value.fix


def test_short_output_is_padded_and_reported(config, video_project, server, tmp_path):   # noqa: F811
    server.file_queue = [make_mp4(tmp_path / "s1.mp4", 73), make_mp4(tmp_path / "s2.mp4", 81)]
    result = stage_with(FakeLLM([dict(PROMPT)])).run(StageContext(config, video_project), Reporter())
    assert result.ok and "第 1 段生成的影格少了 2 幀" in result.warnings[0]
    assert F.probe_video(video_project.path("output", "video_nosub.mp4"))["frames"] == 150


def test_prompt_falls_back_to_default_when_llm_fails(config, video_project, server, raw_clips):   # noqa: F811
    server.file_queue = list(raw_clips)
    result = stage_with(FakeLLM([LLMError("逾時")])).run(StageContext(config, video_project), Reporter())
    assert result.ok and "影像提示詞改用預設內容" in result.warnings[0]
    sent = next(iter(server.prompts.values()))
    assert sent["16"]["inputs"]["positive_prompt"].startswith("A singer stands in front of a studio microphone")


def test_check_inputs_lists_what_is_missing(config, project):   # noqa: F811
    project.data["inputs"]["engine"] = "某個新引擎"
    messages = [p.message for p in VideoStage().check_inputs(StageContext(config, project))]
    assert messages == ["還沒有選擇人像照片", "還沒有歌曲", "還沒有分離後的人聲", "還沒有段落時間表",
                        "對嘴引擎 某個新引擎 尚未實作"]


# ---- LTX-2.3 引擎 ----

def test_ltx23_engine_uses_its_own_size_fps_and_frame_rule(config, video_project, server, tmp_path):   # noqa: F811
    video_project.data["inputs"]["engine"] = "ltx23"
    config.set("engines.ltx23.max_segment_seconds", 4)
    config.set("engines.ltx23.min_segment_seconds", 1)
    # 模型實際輸出的尺寸可能被取整（實機遇過 832×448），裁切時要統一成目標尺寸
    server.file_queue = [make_mp4(tmp_path / "l1.mp4", 73, fps=24, size="832x448"),
                         make_mp4(tmp_path / "l2.mp4", 81, fps=24, size="832x448")]
    llm = FakeLLM([dict(PROMPT)])

    result = VideoStage(llm_factory=lambda cfg: llm).run(StageContext(config, video_project), Reporter())

    assert result.ok
    info = F.probe_video(video_project.path("output", "video_nosub.mp4"))
    assert (info["frames"], info["fps"], info["width"], info["height"]) == (144, 24.0, 896, 512)   # 6 秒 × 24 fps
    plan = json.loads(video_project.path("analysis", "segment_plan.json").read_text(encoding="utf-8"))
    assert plan["engine"] == "ltx23"
    assert [(s["frames"], s["gen_frames"], s["init"]) for s in plan["segments"]] == [(72, 73, "photo"), (72, 73, "prev")]
    assert all((s["gen_frames"] - 1) % 8 == 0 for s in plan["segments"])

    first, second = list(server.prompts.values())
    assert first["302"]["inputs"] == {"width": 448, "height": 256, "length": 73, "batch_size": 1}     # 一半解析度
    assert (first["333"]["inputs"]["width"], first["333"]["inputs"]["height"]) == (896, 512)
    assert first["332"]["inputs"]["duration"] == pytest.approx(73 / 24)
    assert first["307"]["inputs"]["frame_rate"] == 24.0 and first["341"]["inputs"]["frame_rate"] == 24.0
    assert first["306"]["inputs"]["text"] == PROMPT["positive"]
    assert {first[n]["inputs"]["ckpt_name"] for n in ("317", "318", "335")} == {"ltx-2.3-22b-dev-fp8.safetensors"}
    assert first["316"]["inputs"]["tile_size"] == 768
    # 第 1 段的結尾以原始照片引導（兩輪取樣各一次），最後一段不用
    assert plan["segments"][0]["end_anchor"] and not plan["segments"][1]["end_anchor"]
    assert first["342"]["inputs"]["image"]
    assert [first[n]["inputs"]["strength"] for n in ("344", "345")] == [0.8, 0.8]
    assert first["316"]["inputs"]["samples"] == ["346", 2] and "344" not in second
    assert Image.open(video_project.path("input", "photo_prepared.png")).size == (896, 512)
    assert second["286"]["inputs"]["noise_seed"] == first["286"]["inputs"]["noise_seed"] + 1


def test_switching_engine_regenerates_everything(config, video_project, server, raw_clips, tmp_path):   # noqa: F811
    server.file_queue = list(raw_clips)
    VideoStage(llm_factory=lambda cfg: FakeLLM([dict(PROMPT)])).run(StageContext(config, video_project), Reporter())
    video_project.data["inputs"]["engine"] = "ltx23"
    server.file_queue = [make_mp4(tmp_path / "l1.mp4", 145, fps=24)]
    logs: list[str] = []
    VideoStage(llm_factory=lambda cfg: FakeLLM([dict(PROMPT)])).run(
        StageContext(config, video_project), Reporter(on_log=logs.append))
    assert not any("沿用先前的分段計畫" in line for line in logs)
    assert F.probe_video(video_project.path("output", "video_nosub.mp4"))["frames"] == 144


def test_ltx23_out_of_memory_retries_with_smaller_decode_tiles(config, video_project, server, tmp_path):   # noqa: F811
    video_project.data["inputs"]["engine"] = "ltx23"
    server.file_queue = [make_mp4(tmp_path / "l1.mp4", 145, fps=24)]
    server.fail_next = ["torch.OutOfMemoryError: Allocation on device"]
    assert VideoStage(llm_factory=lambda cfg: FakeLLM([dict(PROMPT)])).run(
        StageContext(config, video_project), Reporter()).ok
    assert [p["316"]["inputs"]["tile_size"] for p in server.prompts.values()] == [768, 512]

# ---- 分段銜接方式 ----

def test_join_defaults_come_from_the_engine_config(config):
    ltx, talk = make_engine(config, "ltx23"), make_engine(config, "infinitetalk")
    assert ltx.default_join() == {"join_mode": "anchor", "join_every": 1, "anchor_strength": 0.8}
    assert talk.default_join() == {"join_mode": "photo", "join_every": 4, "anchor_strength": 0.8}
    assert ltx.supports_anchor and not talk.supports_anchor
    ltx.configure_join({})                                   # 專案沒有指定：維持設定檔的行為
    assert (ltx.join_mode, ltx.reset_every, ltx.end_anchor) == ("anchor", 1, True)


def test_project_can_choose_how_segments_are_joined(config):
    engine = make_engine(config, "ltx23")
    engine.configure_join({"join_mode": "prev", "join_every": 3})
    assert (engine.join_mode, engine.reset_every, engine.end_anchor) == ("prev", 0, False)
    engine.configure_join({"join_mode": "photo", "join_every": 3})
    assert (engine.join_mode, engine.reset_every, engine.end_anchor) == ("photo", 3, False)
    engine.configure_join({"join_mode": "anchor", "join_every": 2, "anchor_strength": 0.6})
    assert (engine.join_mode, engine.reset_every, engine.end_anchor) == ("anchor", 2, True)
    assert engine.settings["anchor_strength"] == 0.6
    assert config.get("engines.ltx23.anchor_strength") == 0.8          # 專案的選擇不會改到設定檔的值

    talk = make_engine(config, "infinitetalk")                           # 沒有段尾引導用的工作流：改成定期回到照片
    talk.configure_join({"join_mode": "anchor", "join_every": 2})
    assert (talk.join_mode, talk.reset_every, talk.end_anchor) == ("photo", 2, False)


def test_pure_continuation_never_uses_the_photo_again(config, video_project, server, tmp_path):   # noqa: F811
    video_project.data["inputs"].update(engine="ltx23", join_mode="prev")
    config.set("engines.ltx23.max_segment_seconds", 4)
    config.set("engines.ltx23.min_segment_seconds", 1)
    server.file_queue = [make_mp4(tmp_path / "l1.mp4", 73, fps=24), make_mp4(tmp_path / "l2.mp4", 81, fps=24)]
    assert VideoStage(llm_factory=lambda cfg: FakeLLM([dict(PROMPT)])).run(
        StageContext(config, video_project), Reporter()).ok
    plan = json.loads(video_project.path("analysis", "segment_plan.json").read_text(encoding="utf-8"))
    assert [(s["init"], s["end_anchor"]) for s in plan["segments"]] == [("photo", False), ("prev", False)]
    assert all("344" not in prompt for prompt in server.prompts.values())      # 沒有任何一段用到段尾引導

    # 改回段尾引導：分段計畫的指紋不同，影片整個重新生成
    video_project.data["inputs"].update(join_mode="anchor", anchor_strength=0.6)
    server.file_queue = [make_mp4(tmp_path / "l3.mp4", 73, fps=24), make_mp4(tmp_path / "l4.mp4", 81, fps=24)]
    logs: list[str] = []
    VideoStage(llm_factory=lambda cfg: FakeLLM([dict(PROMPT)])).run(
        StageContext(config, video_project), Reporter(on_log=logs.append))
    assert not any("沿用先前的分段計畫" in line for line in logs)
    third = list(server.prompts.values())[2]
    assert third["344"]["inputs"]["strength"] == 0.6


# ---- Wan 2.2 S2V 與 InfiniteTalk 原生版：一個工作流裡接好幾個區塊 ----

def request_for(frames: int, **extra) -> SegmentRequest:
    return SegmentRequest("photo.png", "clip.wav", frames, 832, 480, "a singer", 100, "mv/x", **extra)


def test_s2v_chains_extend_chunks_inside_one_workflow(config):
    engine = make_engine(config, "wan22_s2v")
    assert (engine.fps, engine.continuation, engine.audio_lead_frames, engine.motion_tail_frames) == (16, "motion", 2, 73)
    assert [engine.chunks(n) for n in (1, 78, 79, 158, 159)] == [1, 1, 2, 2, 3]      # K 個區塊得到 80K − 2 幀
    # 連續接續太久畫面會劣化，所以預設每段都重新從照片開始
    assert engine.default_join()["join_mode"] == "photo" and engine.default_join()["join_every"] == 1
    assert not engine.supports_anchor

    single = engine.build(request_for(78), full_object_info())
    assert "e1" not in single and "20" not in single and "ref_motion" not in single["12"]["inputs"]
    assert single["30"]["inputs"]["samples"] == ["13", 0]

    wf = engine.build(request_for(238, motion_video="mv/tail.mp4"), full_object_info())
    assert wf["e1"]["inputs"]["video_latent"] == ["13", 0] and wf["e2"]["inputs"]["video_latent"] == ["c1", 0]
    assert wf["k2"]["inputs"]["latent_image"] == ["e2", 2] and wf["c2"]["inputs"]["samples2"] == ["k2", 0]
    assert [wf[n]["inputs"]["seed"] for n in ("13", "k1", "k2")] == [100, 101, 102]
    assert wf["k1"]["inputs"]["steps"] == wf["13"]["inputs"]["steps"] == 4
    assert wf["e1"]["inputs"]["ref_image"] == ["11", 0]                                  # 每個區塊都以照片為參考
    assert wf["30"]["inputs"]["samples"] == wf["31"]["inputs"]["samples2"] == ["c2", 0]
    assert wf["20"]["inputs"]["file"] == "mv/tail.mp4" and wf["12"]["inputs"]["ref_motion"] == ["21", 0]
    assert wf["34"]["inputs"]["frame_rate"] == 16


def test_native_infinitetalk_chains_windows_and_trims_the_overlap(config):
    engine = make_engine(config, "infinitetalk_native")
    assert [engine.windows(n) for n in (1, 81, 82, 153, 154)] == [1, 1, 2, 2, 3]       # 81 幀，之後每個視窗多 72 幀
    assert (engine.continuation, engine.fps) == ("prev", 25)
    wf = engine.build(request_for(225), full_object_info())
    assert wf["12"]["inputs"]["mode"] == "single_speaker" and "previous_frames" not in wf["12"]["inputs"]
    assert wf["12w1"]["inputs"]["previous_frames"] == ["18", 0] and wf["12w2"]["inputs"]["previous_frames"] == ["b1", 0]
    assert wf["17w2"]["inputs"]["latent_image"] == ["12w2", 3] and wf["13w2"]["inputs"]["model"] == ["12w2", 0]
    assert wf["t2"]["inputs"]["batch_index"] == ["12w2", 4]                              # 重疊的影格數由節點決定
    assert wf["b2"]["inputs"] == {"image1": ["b1", 0], "image2": ["t2", 0]}
    assert [wf[n]["inputs"]["noise_seed"] for n in ("16", "16w1", "16w2")] == [100, 101, 102]
    assert wf["34"]["inputs"]["images"] == ["b2", 0]
    assert wf["12w1"]["inputs"]["start_image"] == ["11", 0] and wf["18w1"]["inputs"]["vae"] == ["6", 0]   # 共用的節點不複製


def test_s2v_stage_continues_motion_from_the_previous_segment(config, video_project, server, tmp_path):   # noqa: F811
    video_project.data["inputs"].update(engine="wan22_s2v", join_mode="prev")
    config.set("engines.wan22_s2v.max_segment_seconds", 4)
    config.set("engines.wan22_s2v.min_segment_seconds", 1)
    server.file_queue = [make_mp4(tmp_path / "s1.mp4", 78, fps=16, size="832x480"),
                         make_mp4(tmp_path / "s2.mp4", 78, fps=16, size="832x480")]
    logs: list[str] = []
    result = VideoStage(llm_factory=lambda cfg: FakeLLM([dict(PROMPT)])).run(
        StageContext(config, video_project), Reporter(on_log=logs.append))
    assert result.ok
    info = F.probe_video(video_project.path("output", "video_nosub.mp4"))
    assert (info["frames"], info["fps"]) == (96, 16.0)                                   # 6 秒 × 16 fps
    plan = json.loads(video_project.path("analysis", "segment_plan.json").read_text(encoding="utf-8"))
    assert [(s["init"], s["frames"], s["gen_frames"]) for s in plan["segments"]] == [("photo", 48, 78), ("motion", 48, 78)]
    first, second = list(server.prompts.values())
    assert "20" not in first and second["12"]["inputs"]["ref_motion"] == ["21", 0]
    assert first["11"]["inputs"]["image"] == second["11"]["inputs"]["image"]            # 兩段的參考影像都是照片
    assert any("以照片為參考，接續上一段的動作" in line for line in logs)
    assert not list(video_project.path("segments").glob("motion_*.mp4"))                 # 用完就清掉


def test_s2v_starts_every_segment_from_the_photo_by_default(config):
    engine = make_engine(config, "wan22_s2v")
    engine.configure_join({})
    cuts = [0.0, 20.0, 40.0, 60.0]
    plan = G.build_plan(cuts, {40.0}, 60.0, engine.fps, engine.frame_step, engine.frame_offset, engine.reset_every,
                        cuts, engine.end_anchor, engine.continuation)
    # 40 秒是在人聲中強制切開的：那一段改成接續動作，不在唱到一半時跳接
    assert [s.init for s in plan] == ["photo", "photo", "motion"]
    assert all(s.drop_head == 0 for s in plan) and [s.gen_frames for s in plan] == [398, 398, 398]
