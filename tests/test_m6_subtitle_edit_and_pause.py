from __future__ import annotations

import json

import pytest

from app.core import srt_builder as S
from app.core import subtitle_edit as E
from app.core.models import DONE, FAILED, UserFacingError
from app.pipeline import runner as R
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s1_env import EnvStage
from app.pipeline.stages.s5_video import VideoStage
from app.services import ffmpeg_service as F
from tests.helpers import FakeLLM
from tests.test_s3_song_and_runner import project, server, server_info  # noqa: F401  （fixture）
from tests.test_s5_video import PROMPT, raw_clips, video_project  # noqa: F401  （fixture）

SRT = ("1\n00:00:05,000 --> 00:00:08,300\n巷口的燈亮起\n\n"
       "2\n00:00:08,400 --> 00:00:11,300\n晚風吹過這裡\n\n"
       "3\n00:00:14,000 --> 00:00:16,800\n心跳慢慢靠近\n")
REPORT = {"duration": 30.0, "lines": [
    {"id": 1, "text": "巷口的燈亮起", "start": 5.0, "end": 8.0, "confidence": 0.9, "method": "aligned"},
    {"id": 2, "text": "晚風吹過這裡", "start": 8.4, "end": 11.0, "confidence": 0.3, "method": "aligned"},
    {"id": 3, "text": "心跳慢慢靠近", "start": 14.0, "end": 16.5, "confidence": 0.0, "method": "interpolated"}]}


@pytest.fixture
def folder(tmp_path):
    (tmp_path / "lyrics").mkdir()
    (tmp_path / "analysis").mkdir()
    (tmp_path / "lyrics" / "lyrics.srt").write_text(SRT, encoding="utf-8")
    (tmp_path / "analysis" / "alignment_report.json").write_text(json.dumps(REPORT, ensure_ascii=False), encoding="utf-8")
    return tmp_path


def test_load_combines_srt_times_with_alignment_confidence(folder):
    cues, duration = E.load_cues(folder)
    assert duration == 30.0
    assert [(c.id, c.text, c.start, c.end) for c in cues] == [
        (1, "巷口的燈亮起", 5.0, 8.3), (2, "晚風吹過這裡", 8.4, 11.3), (3, "心跳慢慢靠近", 14.0, 16.8)]
    assert [c.flag(0.4) for c in cues] == ["", "yellow", "red"]            # 正常、信心偏低、內插


def test_load_without_report_still_works_and_missing_srt_is_explained(folder, tmp_path):
    (folder / "analysis" / "alignment_report.json").unlink()
    cues, duration = E.load_cues(folder)
    assert duration == 16.8 and [c.flag(0.4) for c in cues] == ["", "", ""]
    with pytest.raises(UserFacingError, match="還沒有字幕檔"):
        E.load_cues(tmp_path / "空的")


def test_edges_cannot_cross_neighbours_or_collapse(folder):
    cues, duration = E.load_cues(folder)
    assert E.move_edge(cues, 1, "start", 7.0, duration) == 8.32             # 不能跨進上一則（8.3 + 間隔）
    assert E.move_edge(cues, 1, "end", 20.0, duration) == 13.98             # 不能跨進下一則（14.0 − 間隔）
    assert E.move_edge(cues, 1, "end", 8.0, duration) == 8.52               # 至少留 0.2 秒
    assert E.move_edge(cues, 0, "start", -3.0, duration) == 0.0
    assert E.move_edge(cues, 2, "end", 99.0, duration) == 30.0              # 最後一則不能超過歌曲結尾
    assert cues[1].edited and cues[0].edited and cues[2].edited


def test_moving_to_the_same_value_does_not_mark_as_edited(folder):
    cues, duration = E.load_cues(folder)
    E.move_edge(cues, 0, "start", 5.0, duration)
    assert not cues[0].edited


def test_shift_keeps_length_and_stops_at_neighbours(folder):
    cues, duration = E.load_cues(folder)
    assert E.shift(cues, 1, 1.0, duration) == 1.0
    assert (cues[1].start, cues[1].end) == (9.4, 12.3)
    assert E.shift(cues, 1, 10.0, duration) == pytest.approx(1.68)          # 被下一則擋住
    assert cues[1].end == 13.98 and cues[1].end - cues[1].start == pytest.approx(2.9)
    assert E.shift(cues, 1, -99.0, duration) == pytest.approx(8.32 - 11.08)


def test_save_rewrites_srt_keeps_text_and_records_manual_lines(folder):
    cues, duration = E.load_cues(folder)
    E.move_edge(cues, 2, "start", 13.2, duration)
    E.save_cues(folder, cues)

    saved = S.parse_srt((folder / "lyrics" / "lyrics.srt").read_text(encoding="utf-8"))
    assert saved == [(5.0, 8.3, "巷口的燈亮起"), (8.4, 11.3, "晚風吹過這裡"), (13.2, 16.8, "心跳慢慢靠近")]
    report = json.loads((folder / "analysis" / "alignment_report.json").read_text(encoding="utf-8"))
    assert [line["method"] for line in report["lines"]] == ["aligned", "aligned", "manual"]
    assert report["manual_count"] == 1
    reloaded, _ = E.load_cues(folder)
    assert reloaded[2].flag(0.4) == ""                                      # 手動調過的行不再標示


def test_save_refuses_overlapping_cues(folder):
    cues, _ = E.load_cues(folder)
    cues[0].end = 9.0
    with pytest.raises(UserFacingError, match="第 1 則與第 2 則字幕的時間重疊"):
        E.save_cues(folder, cues)


# ---- 暫停 ----

def test_pause_stops_after_the_current_segment_and_resume_continues(config, video_project, server, raw_clips):   # noqa: F811
    server.file_queue = list(raw_clips)
    ctx = StageContext(config, video_project)
    stages = {s.id: s for s in (EnvStage(), VideoStage(llm_factory=lambda cfg: FakeLLM([dict(PROMPT)])))}

    def log(text: str) -> None:
        if "第 1／2 段：" in text:
            ctx.pause.set()                            # 第 1 段生成途中按暫停

    paused = R.PipelineRunner(ctx, stages).run(Reporter(on_log=log))
    assert (paused.kind, paused.stage_id, paused.message) == (R.CANCELLED, "s5_video", "已暫停")
    assert video_project.status("s5_video") == FAILED and video_project.stage("s5_video")["error"] == "已暫停"
    assert F.probe_video(video_project.path("segments", "seg_001.mp4"))["frames"] == 75   # 這一段有跑完
    assert not video_project.path("segments", "seg_002.mp4").exists()
    assert server.interrupted == []                    # 暫停不會中斷正在進行的生成

    ctx.pause.clear()
    resumed = R.PipelineRunner(ctx, stages).run(Reporter())
    assert resumed.kind == R.STOPPED and video_project.status("s5_video") == DONE
    assert len([p for p in server.prompts.values() if "17" in p]) == 2      # 第 1 段沒有重做


def test_pause_between_stages(config, video_project, server):   # noqa: F811
    ctx = StageContext(config, video_project)
    ctx.pause.set()
    outcome = R.PipelineRunner(ctx, {"s1_env": EnvStage()}).run(Reporter())
    assert (outcome.kind, outcome.message) == (R.CANCELLED, "已暫停")
