"""流程編排：依序執行七個階段，管理狀態、續跑、暫停點與取消。

每個階段完成後就把狀態寫進 project.json，所以中斷後重新執行會從第一個未完成的階段繼續。
"""
from __future__ import annotations

import shutil
import time
import traceback
from dataclasses import dataclass, field

from app.core.models import (DONE, FAILED, PENDING, RUNNING, STAGE_IDS, STAGE_LABELS, STALE, UserFacingError)
from app.pipeline.stage_base import PauseRequested, Reporter, Stage, StageContext
from app.pipeline.stages.s1_env import EnvStage, can_start
from app.pipeline.stages.s2_lyrics import LyricsStage
from app.pipeline.stages.s3_song import SongStage
from app.pipeline.stages.s4_subtitle import SubtitleStage
from app.pipeline.stages.s5_video import VideoStage
from app.pipeline.stages.s6_compose import ComposeStage
from app.pipeline.stages.s7_export import ExportStage
from app.services.comfy_client import ComfyCancelled

# 逐步確認模式：這些階段完成後停下來讓使用者檢查
PAUSE_AFTER = ("s2_lyrics", "s3_song", "s4_subtitle")

# 結果種類
FINISHED = "finished"      # 全部階段完成
PAUSED = "paused"          # 逐步確認模式的暫停點
STOPPED = "stopped"        # 後面的階段尚未實作
BLOCKED = "blocked"        # 環境檢查不通過或輸入不齊
ERROR = "error"            # 某個階段失敗
CANCELLED = "cancelled"    # 使用者取消


@dataclass
class RunOutcome:
    kind: str
    stage_id: str = ""
    message: str = ""
    fix: str = ""
    warnings: list[str] = field(default_factory=list)


# 換 seed 重試歌曲時，每一版要保存與還原的檔案
TAKE_FILES = ("audio/song.flac", "audio/yue2_prompt.json", "audio/vocals.wav", "audio/instrumental.wav",
              "lyrics/lyrics.srt", "analysis/sections.json", "analysis/alignment_report.json")


def default_stages() -> dict[str, Stage]:
    stages = [EnvStage(), LyricsStage(), SongStage(), SubtitleStage(), VideoStage(), ComposeStage(), ExportStage()]
    return {stage.id: stage for stage in stages}


class PipelineRunner:
    def __init__(self, ctx: StageContext, stages: dict[str, Stage] | None = None):
        self.ctx = ctx
        self.project = ctx.project
        self.stages = stages if stages is not None else default_stages()

    def run(self, reporter: Reporter, on_stage=lambda stage_id, status: None) -> RunOutcome:
        """從第一個未完成的階段跑起。on_stage 在每個階段狀態改變時被呼叫。"""
        project = self.project
        step_mode = project.data.get("mode", "auto") == "step"
        warnings: list[str] = []
        index = -1
        while index + 1 < len(STAGE_IDS):
            index += 1
            stage_id = STAGE_IDS[index]
            # 環境每次都重新檢查（ComfyUI 可能已經關掉）；其餘已完成的階段跳過
            if stage_id != "s1_env" and project.status(stage_id) == DONE:
                continue
            stage = self.stages.get(stage_id)
            label = STAGE_LABELS[stage_id]
            if stage is None:
                return RunOutcome(STOPPED, stage_id, f"「{label}」之後的階段尚未實作，流程先停在這裡",
                                  warnings=warnings)
            if self.ctx.cancel.is_set():
                return RunOutcome(CANCELLED, stage_id, warnings=warnings)
            if self.ctx.pause.is_set():
                return RunOutcome(CANCELLED, stage_id, "已暫停", warnings=warnings)

            problems = stage.check_inputs(self.ctx)
            if problems:
                return RunOutcome(BLOCKED, stage_id, "；".join(p.message for p in problems),
                                  "；".join(p.fix for p in problems if p.fix), warnings)

            # 上次中斷或失敗的階段用 resume，讓有中間狀態的階段接續
            previous = project.status(stage_id)
            action = stage.resume if previous in (RUNNING, FAILED) else stage.run
            project.set_status(stage_id, RUNNING, warnings=[], error="")
            on_stage(stage_id, RUNNING)
            reporter.log(f"── 階段 {STAGE_IDS.index(stage_id) + 1}：{label} ──")
            started = time.monotonic()
            try:
                result = action(self.ctx, reporter)
            except ComfyCancelled as e:
                paused = isinstance(e, PauseRequested)
                # 有中間成果的階段（例如已完成幾段影片）標成失敗，下次用 resume 接續
                project.set_status(stage_id, PENDING if previous in (PENDING, STALE) and not paused else FAILED,
                                   error="已暫停" if paused else "已取消")
                on_stage(stage_id, project.status(stage_id))
                return RunOutcome(CANCELLED, stage_id, "已暫停" if paused else "", warnings=warnings)
            except UserFacingError as e:
                return self._fail(stage_id, e.message, e.fix, started, on_stage, warnings)
            except Exception as e:      # 未預期的錯誤也要回報，不可無聲失敗
                reporter.log(traceback.format_exc())
                return self._fail(stage_id, f"發生未預期的錯誤：{type(e).__name__}: {e}",
                                  "詳細內容已寫在記錄區", started, on_stage, warnings)

            seconds = round(time.monotonic() - started, 1)
            if not result.ok:
                return self._fail(stage_id, result.message, "", started, on_stage, warnings + result.warnings,
                                  kind=BLOCKED if stage_id == "s1_env" else ERROR)
            project.set_status(stage_id, DONE, seconds=seconds, warnings=result.warnings)
            on_stage(stage_id, DONE)
            for warning in result.warnings:
                reporter.log(f"⚠ {warning}")
            warnings.extend(result.warnings)

            if stage_id == "s1_env":
                engine = project.data.get("inputs", {}).get("engine", "infinitetalk")
                if not can_start(stage.report, engine):
                    name = self.ctx.config.get(f"engines.{engine}.label", engine)
                    return RunOutcome(BLOCKED, stage_id, f"選用的對嘴引擎 {name} 在這台電腦上不可用",
                                      "請到環境檢查頁查看原因，或在輸入頁改用其他引擎", warnings)
            if stage_id == "s4_subtitle":
                if step_mode:
                    limit = float(self.ctx.config.get("align.max_unaligned_ratio", 0.15))
                    if result.data.get("unaligned_ratio", 0.0) > limit:
                        warnings.append("對不上人聲的歌詞偏多，歌曲可能沒把歌詞唱完；可按「重新生成歌曲」換一版")
                elif self._retry_song(result.data.get("unaligned_ratio", 0.0), reporter, warnings):
                    # 這一版要重做，它的警告不再適用
                    for old in project.stage("s3_song").get("warnings", []) + result.warnings:
                        if old in warnings:
                            warnings.remove(old)
                    for stale in ("s3_song", "s4_subtitle"):
                        on_stage(stale, STALE)
                    index = STAGE_IDS.index("s3_song") - 1
                    continue
            if step_mode and stage_id in PAUSE_AFTER:
                return RunOutcome(PAUSED, stage_id, f"「{label}」完成，請檢查後按「繼續」", warnings=warnings)
        return RunOutcome(FINISHED, warnings=warnings)

    # ---- 歌詞沒唱完時的重試迴圈（階段 3 ↔ 階段 4）----
    def _retry_song(self, ratio: float, reporter: Reporter, warnings: list[str]) -> bool:
        """字幕階段完成後呼叫。回傳 True 表示要換 seed 重新生成歌曲並重做字幕。

        每一版（歌曲＋字幕產出）先存到 audio/takes/N/；次數用完後取未對齊比例最低的一版還原。
        """
        project, config = self.project, self.ctx.config
        limit = float(config.get("align.max_unaligned_ratio", 0.15))
        max_takes = max(1, int(config.get("song.max_retries", 3)))
        takes: list[dict] = project.data.setdefault("song_takes", [])
        if ratio <= limit and not takes:
            return False                                   # 第一版就合格，最常見的情況
        number = len(takes) + 1
        takes_dir = project.path("audio", "takes")
        for relative in TAKE_FILES:
            source = project.path(*relative.split("/"))
            if source.exists():
                target = takes_dir / str(number) / relative.replace("/", "__")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        takes.append({"number": number, "ratio": round(ratio, 4)})
        project.save()

        if ratio > limit and number < max_takes:
            reporter.log(f"有 {ratio:.0%} 的歌詞對不上人聲（上限 {limit:.0%}），"
                         f"歌曲可能沒把歌詞唱完；換 seed 重新生成（第 {number + 1}／{max_takes} 版）")
            for stage_id in ("s3_song", "s4_subtitle"):
                project.stage(stage_id)["status"] = STALE
            project.save()
            return True

        best = min(takes, key=lambda take: take["ratio"])
        if best["number"] != number:
            reporter.log(f"第 {best['number']} 版對得最好（未對齊 {best['ratio']:.0%}），還原這一版")
            for relative in TAKE_FILES:
                source = takes_dir / str(best["number"]) / relative.replace("/", "__")
                if source.exists():
                    shutil.copyfile(source, project.path(*relative.split("/")))
        if best["ratio"] > limit:
            warnings.append(f"重新生成 {len(takes)} 版後，最好的一版仍有 {best['ratio']:.0%} 的歌詞對不上人聲；"
                            "可減少歌詞行數或到字幕校對頁手動調整")
        shutil.rmtree(takes_dir, ignore_errors=True)
        project.data["song_takes"] = []
        project.save()
        return False

    def _fail(self, stage_id, message, fix, started, on_stage, warnings, kind=ERROR) -> RunOutcome:
        self.project.set_status(stage_id, FAILED, seconds=round(time.monotonic() - started, 1),
                                error=message, warnings=[])
        on_stage(stage_id, FAILED)
        return RunOutcome(kind, stage_id, message, fix, warnings)
