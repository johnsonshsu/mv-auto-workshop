"""階段 4：歌曲解析與 SRT 字幕。

1. 人聲分離：在 ComfyUI 把歌曲拆成人聲與伴奏。
2. 強制對齊：歌詞是已知的，把 lyrics.txt 的每一行對到人聲上的時間（獨立子程序）。
3. 產生 SRT、段落時間表與對齊報告。
產出 audio/vocals.wav、audio/instrumental.wav、lyrics/lyrics.srt、analysis/sections.json、
analysis/alignment_report.json。
"""
from __future__ import annotations

import json
from pathlib import Path

import soundfile

from app.core import lyrics as L
from app.core import srt_builder as S
from app.core.models import Problem, UserFacingError
from app.core.sections import build_sections, detect_silences
from app.pipeline.common import language_info, make_comfy
from app.pipeline.stage_base import Reporter, StageContext, StageResult
from app.pipeline.stages.s1_env import check_requires_option, resolve_models
from app.pipeline.stages.s2_lyrics import load_lyrics
from app.services.align_service import run_alignment
from app.services.comfy_client import Progress
from app.services.workflow_template import WorkflowTemplate, load_requires



def pick_separation(requires: dict, object_info: dict) -> tuple[str, dict]:
    """從需求清單挑第一組齊全的人聲分離方式，回傳（工作流名稱、要注入的模型參數）。"""
    for option in requires.get("any_of", [requires]):
        ok, _, _ = check_requires_option(option, object_info)
        if not ok:
            continue
        return option["workflow"], resolve_models(option, object_info)
    raise UserFacingError("ComfyUI 沒有可用的人聲分離節點或模型", "請到環境檢查頁查看「人聲分離」缺少什麼")


def to_wav(source: Path, dest: Path) -> None:
    data, samplerate = soundfile.read(str(source), dtype="float32", always_2d=True)
    soundfile.write(str(dest), data, samplerate, subtype="PCM_16")


class SubtitleStage:
    id = "s4_subtitle"

    def __init__(self, comfy_factory=make_comfy, aligner=run_alignment):
        self._comfy_factory = comfy_factory
        self._aligner = aligner

    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        problems = []
        if not ctx.project.path("audio", "song.flac").exists():
            problems.append(Problem("還沒有歌曲", "請先完成歌曲生成階段"))
        lyrics = load_lyrics(ctx.project)
        if not lyrics or not L.lyric_lines(lyrics):
            problems.append(Problem("還沒有歌詞", "請先完成歌詞階段"))
        return problems

    def outputs(self, ctx: StageContext) -> list[Path]:
        p = ctx.project
        return [p.path("audio", "vocals.wav"), p.path("audio", "instrumental.wav"), p.path("lyrics", "lyrics.srt"),
                p.path("analysis", "sections.json"), p.path("analysis", "alignment_report.json")]

    # ---- 1. 人聲分離 ----
    def separate(self, ctx: StageContext, reporter: Reporter) -> None:
        project, config = ctx.project, ctx.config
        prefix = config.get("comfyui.output_prefix")
        song = project.path("audio", "song.flac")
        client = self._comfy_factory(ctx)
        try:
            requires = load_requires(config.workflows_dir, config.get("vocal_separation.workflow"))
            name, params = pick_separation(requires, client.object_info())
            template = WorkflowTemplate.load(config.workflows_dir, name)
            problems = template.check_against(client.object_info())
            if problems:
                raise UserFacingError("人聲分離工作流範本與本機 ComfyUI 不相容：" + "；".join(problems),
                                      "請更新 workflows 資料夾內的範本與對應表")
            reporter.log(f"上傳歌曲到 ComfyUI，以 {name} 分離人聲")
            uploaded = client.upload_file(song, prefix, f"{project.folder.name}_song.flac")
            workflow = template.inject({
                **params, "audio": uploaded,
                "vocals_prefix": f"{prefix}/{project.folder.name}_vocals",
                "instrumental_prefix": f"{prefix}/{project.folder.name}_instrumental",
            })

            def on_progress(p: Progress) -> None:
                inner = p.value / p.max if p.max else 0.0
                reporter.progress(0.02 + 0.28 * inner, f"人聲分離 · {p.node_type}")

            entry = client.run(workflow, on_progress, ctx.cancel, log_dir=project.logs_dir, label="separation")
            for key in ("vocals", "instrumental"):
                files = client.output_files(entry, template.output_node(key))
                if not files:
                    raise UserFacingError(f"人聲分離完成，但沒有產生 {key} 音檔", "請到 ComfyUI 介面查看是否有錯誤訊息")
                temp = client.download(files[0], project.path("audio", f"{key}_download{Path(files[0]['filename']).suffix}"))
                to_wav(temp, project.path("audio", f"{key}.wav"))
                temp.unlink()
            client.free()        # 對齊模型在 APP 自己的程序執行，先讓 ComfyUI 卸載模型
        finally:
            client.close()

    # ---- 2、3. 對齊與產出 ----
    def analyse(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        project, config = ctx.project, ctx.config
        lyrics = load_lyrics(project)
        texts = L.lyric_lines(lyrics)
        # lyrics.txt 是對齊的依據，也是字幕文字的唯一來源
        lyrics_txt = project.path("lyrics", "lyrics.txt")
        lyrics_txt.write_text(L.to_plain_text(lyrics), encoding="utf-8")
        vocals = project.path("audio", "vocals.wav")
        language = language_info(config, lyrics.get("language", "zh-TW"))["whisper"]

        reporter.log(f"對齊 {len(texts)} 行歌詞（模型 {config.get('align.model')}）")
        scaled = Reporter(on_log=reporter.log,
                          on_progress=lambda f, t: reporter.progress(0.32 + 0.6 * f, f"字幕對齊 · {t}"))
        raw = self._aligner(config, vocals, lyrics_txt, project.path("analysis", "align_raw.json"),
                            language, scaled, ctx.cancel)

        samples, samplerate = soundfile.read(str(vocals), dtype="float32", always_2d=True)
        duration = len(samples) / samplerate
        assessed = S.assess(raw, texts, duration, float(config.get("align.min_confidence", 0.15)),
                            float(config.get("align.min_seconds_per_char", 0.04)),
                            float(config.get("align.max_word_gap", 2.0)))
        gap = float(config.get("subtitle_timing.gap", 0.05))
        lines = S.interpolate(assessed, texts, duration, gap)
        cues = S.build_cues(lines, float(config.get("subtitle_timing.tail", 0.3)),
                            float(config.get("subtitle_timing.min_duration", 0.8)), gap, duration)
        project.path("lyrics", "lyrics.srt").write_text(S.to_srt(cues), encoding="utf-8")

        silences = detect_silences(samples, samplerate, float(config.get("silence.threshold_db", -40.0)),
                                   float(config.get("silence.min_seconds", 0.4)))
        project.path("analysis", "sections.json").write_text(
            json.dumps(build_sections(lyrics, lines, duration, silences), ensure_ascii=False, indent=2),
            encoding="utf-8")

        interpolated = [line for line in lines if line.method == S.INTERPOLATED]
        low_threshold = float(config.get("align.low_confidence", 0.4))
        low = [line for line in lines if line.method == S.ALIGNED and line.confidence < low_threshold]
        ratio = len(interpolated) / len(lines)
        project.path("analysis", "alignment_report.json").write_text(json.dumps({
            "duration": round(duration, 3),
            "line_count": len(lines),
            "unaligned_count": len(interpolated),
            "unaligned_ratio": round(ratio, 4),
            "low_confidence_count": len(low),
            "lines": [line.to_dict() for line in lines],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        warnings = []
        if interpolated:
            ids = "、".join(str(line.id) for line in interpolated[:10])
            warnings.append(f"有 {len(interpolated)} 行歌詞對不上人聲，時間改用前後行內插（第 {ids} 行）")
        if low:
            warnings.append(f"有 {len(low)} 行的對齊信心偏低，建議在字幕校對頁確認")
        reporter.log(f"字幕完成：{len(lines)} 行，未對齊 {len(interpolated)} 行，低信心 {len(low)} 行")
        reporter.progress(1.0, "完成")
        return StageResult(True, warnings, data={"unaligned_ratio": ratio, "unaligned_count": len(interpolated)})

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        reporter.progress(0.0, "人聲分離")
        self.separate(ctx, reporter)
        return self.analyse(ctx, reporter)

    def resume(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        """中斷後續跑：人聲已經分離好就只重做對齊。"""
        project = ctx.project
        vocals, song = project.path("audio", "vocals.wav"), project.path("audio", "song.flac")
        done = vocals.exists() and project.path("audio", "instrumental.wav").exists() \
            and vocals.stat().st_mtime >= song.stat().st_mtime
        if not done:
            return self.run(ctx, reporter)
        reporter.log("人聲已分離，直接從對齊繼續")
        return self.analyse(ctx, reporter)
