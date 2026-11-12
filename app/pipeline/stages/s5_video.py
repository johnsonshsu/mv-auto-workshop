"""階段 5：對嘴影片生成。

整首歌分段生成再接合：
1. 照片前處理、產生影像提示詞。
2. 依段落時間表決定切點，算出每段的影格數（總和恆等於歌曲長度 × fps）。
3. 逐段送到 ComfyUI；每段完成就存檔並寫入 project.json，中斷後從未完成的段落繼續。
4. 用 FFmpeg 接合各段，捨棄各段自帶的音軌，配回完整歌曲。
產出 segments/seg_NNN.mp4 與 output/video_nosub.mp4。
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import soundfile

from app.core.models import DONE, PENDING, Problem, UserFacingError
from app.core.segmenter import MOTION, PHOTO, PREV, Segment, plan_segments, total_frames
from app.pipeline.common import load_prompt, make_comfy
from app.pipeline.stage_base import PauseRequested, Reporter, StageContext, StageResult
from app.services import ffmpeg_service as F
from app.services.comfy_client import ComfyClient, ComfyWorkflowError, Progress
from app.services.engines import SegmentRequest, make_engine
from app.services.llm_client import LLMError, make_llm
from app.services.photo import prepare_photo

_START_TEXT = {PHOTO: "以原始照片起始", PREV: "以上一段的最後一幀起始", MOTION: "以照片為參考，接續上一段的動作"}

PROMPT_SCHEMA = {"type": "object", "properties": {"positive": {"type": "string"}}, "required": ["positive"]}


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 \
        else f"{seconds // 60}:{seconds % 60:02d}"


def quietest_finder(samples: np.ndarray, samplerate: int, hop_seconds: float = 0.1):
    """回傳函式 f(t0, t1) → 這段時間內人聲最小聲的位置（秒）。"""
    mono = samples.mean(axis=1) if samples.ndim > 1 else samples
    hop = max(int(samplerate * hop_seconds), 1)

    def find(t0: float, t1: float) -> float:
        a, b = int(t0 * samplerate), min(int(t1 * samplerate), len(mono))
        count = (b - a) // hop
        if count <= 0:
            return t1
        energy = (mono[a:a + count * hop].reshape(count, hop).astype(np.float64) ** 2).mean(axis=1)
        return t0 + (int(energy.argmin()) + 0.5) * hop / samplerate

    return find


class VideoStage:
    id = "s5_video"

    def __init__(self, comfy_factory=make_comfy, llm_factory=make_llm, engine_factory=make_engine):
        self._comfy_factory = comfy_factory
        self._llm_factory = llm_factory
        self._engine_factory = engine_factory

    # ---- 輸入與產出 ----
    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        project = ctx.project
        inputs = project.data.get("inputs", {})
        problems = []
        photo = inputs.get("photo", "")
        if not photo or not project.path(photo).exists():
            problems.append(Problem("還沒有選擇人像照片", "請到輸入頁拖放或選擇一張站立或坐在麥克風前的人像照片"))
        for relative, name in (("audio/song.flac", "歌曲"), ("audio/vocals.wav", "分離後的人聲"),
                               ("analysis/sections.json", "段落時間表")):
            if not project.path(*relative.split("/")).exists():
                problems.append(Problem(f"還沒有{name}", "請先完成前面的階段"))
        try:
            self._engine_factory(ctx.config, inputs.get("engine", "infinitetalk"))
        except UserFacingError as e:
            problems.append(Problem(e.message, e.fix))
        return problems

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.path("output", "video_nosub.mp4")]

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        # 不論是第一次跑還是續跑都一樣：分段計畫的指紋相同就沿用已完成的段落，
        # 照片、歌曲、段落時間表、引擎或尺寸有變就整個重來
        return self._execute(ctx, reporter)

    resume = run

    # ---- 準備 ----
    def _signature(self, ctx: StageContext, engine, width: int, height: int) -> str:
        """分段計畫的指紋：照片、歌曲、段落時間表、引擎或尺寸一變，已生成的段落就不能沿用。"""
        project = ctx.project
        photo = project.path(project.data["inputs"]["photo"])
        parts = [engine.id, width, height, engine.fps, engine.max_segment_seconds, engine.reset_every,
                 engine.end_anchor, engine.continuation,
                 ctx.config.get("video.drive_audio", "vocals"), json.dumps(engine.settings, sort_keys=True)]
        for path in (photo, project.path("audio", "song.flac"), project.path("analysis", "sections.json")):
            parts.append(hashlib.sha256(path.read_bytes()).hexdigest())
        return hashlib.sha256(json.dumps(parts).encode("utf-8")).hexdigest()[:16]

    def _video_prompt(self, ctx: StageContext, engine, photo: Path, reporter: Reporter) -> tuple[str, list[str]]:
        """由 LLM 看照片寫影像提示詞；失敗時用引擎的預設提示詞。"""
        inputs = ctx.project.data.get("inputs", {})
        user = load_prompt(ctx.config, "video_prompt", style=inputs.get("style", ""),
                           content=inputs.get("content", ""), vocal=inputs.get("vocal_type", "female"))
        llm = self._llm_factory(ctx.config)
        try:
            data = llm.chat_json("You write prompts for video generation models. Output JSON only.", user,
                                 PROMPT_SCHEMA, photo if llm.supports_vision else None)
            prompt = " ".join(str(data.get("positive", "")).split())
            if len(prompt.split()) < 10:
                raise LLMError("LLM 回傳的影像提示詞太短")
            return prompt, []
        except LLMError as e:
            reporter.log(f"LLM 產生影像提示詞失敗（{e.message}），改用預設提示詞")
            return engine.default_prompt, [f"影像提示詞改用預設內容（LLM 失敗：{e.message}）"]
        finally:
            llm.unload()

    # ---- 主流程 ----
    def _execute(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        project, config = ctx.project, ctx.config
        inputs = project.data.get("inputs", {})
        engine = self._engine_factory(config, inputs.get("engine", "infinitetalk"))
        engine.configure_join(inputs)
        width, height = engine.size(inputs.get("resolution", config.get("video.default_resolution")))
        fps = engine.fps
        warnings: list[str] = []
        segments_dir = project.path("segments")
        plan_path = project.path("analysis", "segment_plan.json")

        # 1. 照片
        reporter.progress(0.0, "處理照片")
        prepared = project.path("input", "photo_prepared.png")
        warnings += prepare_photo(project.path(inputs["photo"]), prepared, width, height)

        # 2. 分段計畫（中斷續跑時，指紋相同就沿用舊計畫與已完成的段落）
        signature = self._signature(ctx, engine, width, height)
        song = project.path("audio", "song.flac")
        duration = soundfile.info(str(song)).frames / soundfile.info(str(song)).samplerate
        drive_name = "song.flac" if config.get("video.drive_audio", "vocals") == "song" else "vocals.wav"
        drive, samplerate = soundfile.read(str(project.path("audio", drive_name)), dtype="float32", always_2d=True)
        old = None
        if plan_path.exists():
            try:
                old = json.loads(plan_path.read_text(encoding="utf-8"))
            except ValueError:
                old = None
        if old and old.get("signature") == signature:
            segments = [Segment.from_dict(s) for s in old["segments"]]
            prompt = old["prompt"]
            reporter.log(f"沿用先前的分段計畫（共 {len(segments)} 段）")
        else:
            if segments_dir.exists():
                shutil.rmtree(segments_dir)
            sections = json.loads(project.path("analysis", "sections.json").read_text(encoding="utf-8"))
            vocals, vocal_rate = soundfile.read(str(project.path("audio", "vocals.wav")), dtype="float32",
                                                always_2d=True)
            segments = plan_segments(duration, fps, sections, engine.max_segment_seconds,
                                     engine.min_segment_seconds, engine.frame_step, engine.frame_offset,
                                     engine.reset_every, quietest_finder(vocals, vocal_rate), engine.end_anchor,
                                     engine.continuation)
            reporter.progress(0.01, "產生影像提示詞")
            prompt, prompt_warnings = self._video_prompt(ctx, engine, prepared, reporter)
            warnings += prompt_warnings
            project.data.setdefault("seeds", {})["video"] = random.randint(0, 2**31 - 1)
            plan_path.write_text(json.dumps({
                "signature": signature, "engine": engine.id, "fps": fps, "width": width, "height": height,
                "duration": round(duration, 3), "total_frames": total_frames(duration, fps), "prompt": prompt,
                "segments": [s.to_dict() for s in segments],
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            project.path("analysis", "video_prompt.json").write_text(
                json.dumps({"positive": prompt, "negative": engine.settings.get("negative_prompt", "")},
                           ensure_ascii=False, indent=2), encoding="utf-8")
        segments_dir.mkdir(exist_ok=True)
        n_total = total_frames(duration, fps)
        if sum(s.frames for s in segments) != n_total:
            raise UserFacingError("分段計畫的影格總數與歌曲長度不符", "這是程式錯誤，請回報並附上 analysis/segment_plan.json")
        forced = [s.index for s in segments if s.forced_cut]
        if forced:
            warnings.append(f"第 {'、'.join(map(str, forced))} 段的起點找不到靜音處，是在人聲中強制切開的")
        reporter.log(f"影片 {width}×{height}、{fps:g} fps，共 {n_total} 影格，分成 {len(segments)} 段："
                     + "、".join(f"{s.t_end - s.t_start:.1f} 秒" for s in segments))
        reporter.log(f"影像提示詞：{prompt}")

        # 3. 逐段生成
        states = {s["index"]: s for s in project.stage(self.id).get("segments", [])}
        project.stage(self.id)["segments"] = [
            states.get(s.index, {"index": s.index, "status": PENDING}) for s in segments]
        seed = int(project.data.get("seeds", {}).get("video", 0))
        done_frames, spent, generated = 0, 0.0, 0
        client = self._comfy_factory(ctx)
        try:
            engine.check(client.object_info())
            for segment in segments:
                target = segments_dir / f"seg_{segment.index:03d}.mp4"
                last_frame = segments_dir / f"last_{segment.index:03d}.png"
                if target.exists() and F.probe_video(target)["frames"] == segment.frames:
                    if not last_frame.exists():
                        F.extract_frame(target, last_frame, segment.frames - 1)
                    self._mark(project, segment, DONE)
                    done_frames += segment.frames
                    reporter.log(f"第 {segment.index} 段已完成，跳過")
                    continue
                started = time.monotonic()
                pad = self._generate(ctx, client, engine, segment, len(segments), prepared, drive, samplerate,
                                     prompt, seed, width, height, target, reporter, done_frames, n_total,
                                     spent, generated)
                if pad:
                    warnings.append(f"第 {segment.index} 段生成的影格少了 {pad} 幀，已用最後一幀補足")
                F.extract_frame(target, last_frame, segment.frames - 1)
                seconds = time.monotonic() - started
                self._mark(project, segment, DONE, seconds=round(seconds, 1), frames=segment.frames)
                done_frames += segment.frames
                spent += seconds
                generated += segment.frames
                reporter.log(f"第 {segment.index}／{len(segments)} 段完成（{segment.frames} 影格，耗時 {_clock(seconds)}）")
                if ctx.pause.is_set() and segment.index < len(segments):
                    client.free()
                    raise PauseRequested()
            client.free()
        finally:
            client.close()

        # 4. 接合並配回完整歌曲
        reporter.progress(0.95, "接合各段並配回歌曲")
        output = project.path("output", "video_nosub.mp4")
        F.concat_with_audio([segments_dir / f"seg_{s.index:03d}.mp4" for s in segments], song, output, segments_dir)
        warnings += self._verify(output, segments, n_total, duration, fps)
        reporter.log(f"影片完成：{n_total} 影格，{duration:.2f} 秒")
        reporter.progress(1.0, "完成")
        return StageResult(True, warnings)

    def _mark(self, project, segment: Segment, status: str, **extra) -> None:
        for entry in project.stage(self.id)["segments"]:
            if entry["index"] == segment.index:
                entry["status"] = status
                entry.update(extra)
        project.save()

    def _generate(self, ctx, client: ComfyClient, engine, segment: Segment, count: int, photo: Path,
                  drive: np.ndarray, samplerate: int, prompt: str, seed: int, width: int, height: int,
                  target: Path, reporter: Reporter, done_frames: int, n_total: int,
                  spent: float, generated: int) -> int:
        """生成一段並裁成精確的影格數，回傳補了幾幀。"""
        project, config = ctx.project, ctx.config
        prefix = config.get("comfyui.output_prefix")
        name = f"{project.folder.name}_seg{segment.index:03d}"
        segments_dir = target.parent

        # 起始影像：原始照片，或上一段的最後一幀
        motion = None
        if segment.init == PREV:
            image = segments_dir / f"last_{segment.index - 1:03d}.png"
            if not image.exists():
                raise UserFacingError(f"找不到第 {segment.index - 1} 段的最後一幀，無法接續第 {segment.index} 段",
                                      "請刪除專案的 segments 資料夾後重跑影片階段")
        else:
            image = photo
            if segment.init == MOTION:
                # 照片仍是參考影像；另外把上一段的尾端交給模型，讓它接續動作
                previous = segments_dir / f"seg_{segment.index - 1:03d}.mp4"
                if not previous.exists():
                    raise UserFacingError(f"找不到第 {segment.index - 1} 段的影片，無法接續第 {segment.index} 段",
                                          "請刪除專案的 segments 資料夾後重跑影片階段")
                motion = segments_dir / f"motion_{segment.index:03d}.mp4"
                F.tail_clip(previous, motion, engine.motion_tail_frames, engine.fps)
        # 音訊片段：以影格邊界換算取樣點，長度等於要生成的影格數；超出歌曲開頭或結尾的部分補靜音
        lead = engine.audio_lead_frames
        begin = round((segment.audio_start_frame - lead) / engine.fps * samplerate)
        length = round((segment.gen_frames + lead) / engine.fps * samplerate)
        clip = drive[max(begin, 0):max(begin + length, 0)]
        if begin < 0:
            clip = np.concatenate([np.zeros((min(-begin, length), drive.shape[1]), dtype=drive.dtype), clip])
        if len(clip) < length:
            clip = np.concatenate([clip, np.zeros((length - len(clip), drive.shape[1]), dtype=drive.dtype)])
        audio_file = segments_dir / f"audio_{segment.index:03d}.wav"
        soundfile.write(str(audio_file), clip, samplerate, subtype="PCM_16")

        uploaded_image = client.upload_file(image, prefix, f"{name}_init.png")
        uploaded_audio = client.upload_file(audio_file, prefix, f"{name}.wav")
        request = SegmentRequest(uploaded_image, uploaded_audio, segment.gen_frames, width, height, prompt,
                                 seed + segment.index, f"{prefix}/{name}")
        if motion is not None:
            request.motion_video = client.upload_file(motion, prefix, f"{name}_motion.mp4")
        if segment.end_anchor:
            request.anchor_image = client.upload_file(photo, prefix, f"{project.folder.name}_photo.png")

        started = time.monotonic()

        state = {"passes": 0, "last": 0, "inner": 0.0}
        passes = max(engine.passes(segment.gen_frames), 1)

        def on_progress(p: Progress) -> None:
            # 取樣節點的步數歸零代表進到下一輪（LTX-2.3 有兩輪：低解析度生成、放大後精修）
            if p.max and "Sampler" in p.node_type:
                if p.value < state["last"]:
                    state["passes"] += 1
                state["last"] = p.value
                state["inner"] = max(state["inner"], min(
                    (state["passes"] + p.value / p.max) / passes, 1.0))
            inner = state["inner"]
            frames_now = done_frames + inner * segment.frames
            # 以這次執行實際生成的速度估剩餘時間
            elapsed = spent + (time.monotonic() - started)
            made = generated + inner * segment.frames
            eta = f"，預估剩餘 {_clock(elapsed / made * (n_total - frames_now))}" if made > segment.frames * 0.05 else ""
            reporter.progress(0.03 + 0.92 * frames_now / n_total,
                              f"第 {segment.index}／{count} 段 · {p.node_type}"
                              + (f" {p.value}/{p.max}" if p.max else "") + eta)

        reporter.log(f"第 {segment.index}／{count} 段：{segment.t_start:.2f}–{segment.t_end:.2f} 秒，"
                     f"生成 {segment.gen_frames} 影格（{_START_TEXT[segment.init]}"
                     + ("，結尾回到原始照片的樣子）" if segment.end_anchor else "）"))
        logs = project.logs_dir
        try:
            entry = client.run(engine.build(request, client.object_info()), on_progress, ctx.cancel,
                               log_dir=logs, label=f"video_seg{segment.index:03d}")
        except ComfyWorkflowError as e:
            if not e.is_oom:
                raise
            # 顯示記憶體不足：釋放模型後改用更省 VRAM 的設定重試一次
            reporter.log("顯示記憶體不足：釋放模型後，改把更多模型區塊留在系統記憶體（較慢）重試這一段")
            client.free()
            request.low_vram = True
            state.update(passes=0, last=0, inner=0.0)
            try:
                entry = client.run(engine.build(request, client.object_info()), on_progress, ctx.cancel,
                                   log_dir=logs, label=f"video_seg{segment.index:03d}_lowvram")
            except ComfyWorkflowError as again:
                if again.is_oom:
                    raise UserFacingError(
                        "顯示記憶體不足，改用省 VRAM 的設定後仍然失敗",
                        "請關閉其他佔用顯示卡的程式，或到輸入頁改用較低的解析度後重跑")
                raise
        files = client.output_files(entry, engine.output_node)
        if not files:
            raise UserFacingError(f"第 {segment.index} 段執行完成，但沒有產生影片", "請到 ComfyUI 介面查看是否有錯誤訊息")
        raw = client.download(files[0], segments_dir / f"raw_{segment.index:03d}.mp4")
        pad = F.trim_segment(raw, target, segment.drop_head, segment.frames, engine.fps,
                             int(config.get("video.crf", 16)), width, height)
        got = F.probe_video(target)["frames"]
        if got != segment.frames:
            target.unlink(missing_ok=True)
            raise UserFacingError(f"第 {segment.index} 段裁切後是 {got} 影格，應為 {segment.frames} 影格",
                                  "請重跑影片階段；若持續發生請回報")
        raw.unlink(missing_ok=True)
        audio_file.unlink(missing_ok=True)
        if motion is not None:
            motion.unlink(missing_ok=True)
        return pad

    def _verify(self, output: Path, segments: list[Segment], n_total: int, duration: float, fps: float) -> list[str]:
        info = F.probe_video(output)
        if info["frames"] != n_total:
            raise UserFacingError(f"接合後的影片是 {info['frames']} 影格，應為 {n_total} 影格",
                                  "請刪除專案的 segments 資料夾後重跑影片階段")
        length = info["frames"] / fps
        if abs(length - duration) > 1 / fps + 1e-6:
            raise UserFacingError(f"影片長度 {length:.3f} 秒與歌曲長度 {duration:.3f} 秒相差超過一個影格",
                                  "請檢查設定檔中引擎的 fps 是否與工作流一致")
        warnings = []
        joins = [s.f_start / fps for s in segments[1:]]
        margin = 3 / fps
        for start, end in F.detect_black(output):
            near = [j for j in joins if start - margin <= j <= end + margin]
            if near:
                warnings.append(f"段落交界 {near[0]:.2f} 秒附近偵測到黑畫面（{start:.2f}–{end:.2f} 秒）")
        return warnings
