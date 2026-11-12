"""階段 3：YuE2 歌曲生成。

1. 請 LLM 選定音樂特徵，由程式組成一行 YuE2 Style（語言與人聲類型由程式決定，確保與輸入相符）。
2. 把 lyrics.json 轉成 YuE2 歌詞格式，注入工作流範本，送到 ComfyUI。
3. 檢查實際長度；太短就換 seed 重試，最後保留最接近目標長度的一版。
產出 audio/song.flac 與 audio/yue2_prompt.json。
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import soundfile

from app.core import lyrics as L
from app.core.models import Problem, UserFacingError
from app.pipeline.common import duration_mode, language_info, load_prompt, make_comfy, style_hint
from app.pipeline.stage_base import Reporter, StageContext, StageResult
from app.pipeline.stages.s2_lyrics import load_lyrics
from app.services.comfy_client import ComfyClient, ComfyWorkflowError, Progress
from app.services.llm_client import LLMClient, LLMError, make_llm
from app.services.workflow_template import WorkflowTemplate
from app.storage.app_config import AppConfig

STYLE_SCHEMA = {
    "type": "object",
    "properties": {
        "vocal_character": {"type": "string"},
        "genre": {"type": "string"},
        "bpm": {"type": "integer"},
        "instruments": {"type": "string"},
        "production": {"type": "string"},
    },
    "required": ["vocal_character", "genre", "bpm", "instruments", "production"],
}

LANGUAGE_NAMES = {"zh-TW": "Mandarin", "zh": "Mandarin", "en": "English", "ja": "Japanese", "yue": "Cantonese"}
VOCAL_NAMES = {"female": "female", "male": "male"}

# 各節點在單次生成中大約佔的進度區間，用來把節點進度換算成階段進度
_NODE_SPAN = {"YuE2GenerateABC": (0.02, 0.30), "YuE2GenerateMusic": (0.30, 0.85), "KSampler": (0.85, 0.97)}


def _ascii_words(text: object, limit: int) -> str:
    """只留英數與基本符號，並限制字數，避免 LLM 多寫的內容污染 Style。"""
    text = "" if text is None else str(text)
    words = "".join(ch if ch.isascii() and (ch.isalnum() or ch in " ,-&'") else " " for ch in text).split()
    return " ".join(words[:limit]).strip(" ,")


def compose_style(features: dict, language: str, vocal_type: str, has_intro: bool,
                  language_name: str = "", voice: str = "") -> str:
    """依官方建議的順序組成 Style：語言、人聲、曲風、速度、樂器、其他特徵。

    語言、男女聲與使用者選的聲線由程式填入，確保與輸入頁的選擇一致；
    language_name 是寫進 Style 的語言名稱，voice 是使用者選的聲線（留空就用 LLM 建議的）。
    """
    character = _ascii_words(voice or features.get("vocal_character", ""), 2)
    try:
        bpm = min(max(int(features.get("bpm", 96)), 60), 180)
    except (TypeError, ValueError):
        bpm = 96
    parts = [
        language_name or LANGUAGE_NAMES.get(language, "Mandarin"),
        f"{character} {VOCAL_NAMES.get(vocal_type, 'female')} vocal".strip(),
        _ascii_words(features.get("genre", ""), 8) or "pop",
        f"{bpm} BPM",
        _ascii_words(features.get("instruments", ""), 12),
        _ascii_words(features.get("production", ""), 6),
        "instrumental intro" if has_intro else "",
    ]
    return ", ".join(p for p in parts if p)


def make_style(llm: LLMClient, lyrics: dict, inputs: dict, config: AppConfig, reporter: Reporter) -> tuple[str, list[str]]:
    """回傳（Style、警告）。LLM 失敗時退回曲風範本的提示，不讓整個階段失敗。"""
    style = str(inputs.get("style", ""))
    hint = style_hint(config, style)
    has_intro = bool(lyrics["sections"]) and not lyrics["sections"][0]["lines"]
    # 歌詞實際的語言優先（人工輸入歌詞時，歌詞檔記的就是輸入頁選的語言）
    language = language_info(config, lyrics.get("language") or inputs.get("language", "zh-TW"))
    code, name = language["code"], language["yue2"]
    vocal, voice = inputs.get("vocal_type", "female"), str(inputs.get("voice_character", ""))
    user = load_prompt(config, "yue2_style", content=inputs.get("content", ""), style=style,
                       hint=hint or "(none)", language=name,
                       lyrics_first_lines="\n".join(L.lyric_lines(lyrics)[:4]))
    try:
        features = llm.chat_json("You are a music producer. Output JSON only.", user, STYLE_SCHEMA)
        return compose_style(features, code, vocal, has_intro, name, voice), []
    except LLMError as e:
        reporter.log(f"LLM 產生風格描述失敗（{e.message}），改用曲風範本的描述")
        fallback = {"genre": _ascii_words(hint, 30) or "pop", "bpm": 96}
        return (compose_style(fallback, code, vocal, has_intro, name, voice),
                [f"風格描述改用曲風範本的預設內容（LLM 失敗：{e.message}）"])


def audio_seconds(path: Path) -> float:
    info = soundfile.info(str(path))
    return info.frames / info.samplerate


def fade_out(path: Path, seconds: float) -> None:
    """把音檔最後幾秒淡出（歌曲被長度上限切斷時用，避免突然中斷）。"""
    info = soundfile.info(str(path))
    data, samplerate = soundfile.read(str(path), dtype="float32", always_2d=True)
    count = min(int(seconds * samplerate), len(data))
    if count > 0:
        ramp = (1.0 - (np.arange(count, dtype="float32") + 1) / count)[:, None]
        data[-count:] *= ramp
        soundfile.write(str(path), data, samplerate, format=info.format, subtype=info.subtype)


class SongStage:
    id = "s3_song"

    def __init__(self, llm_factory=make_llm, comfy_factory=make_comfy):
        self._llm_factory = llm_factory
        self._comfy_factory = comfy_factory

    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        lyrics = load_lyrics(ctx.project)
        if not lyrics or not L.lyric_lines(lyrics):
            return [Problem("還沒有歌詞", "請先完成歌詞階段")]
        return []

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.path("audio", "song.flac"), ctx.project.path("audio", "yue2_prompt.json")]

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        project, config = ctx.project, ctx.config
        inputs = project.data.get("inputs", {})
        lyrics = load_lyrics(project)
        warnings: list[str] = []

        # 1. Style
        reporter.progress(0.0, "產生風格描述")
        style = str(inputs.get("yue2_style_override", "")).strip()
        if not style:
            llm = self._llm_factory(config)
            try:
                style, warnings = make_style(llm, lyrics, inputs, config, reporter)
            finally:
                llm.unload()         # 把 VRAM 讓給 ComfyUI
        reporter.log(f"Style：{style}")
        yue2_lyrics = L.to_yue2(lyrics, config.get("yue2.tag_map", {}))

        # 2. 生成（不合格就換 seed 重試）
        # 長度模式：auto＝給寬鬆的上限，讓歌曲唱完後自己結束；max＝不超過使用者設定的上限
        auto = duration_mode(inputs, config) == "auto"
        limit = float(config.get("song.auto_max_limit", 600))
        cap = float(config.get("song.auto_max_seconds", 300)) if auto else \
            float(inputs.get("target_seconds", config.get("song.target_seconds", 150)))
        needed = L.lyrics_seconds(lyrics, L.bpm_from_text(style))
        minimum = min(needed * float(config.get("song.min_duration_ratio", 0.5)), cap * 0.9)
        reporter.log(f"唱完這份歌詞估計需要約 {needed:.0f} 秒；"
                     + (f"長度由 AI 決定（上限先給 {cap:.0f} 秒）" if auto else f"長度上限 {cap:.0f} 秒"))
        if not auto and needed > cap:
            warnings.append(f"歌詞估計需要約 {needed:.0f} 秒，超過設定的上限 {cap:.0f} 秒，歌曲可能唱不完；"
                            "建議提高上限、減少歌詞，或把長度改成「自動」")
        max_tries = max(1, int(config.get("song.max_retries", 3)))
        template = WorkflowTemplate.load(config.workflows_dir, config.get("yue2.workflow"))
        audio_dir = project.path("audio")
        attempts: list[dict] = []
        client = self._comfy_factory(ctx)
        try:
            problems = template.check_against(client.object_info())
            if problems:
                raise UserFacingError("YuE2 工作流範本與本機 ComfyUI 不相容：" + "；".join(problems),
                                      "請更新 ComfyUI，或更新 workflows 資料夾內的範本與對應表")
            for number in range(1, max_tries + 1):
                seed = random.randint(0, 2**31 - 1)
                workflow = template.inject({
                    "ckpt_name": config.get("yue2.ckpt_name"),
                    "style": style,
                    "lyrics": yue2_lyrics,
                    "plan_mode": config.get("yue2.plan_mode", "full"),
                    "seed": seed,
                    "max_duration": cap,
                    "filename_prefix": f"{config.get('comfyui.output_prefix')}/{project.folder.name}_song",
                })
                reporter.log(f"送出歌曲生成（第 {number}／{max_tries} 次，seed {seed}，最長 {cap:.0f} 秒）")
                entry = self._run_once(client, workflow, ctx, reporter, number, max_tries)
                files = client.output_files(entry, template.output_node("audio"))
                if not files:
                    raise UserFacingError("工作流執行完成，但沒有產生音檔", "請到 ComfyUI 介面查看是否有錯誤訊息")
                path = client.download(files[0], audio_dir / f"song_try{number}{Path(files[0]['filename']).suffix}")
                seconds = audio_seconds(path)
                # 長度剛好等於上限，代表是被上限切斷的，不是自己唱完的
                capped = seconds >= cap - 1.0
                attempts.append({"seed": seed, "seconds": round(seconds, 2), "file": path.name,
                                 "max_seconds": cap, "capped": capped})
                reporter.log(f"第 {number} 次：長度 {seconds:.1f} 秒" + ("（達到上限，可能沒唱完）" if capped else ""))
                if not capped and seconds >= minimum:
                    break
                if number == max_tries:
                    break
                if capped and auto and cap < limit:
                    cap = min(cap * 1.5, limit)
                    reporter.log(f"把上限放寬到 {cap:.0f} 秒再生成一次")
                elif capped:
                    reporter.log("換 seed 重試，看能不能在上限內唱完")
                else:
                    reporter.log(f"比唱完歌詞需要的時間短很多（低於 {minimum:.0f} 秒），換 seed 重試")
            client.free()
        finally:
            client.close()

        # 3. 選一版：優先選自己唱完的（沒碰到上限），其中取最長的；都碰到上限就取最長的
        finished = [a for a in attempts if not a["capped"]]
        best = max(finished or attempts, key=lambda a: a["seconds"])
        final = audio_dir / "song.flac"
        best_file = audio_dir / best["file"]
        if best_file.suffix.lower() != ".flac":
            raise UserFacingError(f"ComfyUI 產生的音檔不是 FLAC（{best_file.name}）",
                                  "請確認 YuE2 工作流的存檔節點輸出 FLAC")
        shutil.copyfile(best_file, final)
        for attempt in attempts:
            (audio_dir / attempt["file"]).unlink(missing_ok=True)
        if best["capped"]:
            fade = float(config.get("song.fade_seconds", 3.0))
            fade_out(final, fade)
            warnings.append(
                f"歌曲長度達到上限 {best['max_seconds']:.0f} 秒，結尾可能沒唱完，已加上 {fade:.0f} 秒淡出；"
                + ("可減少歌詞後重新生成" if auto else "建議提高上限、減少歌詞，或把長度改成「自動」"))
        elif best["seconds"] < minimum:
            warnings.append(f"重試 {len(attempts)} 次後歌曲仍只有 {best['seconds']:.0f} 秒，"
                            f"比唱完歌詞估計需要的 {needed:.0f} 秒短很多，可能沒唱完；可按「重新生成歌曲」再試")
        project.data.setdefault("seeds", {})["yue2"] = best["seed"]
        project.path("audio", "yue2_prompt.json").write_text(json.dumps({
            "style": style,
            "lyrics": yue2_lyrics,
            "seed": best["seed"],
            "seconds": best["seconds"],
            "duration_mode": "auto" if auto else "max",
            "max_seconds": best["max_seconds"],
            "capped": best["capped"],
            "estimated_seconds": round(needed, 1),
            "ckpt_name": config.get("yue2.ckpt_name"),
            "plan_mode": config.get("yue2.plan_mode", "full"),
            "attempts": attempts,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        project.stage(self.id)["attempts"] = len(attempts)
        project.save()
        reporter.log(f"歌曲完成：{best['seconds']:.1f} 秒")
        reporter.progress(1.0, "完成")
        return StageResult(True, warnings)

    def _run_once(self, client: ComfyClient, workflow: dict, ctx: StageContext, reporter: Reporter,
                  number: int, max_tries: int) -> dict:
        last = [0.0]

        def on_progress(p: Progress) -> None:
            text = f"第 {number}／{max_tries} 次 · {p.node_type}" + (f" {p.value}/{p.max}" if p.max else "")
            span = _NODE_SPAN.get(p.node_type)
            if span:
                # YuE2GenerateABC 的上限是最大 token 數，通常遠早於上限就結束，所以只顯示計數
                inner = p.value / p.max if p.max and p.node_type != "YuE2GenerateABC" else 0.0
                last[0] = max(last[0], span[0] + (span[1] - span[0]) * inner)
            reporter.progress(last[0], text)      # 其他節點（載入模型、解碼、存檔）沿用目前的進度

        logs = ctx.project.logs_dir
        try:
            return client.run(workflow, on_progress, ctx.cancel, log_dir=logs, label="yue2")
        except ComfyWorkflowError as e:
            if not e.is_oom:
                raise
            # 顯示記憶體不足：釋放模型後原樣重試一次
            reporter.log("顯示記憶體不足，釋放 ComfyUI 已載入的模型後重試一次")
            client.free()
            try:
                return client.run(workflow, on_progress, ctx.cancel, log_dir=logs, label="yue2_retry")
            except ComfyWorkflowError as again:
                if again.is_oom:
                    raise UserFacingError("顯示記憶體不足，釋放模型後重試仍然失敗",
                                          "請關閉其他佔用顯示卡的程式（包含 ComfyUI 內其他工作）後再試")
                raise

    resume = run
