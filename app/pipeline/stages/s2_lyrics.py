"""階段 2：歌詞產生。

AI 模式：依歌詞範本請 LLM 產生原創歌詞，程式驗證行數與字數，不合格就附上問題重試。
人工模式：解析使用者貼上的歌詞（中英文段落標記都接受）。
產出 lyrics/lyrics.json（結構化）與 lyrics/lyrics.txt（一行一句，供字幕對齊）。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.core import lyrics as L
from app.core.lyric_templates import LyricTemplateStore
from app.core.models import Problem, UserFacingError
from app.pipeline.common import duration_mode, language_info, load_prompt, style_hint
from app.pipeline.stage_base import Reporter, StageContext, StageResult
from app.services.llm_client import LLMClient, LLMFormatError, make_llm
from app.storage.app_config import AppConfig
from app.storage.project_store import Project

LYRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
                "required": ["lines"],
            },
        },
    },
    "required": ["title", "sections"],
}

SECTION_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {"type": "string"}}},
    "required": ["lines"],
}


def template_store(config: AppConfig) -> LyricTemplateStore:
    return LyricTemplateStore(config.root / "config" / "lyric_templates",
                              config.user_path.parent / "lyric_templates")


# ---- 存取 ----

def lyrics_path(project: Project) -> Path:
    return project.path("lyrics", "lyrics.json")


def load_lyrics(project: Project) -> dict | None:
    try:
        return json.loads(lyrics_path(project).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def save_lyrics(project: Project, lyrics: dict) -> bool:
    """寫入 lyrics.json 與 lyrics.txt。內容有變時把下游階段標為已過期，並回傳 True。"""
    changed = load_lyrics(project) != lyrics
    target = lyrics_path(project)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(lyrics, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    project.path("lyrics", "lyrics.txt").write_text(L.to_plain_text(lyrics), encoding="utf-8")
    if changed:
        project.data["song_takes"] = []      # 歌詞變了，之前為了「歌詞沒唱完」保留的各版歌曲不再適用
        project.mark_stale_after("s2_lyrics")
    return changed


# ---- 範本 → 提示詞 ----

def unique_sections(template: dict) -> list[dict]:
    """需要 LLM 寫的段落：有歌詞而且不是重複段。"""
    return [s for s in template["sections"] if s["lines"] and not s["repeat_of"]]


def structure_text(template: dict) -> str:
    parts = []
    for section in template["sections"]:
        label = L.SECTION_LABELS[section["type"]]
        if section["repeat_of"]:
            label += "（重複）"
        elif not section["lines"]:
            label += "（純音樂）"
        parts.append(label)
    return " → ".join(parts)


def sections_spec(template: dict, unit: str = "字") -> str:
    return "\n".join(
        f"{i}. {L.SECTION_LABELS[s['type']]}：{s['lines']} 行，每行 {s['chars_min']}–{s['chars_max']} {unit}"
        for i, s in enumerate(unique_sections(template), start=1))


def localize(template: dict, language: dict) -> dict:
    """把範本的「每行字數」換算成這個語言的長度。

    範本是以中文字數寫的；英語一行的單字數比較少、日語一行的假名比較多，
    乘上語言的 length_factor 才會是唱起來差不多長的一句。
    """
    factor = float(language.get("length_factor", 1.0))
    if factor == 1.0:
        return template
    sections = [{**s, "chars_min": max(round(s["chars_min"] * factor), 1) if s["lines"] else 0,
                 "chars_max": max(round(s["chars_max"] * factor), 2) if s["lines"] else 0}
                for s in template["sections"]]
    return {**template, "sections": sections}


def fit_template(template: dict, max_seconds: float, bpm: int, margin: float = 1.1) -> tuple[dict, int]:
    """把歌詞範本縮到在 max_seconds 內唱得完，回傳（範本、拿掉了幾個段落）。

    從最後面一段一段拿掉（通常是重複的副歌、過渡），直到估計長度（留一成餘裕）不超過上限。
    至少保留兩個有歌詞的段落；還是太長就維持這個最小結構，由歌詞階段另外提出警告。
    """
    sections = list(template["sections"])

    def needed(items: list[dict]) -> float:
        sung = [s for s in items if s["lines"]]
        return L.song_seconds(sum(s["lines"] for s in sung), len(sung), bpm) * margin

    removed = 0
    while needed(sections) > max_seconds and sum(1 for s in sections[:-1] if s["lines"]) >= 2:
        sections.pop()
        removed += 1
    return ({**template, "sections": sections}, removed) if removed else (template, 0)


# ---- 驗證與組裝 ----

def check_section(lines: list, spec: dict, number: int, tolerance: int, chinese: bool = True,
                  unit: str = "字") -> tuple[list[str], list[str]]:
    """清理一個段落並檢查行數與字數，回傳（清理後的行、違規說明）。"""
    cleaned = [L.clean_line(str(line)) for line in lines]
    if chinese:
        # 先把混入的簡體字轉成繁體；轉不掉的（沒裝 OpenCC）才會被下面的檢查當成違規
        cleaned = [L.to_traditional(line) for line in cleaned]
    cleaned = [line for line in cleaned if line]
    label = L.SECTION_LABELS[spec["type"]]
    problems = []
    if len(cleaned) != spec["lines"]:
        problems.append(f"第 {number} 段（{label}）應有 {spec['lines']} 行，實際 {len(cleaned)} 行")
    for row, line in enumerate(cleaned, start=1):
        count = L.count_chars(line)
        if not spec["chars_min"] - tolerance <= count <= spec["chars_max"] + tolerance:
            problems.append(f"第 {number} 段第 {row} 行「{line}」有 {count} {unit}，"
                            f"應為 {spec['chars_min']}–{spec['chars_max']} {unit}")
        simplified = L.find_simplified(line) if chinese else []
        if simplified:
            problems.append(f"第 {number} 段第 {row} 行「{line}」含有簡體字（{'、'.join(simplified)}），"
                            "請整行改用繁體中文")
    return cleaned, problems


def validate_generated(data: dict, template: dict, tolerance: int, chinese: bool = True,
                       unit: str = "字") -> tuple[list[list[str]], list[str]]:
    """檢查 LLM 回傳的段落，回傳（各段清理後的行、違規說明）。"""
    specs = unique_sections(template)
    sections = data.get("sections")
    if not isinstance(sections, list) or len(sections) != len(specs):
        got = len(sections) if isinstance(sections, list) else 0
        return [], [f"sections 應有 {len(specs)} 個段落，實際 {got} 個"]
    result, problems = [], []
    for number, (section, spec) in enumerate(zip(sections, specs), start=1):
        lines = section.get("lines") if isinstance(section, dict) else None
        cleaned, found = check_section(lines if isinstance(lines, list) else [], spec, number, tolerance,
                                       chinese, unit)
        result.append(cleaned)
        problems.extend(found)
    return result, problems


def assemble(template: dict, written: list[list[str]], title: str, language: str) -> dict:
    """把 LLM 寫的段落放回範本的完整順序：重複段複製、純音樂段留空。"""
    sections: list[dict] = []
    remaining = iter(written)
    for spec in template["sections"]:
        if spec["repeat_of"]:
            lines = list(sections[spec["repeat_of"] - 1]["lines"])
        elif spec["lines"]:
            lines = next(remaining)
        else:
            lines = []
        sections.append({"type": spec["type"], "lines": lines})
    return {"title": L.clean_line(title) or "未命名", "language": language, "sections": sections}


# ---- 產生 ----

def _ask_with_retries(llm: LLMClient, system: str, user: str, schema: dict, validate, retries: int,
                      reporter: Reporter, what: str):
    """呼叫 LLM 並驗證；格式不符或驗證不過時附上問題重試。validate 回傳（結果、違規說明）。"""
    feedback, last_problems = "", ["沒有回應"]
    for attempt in range(1, retries + 2):
        if attempt > 1:
            reporter.log(f"{what}第 {attempt} 次嘗試（上次的問題：{'；'.join(last_problems[:3])}）")
        try:
            data = llm.chat_json(system, user + feedback, schema)
        except LLMFormatError as e:
            last_problems = [e.message]
            feedback = "\n\n上一次的輸出不是合法的 JSON。請只輸出 JSON 物件，不要有其他文字。"
            continue
        result, problems = validate(data)
        if not problems:
            return result, data
        last_problems = problems
        feedback = ("\n\n上一次的輸出如下：\n" + json.dumps(data, ensure_ascii=False)
                    + "\n它有以下問題，請修正後重新輸出完整的 JSON：\n- " + "\n- ".join(problems[:8]))
    raise UserFacingError(
        f"LLM 連續 {retries + 1} 次沒有產生符合要求的{what}（{'；'.join(last_problems[:3])}）",
        "可以再試一次、換一個歌詞範本或模型，或改為自己輸入歌詞")


def generate_lyrics(llm: LLMClient, template: dict, inputs: dict, config: AppConfig,
                    reporter: Reporter | None = None) -> dict:
    reporter = reporter or Reporter()
    tolerance = int(config.get("lyrics.char_tolerance", 2))
    language = language_info(config, inputs.get("language", "zh-TW"))
    template = localize(template, language)
    specs = unique_sections(template)
    user = load_prompt(config, "lyrics", content=inputs.get("content", ""), style=inputs.get("style", ""),
                       language_rule=language["rule"],
                       structure=structure_text(template), style_hint=template["style_hint"] or "自然口語",
                       rhyme=template["rhyme"] or "自由", count=len(specs),
                       sections_spec=sections_spec(template, language["unit"]))
    reporter.log(f"請 {llm.label} 產生{language['label']}歌詞（{len(specs)} 個段落）")
    written, data = _ask_with_retries(
        llm, "你是專業的流行歌曲作詞人，只輸出 JSON。", user, LYRICS_SCHEMA,
        lambda d: validate_generated(d, template, tolerance, language["chinese"], language["unit"]),
        int(config.get("llm.max_retries", 2)), reporter, "歌詞")
    title = str(data.get("title", ""))
    lyrics = assemble(template, written, L.to_traditional(title) if language["chinese"] else title, language["code"])
    return lyrics


def regenerate_section(llm: LLMClient, lyrics: dict, index: int, inputs: dict, config: AppConfig,
                       template: dict | None = None, reporter: Reporter | None = None) -> dict:
    """只重寫第 index 段（從 0 起）；歌詞相同的重複段落會一起更新。"""
    reporter = reporter or Reporter()
    section = lyrics["sections"][index]
    if not section["lines"]:
        raise UserFacingError("這個段落是純音樂，沒有歌詞可以重新產生", "請把游標放在有歌詞的段落")
    language = language_info(config, lyrics.get("language") or inputs.get("language", "zh-TW"))
    factor = float(language["length_factor"])
    spec = {"type": section["type"], "lines": len(section["lines"]),
            "chars_min": max(round(5 * factor), 1), "chars_max": max(round(12 * factor), 2)}
    # 歌詞結構仍與範本一致時，沿用範本對這一段的字數要求
    if template and len(template["sections"]) == len(lyrics["sections"]):
        candidate = localize(template, language)["sections"][index]
        if candidate["type"] == section["type"] and candidate["lines"] == spec["lines"]:
            spec.update(chars_min=candidate["chars_min"], chars_max=candidate["chars_max"])
    user = load_prompt(config, "lyrics_section", content=inputs.get("content", ""), style=inputs.get("style", ""),
                       language_rule=language["rule"], unit=language["unit"],
                       lyrics=L.to_editor_text(lyrics), number=index + 1,
                       label=L.SECTION_LABELS[section["type"]], lines=spec["lines"],
                       chars_min=spec["chars_min"], chars_max=spec["chars_max"])
    tolerance = int(config.get("lyrics.char_tolerance", 2))
    reporter.log(f"請 {llm.label} 重寫第 {index + 1} 段（{L.SECTION_LABELS[section['type']]}）")
    new_lines, _ = _ask_with_retries(
        llm, "你是專業的流行歌曲作詞人，只輸出 JSON。", user, SECTION_SCHEMA,
        lambda d: check_section(d.get("lines") if isinstance(d.get("lines"), list) else [], spec, index + 1,
                                tolerance, language["chinese"], language["unit"]),
        int(config.get("llm.max_retries", 2)), reporter, "段落")
    old_lines = section["lines"]
    sections = [{"type": s["type"], "lines": list(new_lines) if s["lines"] == old_lines else list(s["lines"])}
                for s in lyrics["sections"]]
    return {**lyrics, "sections": sections}


class LyricsStage:
    id = "s2_lyrics"

    def __init__(self, llm_factory=make_llm):
        self._llm_factory = llm_factory

    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        inputs = ctx.project.data.get("inputs", {})
        if inputs.get("lyrics_source", "ai") == "manual":
            if not str(inputs.get("manual_lyrics", "")).strip():
                return [Problem("歌詞來源選了「自己輸入」，但還沒有貼上歌詞", "請到歌詞頁貼上歌詞並儲存")]
            return []
        problems = []
        if not str(inputs.get("content", "")).strip():
            problems.append(Problem("還沒有填寫音樂內容", "請到輸入頁描述歌曲的主題、故事與情緒"))
        if not str(inputs.get("style", "")).strip():
            problems.append(Problem("還沒有選擇曲風", "請到輸入頁選擇或輸入曲風"))
        return problems

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [lyrics_path(ctx.project), ctx.project.path("lyrics", "lyrics.txt")]

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        project, config = ctx.project, ctx.config
        inputs = project.data.get("inputs", {})
        warnings: list[str] = []
        if inputs.get("lyrics_source", "ai") == "manual":
            reporter.progress(0.5, "解析歌詞")
            lyrics = L.parse_lyrics_text(inputs["manual_lyrics"], project.name, inputs.get("language", "zh-TW"))
        else:
            template = template_store(config).get(inputs.get("lyric_template", "verse_chorus_bridge"))
            if duration_mode(inputs, config) == "max":
                # 有長度上限：先把歌詞結構縮到上限內唱得完，免得生成一首唱到一半的歌
                limit = float(inputs.get("target_seconds", config.get("song.target_seconds", 150)))
                bpm = L.bpm_from_text(style_hint(config, str(inputs.get("style", ""))))
                template, removed = fit_template(template, limit, bpm)
                if removed:
                    warnings.append(f"為了在上限 {limit:.0f} 秒內唱完，歌詞結構縮短為「{structure_text(template)}」"
                                    f"（拿掉最後 {removed} 個段落）；想要完整結構可提高上限或把長度改成「自動」")
                    reporter.log(f"⚠ {warnings[-1]}")
            reporter.progress(0.1, "產生歌詞")
            llm = self._llm_factory(config)
            try:
                lyrics = generate_lyrics(llm, template, inputs, config, reporter)
            finally:
                llm.unload()
        if duration_mode(inputs, config) == "max":
            limit = float(inputs.get("target_seconds", config.get("song.target_seconds", 150)))
            needed = L.lyrics_seconds(lyrics, L.bpm_from_text(style_hint(config, str(inputs.get("style", "")))))
            if needed > limit:
                warnings.append(f"歌詞共 {len(L.lyric_lines(lyrics))} 行，估計需要約 {needed:.0f} 秒，"
                                f"超過上限 {limit:.0f} 秒，歌曲可能唱不完；"
                                "建議提高上限、減少歌詞，或把長度改成「自動」")
        save_lyrics(project, lyrics)
        reporter.log(f"歌詞完成：《{lyrics['title']}》共 {len(L.lyric_lines(lyrics))} 行")
        reporter.progress(1.0, "完成")
        return StageResult(True, warnings)

    resume = run
