"""執行結果報告：整理專案的輸入、各階段的執行方式與耗時，輸出成 Markdown 或 HTML。

資料全部來自專案資料夾（project.json 與各階段留下的 JSON），不需要 ComfyUI 或 LLM 在執行中。
"""
from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path

from app.core.models import DONE, STAGE_IDS, STAGE_LABELS, STATUS_LABELS
from app.pipeline.common import duration_mode, language_info, voice_options
from app.pipeline.stages.s7_export import LICENSE_NOTICE
from app.storage.app_config import AppConfig

FORMATS = {"md": "Markdown", "html": "HTML"}
TIMING_NOTE = "各階段的耗時是該階段最後一次執行所花的時間；重跑過的階段不會累計先前的時間，暫停與等待確認的時間也不算在內。"


def clock(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    whole = int(round(seconds))
    hours, minutes, secs = whole // 3600, whole % 3600 // 60, whole % 60
    return f"{hours} 小時 {minutes:02d} 分 {secs:02d} 秒" if hours else f"{minutes} 分 {secs:02d} 秒"


def _load(project, *parts: str) -> dict:
    path = project.path(*parts)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}


def _when(text: str | None) -> str:
    if not text:
        return "—"
    try:
        return datetime.fromisoformat(text).strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return text


_SECTION_LABELS = {"intro": "前奏", "verse": "主歌", "pre_chorus": "導歌", "pre-chorus": "導歌", "chorus": "副歌",
                   "bridge": "橋段", "outro": "尾奏"}


def _lyrics(project) -> dict:
    """這次的歌詞：歌名與各段內容。沒有分段資料時，把 lyrics.txt 當成一整段。"""
    data = _load(project, "lyrics", "lyrics.json")
    sections = [{"label": _SECTION_LABELS.get(str(s.get("type", "")).lower(), str(s.get("type", ""))),
                 "lines": [str(line) for line in s.get("lines", [])]}
                for s in data.get("sections", []) if s.get("lines")]
    plain = project.path("lyrics", "lyrics.txt")
    if not sections and plain.exists():
        lines = [line.strip() for line in plain.read_text(encoding="utf-8").splitlines() if line.strip()]
        sections = [{"label": "", "lines": lines}] if lines else []
    return {"title": data.get("title", ""), "sections": sections,
            "line_count": sum(len(s["lines"]) for s in sections)}


def _prompts(project) -> list[tuple[str, str]]:
    """這次用到的提示詞：使用者一開始輸入的音樂內容，以及 AI 據此寫出的歌曲風格與影像提示詞。"""
    inputs = project.data.get("inputs", {})
    rows = [("音樂內容（一開始輸入的提示詞）", inputs.get("content", "")), ("曲風", inputs.get("style", "")),
            ("AI 寫的歌曲風格描述（送給 YuE2）", _load(project, "audio", "yue2_prompt.json").get("style", "")),
            ("AI 寫的影像提示詞（送給對嘴引擎）", _load(project, "analysis", "video_prompt.json").get("positive", ""))]
    return [(label, str(text).strip()) for label, text in rows if str(text).strip()]


def _methods(config: AppConfig, project) -> dict[str, str]:
    """每個階段是用什麼方式執行的。只描述專案裡查得到的事實，查不到的就不寫。"""
    inputs = project.data.get("inputs", {})
    song, plan = _load(project, "audio", "yue2_prompt.json"), _load(project, "analysis", "segment_plan.json")
    align = _load(project, "analysis", "alignment_report.json")
    methods = {"s1_env": "偵測 ComfyUI、硬體，檢查工作流需要的節點與模型",
               "s7_export": "把成品放到專案資料夾最上層"}

    if inputs.get("lyrics_source") == "manual":
        methods["s2_lyrics"] = "使用者自行輸入歌詞"
    else:
        provider = config.get("llm.provider", "")
        model = {"ollama": config.get("llm.ollama_text_model"), "claude": config.get("llm.claude_model"),
                 "openai_compatible": config.get("llm.openai_model")}.get(provider, "")
        methods["s2_lyrics"] = f"由 LLM 依歌詞範本產生（目前設定的 LLM：{provider}／{model}）"

    if song:
        mode = "自動長度（唱完為止）" if song.get("duration_mode", duration_mode(inputs, config)) == "auto" \
            else f"長度上限 {song.get('max_seconds', song.get('target_seconds', 0)):g} 秒"
        parts = [f"YuE2（{song.get('ckpt_name', '')}）", mode, f"生成 {len(song.get('attempts', [])) or 1} 次",
                 f"歌曲長度 {song.get('seconds', 0):.1f} 秒", f"seed {song.get('seed')}"]
        if song.get("capped"):
            parts.append("碰到長度上限，結尾淡出")
        methods["s3_song"] = "，".join(parts)

    if align:
        total, missed = align.get("line_count", 0), align.get("unaligned_count", 0)
        methods["s4_subtitle"] = (f"分離人聲後，以 Whisper {config.get('align.model')} 強制對齊歌詞；"
                                  f"{total} 行中 {total - missed} 行對上人聲"
                                  + (f"，{missed} 行以內插補上時間" if missed else "")
                                  + (f"，{align['low_confidence_count']} 行信心偏低" if align.get("low_confidence_count") else ""))

    if plan:
        label = config.get(f"engines.{plan.get('engine')}.label", plan.get("engine", ""))
        anchored = sum(1 for s in plan.get("segments", []) if s.get("end_anchor"))
        methods["s5_video"] = (f"{label}，{plan.get('width')}×{plan.get('height')}、{plan.get('fps'):g} fps，"
                               f"分 {len(plan.get('segments', []))} 段生成後接合"
                               + (f"（其中 {anchored} 段的結尾回到原始照片的樣子）" if anchored else ""))

    style = project.stage("s6_compose").get("style")
    if style:
        methods["s6_compose"] = (f"FFmpeg 燒入字幕；字型 {style.get('font_used', style.get('font'))}，"
                                 f"大小為畫面高度的 {style.get('size_percent'):g}%")
    return methods


def collect(config: AppConfig, project) -> dict:
    """把報告需要的資料整理成一個字典；Markdown 與 HTML 都由它產生。"""
    inputs = project.data.get("inputs", {})
    methods = _methods(config, project)
    stages = []
    for stage_id in STAGE_IDS:
        entry = project.stage(stage_id)
        status = entry.get("status", "pending")
        stages.append({
            "id": stage_id, "label": STAGE_LABELS[stage_id], "status": status,
            "status_label": STATUS_LABELS.get(status, status),
            "method": methods.get(stage_id, "—"),
            "started_at": _when(entry.get("started_at")),
            "seconds": entry.get("seconds") if status == DONE else None,
            "warnings": list(entry.get("warnings", [])), "error": entry.get("error", ""),
        })
    total = sum(s["seconds"] or 0 for s in stages)
    for stage in stages:
        stage["percent"] = round(100 * stage["seconds"] / total, 1) if stage["seconds"] and total else 0.0

    # 對嘴影片各段：分段計畫的時間範圍 + 實際耗時
    plan = _load(project, "analysis", "segment_plan.json")
    spent = {s["index"]: s for s in project.stage("s5_video").get("segments", [])}
    fps = float(plan.get("fps") or 0)
    segments = []
    for item in plan.get("segments", []):
        seconds = spent.get(item["index"], {}).get("seconds")
        length = item["frames"] / fps if fps else 0
        segments.append({"index": item["index"], "t_start": item["t_start"], "t_end": item["t_end"],
                         "frames": item["frames"], "seconds": seconds,
                         "ratio": round(seconds / length, 1) if seconds and length else None})

    video = _load(project, "logs", "summary.json").get("video", {})
    language = language_info(config, inputs.get("language", ""))
    voice = next((v["label"] for v in voice_options(config) if v.get("value", "") == inputs.get("voice_character", "")),
                 inputs.get("voice_character", ""))
    finished = all(s["status"] == DONE for s in stages)
    settings = [
        ("音樂內容", inputs.get("content", "")),
        ("曲風", inputs.get("style", "")),
        ("歌曲語言", language["label"]),
        ("人聲類型", "男聲" if inputs.get("vocal_type") == "male" else "女聲"),
        ("聲線", voice),
        ("歌曲長度", "自動（由 AI 決定，唱完為止）" if duration_mode(inputs, config) == "auto"
            else f"最多 {inputs.get('target_seconds', 0)} 秒"),
        ("歌詞來源", "自行輸入" if inputs.get("lyrics_source") == "manual" else "AI 產生"),
        ("對嘴引擎", config.get(f"engines.{inputs.get('engine')}.label", inputs.get("engine", ""))),
        ("解析度", inputs.get("resolution", "")),
        ("執行模式", "逐步確認" if project.data.get("mode") == "step" else "全自動"),
    ]
    return {
        "name": project.name,
        "created_at": _when(project.data.get("created_at")),
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished": finished,
        "state": "已完成" if finished else "尚未完成",
        "mode": "逐步確認" if project.data.get("mode") == "step" else "全自動",
        "total_seconds": round(total, 1),
        "video": video,
        "speed": round(total / video["seconds"], 1) if video.get("seconds") else None,
        "settings": settings, "stages": stages, "segments": segments, "lyrics": _lyrics(project),
        "prompts": _prompts(project),
        "warnings": [(s["label"], w) for s in stages for w in s["warnings"]],
        "folder": str(project.folder),
        "license": LICENSE_NOTICE,
    }


def _overview(data: dict) -> list[tuple[str, str]]:
    rows = [("專案", data["name"]), ("狀態", data["state"]), ("執行模式", data["mode"]),
            ("總耗時", clock(data["total_seconds"]))]
    video = data["video"]
    if video:
        rows.append(("成品", f"{video['width']}×{video['height']}、{video['fps']:g} fps、長度 {clock(video['seconds'])}"))
    if data["speed"]:
        rows.append(("製作速度", f"每 1 秒影片約花 {data['speed']:g} 秒"))
    rows += [("專案建立時間", data["created_at"]), ("報告產生時間", data["generated_at"]),
             ("專案資料夾", data["folder"])]
    return rows


# ---- Markdown ----

def _cell(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _escape_md(text: str) -> str:
    """歌詞原樣顯示：行首或行內的 Markdown 符號不要被當成格式。"""
    for mark in "\\`*_#<>[]|":
        text = text.replace(mark, "\\" + mark)
    return text


def to_markdown(data: dict) -> str:
    lines = [f"# 《{data['name']}》執行結果報告", ""]
    lines += ["| 項目 | 內容 |", "| --- | --- |"] + [f"| {k} | {_cell(v)} |" for k, v in _overview(data)]
    lines += ["", "## 各階段的執行方式與耗時", "",
              "| 階段 | 狀態 | 執行方式 | 開始時間 | 耗時 | 佔比 |", "| --- | --- | --- | --- | ---: | ---: |"]
    for s in data["stages"]:
        lines.append(f"| {s['label']} | {s['status_label']} | {_cell(s['method'])} | {s['started_at']} | "
                     f"{clock(s['seconds'])} | {s['percent']:g}% |" if s["seconds"] is not None else
                     f"| {s['label']} | {s['status_label']} | {_cell(s['method'])} | {s['started_at']} | — | — |")
    lines += [f"| **合計** | | | | **{clock(data['total_seconds'])}** | 100% |", "", TIMING_NOTE]
    if data["segments"]:
        lines += ["", "## 對嘴影片各段", "",
                  "| 段 | 歌曲時間 | 影格數 | 耗時 | 每 1 秒影片所花秒數 |", "| ---: | --- | ---: | ---: | ---: |"]
        for g in data["segments"]:
            lines.append(f"| {g['index']} | {g['t_start']:.2f}–{g['t_end']:.2f} 秒 | {g['frames']} | "
                         f"{clock(g['seconds'])} | {g['ratio'] if g['ratio'] is not None else '—'} |")
    lines += ["", "## 輸入設定", "", "| 項目 | 內容 |", "| --- | --- |"]
    lines += [f"| {k} | {_cell(v) or '—'} |" for k, v in data["settings"]]
    lines += ["", "## 提示詞", ""]
    for label, text in data["prompts"]:
        lines += [f"**{label}**", "", *[f"> {_escape_md(line)}" for line in text.splitlines()], ""]
    if not data["prompts"]:
        lines += ["還沒有填寫音樂內容。", ""]
    lyrics = data["lyrics"]
    lines += ["## 歌詞", ""]
    if lyrics["sections"]:
        lines += [f"歌名：{lyrics['title'] or '（未命名）'}　共 {lyrics['line_count']} 行"]
        for section in lyrics["sections"]:
            lines.append("")
            if section["label"]:
                lines.append(f"**［{section['label']}］**  ")
            # 行尾兩個空白是 Markdown 的換行
            lines += [f"{_escape_md(line)}  " for line in section["lines"]]
    else:
        lines.append("還沒有歌詞。")
    lines += ["", "## 警告", ""]
    lines += [f"- {label}：{text}" for label, text in data["warnings"]] or ["沒有警告。"]
    failed = [s for s in data["stages"] if s["error"]]
    if failed:
        lines += ["", "## 錯誤", ""] + [f"- {s['label']}：{s['error']}" for s in failed]
    lines += ["", "---", "", data["license"], ""]
    return "\n".join(lines)


# ---- HTML ----

_CSS = """
:root { --bg:#ffffff; --fg:#1f2328; --muted:#656d76; --line:#d8dee4; --card:#f6f8fa; --accent:#2f6fde; --warn:#9a6700; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#16181c; --fg:#e6e8eb; --muted:#9aa4af; --line:#30363d; --card:#1f2329; --accent:#6ea3ff; --warn:#d8a530; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); line-height:1.6;
       font-family:"Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif; }
main { max-width:960px; margin:0 auto; padding:32px 16px 48px; }
h1 { font-size:26px; margin:0 0 4px; }
h2 { font-size:18px; margin:36px 0 12px; }
.sub { color:var(--muted); margin:0 0 24px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.card .k { color:var(--muted); font-size:13px; }
.lyrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
.prompt { margin-bottom:12px; white-space:pre-wrap; }
.card .v { font-size:20px; font-weight:600; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; white-space:nowrap; }
td.n, th.n { text-align:right; white-space:nowrap; font-variant-numeric:tabular-nums; }
tr.total td { font-weight:700; border-top:2px solid var(--line); }
.bar { height:8px; background:var(--line); border-radius:4px; min-width:90px; }
.bar i { display:block; height:100%; background:var(--accent); border-radius:4px; }
.note, footer { color:var(--muted); font-size:13px; }
.warn { color:var(--warn); }
footer { margin-top:40px; border-top:1px solid var(--line); padding-top:12px; }
"""


def to_html(data: dict) -> str:
    e = html.escape
    overview = dict(_overview(data))
    cards = [("狀態", data["state"]), ("總耗時", clock(data["total_seconds"])), ("執行模式", data["mode"])]
    if "成品" in overview:
        cards.append(("成品長度", clock(data["video"]["seconds"])))
    if data["speed"]:
        cards.append(("每 1 秒影片", f"約 {data['speed']:g} 秒"))
    out = ["<!doctype html>", '<html lang="zh-Hant"><head><meta charset="utf-8">',
           '<meta name="viewport" content="width=device-width, initial-scale=1">',
           f"<title>《{e(data['name'])}》執行結果報告</title>", f"<style>{_CSS}</style></head><body><main>",
           f"<h1>《{e(data['name'])}》執行結果報告</h1>",
           f'<p class="sub">報告產生時間 {e(data["generated_at"])}　專案建立時間 {e(data["created_at"])}</p>',
           '<div class="cards">']
    out += [f'<div class="card"><div class="k">{e(k)}</div><div class="v">{e(v)}</div></div>' for k, v in cards]
    out.append("</div>")

    out += ["<h2>各階段的執行方式與耗時</h2>", '<div class="wrap"><table>',
            '<tr><th>階段</th><th>狀態</th><th>執行方式</th><th>開始時間</th><th class="n">耗時</th>'
            '<th class="n">佔比</th><th></th></tr>']
    for s in data["stages"]:
        done = s["seconds"] is not None
        share = f"{s['percent']:g}%" if done else "—"
        out.append(f"<tr><td>{e(s['label'])}</td><td>{e(s['status_label'])}</td><td>{e(s['method'])}</td>"
                   f"<td>{e(s['started_at'])}</td><td class=\"n\">{e(clock(s['seconds']))}</td>"
                   f"<td class=\"n\">{share}</td>"
                   f"<td><div class=\"bar\"><i style=\"width:{s['percent'] if done else 0}%\"></i></div></td></tr>")
    out += [f'<tr class="total"><td>合計</td><td></td><td></td><td></td><td class="n">{e(clock(data["total_seconds"]))}</td>'
            '<td class="n">100%</td><td></td></tr>', "</table></div>", f'<p class="note">{e(TIMING_NOTE)}</p>']

    if data["segments"]:
        out += ["<h2>對嘴影片各段</h2>", '<div class="wrap"><table>',
                '<tr><th class="n">段</th><th>歌曲時間</th><th class="n">影格數</th><th class="n">耗時</th>'
                '<th class="n">每 1 秒影片所花秒數</th></tr>']
        for g in data["segments"]:
            out.append(f'<tr><td class="n">{g["index"]}</td><td>{g["t_start"]:.2f}–{g["t_end"]:.2f} 秒</td>'
                       f'<td class="n">{g["frames"]}</td><td class="n">{e(clock(g["seconds"]))}</td>'
                       f'<td class="n">{g["ratio"] if g["ratio"] is not None else "—"}</td></tr>')
        out.append("</table></div>")

    out += ["<h2>輸入設定</h2>", '<div class="wrap"><table>']
    out += [f"<tr><th>{e(k)}</th><td>{e(str(v)) or '—'}</td></tr>" for k, v in data["settings"]]
    out += [f"<tr><th>專案資料夾</th><td>{e(data['folder'])}</td></tr>", "</table></div>", "<h2>提示詞</h2>"]
    for label, text in data["prompts"]:
        out.append(f'<div class="card prompt"><div class="k">{e(label)}</div>{e(text)}</div>')
    if not data["prompts"]:
        out.append("<p>還沒有填寫音樂內容。</p>")
    out.append("<h2>歌詞</h2>")
    lyrics = data["lyrics"]
    if lyrics["sections"]:
        out.append(f'<p class="note">歌名：{e(lyrics["title"] or "（未命名）")}　共 {lyrics["line_count"]} 行</p>')
        out.append('<div class="lyrics">')
        for section in lyrics["sections"]:
            label = f'<div class="k">{e(section["label"])}</div>' if section["label"] else ""
            out.append(f'<div class="card">{label}{"<br>".join(e(line) for line in section["lines"])}</div>')
        out.append("</div>")
    else:
        out.append("<p>還沒有歌詞。</p>")
    out.append("<h2>警告</h2>")
    if data["warnings"]:
        out.append("<ul>" + "".join(f'<li class="warn">{e(label)}：{e(text)}</li>' for label, text in data["warnings"])
                   + "</ul>")
    else:
        out.append("<p>沒有警告。</p>")
    failed = [s for s in data["stages"] if s["error"]]
    if failed:
        out += ["<h2>錯誤</h2>", "<ul>" + "".join(f"<li>{e(s['label'])}：{e(s['error'])}</li>" for s in failed) + "</ul>"]
    out += [f"<footer>{e(data['license'])}</footer>", "</main></body></html>", ""]
    return "\n".join(out)


def write_report(config: AppConfig, project, kind: str, dest: Path | None = None) -> Path:
    """產生報告並寫檔，回傳檔案路徑。kind 是 md 或 html；沒有指定位置就放在專案資料夾最上層。"""
    data = collect(config, project)
    text = to_html(data) if kind == "html" else to_markdown(data)
    dest = dest or project.folder / f"report.{'html' if kind == 'html' else 'md'}"
    dest.write_text(text, encoding="utf-8")
    return dest
