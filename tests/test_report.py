"""執行結果報告：資料整理、Markdown 與 HTML 輸出。"""
import json

from app.pipeline import report as R
from tests.test_s3_song_and_runner import project, server, server_info  # noqa: F401  （fixture）


def fill(project) -> None:
    """一個跑完的專案：七個階段都有耗時，歌曲、對齊、分段各自留下記錄。"""
    project.data["inputs"].update(content="海邊 <b>日落</b> | 散步", style="抒情流行", engine="ltx23",
                                  resolution="480p", duration_mode="auto", lyrics_source="ai")
    seconds = {"s1_env": 1.0, "s2_lyrics": 9.0, "s3_song": 180.0, "s4_subtitle": 40.0, "s5_video": 3750.0,
               "s6_compose": 15.0, "s7_export": 5.0}
    for stage_id, spent in seconds.items():
        project.stage(stage_id).update(status="done", seconds=spent, started_at="2026-11-11T08:00:00+08:00")
    project.stage("s4_subtitle")["warnings"] = ["第 3 行的時間是內插的"]
    project.stage("s5_video")["segments"] = [{"index": 1, "status": "done", "seconds": 60.0, "frames": 240},
                                             {"index": 2, "status": "done", "seconds": 30.0, "frames": 240}]
    project.stage("s6_compose")["style"] = {"font": "標楷體", "font_used": "標楷體", "size_percent": 9.0}
    project.path("audio", "yue2_prompt.json").write_text(json.dumps(
        {"ckpt_name": "yue2.safetensors", "seed": 7, "seconds": 20.0, "duration_mode": "auto",
         "attempts": [{}, {}]}), encoding="utf-8")
    project.path("analysis", "alignment_report.json").write_text(json.dumps(
        {"line_count": 10, "unaligned_count": 1, "low_confidence_count": 2}), encoding="utf-8")
    project.path("analysis", "segment_plan.json").write_text(json.dumps(
        {"engine": "ltx23", "fps": 24.0, "width": 896, "height": 512, "segments": [
            {"index": 1, "t_start": 0.0, "t_end": 10.0, "frames": 240, "end_anchor": True},
            {"index": 2, "t_start": 10.0, "t_end": 20.0, "frames": 240}]}), encoding="utf-8")
    project.path("logs").mkdir(exist_ok=True)
    project.path("logs", "summary.json").write_text(json.dumps(
        {"video": {"width": 896, "height": 512, "fps": 24.0, "frames": 480, "seconds": 20.0}}), encoding="utf-8")


def test_clock_formats():
    assert [R.clock(v) for v in (None, 1.26, 75, 3600 + 61)] == ["—", "1.3 秒", "1 分 15 秒", "1 小時 01 分 01 秒"]


def test_collect_sums_time_and_describes_each_stage(config, project):
    fill(project)
    data = R.collect(config, project)
    assert data["finished"] and data["total_seconds"] == 4000.0 and data["speed"] == 200.0
    stages = {s["id"]: s for s in data["stages"]}
    assert stages["s5_video"]["percent"] == 93.8 and stages["s1_env"]["percent"] == 0.0
    assert "生成 2 次" in stages["s3_song"]["method"] and "自動長度" in stages["s3_song"]["method"]
    assert "10 行中 9 行對上人聲，1 行以內插補上時間，2 行信心偏低" in stages["s4_subtitle"]["method"]
    assert "LTX-2.3，896×512、24 fps，分 2 段" in stages["s5_video"]["method"] and "其中 1 段" in stages["s5_video"]["method"]
    assert "標楷體" in stages["s6_compose"]["method"]
    assert [(g["seconds"], g["ratio"]) for g in data["segments"]] == [(60.0, 6.0), (30.0, 3.0)]
    assert data["warnings"] == [("字幕對齊", "第 3 行的時間是內插的")]


def test_unfinished_project_still_reports(config, project):
    project.stage("s1_env").update(status="done", seconds=2.0)
    project.stage("s2_lyrics").update(status="failed", error="LLM 沒有回應")
    data = R.collect(config, project)
    assert not data["finished"] and data["state"] == "尚未完成" and data["total_seconds"] == 2.0
    text = R.to_markdown(data)
    assert "| 歌詞產生 | 失敗 |" in text and "- 歌詞產生：LLM 沒有回應" in text and "沒有警告。" in text
    assert "對嘴影片各段" not in text                      # 還沒有分段計畫就不列這一節


def test_markdown_and_html_outputs(config, project, tmp_path):
    fill(project)
    md = R.write_report(config, project, "md")
    assert md == project.folder / "report.md"
    text = md.read_text(encoding="utf-8")
    assert text.startswith(f"# 《{project.name}》執行結果報告")
    assert "| **合計** | | | | **1 小時 06 分 40 秒** | 100% |" in text
    assert "| 對嘴影片 | 完成 |" in text and "1 小時 02 分 30 秒 | 93.8% |" in text
    assert "海邊 <b>日落</b> \\| 散步" in text              # 表格裡的直線要跳脫
    assert "- 字幕對齊：第 3 行的時間是內插的" in text and "CC-BY-NC-4.0" in text

    page = R.write_report(config, project, "html", tmp_path / "報告.html").read_text(encoding="utf-8")
    assert page.startswith("<!doctype html>") and '<meta charset="utf-8">' in page
    assert "海邊 &lt;b&gt;日落&lt;/b&gt; | 散步" in page and "<b>日落</b>" not in page     # 使用者輸入要跳脫
    assert "1 小時 06 分 40 秒" in page and 'style="width:93.8%"' in page and "CC-BY-NC-4.0" in page


def test_report_includes_this_runs_lyrics(config, project):
    assert "還沒有歌詞。" in R.to_markdown(R.collect(config, project))
    project.path("lyrics", "lyrics.json").write_text(json.dumps({"title": "海風", "sections": [
        {"type": "verse", "lines": ["巷口的燈亮起", "*晚風* <吹過>"]},
        {"type": "chorus", "lines": ["讓風帶我走"]}]}, ensure_ascii=False), encoding="utf-8")
    data = R.collect(config, project)
    assert data["lyrics"]["line_count"] == 3
    text = R.to_markdown(data)
    assert "歌名：海風　共 3 行" in text and "**［主歌］**" in text and "**［副歌］**" in text
    assert "巷口的燈亮起  \n" in text and "\\*晚風\\* \\<吹過\\>" in text       # 每行各自換行；符號不被當成格式
    page = R.to_html(data)
    assert "<h2>歌詞</h2>" in page and "巷口的燈亮起<br>*晚風* &lt;吹過&gt;" in page

    # 只有純文字歌詞（例如自行輸入）時也列得出來
    project.path("lyrics", "lyrics.json").unlink()
    project.path("lyrics", "lyrics.txt").write_text("第一行\n\n第二行\n", encoding="utf-8")
    assert R.collect(config, project)["lyrics"] == {"title": "", "sections": [{"label": "", "lines": ["第一行", "第二行"]}],
                                                    "line_count": 2}


def test_report_includes_the_original_prompt_and_generated_prompts(config, project):
    project.data["inputs"].update(content="", style="")
    assert "還沒有填寫音樂內容。" in R.to_markdown(R.collect(config, project))
    fill(project)
    project.path("audio", "yue2_prompt.json").write_text(json.dumps(
        {"style": "Mandarin, warm female vocal, pop ballad", "attempts": [{}]}), encoding="utf-8")
    project.path("analysis", "video_prompt.json").write_text(json.dumps({"positive": "A singer sings."}), encoding="utf-8")
    data = R.collect(config, project)
    assert [label for label, _ in data["prompts"]] == [
        "音樂內容（一開始輸入的提示詞）", "曲風", "AI 寫的歌曲風格描述（送給 YuE2）", "AI 寫的影像提示詞（送給對嘴引擎）"]
    text = R.to_markdown(data)
    assert "## 提示詞" in text and text.index("## 提示詞") < text.index("## 歌詞")
    assert "**音樂內容（一開始輸入的提示詞）**" in text and "> 海邊 \\<b\\>日落\\</b\\> \\| 散步" in text
    assert "> Mandarin, warm female vocal, pop ballad" in text and "> A singer sings." in text
    page = R.to_html(data)
    assert "<h2>提示詞</h2>" in page and "海邊 &lt;b&gt;日落&lt;/b&gt; | 散步</div>" in page


def test_segment_hint_tells_how_long_each_segment_is(config, project):
    from app.ui.pages.input_page import segment_hint
    assert segment_hint(config, None, "ltx23") == \
        "每段最長 10 秒（切點落在換氣或段落交界處，實際會短一些）。以 150 秒的歌估算，大約分成 15 到 24 段。"
    assert "每段最長 60 秒" in segment_hint(config, None, "infinitetalk") and "3 到 4 段" in segment_hint(config, None, "infinitetalk")
    assert "每段最長 30 秒" in segment_hint(config, None, "wan22_s2v")
    assert segment_hint(config, None, "沒有這個引擎") == ""

    # 有歌曲：用實際長度估；已經用同一個引擎分過段：直接顯示實際段數
    import numpy as np
    import soundfile
    soundfile.write(str(project.path("audio", "song.flac")), np.zeros(8000 * 60, dtype="float32"), 8000)
    assert "這首歌 60 秒，大約分成 6 到 10 段" in segment_hint(config, project, "ltx23")
    project.path("analysis", "segment_plan.json").write_text(json.dumps(
        {"engine": "ltx23", "duration": 60.0, "segments": [{}] * 8}), encoding="utf-8")
    assert "上次用這個引擎分成 8 段（平均每段 7.5 秒），有 7 個交界" in segment_hint(config, project, "ltx23")
    assert "大約分成 1 到 2 段" in segment_hint(config, project, "infinitetalk")     # 別的引擎的分段計畫不算


def test_time_hint_estimates_from_reference_speed_or_the_projects_own_run(config, project):
    from app.ui.pages.input_page import time_hint
    assert time_hint(config, None, "infinitetalk", "480p") == \
        "預估時間：30 秒的影片約 47 分鐘（只算對嘴影片階段，依 RTX 5090 Laptop 的實測值估算）。"
    assert "30 秒的影片約 3 分鐘" in time_hint(config, None, "ltx23", "480p")
    assert "30 秒的影片約 4 分鐘" in time_hint(config, None, "ltx23", "720p")
    assert "30 秒的影片約 10 分鐘" in time_hint(config, None, "wan22_s2v", "480p")
    assert time_hint(config, None, "wan22_s2v", "720p") == "預估時間：這個引擎的 720p 沒有實測過，無法估算。"

    import numpy as np
    import soundfile
    soundfile.write(str(project.path("audio", "song.flac")), np.zeros(8000 * 150, dtype="float32"), 8000)
    assert "這首歌 150 秒約 3 小時 55 分" in time_hint(config, project, "infinitetalk", "480p")
    # 這個專案用 LTX-2.3 480p 生成過：10 秒影片花了 100 秒，就用每秒 10 秒來估，不用參考值
    project.path("analysis", "segment_plan.json").write_text(json.dumps(
        {"engine": "ltx23", "fps": 24.0, "width": 896, "height": 512, "duration": 150.0, "segments": [{}]}), encoding="utf-8")
    project.stage("s5_video")["segments"] = [{"index": 1, "seconds": 100.0, "frames": 240}]
    hint = time_hint(config, project, "ltx23", "480p")
    assert "30 秒的影片約 5 分鐘；這首歌 150 秒約 25 分鐘" in hint and "依這個專案上次生成的實際速度估算" in hint
    assert "依 RTX 5090 Laptop 的實測值估算" in time_hint(config, project, "ltx23", "720p")     # 尺寸不同就不能沿用
