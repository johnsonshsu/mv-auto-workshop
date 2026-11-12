"""階段 7：完成與匯出。把交付檔案放到專案資料夾最上層，並寫出各階段耗時與警告摘要。"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

from app.core.models import STAGE_IDS, STAGE_LABELS, Problem
from app.pipeline.stage_base import Reporter, StageContext, StageResult
from app.services import ffmpeg_service as F

LICENSE_NOTICE = "歌曲由 YuE2-3B 產生，該模型為 CC-BY-NC-4.0 研究授權，產出僅供非商業用途。"

# 交付檔名 → 專案內的來源
DELIVERABLES = {
    "final_mv.mp4": "output/final_mv.mp4",
    "video_nosub.mp4": "output/video_nosub.mp4",
    "song.flac": "audio/song.flac",
    "lyrics.txt": "lyrics/lyrics.txt",
    "lyrics.srt": "lyrics/lyrics.srt",
}


def place(source: Path, dest: Path) -> None:
    """放一份到 dest：能用硬連結就不佔額外空間，不行（例如跨磁碟）就複製。"""
    dest.unlink(missing_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copyfile(source, dest)


def build_summary(project) -> dict:
    stages = []
    for stage_id in STAGE_IDS:
        entry = project.stage(stage_id)
        stages.append({"id": stage_id, "label": STAGE_LABELS[stage_id], "status": entry.get("status"),
                       "seconds": entry.get("seconds"), "warnings": entry.get("warnings", [])})
    return {
        "name": project.name,
        "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_seconds": round(sum(s["seconds"] or 0 for s in stages), 1),
        "stages": stages,
        "license": LICENSE_NOTICE,
    }


class ExportStage:
    id = "s7_export"

    def check_inputs(self, ctx: StageContext) -> list[Problem]:
        missing = [name for name, source in DELIVERABLES.items()
                   if not ctx.project.path(*source.split("/")).exists()]
        return [Problem(f"缺少要匯出的檔案：{'、'.join(missing)}", "請先完成前面的階段")] if missing else []

    def outputs(self, ctx: StageContext) -> list[Path]:
        return [ctx.project.folder / name for name in DELIVERABLES]

    def run(self, ctx: StageContext, reporter: Reporter) -> StageResult:
        project = ctx.project
        for index, (name, source) in enumerate(DELIVERABLES.items()):
            reporter.progress(index / len(DELIVERABLES), f"匯出 {name}")
            place(project.path(*source.split("/")), project.folder / name)
        info = F.probe_video(project.folder / "final_mv.mp4")
        summary = build_summary(project)
        summary["video"] = {"width": info["width"], "height": info["height"], "fps": info["fps"],
                            "frames": info["frames"], "seconds": round(info["frames"] / info["fps"], 3)}
        project.path("logs", "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
        reporter.log(f"已匯出到 {project.folder}：{'、'.join(DELIVERABLES)}")
        reporter.log(LICENSE_NOTICE)
        reporter.progress(1.0, "完成")
        return StageResult(True)

    resume = run
