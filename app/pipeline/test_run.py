"""測試工作流：送出一段很短的 YuE2 生成，確認「注入 → 送出 → 等待 → 下載」整條路是通的。"""
from __future__ import annotations

import random
import threading
from pathlib import Path

from app.core.models import UserFacingError
from app.pipeline.stage_base import Reporter
from app.services.comfy_client import ComfyClient, Progress
from app.services.workflow_template import WorkflowTemplate
from app.storage.app_config import AppConfig

TEST_STYLE = "Mandarin, warm female vocal, contemporary pop, 96 BPM, piano, soft drums"
TEST_LYRICS = "[Verse]\n晚風輕輕吹過窗台\n我在這裡等你回來"


def run_yue2_test(config: AppConfig, comfy_url: str, dest_dir: Path,
                  reporter: Reporter | None = None, cancel: threading.Event | None = None) -> Path:
    """回傳下載回來的音檔路徑。"""
    reporter = reporter or Reporter()
    template = WorkflowTemplate.load(config.workflows_dir, config.get("yue2.workflow"))
    client = ComfyClient(comfy_url, ws_poll_seconds=config.get("comfyui.ws_poll_seconds", 5.0))
    try:
        problems = template.check_against(client.object_info())
        if problems:
            raise UserFacingError("工作流範本與本機 ComfyUI 不相容：" + "；".join(problems),
                                  "請更新 ComfyUI，或更新 workflows 資料夾內的範本與對應表")
        seconds = float(config.get("yue2.test_seconds", 10))
        workflow = template.inject({
            "ckpt_name": config.get("yue2.ckpt_name"),
            "style": TEST_STYLE,
            "lyrics": TEST_LYRICS,
            "plan_mode": config.get("yue2.plan_mode", "full"),
            "seed": random.randint(0, 2**31 - 1),
            "max_duration": seconds,
            "filename_prefix": f"{config.get('comfyui.output_prefix')}/test",
        })
        reporter.log(f"送出測試工作流（YuE2，最長 {seconds:.0f} 秒）到 {comfy_url}")

        def on_progress(p: Progress) -> None:
            if p.max:
                reporter.progress(p.value / p.max, f"{p.node_type} {p.value}/{p.max}")
            else:
                reporter.progress(0.0, f"執行節點 {p.node_id}：{p.node_type}")

        entry = client.run(workflow, on_progress, cancel, log_dir=dest_dir, label="test_yue2")
        files = client.output_files(entry, template.output_node("audio"))
        if not files:
            raise UserFacingError("工作流執行完成，但沒有產生音檔", "請到 ComfyUI 介面查看是否有錯誤訊息")
        dest = dest_dir / ("test_yue2" + Path(files[0]["filename"]).suffix)
        client.download(files[0], dest)
        reporter.log(f"已取回音檔：{dest}")
        client.free()
        reporter.log("已請 ComfyUI 釋放模型")
        return dest
    finally:
        client.close()
