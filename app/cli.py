"""命令列工具：不開介面就能做環境檢查與測試工作流，方便除錯。

用法：
    python -m app.cli env
    python -m app.cli test-workflow [輸出資料夾]
"""
from __future__ import annotations

import sys
from pathlib import Path

from app.core.models import DEGRADE, FAIL, PASS, UserFacingError
from app.pipeline.stage_base import Reporter, StageContext
from app.pipeline.stages.s1_env import EnvStage
from app.pipeline.test_run import run_yue2_test
from app.storage.app_config import AppConfig

_MARKS = {PASS: "✓", DEGRADE: "⚠", FAIL: "✗"}


def _reporter() -> Reporter:
    return Reporter(on_log=print, on_progress=lambda f, t: print(f"  [{f * 100:3.0f}%] {t}"))


def cmd_env(config: AppConfig) -> int:
    stage = EnvStage()
    result = stage.run(StageContext(config), Reporter())
    for item in stage.report["items"]:
        print(f"{_MARKS[item['level']]} {item['label']}：{item['detail']}")
        if item["fix"]:
            for line in item["fix"].splitlines():
                print(f"    → {line}")
    print(f"\n總結：{'可以開始製作' if result.ok else result.message}")
    return 0 if result.ok else 1


def cmd_test_workflow(config: AppConfig, dest: Path) -> int:
    stage = EnvStage()
    stage.run(StageContext(config), Reporter())
    url = stage.report["comfy_url"]
    if not url:
        print("✗ 無法連線到 ComfyUI，請先啟動 ComfyUI Desktop")
        return 1
    path = run_yue2_test(config, url, dest, _reporter())
    print(f"✓ 測試完成：{path}")
    return 0


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    config = AppConfig()
    try:
        if argv[:1] == ["env"]:
            return cmd_env(config)
        if argv[:1] == ["test-workflow"]:
            return cmd_test_workflow(config, Path(argv[1]) if len(argv) > 1 else Path("test_output"))
    except UserFacingError as e:
        print(f"✗ {e.message}")
        if e.fix:
            print(f"    → {e.fix}")
        return 1
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
